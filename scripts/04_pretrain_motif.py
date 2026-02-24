from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ModuleNotFoundError:
    tqdm = None
    TQDM_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checks import assert_ho_train_only  # noqa: E402
from src.data import MotifPretrainBundle, prepare_motif_pretrain_data  # noqa: E402
from src.model_rgcn import BaseRGCNPairModel, ModelConfig  # noqa: E402
from src.motif import (  # noqa: E402
    MOTIF_COMPONENTS,
    MotifScorer,
    sample_corrupted_quadruplets,
)

_COMPONENT_TO_KEY = {
    "drug": "drug_index",
    "protein": "protein_index",
    "pathway": "pathway_index",
    "disease": "disease_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Motif-pattern self-supervised pretraining on HO_train only "
            "with single-component corruption."
        )
    )
    parser.add_argument(
        "--kg-edges",
        default="data/KG",
        help=(
            "KG edges CSV/TSV path or a directory containing KG CSV files. "
            "Current project default: data/KG"
        ),
    )
    parser.add_argument(
        "--node-types",
        default="data/KG/nodes.csv",
        help=(
            "Node-type mapping CSV/TSV path. Supports id,type (current project) "
            "and node_id,node_type. Default: data/KG/nodes.csv"
        ),
    )
    parser.add_argument(
        "--split-dir",
        default="outputs/splits/random",
        help=(
            "Directory containing split files. This pretrainer reads only "
            "kg_pos_train.csv and ho_train.csv."
        ),
    )
    parser.add_argument("--indication-relation", default="indication")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ho-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--motif-hidden-dim", type=int, default=128)
    parser.add_argument("--num-corruptions", type=int, default=4)
    parser.add_argument(
        "--corruption-modes",
        nargs="+",
        default=["protein", "pathway"],
        choices=list(MOTIF_COMPONENTS),
        help=(
            "Allowed single-component corruption modes. "
            "Default applies Scheme-A: protein/pathway only."
        ),
    )
    parser.add_argument(
        "--mode-sampling",
        default="balanced",
        choices=["balanced", "uniform"],
        help=(
            "How to sample corruption modes per batch. "
            "'balanced' keeps mode usage nearly 50/50 when two modes are used."
        ),
    )
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--balance-key",
        default="drug",
        choices=["drug", "disease"],
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or cuda[:index].",
    )
    parser.add_argument(
        "--save-ckpt",
        default="outputs/pretrain/motif_pretrain.pt",
        help="Path to save pretrained encoder + motif scorer weights.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON path for final pretraining summary.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def motif_infonce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be > 0 for motif InfoNCE.")
    if pos_scores.dim() != 1:
        raise ValueError(f"Expected pos_scores shape [B], got {tuple(pos_scores.shape)}")
    if neg_scores.dim() != 2:
        raise ValueError(f"Expected neg_scores shape [B, C], got {tuple(neg_scores.shape)}")
    if neg_scores.size(0) != pos_scores.size(0):
        raise ValueError(
            "Positive/negative score batch mismatch: "
            f"pos={tuple(pos_scores.shape)}, neg={tuple(neg_scores.shape)}"
        )

    logits = torch.cat((pos_scores.unsqueeze(1), neg_scores), dim=1) / tau
    targets = torch.zeros(pos_scores.size(0), dtype=torch.long, device=pos_scores.device)
    return F.cross_entropy(logits, targets)


def _to_ho_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    drug_index, protein_index, pathway_index, disease_index = batch
    return {
        "drug_index": drug_index.to(device),
        "protein_index": protein_index.to(device),
        "pathway_index": pathway_index.to(device),
        "disease_index": disease_index.to(device),
    }


def _score_corrupted(
    motif_scorer: MotifScorer,
    node_embeddings: torch.Tensor,
    corrupted_batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    batch_size, num_corruptions = corrupted_batch["drug_index"].shape
    flat_scores = motif_scorer.score(
        node_embeddings=node_embeddings,
        drug_index=corrupted_batch["drug_index"].reshape(-1),
        protein_index=corrupted_batch["protein_index"].reshape(-1),
        pathway_index=corrupted_batch["pathway_index"].reshape(-1),
        disease_index=corrupted_batch["disease_index"].reshape(-1),
    )
    return flat_scores.reshape(batch_size, num_corruptions)


def _coverage_stats(
    seen_true: Mapping[str, set[int]],
    seen_total: Mapping[str, set[int]],
    pool_sizes: Mapping[str, int],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for component in MOTIF_COMPONENTS:
        total = float(pool_sizes[component])
        true_count = len(seen_true[component])
        seen_count = len(seen_total[component])
        stats[component] = {
            "unique_true": float(true_count),
            "true_coverage": float(true_count / total if total > 0 else 0.0),
            "unique_seen": float(seen_count),
            "seen_coverage": float(seen_count / total if total > 0 else 0.0),
        }
    return stats


def _extract_encoder_state_dict(model: BaseRGCNPairModel) -> Dict[str, torch.Tensor]:
    encoder_prefixes = ("node_embedding.", "layers.")
    state = model.state_dict()
    extracted: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.startswith(encoder_prefixes):
            extracted[key] = tensor.detach().cpu().clone()
    if not extracted:
        raise ValueError("No encoder weights were extracted for checkpointing.")
    return extracted


def train_one_epoch(
    encoder_model: BaseRGCNPairModel,
    motif_scorer: MotifScorer,
    bundle: MotifPretrainBundle,
    ho_loader,
    candidate_pools: Mapping[str, torch.LongTensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tau: float,
    num_corruptions: int,
    corruption_modes: Sequence[str],
    mode_sampling: str,
    rng: torch.Generator,
    epoch: int,
    total_epochs: int,
    use_tqdm: bool,
) -> Dict[str, object]:
    encoder_model.train()
    motif_scorer.train()

    edge_index = bundle.graph.edge_index.to(device)
    edge_type = bundle.graph.edge_type.to(device)
    pool_sizes = {name: int(pool.numel()) for name, pool in candidate_pools.items()}

    total_loss = 0.0
    total_count = 0
    mode_counts = {mode: 0 for mode in corruption_modes}
    unchanged_count = 0
    seen_true = {component: set() for component in MOTIF_COMPONENTS}
    seen_total = {component: set() for component in MOTIF_COMPONENTS}

    batch_iter = ho_loader
    if use_tqdm:
        batch_iter = tqdm(
            ho_loader,
            desc=f"Epoch {epoch}/{total_epochs} pretrain",
            leave=False,
            dynamic_ncols=True,
        )

    for batch in batch_iter:
        ho_batch_cpu = _to_ho_batch(batch=batch, device=torch.device("cpu"))
        corruption = sample_corrupted_quadruplets(
            ho_batch=ho_batch_cpu,
            candidate_indices=candidate_pools,
            num_corruptions=num_corruptions,
            generator=rng,
            corruption_modes=corruption_modes,
            mode_sampling=mode_sampling,
        )
        ho_batch = {key: value.to(device) for key, value in ho_batch_cpu.items()}
        corrupted_batch = {key: value.to(device) for key, value in corruption.corrupted.items()}

        optimizer.zero_grad(set_to_none=True)
        node_embeddings = encoder_model.encode(edge_index=edge_index, edge_type=edge_type)
        pos_scores = motif_scorer.score_from_batch(node_embeddings=node_embeddings, quad_batch=ho_batch)
        neg_scores = _score_corrupted(
            motif_scorer=motif_scorer,
            node_embeddings=node_embeddings,
            corrupted_batch=corrupted_batch,
        )
        loss = motif_infonce_loss(pos_scores=pos_scores, neg_scores=neg_scores, tau=tau)
        loss.backward()
        optimizer.step()

        batch_size = int(pos_scores.size(0))
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        unchanged_count += int(corruption.unchanged_count)
        for mode, count in corruption.mode_counts.items():
            mode_counts[mode] += int(count)

        for component, key in _COMPONENT_TO_KEY.items():
            true_values = ho_batch_cpu[key].tolist()
            seen_true[component].update(int(x) for x in true_values)
            seen_total[component].update(int(x) for x in true_values)
            corrupt_values = corruption.corrupted[key].reshape(-1).tolist()
            seen_total[component].update(int(x) for x in corrupt_values)

        if use_tqdm:
            mean_loss = total_loss / max(total_count, 1)
            batch_iter.set_postfix(loss=f"{mean_loss:.4f}")

    if total_count == 0:
        raise ValueError("HO pretrain loader is empty.")

    total_corruptions = int(sum(mode_counts.values()))
    return {
        "loss": total_loss / total_count,
        "num_quads": float(total_count),
        "num_corruptions": float(total_corruptions),
        "mode_sampling": mode_sampling,
        "corruption_modes": {mode: float(mode_counts[mode]) for mode in corruption_modes},
        "unchanged_corruptions": float(unchanged_count),
        "unchanged_rate": float(unchanged_count / total_corruptions if total_corruptions else 0.0),
        "coverage": _coverage_stats(
            seen_true=seen_true,
            seen_total=seen_total,
            pool_sizes=pool_sizes,
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_tqdm = (not args.no_tqdm) and TQDM_AVAILABLE
    if (not args.no_tqdm) and (not TQDM_AVAILABLE):
        print(
            "tqdm is not installed; running without progress bars. "
            "Install it with: pip install tqdm (or pip install -r requirements.txt)."
        )

    assert_ho_train_only(("train",))
    bundle = prepare_motif_pretrain_data(
        node_types_path=args.node_types,
        kg_edges_path=args.kg_edges,
        split_dir=args.split_dir,
        indication_relation=args.indication_relation,
        keep_only_train_indication=True,
    )
    ho_loader = bundle.make_ho_pretrain_loader(
        batch_size=args.ho_batch_size,
        seed=args.seed,
        balance_key=args.balance_key,
        num_workers=args.num_workers,
    )
    candidate_pools = bundle.get_corruption_node_pools()

    config = ModelConfig(
        num_nodes=bundle.graph.num_nodes,
        num_relations=bundle.graph.num_relations,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pair_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    encoder_model = BaseRGCNPairModel(config).to(device)
    motif_scorer = MotifScorer(
        hidden_dim=args.hidden_dim,
        mlp_hidden_dim=args.motif_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    trainable_params = (
        list(encoder_model.node_embedding.parameters())
        + list(encoder_model.layers.parameters())
        + list(motif_scorer.parameters())
    )
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed + 999)

    history: list[Dict[str, object]] = []
    best_epoch = 0
    best_loss = float("inf")
    best_encoder_state = _extract_encoder_state_dict(encoder_model)
    best_motif_state = {
        key: value.detach().cpu().clone()
        for key, value in motif_scorer.state_dict().items()
    }

    epoch_iter = range(1, args.epochs + 1)
    if use_tqdm:
        epoch_iter = tqdm(
            epoch_iter,
            desc="Pretrain epochs",
            dynamic_ncols=True,
        )

    for epoch in epoch_iter:
        ho_loader.set_epoch(epoch - 1)
        stats = train_one_epoch(
            encoder_model=encoder_model,
            motif_scorer=motif_scorer,
            bundle=bundle,
            ho_loader=ho_loader,
            candidate_pools=candidate_pools,
            optimizer=optimizer,
            device=device,
            tau=args.tau,
            num_corruptions=args.num_corruptions,
            corruption_modes=args.corruption_modes,
            mode_sampling=args.mode_sampling,
            rng=rng,
            epoch=epoch,
            total_epochs=args.epochs,
            use_tqdm=use_tqdm,
        )
        record = {
            "epoch": float(epoch),
            "loss": stats["loss"],
            "num_quads": stats["num_quads"],
            "num_corruptions": stats["num_corruptions"],
            "mode_sampling": stats["mode_sampling"],
            "corruption_modes": stats["corruption_modes"],
            "unchanged_corruptions": stats["unchanged_corruptions"],
            "unchanged_rate": stats["unchanged_rate"],
            "coverage": stats["coverage"],
        }
        history.append(record)
        if use_tqdm:
            tqdm.write(json.dumps(record))
            epoch_iter.set_postfix(loss=f"{float(stats['loss']):.4f}")
        else:
            print(json.dumps(record))

        current_loss = float(stats["loss"])
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_encoder_state = _extract_encoder_state_dict(encoder_model)
            best_motif_state = {
                key: value.detach().cpu().clone()
                for key, value in motif_scorer.state_dict().items()
            }

    ckpt_path = Path(args.save_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder_state_dict": best_encoder_state,
        "motif_scorer_state_dict": best_motif_state,
        "model_config": asdict(config),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "history": history,
    }
    torch.save(checkpoint, ckpt_path)

    summary = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "mode_sampling": args.mode_sampling,
        "corruption_modes": list(args.corruption_modes),
        "split_dir": args.split_dir,
        "save_ckpt": str(ckpt_path),
        "ho_train_size": bundle.ho_train.total,
        "graph": {
            "num_nodes": bundle.graph.num_nodes,
            "num_edges": int(bundle.graph.edge_index.size(1)),
            "num_relations": bundle.graph.num_relations,
        },
        "history": history,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
