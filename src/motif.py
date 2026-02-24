from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

MOTIF_COMPONENTS = ("drug", "protein", "pathway", "disease")
_COMPONENT_TO_KEY = {
    "drug": "drug_index",
    "protein": "protein_index",
    "pathway": "pathway_index",
    "disease": "disease_index",
}


class MotifScorer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim
        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")

        self.hidden_dim = int(hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def score(
        self,
        node_embeddings: torch.Tensor,
        drug_index: torch.LongTensor,
        protein_index: torch.LongTensor,
        pathway_index: torch.LongTensor,
        disease_index: torch.LongTensor,
    ) -> torch.Tensor:
        drug_h = node_embeddings[drug_index]
        protein_h = node_embeddings[protein_index]
        pathway_h = node_embeddings[pathway_index]
        disease_h = node_embeddings[disease_index]
        features = torch.cat((drug_h, protein_h, pathway_h, disease_h), dim=-1)
        return self.scorer(features).squeeze(-1)

    def score_from_batch(
        self,
        node_embeddings: torch.Tensor,
        quad_batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.score(
            node_embeddings=node_embeddings,
            drug_index=quad_batch["drug_index"],
            protein_index=quad_batch["protein_index"],
            pathway_index=quad_batch["pathway_index"],
            disease_index=quad_batch["disease_index"],
        )


@dataclass(frozen=True)
class CorruptionResult:
    corrupted: Dict[str, torch.LongTensor]
    mode_index: torch.LongTensor
    mode_counts: Dict[str, int]
    unchanged_count: int


def sample_corrupted_quadruplets(
    ho_batch: Mapping[str, torch.Tensor],
    candidate_indices: Mapping[str, torch.LongTensor],
    num_corruptions: int,
    generator: torch.Generator,
    corruption_modes: Sequence[str] = MOTIF_COMPONENTS,
    mode_sampling: str = "uniform",
    max_resample_attempts: int = 8,
) -> CorruptionResult:
    if num_corruptions <= 0:
        raise ValueError("num_corruptions must be > 0")
    if not corruption_modes:
        raise ValueError("corruption_modes must be non-empty")
    if mode_sampling not in {"uniform", "balanced"}:
        raise ValueError("mode_sampling must be one of {'uniform', 'balanced'}.")

    for mode in corruption_modes:
        if mode not in _COMPONENT_TO_KEY:
            raise ValueError(f"Unknown corruption mode '{mode}'.")
        if mode not in candidate_indices:
            raise ValueError(f"Missing candidate node pool for mode '{mode}'.")

    reference = ho_batch["drug_index"]
    if reference.dim() != 1:
        raise ValueError(
            f"Expected ho_batch tensors to be rank-1; got shape={tuple(reference.shape)}"
        )
    batch_size = reference.size(0)
    if batch_size == 0:
        raise ValueError("Cannot corrupt an empty HO batch.")

    for key in _COMPONENT_TO_KEY.values():
        tensor = ho_batch[key]
        if tensor.dim() != 1 or tensor.size(0) != batch_size:
            raise ValueError(
                f"Expected {key} to have shape ({batch_size},), got {tuple(tensor.shape)}"
            )

    mode_count = len(corruption_modes)
    if mode_sampling == "uniform":
        mode_index_cpu = torch.randint(
            low=0,
            high=mode_count,
            size=(batch_size, num_corruptions),
            generator=generator,
            dtype=torch.long,
        )
    else:
        total_slots = batch_size * num_corruptions
        repeated_modes = torch.arange(total_slots, dtype=torch.long) % mode_count
        shuffle_index = torch.randperm(total_slots, generator=generator)
        mode_index_cpu = repeated_modes[shuffle_index].reshape(batch_size, num_corruptions)

    expanded_cpu: Dict[str, torch.LongTensor] = {}
    for key in _COMPONENT_TO_KEY.values():
        expanded_cpu[key] = ho_batch[key].detach().cpu().unsqueeze(1).repeat(1, num_corruptions)

    mode_counts = {mode: 0 for mode in corruption_modes}
    unchanged_count = 0
    for mode_id, mode in enumerate(corruption_modes):
        mask = mode_index_cpu == mode_id
        n_replace = int(mask.sum().item())
        if n_replace == 0:
            continue
        mode_counts[mode] = n_replace

        key = _COMPONENT_TO_KEY[mode]
        pool = candidate_indices[mode].detach().cpu()
        if pool.numel() == 0:
            raise ValueError(f"Candidate pool for mode '{mode}' is empty.")

        sampled_pool_idx = torch.randint(
            low=0,
            high=pool.numel(),
            size=(n_replace,),
            generator=generator,
            dtype=torch.long,
        )
        replacements = pool[sampled_pool_idx]
        originals = expanded_cpu[key][mask]

        if pool.numel() > 1:
            same_as_original = replacements == originals
            attempts = 0
            while bool(same_as_original.any()) and attempts < max_resample_attempts:
                reroll_idx = torch.randint(
                    low=0,
                    high=pool.numel(),
                    size=(int(same_as_original.sum().item()),),
                    generator=generator,
                    dtype=torch.long,
                )
                replacements[same_as_original] = pool[reroll_idx]
                same_as_original = replacements == originals
                attempts += 1

        unchanged_count += int((replacements == originals).sum().item())
        expanded_cpu[key][mask] = replacements

    target_device = reference.device
    corrupted = {key: value.to(device=target_device) for key, value in expanded_cpu.items()}
    return CorruptionResult(
        corrupted=corrupted,
        mode_index=mode_index_cpu.to(device=target_device),
        mode_counts=mode_counts,
        unchanged_count=unchanged_count,
    )
