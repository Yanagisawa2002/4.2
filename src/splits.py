from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Edge = Tuple[str, str]
HOQuad = Tuple[str, str, str, str]
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class KGSplit:
    split_type: str
    seed: int
    pos: Dict[str, List[Edge]]
    neg: Dict[str, List[Edge]]


def split_kg_indications(
    positive_edges: Sequence[Edge],
    split_type: str,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    negative_k: int = 1,
) -> KGSplit:
    if negative_k != 1:
        raise ValueError("SPEC requires exactly K=1 negative per positive edge.")
    if split_type not in {"random", "cross-drug", "cross-disease"}:
        raise ValueError("split_type must be one of: random, cross-drug, cross-disease")
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    deduped_edges = _dedupe_edges(positive_edges)
    if not deduped_edges:
        raise ValueError("No KG positive edges provided.")

    rng = random.Random(seed)
    pos_splits = _split_positive_edges(
        deduped_edges,
        split_type=split_type,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
    )

    all_drugs = sorted({d for d, _ in deduped_edges})
    all_diseases = sorted({dis for _, dis in deduped_edges})
    known_positive = set(deduped_edges)

    neg_splits: Dict[str, List[Edge]] = {}
    for split_name in SPLIT_NAMES:
        neg_splits[split_name] = _sample_negatives_for_positives(
            positives=pos_splits[split_name],
            all_drugs=all_drugs,
            all_diseases=all_diseases,
            known_positive=known_positive,
            rng=rng,
        )

    return KGSplit(
        split_type=split_type,
        seed=seed,
        pos=pos_splits,
        neg=neg_splits,
    )


def derive_ho_splits(
    ho_quads: Sequence[HOQuad],
    kg_pos_splits: Mapping[str, Sequence[Edge]],
) -> Dict[str, List[HOQuad]]:
    _require_split_keys(kg_pos_splits)

    pair_to_split: Dict[Edge, str] = {}
    for split_name in SPLIT_NAMES:
        for edge in kg_pos_splits[split_name]:
            pair = (edge[0], edge[1])
            if pair in pair_to_split:
                raise ValueError(
                    f"KG positive edge appears in multiple splits: {pair} "
                    f"({pair_to_split[pair]}, {split_name})"
                )
            pair_to_split[pair] = split_name

    derived: Dict[str, List[HOQuad]] = {name: [] for name in SPLIT_NAMES}
    for quad in ho_quads:
        if len(quad) != 4:
            raise ValueError(f"Invalid HO quadruplet (need 4 values): {quad}")
        pair = (quad[0], quad[3])
        split_name = pair_to_split.get(pair)
        if split_name is None:
            raise ValueError(
                f"HO quadruplet maps to non-positive KG pair {pair}. "
                "SPEC requires all HO pairs to be KG positives."
            )
        derived[split_name].append((quad[0], quad[1], quad[2], quad[3]))

    return derived


def _split_positive_edges(
    edges: Sequence[Edge],
    split_type: str,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Edge]]:
    if split_type == "random":
        shuffled = list(edges)
        rng.shuffle(shuffled)
        return _slice_by_ratios(shuffled, val_ratio=val_ratio, test_ratio=test_ratio)

    if split_type == "cross-drug":
        return _split_by_entity(
            edges=edges,
            entity_selector=lambda e: e[0],
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )

    return _split_by_entity(
        edges=edges,
        entity_selector=lambda e: e[1],
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
    )


def _split_by_entity(
    edges: Sequence[Edge],
    entity_selector,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Edge]]:
    entities = sorted({entity_selector(edge) for edge in edges})
    rng.shuffle(entities)
    entity_splits = _slice_by_ratios(entities, val_ratio=val_ratio, test_ratio=test_ratio)

    entity_to_split: Dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for entity in entity_splits[split_name]:
            entity_to_split[entity] = split_name

    out: Dict[str, List[Edge]] = {name: [] for name in SPLIT_NAMES}
    for edge in edges:
        split_name = entity_to_split[entity_selector(edge)]
        out[split_name].append(edge)
    return out


def _slice_by_ratios(
    values: Sequence,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List]:
    n_total = len(values)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    if n_train < 1:
        deficit = 1 - n_train
        reduce_test = min(deficit, n_test)
        n_test -= reduce_test
        deficit -= reduce_test
        if deficit > 0:
            n_val -= min(deficit, n_val)
        n_train = n_total - n_val - n_test

    train_end = n_train
    val_end = n_train + n_val
    return {
        "train": list(values[:train_end]),
        "val": list(values[train_end:val_end]),
        "test": list(values[val_end:]),
    }


def _sample_negatives_for_positives(
    positives: Sequence[Edge],
    all_drugs: Sequence[str],
    all_diseases: Sequence[str],
    known_positive: set[Edge],
    rng: random.Random,
) -> List[Edge]:
    pos_by_drug: Dict[str, set[str]] = {}
    pos_by_disease: Dict[str, set[str]] = {}
    for drug, disease in known_positive:
        pos_by_drug.setdefault(drug, set()).add(disease)
        pos_by_disease.setdefault(disease, set()).add(drug)

    negatives: List[Edge] = []
    for drug, disease in positives:
        negatives.append(
            _sample_one_negative(
                drug=drug,
                disease=disease,
                all_drugs=all_drugs,
                all_diseases=all_diseases,
                known_positive=known_positive,
                pos_by_drug=pos_by_drug,
                pos_by_disease=pos_by_disease,
                rng=rng,
            )
        )
    return negatives


def _sample_one_negative(
    drug: str,
    disease: str,
    all_drugs: Sequence[str],
    all_diseases: Sequence[str],
    known_positive: set[Edge],
    pos_by_drug: Mapping[str, set[str]],
    pos_by_disease: Mapping[str, set[str]],
    rng: random.Random,
) -> Edge:
    blocked_diseases = pos_by_drug.get(drug, set())
    candidate_diseases = [dis for dis in all_diseases if dis not in blocked_diseases]
    if candidate_diseases:
        return (drug, rng.choice(candidate_diseases))

    blocked_drugs = pos_by_disease.get(disease, set())
    candidate_drugs = [d for d in all_drugs if d not in blocked_drugs]
    if candidate_drugs:
        return (rng.choice(candidate_drugs), disease)

    max_tries = max(1_000, len(all_drugs) * len(all_diseases) * 2)
    for _ in range(max_tries):
        candidate = (rng.choice(all_drugs), rng.choice(all_diseases))
        if candidate not in known_positive:
            return candidate

    raise ValueError(
        "Unable to sample a negative edge: all candidate drug-disease pairs appear positive."
    )


def _dedupe_edges(edges: Iterable[Edge]) -> List[Edge]:
    seen: set[Edge] = set()
    out: List[Edge] = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Invalid edge (need 2 values): {edge}")
        normalized = (edge[0], edge[1])
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _require_split_keys(mapping: Mapping[str, Sequence]) -> None:
    missing = [k for k in SPLIT_NAMES if k not in mapping]
    if missing:
        raise ValueError(f"Missing split keys: {missing}. Required: {SPLIT_NAMES}")
