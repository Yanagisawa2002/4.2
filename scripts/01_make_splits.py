from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checks import (  # noqa: E402
    assert_cross_disease_disjointness,
    assert_cross_drug_disjointness,
    assert_edge_disjointness,
    assert_ho_alignment_with_kg,
)
from src.splits import derive_ho_splits, split_kg_indications  # noqa: E402

Edge = Tuple[str, str]
HOQuad = Tuple[str, str, str, str]
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create KG train/val/test splits and derive HO splits from KG."
    )
    parser.add_argument(
        "--kg-positive",
        required=True,
        help="Path to CSV/TSV with columns: drug,disease",
    )
    parser.add_argument(
        "--ho",
        required=True,
        help="Path to CSV/TSV with columns: drug,protein,pathway,disease",
    )
    parser.add_argument(
        "--split-type",
        required=True,
        choices=["random", "cross-drug", "cross-disease"],
        help="Split strategy for KG positives.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        default="outputs/splits",
        help="Directory where split files are written.",
    )
    return parser.parse_args()


def _guess_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def _read_rows(path: Path) -> List[dict]:
    delimiter = _guess_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header detected in file: {path}")
        rows: List[dict] = []
        for row in reader:
            clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            rows.append(clean)
    return rows


def _read_kg_positive_edges(path: Path) -> List[Edge]:
    rows = _read_rows(path)
    required = ("drug", "disease")
    _validate_columns(path, rows, required)
    edges: List[Edge] = []
    for i, row in enumerate(rows, start=2):
        drug = row["drug"]
        disease = row["disease"]
        if not drug or not disease:
            raise ValueError(f"Empty drug/disease at {path}:{i}")
        edges.append((drug, disease))
    return edges


def _read_ho_quads(path: Path) -> List[HOQuad]:
    rows = _read_rows(path)
    required = ("drug", "protein", "pathway", "disease")
    _validate_columns(path, rows, required)
    quads: List[HOQuad] = []
    for i, row in enumerate(rows, start=2):
        quad = (row["drug"], row["protein"], row["pathway"], row["disease"])
        if any(not item for item in quad):
            raise ValueError(f"Empty HO value at {path}:{i}")
        quads.append(quad)
    return quads


def _validate_columns(path: Path, rows: Sequence[dict], required: Iterable[str]) -> None:
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    keys = set(rows[0].keys())
    missing = [col for col in required if col not in keys]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    kg_positive_path = Path(args.kg_positive)
    ho_path = Path(args.ho)
    out_dir = Path(args.out_dir)

    kg_positive = _read_kg_positive_edges(kg_positive_path)
    ho_quads = _read_ho_quads(ho_path)

    kg_split = split_kg_indications(
        positive_edges=kg_positive,
        split_type=args.split_type,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        negative_k=1,
    )
    ho_splits = derive_ho_splits(ho_quads, kg_split.pos)

    assert_edge_disjointness(kg_split.pos)
    if args.split_type == "cross-drug":
        assert_cross_drug_disjointness(kg_split.pos)
    if args.split_type == "cross-disease":
        assert_cross_disease_disjointness(kg_split.pos)
    assert_ho_alignment_with_kg(ho_splits, kg_split.pos)

    for split_name in SPLIT_NAMES:
        _write_csv(
            out_dir / f"kg_pos_{split_name}.csv",
            ("drug", "disease"),
            kg_split.pos[split_name],
        )
        _write_csv(
            out_dir / f"kg_neg_{split_name}.csv",
            ("drug", "disease"),
            kg_split.neg[split_name],
        )
        _write_csv(
            out_dir / f"ho_{split_name}.csv",
            ("drug", "protein", "pathway", "disease"),
            ho_splits[split_name],
        )

    metadata = {
        "split_type": args.split_type,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "counts": {
            "kg_pos": {name: len(kg_split.pos[name]) for name in SPLIT_NAMES},
            "kg_neg": {name: len(kg_split.neg[name]) for name in SPLIT_NAMES},
            "ho": {name: len(ho_splits[name]) for name in SPLIT_NAMES},
        },
    }
    (out_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
