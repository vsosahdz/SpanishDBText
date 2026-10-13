#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute metrics_combined.json after the correction run.

It reproduces the cells of notebook 06 that compute the combined metrics: it
loads the three datasets from output/ and writes a JSON file with the same
structure as corpus/metrics/metrics_combined.json. The per-dataset and semantic
metrics are recomputed with `python3 -m metrics.run_metrics --all` and notebook 06.

Requires the expanded corpus:  python3 scripts/unpack_corpus.py --final --pipeline
Usage (from the repository root):
    python3 scripts/correction/recompute_combined.py corpus/metrics/metrics_combined.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metrics.loader import load_dataset, find_latest_output  # noqa: E402
from metrics.combined_metrics import (  # noqa: E402
    compute_combined_totals, compute_vocabulary_overlap, compute_cross_dataset_duplicates,
    compute_combined_growth_curve, compute_combined_classification)

DATASETS = ["wikitext", "fever", "europarl"]


def main(out: Path) -> None:
    data = {ds: load_dataset(ds, find_latest_output(ds)) for ds in DATASETS}
    result = {
        "metadata": {"datasets": DATASETS,
                     "total_entries": sum(data[ds]["total_entries"] for ds in DATASETS),
                     "generated_at": datetime.now().isoformat()},
        "combined_totals": compute_combined_totals(data),
        "vocabulary_overlap": compute_vocabulary_overlap(data),
        "cross_dataset_duplicates": compute_cross_dataset_duplicates(data),
        "combined_growth_curve": compute_combined_growth_curve(data, DATASETS),
        "combined_classification": compute_combined_classification(data),
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
