#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the small, human-readable samples under corpus/samples/.

The full corpus files are tens of megabytes of compressed JSON, too large to
browse on GitHub. This script extracts a handful of entries from each dataset
(and from one pipeline trace) and writes them pretty-printed so that the record
layout can be inspected without downloading anything.

Usage:
    python3 scripts/make_samples.py
"""

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "corpus" / "final"
TRACES = ROOT / "corpus" / "pipeline"
SAMPLES = ROOT / "corpus" / "samples"

# dataset -> (compressed corpus file, number of entries to keep)
CORPUS_FILES = {
    "wikitext": ("wikitext-103-raw.wiki.train.selected_1000_260620_235838.json.gz", 3),
    "fever": ("fever_selected_260712_212751.json.gz", 5),
    "europarl": ("selected_260710_084528.json.gz", 2),
}

# one trace sample only: FEVER entries are short enough to read end to end
TRACE_FILE = ("fever", "fever_selected_pipeline_260712_212751.json.gz", 2)


def load_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def spread(entries: list, n: int) -> list:
    """Pick n entries at evenly spaced length percentiles.

    Taking the first n records would misrepresent the corpus: in Europarl the
    opening entries are short procedural debates, while the median debate is an
    order of magnitude longer. Selecting by length percentile gives a sample
    whose records look like the ones a user will actually meet.
    """
    if n >= len(entries):
        return entries
    ranked = sorted(entries, key=lambda e: len(e["english"]["text"]))
    picks = [ranked[int((i + 0.5) / n * len(ranked))] for i in range(n)]
    return sorted(picks, key=lambda e: e["id"])


def sample_corpus(dataset: str, filename: str, n: int) -> None:
    data = load_gz(FINAL / filename)
    payload = data["payload"][0]
    out = {
        **{k: v for k, v in data.items() if k != "payload"},
        "payload": [
            {
                **{k: v for k, v in payload.items() if k != "data"},
                "data": spread(payload["data"], n),
            }
        ],
        "_sample_note": (
            f"{n} of {len(payload['data'])} entries, sampled at evenly spaced length percentiles. "
            f"Full file: corpus/final/{filename}"
        ),
    }
    write_json(SAMPLES / f"{dataset}_sample.json", out)


def sample_trace(dataset: str, filename: str, n: int) -> None:
    data = load_gz(TRACES / filename)
    out = {
        **{k: v for k, v in data.items() if k != "pipeline"},
        "pipeline": data["pipeline"][:n],
        "_sample_note": (
            f"{n} of {len(data['pipeline'])} traced entries. "
            f"Full file: corpus/pipeline/{filename}. The 'summary' block above "
            f"reports totals for the complete run, not for this sample."
        ),
    }
    write_json(SAMPLES / f"{dataset}_pipeline_sample.json", out)


def main() -> None:
    print("Writing samples to corpus/samples/")
    for dataset, (filename, n) in CORPUS_FILES.items():
        sample_corpus(dataset, filename, n)
    sample_trace(*TRACE_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
