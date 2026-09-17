#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the compressed archives under corpus/ from output/ and data/.

This is the inverse of unpack_corpus.py and is meant for maintainers preparing a
new release: run the pipeline, then run this to refresh the distributed files.
Ordinary users reproducing the paper never need it.

Usage:
    python3 scripts/pack_corpus.py --all
    python3 scripts/pack_corpus.py --final --pipeline
"""

import argparse
import gzip
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPRESS_LEVEL = 6

# (source relative to repository root, archive relative to corpus/)
FINAL = [
    ("output/wikitext/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json",
     "final/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json.gz"),
    ("output/FEVER/fever_selected_260712_212751.json",
     "final/fever_selected_260712_212751.json.gz"),
    ("output/Europarl/selected_260710_084528.json",
     "final/selected_260710_084528.json.gz"),
]

PIPELINE = [
    ("output/wikitext/wikitext-103-raw.wiki.train.selected_1000_pipeline_260620_235838.json",
     "pipeline/wikitext-103-raw.wiki.train.selected_1000_pipeline_260620_235838.json.gz"),
    ("output/FEVER/fever_selected_pipeline_260712_212751.json",
     "pipeline/fever_selected_pipeline_260712_212751.json.gz"),
    ("output/Europarl/selected_pipeline_260710_084528.json",
     "pipeline/selected_pipeline_260710_084528.json.gz"),
]

INPUTS = [
    ("data/WikiText/wikitext-103-raw.wiki.train.selected_1000.raw",
     "inputs/wikitext-103-raw.wiki.train.selected_1000.raw.gz"),
    ("data/FEVER/selected/fever_selected.jsonl",
     "inputs/fever_selected.jsonl.gz"),
]

PILOT = [
    ("output/wikitext/tests/wikitext-103-raw.wiki.test_260612_032122.json",
     "pilot/wikitext-103-raw.wiki.test_260612_032122.json.gz"),
    ("output/wikitext/tests/wikitext-103-raw.wiki.test_pipeline_260612_032122.json",
     "pilot/wikitext-103-raw.wiki.test_pipeline_260612_032122.json.gz"),
    ("output/wikitext/translation_260618_015112.log",
     "pilot/translation_260618_015112.log.gz"),
]

# (source directory relative to root, archive relative to corpus/, name inside the tar)
INPUT_TARBALLS = [
    ("data/Europarl/selected", "inputs/europarl_selected_1006.tar.gz", "selected"),
]

GITHUB_FILE_LIMIT = 100 * 1024 * 1024


def human(n_bytes: int) -> str:
    return f"{n_bytes / 1048576:.1f} MB"


def compress(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as src, gzip.open(archive, "wb", compresslevel=COMPRESS_LEVEL) as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    size = archive.stat().st_size
    warn = "  <-- OVER GITHUB'S 100 MB FILE LIMIT" if size > GITHUB_FILE_LIMIT else ""
    print(f"  {archive.name}  {human(source.stat().st_size)} -> {human(size)}{warn}")


def compress_dir(source_dir: Path, archive: Path, arc_root: str) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source_dir.iterdir() if p.is_file())
    with tarfile.open(archive, "w:gz", compresslevel=COMPRESS_LEVEL) as tf:
        for path in files:
            tf.add(path, arcname=f"{arc_root}/{path.name}")
    print(f"  {archive.name}  {len(files)} files -> {human(archive.stat().st_size)}")


def run_group(label: str, entries: list) -> None:
    print(f"\n{label}")
    for rel_source, rel_archive in entries:
        source = ROOT / rel_source
        if not source.exists():
            print(f"  MISSING  {rel_source}  (run unpack_corpus.py or the pipeline first)")
            continue
        compress(source, ROOT / "corpus" / rel_archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="pack everything")
    parser.add_argument("--final", action="store_true", help="pack the final bilingual corpus")
    parser.add_argument("--pipeline", action="store_true", help="pack the pipeline traces")
    parser.add_argument("--inputs", action="store_true", help="pack the selected source subsets")
    parser.add_argument("--pilot", action="store_true", help="pack the WikiText pilot run")
    args = parser.parse_args()

    if not any([args.all, args.final, args.pipeline, args.inputs, args.pilot]):
        parser.print_help()
        sys.exit(1)

    if args.all or args.final:
        run_group("Final bilingual corpus -> corpus/final/", FINAL)
    if args.all or args.pipeline:
        run_group("Pipeline traces -> corpus/pipeline/", PIPELINE)
    if args.all or args.inputs:
        run_group("Selected source subsets -> corpus/inputs/", INPUTS)
        for rel_dir, rel_archive, arc_root in INPUT_TARBALLS:
            source_dir = ROOT / rel_dir
            if not source_dir.is_dir():
                print(f"  MISSING  {rel_dir}/")
                continue
            compress_dir(source_dir, ROOT / "corpus" / rel_archive, arc_root)
    if args.all or args.pilot:
        run_group("WikiText pilot run -> corpus/pilot/", PILOT)

    print("\nDone.")


if __name__ == "__main__":
    main()
