#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decompress the released corpus into the layout the pipeline code expects.

Everything under corpus/ is shipped compressed so that the repository stays
within GitHub's per-file limits. The agents, the metrics package and the
notebooks all read plain JSON from output/ and data/, exactly as they did during
the production run, and none of that code was modified for release. This script
is the bridge: it expands the archives into those two directories.

    corpus/final/*.json.gz     ->  output/{wikitext,FEVER,Europarl}/*.json
    corpus/pipeline/*.json.gz  ->  output/{wikitext,FEVER,Europarl}/*.json
    corpus/inputs/*            ->  data/{WikiText,FEVER/selected,Europarl/selected}/
    corpus/pilot/*.gz          ->  output/wikitext/tests/ and output/wikitext/

Expanded size is roughly 530 MB, so make sure there is room on disk before
running it with --all.

Usage:
    python3 scripts/unpack_corpus.py --all
    python3 scripts/unpack_corpus.py --final           # corpus only (159 MB)
    python3 scripts/unpack_corpus.py --final --inputs  # corpus + pipeline inputs
    python3 scripts/unpack_corpus.py --all --force     # overwrite existing files
"""

import argparse
import gzip
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (archive relative to corpus/, destination relative to repository root)
FINAL = [
    ("final/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json.gz",
     "output/wikitext/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json"),
    ("final/fever_selected_260712_212751.json.gz",
     "output/FEVER/fever_selected_260712_212751.json"),
    ("final/selected_260710_084528.json.gz",
     "output/Europarl/selected_260710_084528.json"),
]

PIPELINE = [
    ("pipeline/wikitext-103-raw.wiki.train.selected_1000_pipeline_260620_235838.json.gz",
     "output/wikitext/wikitext-103-raw.wiki.train.selected_1000_pipeline_260620_235838.json"),
    ("pipeline/fever_selected_pipeline_260712_212751.json.gz",
     "output/FEVER/fever_selected_pipeline_260712_212751.json"),
    ("pipeline/selected_pipeline_260710_084528.json.gz",
     "output/Europarl/selected_pipeline_260710_084528.json"),
]

INPUTS = [
    ("inputs/wikitext-103-raw.wiki.train.selected_1000.raw.gz",
     "data/WikiText/wikitext-103-raw.wiki.train.selected_1000.raw"),
    ("inputs/fever_selected.jsonl.gz",
     "data/FEVER/selected/fever_selected.jsonl"),
]

# The June 2026 WikiText trial runs. The 11 June pilot is read by notebook 05;
# the 7 June run is the input of the evaluator ablation in ablation/. Both are
# small and they are the only record of the pipeline behaviour before the
# production configuration was frozen, so they ship with the release.
PILOT = [
    ("pilot/wikitext-103-raw.wiki.test_260607.json.gz",
     "output/wikitext/tests/wikitext-103-raw.wiki.test_260607.json"),
    ("pilot/wikitext-103-raw.wiki.test_pipeline_260607.json.gz",
     "output/wikitext/tests/wikitext-103-raw.wiki.test_pipeline_260607.json"),
    ("pilot/wikitext-103-raw.wiki.test_260612_032122.json.gz",
     "output/wikitext/tests/wikitext-103-raw.wiki.test_260612_032122.json"),
    ("pilot/wikitext-103-raw.wiki.test_pipeline_260612_032122.json.gz",
     "output/wikitext/tests/wikitext-103-raw.wiki.test_pipeline_260612_032122.json"),
    ("pilot/translation_260618_015112.log.gz",
     "output/wikitext/translation_260618_015112.log"),
]

# tar archives: (archive relative to corpus/, directory it expands into)
INPUT_TARBALLS = [
    ("inputs/europarl_selected_1006.tar.gz", "data/Europarl"),
]


def human(n_bytes: int) -> str:
    return f"{n_bytes / 1048576:.1f} MB"


def expand_gz(archive: Path, target: Path, force: bool) -> bool:
    if target.exists() and not force:
        print(f"  skip  {target.relative_to(ROOT)} (already there, use --force to overwrite)")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    print(f"  ok    {target.relative_to(ROOT)} ({human(target.stat().st_size)})")
    return True


def expand_tar(archive: Path, target_dir: Path, force: bool) -> bool:
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
        first = target_dir / names[0]
        if first.exists() and not force:
            print(f"  skip  {target_dir.relative_to(ROOT)}/ (already there, use --force to overwrite)")
            return False
        target_dir.mkdir(parents=True, exist_ok=True)
        # The archive is one we build ourselves in pack_corpus.py and contains
        # only regular files under a single relative directory, but check anyway
        # rather than trusting the paths inside a downloaded file.
        for member in tf.getmembers():
            path = (target_dir / member.name).resolve()
            if not str(path).startswith(str(target_dir.resolve())):
                sys.exit(f"ERROR: archive {archive.name} contains an unsafe path: {member.name}")
            if not (member.isfile() or member.isdir()):
                sys.exit(f"ERROR: archive {archive.name} contains a non-regular entry: {member.name}")
        tf.extractall(target_dir)
    print(f"  ok    {target_dir.relative_to(ROOT)}/ ({len(names)} files)")
    return True


def run_group(label: str, entries: list, force: bool) -> None:
    print(f"\n{label}")
    for rel_archive, rel_target in entries:
        archive = ROOT / "corpus" / rel_archive
        if not archive.exists():
            print(f"  MISSING  corpus/{rel_archive}")
            continue
        expand_gz(archive, ROOT / rel_target, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="expand everything (final corpus, pipeline traces, inputs)")
    parser.add_argument("--final", action="store_true",
                        help="expand the final bilingual corpus (159 MB)")
    parser.add_argument("--pipeline", action="store_true",
                        help="expand the per-entry pipeline traces (278 MB)")
    parser.add_argument("--inputs", action="store_true",
                        help="expand the selected English source subsets (87 MB)")
    parser.add_argument("--pilot", action="store_true",
                        help="expand the June 2026 WikiText trial runs (13 MB), used by notebook 05 and ablation/")
    parser.add_argument("--force", action="store_true",
                        help="overwrite files that already exist")
    args = parser.parse_args()

    if not any([args.all, args.final, args.pipeline, args.inputs, args.pilot]):
        parser.print_help()
        sys.exit(1)

    if args.all or args.final:
        run_group("Final bilingual corpus -> output/", FINAL, args.force)
    if args.all or args.pipeline:
        run_group("Pipeline traces -> output/", PIPELINE, args.force)
    if args.all or args.inputs:
        run_group("Selected source subsets -> data/", INPUTS, args.force)
        for rel_archive, rel_dir in INPUT_TARBALLS:
            archive = ROOT / "corpus" / rel_archive
            if not archive.exists():
                print(f"  MISSING  corpus/{rel_archive}")
                continue
            expand_tar(archive, ROOT / rel_dir, args.force)
    if args.all or args.pilot:
        run_group("WikiText pilot run -> output/", PILOT, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
