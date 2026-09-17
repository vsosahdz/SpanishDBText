#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the three English source datasets into data/.

You only need this if you want to redo the subset selection from scratch, that
is, to run notebooks 01-03 and reproduce the stratified samples rather than
reuse the ones shipped in corpus/inputs/. To rerun the translation pipeline on
the exact subsets used in the paper, run scripts/unpack_corpus.py --inputs
instead; it is far smaller and far faster.

The three sources are large. Full download is about 2.2 GB compressed and about
7 GB once expanded.

    WikiText-103  ~190 MB zip     https://huggingface.co/datasets/Salesforce/wikitext
    FEVER wiki    ~1.7 GB zip     https://fever.ai/dataset/fever.html
    Europarl v7   ~630 MB tgz     https://www.statmt.org/europarl/

Usage:
    python3 scripts/download_datasets.py --list
    python3 scripts/download_datasets.py --dataset wikitext
    python3 scripts/download_datasets.py --all
    python3 scripts/download_datasets.py --all --keep-archives
"""

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOWNLOADS = DATA / "_downloads"

DATASETS = {
    "wikitext": {
        "name": "WikiText-103 (raw character level)",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1.zip",
        "archive": "wikitext-103-raw-v1.zip",
        "kind": "zip",
        "target": DATA / "WikiText",
        "note": (
            "The pipeline reads wikitext-103-raw.wiki.train.raw. The archive expands to a\n"
            "  wikitext-103-raw/ directory holding wiki.train.raw, wiki.valid.raw and\n"
            "  wiki.test.raw; this script renames them to the flat wikitext-103-raw.wiki.*.raw\n"
            "  form the notebooks expect."
        ),
    },
    "fever": {
        "name": "FEVER Wikipedia dump (wiki-pages)",
        "url": "https://fever.ai/download/fever/wiki-pages.zip",
        "archive": "wiki-pages.zip",
        "kind": "zip",
        "target": DATA / "FEVER",
        "note": (
            "The archive expands to 109 shards (wiki-001.jsonl ... wiki-109.jsonl) of about\n"
            "  50,000 pages each. The run reported in the paper used a single 50,000-page\n"
            "  shard, saved as data/FEVER/fever.jsonl. See docs/reproduction.md."
        ),
    },
    "europarl": {
        "name": "Europarl v7 source release (English side)",
        "url": "https://www.statmt.org/europarl/v7/europarl.tgz",
        "archive": "europarl.tgz",
        "kind": "tgz",
        "target": DATA / "Europarl",
        "note": (
            "The pipeline reads one .txt per session chapter from data/Europarl/en/. The run\n"
            "  reported in the paper used 3,656 chapter files covering the 2009-2011 sessions\n"
            "  (ep-09-05-07-*.txt through ep-11-11-16-*.txt). See docs/reproduction.md."
        ),
    },
}


def human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def download(url: str, target: Path) -> Path:
    """Stream a URL to disk with a progress line. Resumes nothing: a partial file
    is deleted so that a failed run never leaves a truncated archive behind."""
    try:
        import httpx
    except ImportError:
        sys.exit("ERROR: httpx is required. Run: pip install -r requirements.txt")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"  archive already downloaded: {target.name} ({human(target.stat().st_size)})")
        return target

    partial = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {url}")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            done = 0
            with open(partial, "wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {human(done)} / {human(total)} "
                              f"({done * 100 // total}%)", end="", flush=True)
                    else:
                        print(f"\r  {human(done)}", end="", flush=True)
            print()
    except Exception as exc:
        partial.unlink(missing_ok=True)
        sys.exit(f"\nERROR: download failed: {exc}\n"
                 f"       You can fetch it by hand and place it at {target}")

    partial.rename(target)
    return target


def _safe_members(names, root: Path):
    """Reject absolute paths and paths escaping the extraction directory."""
    for name in names:
        resolved = (root / name).resolve()
        if not str(resolved).startswith(str(root.resolve())):
            sys.exit(f"ERROR: archive contains an unsafe path: {name}")


def extract(archive: Path, kind: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"  extracting into {target.relative_to(ROOT)}/ (this takes a while)")
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            _safe_members(zf.namelist(), target)
            zf.extractall(target)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            _safe_members(tf.getnames(), target)
            tf.extractall(target)


def flatten_wikitext(target: Path) -> None:
    """Rename wikitext-103-raw/wiki.train.raw to the flat name the code uses."""
    nested = target / "wikitext-103-raw"
    if not nested.is_dir():
        return
    for split in ("train", "valid", "test"):
        source = nested / f"wiki.{split}.raw"
        if source.exists():
            dest = target / f"wikitext-103-raw.wiki.{split}.raw"
            if not dest.exists():
                shutil.move(str(source), str(dest))
                print(f"  -> {dest.name}")
    if not any(nested.iterdir()):
        nested.rmdir()


def fetch(key: str, keep_archives: bool) -> None:
    spec = DATASETS[key]
    print(f"\n{spec['name']}")
    archive = download(spec["url"], DOWNLOADS / spec["archive"])
    extract(archive, spec["kind"], spec["target"])
    if key == "wikitext":
        flatten_wikitext(spec["target"])
    if not keep_archives:
        archive.unlink()
        print(f"  removed archive {archive.name} (use --keep-archives to keep it)")
    print(f"  NOTE: {spec['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", choices=sorted(DATASETS),
                        help="download a single dataset")
    parser.add_argument("--all", "-a", action="store_true", help="download all three")
    parser.add_argument("--list", "-l", action="store_true",
                        help="print the source URLs without downloading anything")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the downloaded archives under data/_downloads/")
    args = parser.parse_args()

    if args.list:
        for key, spec in sorted(DATASETS.items()):
            host = urlparse(spec["url"]).netloc
            print(f"{key:10s} {spec['name']}\n           {spec['url']}  ({host})\n")
        return

    if not args.dataset and not args.all:
        parser.print_help()
        sys.exit(1)

    for key in (sorted(DATASETS) if args.all else [args.dataset]):
        fetch(key, args.keep_archives)

    print("\nDone. Next: run notebooks/01-03 to rebuild the stratified subsets,")
    print("or skip straight to the shipped ones with scripts/unpack_corpus.py --inputs")


if __name__ == "__main__":
    main()
