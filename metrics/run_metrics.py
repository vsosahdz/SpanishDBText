"""Main entry point: compute all metrics for a dataset and save to JSON.

Usage:
    python3 -m metrics.run_metrics --dataset fever
    python3 -m metrics.run_metrics --dataset wikitext
    python3 -m metrics.run_metrics --dataset europarl
    python3 -m metrics.run_metrics --all
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .loader import load_dataset, find_latest_output
from .general_metrics import compute_general_metrics, compute_frequency_metrics
from .lexical_metrics import compute_lexical_metrics
from .syntactic_metrics import compute_syntactic_metrics
from .quality_metrics import compute_quality_metrics


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def run_metrics(dataset: str) -> dict:
    """Compute all metrics for a single dataset and return the results dict."""

    print(f"\n{'='*60}")
    print(f"  Computing metrics for: {dataset.upper()}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load data
    source_path = find_latest_output(dataset)
    print(f"  Source file: {source_path.name}")
    data = load_dataset(dataset, source_path)
    print(f"  Entries loaded: {data['total_entries']}")

    en_texts = [e["en_text"] for e in data["entries"]]
    es_texts = [e["es_text"] for e in data["entries"]]

    # 1. General metrics
    print("\n[1/5] General corpus metrics...")
    t0 = time.time()
    general_en = compute_general_metrics(en_texts, "en")
    general_es = compute_general_metrics(es_texts, "es")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # 2. Frequency metrics
    print("[2/5] Word frequency metrics...")
    t0 = time.time()
    frequency_en = compute_frequency_metrics(en_texts, "en")
    frequency_es = compute_frequency_metrics(es_texts, "es")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # 3. Lexical diversity
    print("[3/5] Lexical diversity metrics...")
    t0 = time.time()
    lexical_en = compute_lexical_metrics(en_texts, "en")
    lexical_es = compute_lexical_metrics(es_texts, "es")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # 4. Syntactic metrics
    print("[4/5] Syntactic metrics...")
    t0 = time.time()
    syntactic_en = compute_syntactic_metrics(en_texts, "en")
    syntactic_es = compute_syntactic_metrics(es_texts, "es")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # 5. Quality metrics
    print("[5/5] Quality & semantic metrics...")
    t0 = time.time()
    quality = compute_quality_metrics(data["entries"])
    print(f"  Done ({time.time()-t0:.1f}s)")

    elapsed = time.time() - start_time

    # Assemble result
    result = {
        "metadata": {
            "dataset": dataset,
            "source_file": source_path.name,
            "source_name": data["source_name"],
            "total_entries": data["total_entries"],
            "generated_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "general_metrics": {
            "english": general_en,
            "spanish": general_es,
        },
        "frequency_metrics": {
            "english": frequency_en,
            "spanish": frequency_es,
        },
        "lexical_diversity": {
            "english": lexical_en,
            "spanish": lexical_es,
        },
        "syntactic_metrics": {
            "english": syntactic_en,
            "spanish": syntactic_es,
        },
        "quality_metrics": quality,
    }

    return result


def save_results(result: dict, dataset: str) -> Path:
    """Save metrics result to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"metrics_{dataset}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compute corpus metrics for translation datasets")
    parser.add_argument("--dataset", "-d", choices=["wikitext", "fever", "europarl"],
                        help="Dataset to process")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Process all datasets")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        sys.exit(1)

    datasets = ["wikitext", "fever", "europarl"] if args.all else [args.dataset]

    for dataset in datasets:
        try:
            result = run_metrics(dataset)
            save_results(result, dataset)
        except Exception as e:
            print(f"\n  ERROR processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
