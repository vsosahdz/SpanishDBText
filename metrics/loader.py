"""Load translation output JSON files and extract text data."""

import json
import re
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Map dataset names to their output subdirectories
DATASET_DIRS = {
    "wikitext": OUTPUT_DIR / "wikitext",
    "fever": OUTPUT_DIR / "FEVER",
    "europarl": OUTPUT_DIR / "Europarl",
}


def find_latest_output(dataset: str) -> Path:
    """Find the most recent main output JSON (not pipeline, not log) for a dataset."""
    dir_path = DATASET_DIRS[dataset]
    candidates = []
    for f in dir_path.glob("*.json"):
        if "pipeline" in f.name or "progress" in f.name:
            continue
        candidates.append(f)
    if not candidates:
        raise FileNotFoundError(f"No output JSON found in {dir_path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_dataset(dataset: str, path: Path | None = None) -> dict:
    """Load a dataset and return structured data.

    Returns:
        {
            "dataset": str,
            "file": str,
            "source_name": str,
            "total_entries": int,
            "entries": [
                {
                    "id": int,
                    "source_language": str,
                    "en_text": str,
                    "es_text": str,
                    "classification": str,
                    "keywords_en": list[str],
                    "keywords_es": list[str],
                }
            ]
        }
    """
    if path is None:
        path = find_latest_output(dataset)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw["payload"][0]["data"]

    entries = []
    for item in data:
        en = item.get("english", {})
        es = item.get("spanish", {})
        entries.append({
            "id": item["id"],
            "source_language": item.get("source_language", "en"),
            "en_text": en.get("text", ""),
            "es_text": es.get("text", ""),
            "classification": en.get("classification", es.get("classification", "")),
            "keywords_en": en.get("keywords", []),
            "keywords_es": es.get("keywords", []),
        })

    return {
        "dataset": dataset,
        "file": path.name,
        "source_name": raw.get("source_name", dataset),
        "total_entries": len(entries),
        "entries": entries,
    }


def tokenize_simple(text: str) -> list[str]:
    """Simple word tokenizer using regex. Splits on non-alphanumeric characters."""
    return re.findall(r'\b\w+\b', text.lower())


def sentence_split(text: str) -> list[str]:
    """Simple sentence splitter using punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 0]
