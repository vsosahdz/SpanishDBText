"""Combined corpus metrics: totals, vocabulary overlap, cross-dataset duplicates, growth curve."""

import hashlib
from collections import Counter

import numpy as np

from .loader import tokenize_simple


def _text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def compute_combined_totals(datasets: dict[str, dict]) -> dict:
    """Compute aggregate totals across all datasets.

    Args:
        datasets: {dataset_name: {"entries": [...], ...}}

    Returns:
        Totals for EN and ES across all datasets.
    """
    totals = {}
    for lang_key, text_key in [("english", "en_text"), ("spanish", "es_text")]:
        all_words = []
        total_chars = 0
        total_docs = 0
        per_dataset = {}

        for ds_name, ds_data in datasets.items():
            ds_words = []
            ds_chars = 0
            for entry in ds_data["entries"]:
                text = entry[text_key]
                words = tokenize_simple(text)
                ds_words.extend(words)
                ds_chars += len(text)
            all_words.extend(ds_words)
            total_chars += ds_chars
            total_docs += len(ds_data["entries"])
            per_dataset[ds_name] = {
                "tokens": len(ds_words),
                "types": len(set(ds_words)),
                "chars": ds_chars,
                "docs": len(ds_data["entries"]),
            }

        totals[lang_key] = {
            "total_documents": total_docs,
            "total_tokens": len(all_words),
            "total_types": len(set(all_words)),
            "total_characters": total_chars,
            "per_dataset": per_dataset,
        }

    return totals


def compute_vocabulary_overlap(datasets: dict[str, dict]) -> dict:
    """Compute vocabulary overlap between datasets.

    For each pair of datasets, compute:
    - Shared types (intersection)
    - Exclusive types (in A but not B)
    - Jaccard similarity of vocabularies
    """
    result = {}
    for lang_key, text_key in [("english", "en_text"), ("spanish", "es_text")]:
        vocabs = {}
        for ds_name, ds_data in datasets.items():
            words = set()
            for entry in ds_data["entries"]:
                words.update(tokenize_simple(entry[text_key]))
            vocabs[ds_name] = words

        ds_names = list(vocabs.keys())

        # Pairwise overlap
        pairwise = {}
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                a, b = ds_names[i], ds_names[j]
                shared = len(vocabs[a] & vocabs[b])
                only_a = len(vocabs[a] - vocabs[b])
                only_b = len(vocabs[b] - vocabs[a])
                union = len(vocabs[a] | vocabs[b])
                jaccard = round(shared / union, 4) if union > 0 else 0
                pairwise[f"{a}_vs_{b}"] = {
                    "shared_types": shared,
                    f"only_{a}": only_a,
                    f"only_{b}": only_b,
                    "jaccard_similarity": jaccard,
                    "overlap_pct_of_smaller": round(
                        shared / min(len(vocabs[a]), len(vocabs[b])) * 100, 2
                    ) if min(len(vocabs[a]), len(vocabs[b])) > 0 else 0,
                }

        # Per-dataset exclusive contribution to combined vocabulary
        combined = set()
        for v in vocabs.values():
            combined.update(v)

        exclusive = {}
        for ds_name, vocab in vocabs.items():
            others = set()
            for other_name, other_vocab in vocabs.items():
                if other_name != ds_name:
                    others.update(other_vocab)
            excl = vocab - others
            exclusive[ds_name] = {
                "exclusive_types": len(excl),
                "exclusive_pct": round(len(excl) / len(vocab) * 100, 2) if vocab else 0,
                "contribution_to_combined_pct": round(len(excl) / len(combined) * 100, 2) if combined else 0,
            }

        result[lang_key] = {
            "per_dataset_vocab_size": {ds: len(v) for ds, v in vocabs.items()},
            "combined_vocab_size": len(combined),
            "pairwise_overlap": pairwise,
            "exclusive_contributions": exclusive,
        }

    return result


def compute_cross_dataset_duplicates(datasets: dict[str, dict]) -> dict:
    """Detect exact duplicate texts across different datasets."""

    result = {}
    for lang_key, text_key in [("english", "en_text"), ("spanish", "es_text")]:
        # Build hash -> (dataset, id) mapping
        hash_map = {}
        cross_duplicates = []

        for ds_name, ds_data in datasets.items():
            for entry in ds_data["entries"]:
                h = _text_hash(entry[text_key])
                if h in hash_map:
                    prev_ds, prev_id = hash_map[h]
                    if prev_ds != ds_name:
                        cross_duplicates.append({
                            "dataset_1": prev_ds,
                            "id_1": prev_id,
                            "dataset_2": ds_name,
                            "id_2": entry["id"],
                        })
                else:
                    hash_map[h] = (ds_name, entry["id"])

        result[lang_key] = {
            "cross_dataset_duplicate_pairs": len(cross_duplicates),
            "details": cross_duplicates[:50],  # Cap for JSON size
        }

    return result


def compute_combined_growth_curve(datasets: dict[str, dict], order: list[str]) -> dict:
    """Compute vocabulary growth curve processing datasets in sequence.

    This shows whether adding each dataset contributes new vocabulary.
    """
    result = {}
    for lang_key, text_key in [("english", "en_text"), ("spanish", "es_text")]:
        curve = []
        seen = set()
        cumulative_tokens = 0

        for ds_name in order:
            ds_data = datasets[ds_name]
            ds_start_types = len(seen)
            ds_start_tokens = cumulative_tokens

            for entry in ds_data["entries"]:
                words = tokenize_simple(entry[text_key])
                for w in words:
                    seen.add(w)
                    cumulative_tokens += 1

            curve.append({
                "dataset": ds_name,
                "tokens_after": cumulative_tokens,
                "types_after": len(seen),
                "new_types_added": len(seen) - ds_start_types,
                "tokens_in_dataset": cumulative_tokens - ds_start_tokens,
            })

        # Also generate a fine-grained curve for plotting
        seen_fine = set()
        tokens_fine = 0
        fine_curve = []
        total_tokens = sum(
            sum(len(tokenize_simple(e[text_key])) for e in datasets[ds]["entries"])
            for ds in order
        )
        step = max(1, total_tokens // 300)

        for ds_name in order:
            for entry in datasets[ds_name]["entries"]:
                words = tokenize_simple(entry[text_key])
                for w in words:
                    seen_fine.add(w)
                    tokens_fine += 1
                    if tokens_fine % step == 0:
                        fine_curve.append({
                            "tokens": tokens_fine,
                            "types": len(seen_fine),
                            "dataset": ds_name,
                        })

        result[lang_key] = {
            "processing_order": order,
            "per_dataset_contribution": curve,
            "growth_curve": fine_curve,
        }

    return result


def compute_combined_classification(datasets: dict[str, dict]) -> dict:
    """Compute classification distribution across all datasets combined."""
    all_classifications = Counter()
    per_dataset = {}

    for ds_name, ds_data in datasets.items():
        ds_counter = Counter()
        for entry in ds_data["entries"]:
            cls = entry.get("classification", "unknown").lower()
            all_classifications[cls] += 1
            ds_counter[cls] += 1
        per_dataset[ds_name] = dict(ds_counter.most_common())

    return {
        "combined": dict(all_classifications.most_common()),
        "total_entries": sum(all_classifications.values()),
        "total_categories": len(all_classifications),
        "per_dataset": per_dataset,
    }
