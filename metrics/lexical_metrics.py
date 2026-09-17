"""3. Lexical diversity metrics: TTR, RTTR, CTTR, MATTR, MTLD."""

import math

import numpy as np

from .loader import tokenize_simple


def _ttr(types: int, tokens: int) -> float:
    return types / tokens if tokens > 0 else 0


def _rttr(types: int, tokens: int) -> float:
    return types / math.sqrt(tokens) if tokens > 0 else 0


def _cttr(types: int, tokens: int) -> float:
    return types / math.sqrt(2 * tokens) if tokens > 0 else 0


def _mattr(words: list[str], window: int = 100) -> float:
    """Moving Average Type-Token Ratio with a sliding window."""
    if len(words) < window:
        return _ttr(len(set(words)), len(words))

    ttrs = []
    for i in range(len(words) - window + 1):
        segment = words[i:i + window]
        ttrs.append(len(set(segment)) / window)
    return sum(ttrs) / len(ttrs)


def _mtld(words: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity.

    Counts how many consecutive segments maintain TTR above the threshold.
    Runs forward and backward, averages the result.
    """
    def _mtld_one_direction(word_list):
        factor_count = 0
        current_start = 0

        for i in range(1, len(word_list) + 1):
            segment = word_list[current_start:i]
            ttr = len(set(segment)) / len(segment)
            if ttr <= threshold:
                factor_count += 1
                current_start = i

        # Partial factor for remaining tokens
        if current_start < len(word_list):
            remaining = word_list[current_start:]
            remaining_ttr = len(set(remaining)) / len(remaining)
            if remaining_ttr < 1.0:
                factor_count += (1 - remaining_ttr) / (1 - threshold)

        return len(word_list) / factor_count if factor_count > 0 else len(word_list)

    if len(words) < 10:
        return 0

    forward = _mtld_one_direction(words)
    backward = _mtld_one_direction(words[::-1])
    return (forward + backward) / 2


def compute_lexical_metrics(texts: list[str], lang: str) -> dict:
    """Compute lexical diversity metrics for the full corpus and per-document averages."""

    # Corpus-level
    all_words = []
    for text in texts:
        all_words.extend(tokenize_simple(text))

    total_tokens = len(all_words)
    total_types = len(set(all_words))

    corpus_ttr = round(_ttr(total_types, total_tokens), 4)
    corpus_rttr = round(_rttr(total_types, total_tokens), 4)
    corpus_cttr = round(_cttr(total_types, total_tokens), 4)
    corpus_mattr = round(_mattr(all_words, window=100), 4)
    corpus_mtld = round(_mtld(all_words), 2)

    # Per-document metrics (for distributions)
    doc_ttrs = []
    doc_mattrs = []
    doc_mtlds = []

    for text in texts:
        words = tokenize_simple(text)
        if len(words) < 10:
            continue
        types = len(set(words))
        doc_ttrs.append(round(_ttr(types, len(words)), 4))
        doc_mattrs.append(round(_mattr(words, window=min(100, len(words))), 4))
        doc_mtlds.append(round(_mtld(words), 2))

    return {
        "language": lang,
        "corpus_level": {
            "vocabulary_size": total_types,
            "total_tokens": total_tokens,
            "ttr": corpus_ttr,
            "rttr": corpus_rttr,
            "cttr": corpus_cttr,
            "mattr_w100": corpus_mattr,
            "mtld": corpus_mtld,
        },
        "per_document_averages": {
            "ttr_mean": round(float(np.mean(doc_ttrs)), 4) if doc_ttrs else 0,
            "ttr_std": round(float(np.std(doc_ttrs)), 4) if doc_ttrs else 0,
            "mattr_mean": round(float(np.mean(doc_mattrs)), 4) if doc_mattrs else 0,
            "mattr_std": round(float(np.std(doc_mattrs)), 4) if doc_mattrs else 0,
            "mtld_mean": round(float(np.mean(doc_mtlds)), 2) if doc_mtlds else 0,
            "mtld_std": round(float(np.std(doc_mtlds)), 2) if doc_mtlds else 0,
        },
        # Plot data
        "distribution_ttr": doc_ttrs,
        "distribution_mattr": doc_mattrs,
        "distribution_mtld": doc_mtlds,
    }
