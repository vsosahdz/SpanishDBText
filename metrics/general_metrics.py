"""1. General corpus metrics and 2. Word frequency metrics."""

import math
from collections import Counter

import numpy as np

from .loader import tokenize_simple, sentence_split


def compute_general_metrics(texts: list[str], lang: str) -> dict:
    """Compute general corpus statistics for a list of texts."""

    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise ValueError(f"No non-empty texts provided for language '{lang}'")

    all_words = []
    doc_word_counts = []
    doc_sentence_counts = []
    doc_char_counts = []

    for text in texts:
        words = tokenize_simple(text)
        sentences = sentence_split(text)
        all_words.extend(words)
        doc_word_counts.append(len(words))
        doc_sentence_counts.append(len(sentences))
        doc_char_counts.append(len(text))

    word_counts_arr = np.array(doc_word_counts)
    sent_counts_arr = np.array(doc_sentence_counts)
    char_counts_arr = np.array(doc_char_counts)

    total_tokens = len(all_words)
    unique_types = len(set(all_words))

    return {
        "language": lang,
        "num_documents": len(texts),
        "total_tokens": total_tokens,
        "unique_types": unique_types,
        "total_characters": int(char_counts_arr.sum()),
        "words_per_document": {
            "mean": round(float(word_counts_arr.mean()), 1),
            "median": round(float(np.median(word_counts_arr)), 1),
            "std": round(float(word_counts_arr.std()), 1),
            "min": int(word_counts_arr.min()),
            "max": int(word_counts_arr.max()),
        },
        "sentences_per_document": {
            "mean": round(float(sent_counts_arr.mean()), 1),
            "median": round(float(np.median(sent_counts_arr)), 1),
            "std": round(float(sent_counts_arr.std()), 1),
            "min": int(sent_counts_arr.min()),
            "max": int(sent_counts_arr.max()),
        },
        "characters_per_document": {
            "mean": round(float(char_counts_arr.mean()), 1),
            "median": round(float(np.median(char_counts_arr)), 1),
            "std": round(float(char_counts_arr.std()), 1),
            "min": int(char_counts_arr.min()),
            "max": int(char_counts_arr.max()),
        },
        # Distribution data for plotting
        "distribution_word_counts": doc_word_counts,
        "distribution_sentence_counts": doc_sentence_counts,
        "distribution_char_counts": doc_char_counts,
    }


def compute_frequency_metrics(texts: list[str], lang: str) -> dict:
    """Compute word frequency metrics: Zipf, hapax, dis legomena, coverage, vocabulary growth."""

    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise ValueError(f"No non-empty texts provided for language '{lang}'")

    all_words = []
    for text in texts:
        all_words.extend(tokenize_simple(text))

    freq = Counter(all_words)
    total_tokens = len(all_words)
    total_types = len(freq)

    if total_tokens == 0:
        raise ValueError(f"No tokens found in texts for language '{lang}'")

    # Hapax legomena (frequency == 1) and dis legomena (frequency == 2)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    dis = sum(1 for w, c in freq.items() if c == 2)

    # Zipf distribution: rank vs frequency (top 500 for plotting)
    sorted_freqs = sorted(freq.values(), reverse=True)
    zipf_data = []
    for rank, f in enumerate(sorted_freqs[:500], start=1):
        zipf_data.append({"rank": rank, "frequency": f,
                          "log_rank": round(math.log10(rank), 4),
                          "log_freq": round(math.log10(f), 4) if f > 0 else 0})

    # Zipf exponent estimation (linear regression on log-log)
    if len(zipf_data) > 10:
        log_ranks = np.array([z["log_rank"] for z in zipf_data])
        log_freqs = np.array([z["log_freq"] for z in zipf_data])
        # slope of log(freq) = -alpha * log(rank) + c
        coeffs = np.polyfit(log_ranks, log_freqs, 1)
        zipf_exponent = round(-coeffs[0], 3)
    else:
        zipf_exponent = None

    # Vocabulary coverage: what % of tokens do the top-N words cover
    coverage = {}
    cumulative = 0
    for i, (word, count) in enumerate(freq.most_common()):
        cumulative += count
        if (i + 1) in (100, 500, 1000, 5000, 10000):
            coverage[str(i + 1)] = round(cumulative / total_tokens * 100, 2)

    # Vocabulary growth curve: types seen as tokens accumulate
    # Sample at intervals to keep data manageable
    growth_data = []
    seen = set()
    step = max(1, total_tokens // 200)  # ~200 data points
    for i, word in enumerate(all_words):
        seen.add(word)
        if (i + 1) % step == 0 or i == total_tokens - 1:
            growth_data.append({"tokens": i + 1, "types": len(seen)})

    return {
        "language": lang,
        "total_tokens": total_tokens,
        "total_types": total_types,
        "hapax_legomena": hapax,
        "hapax_legomena_pct": round(hapax / total_types * 100, 2) if total_types > 0 else 0,
        "dis_legomena": dis,
        "dis_legomena_pct": round(dis / total_types * 100, 2) if total_types > 0 else 0,
        "zipf_exponent": zipf_exponent,
        "vocabulary_coverage_by_top_n": coverage,
        # Plot data
        "zipf_distribution": zipf_data,
        "vocabulary_growth_curve": growth_data,
    }
