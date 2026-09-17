"""4. Syntactic metrics: sentence length, POS distribution."""

from collections import Counter

import numpy as np

from .loader import tokenize_simple, sentence_split

# Try to import spaCy; if not available, POS metrics will be skipped
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


_SPACY_MODELS = {}


def _get_spacy_model(lang: str):
    """Load spaCy model lazily. Returns None if not available."""
    if not SPACY_AVAILABLE:
        return None

    model_name = "en_core_web_sm" if lang == "en" else "es_core_news_sm"

    if model_name not in _SPACY_MODELS:
        try:
            _SPACY_MODELS[model_name] = spacy.load(model_name)
        except OSError:
            print(f"  [WARN] spaCy model '{model_name}' not installed. "
                  f"Run: python3 -m spacy download {model_name}")
            _SPACY_MODELS[model_name] = None

    return _SPACY_MODELS[model_name]


def compute_syntactic_metrics(texts: list[str], lang: str) -> dict:
    """Compute syntactic metrics: sentence lengths and POS distribution."""

    # Sentence length metrics (no spaCy needed)
    doc_avg_sent_lengths = []
    all_sent_lengths = []

    for text in texts:
        sentences = sentence_split(text)
        sent_lengths = [len(tokenize_simple(s)) for s in sentences]
        if sent_lengths:
            doc_avg_sent_lengths.append(np.mean(sent_lengths))
            all_sent_lengths.extend(sent_lengths)

    sent_arr = np.array(all_sent_lengths) if all_sent_lengths else np.array([0])

    result = {
        "language": lang,
        "sentence_length": {
            "mean": round(float(sent_arr.mean()), 1),
            "median": round(float(np.median(sent_arr)), 1),
            "std": round(float(sent_arr.std()), 1),
            "min": int(sent_arr.min()),
            "max": int(sent_arr.max()),
        },
        "avg_sentence_length_per_doc": {
            "mean": round(float(np.mean(doc_avg_sent_lengths)), 1) if doc_avg_sent_lengths else 0,
            "std": round(float(np.std(doc_avg_sent_lengths)), 1) if doc_avg_sent_lengths else 0,
        },
        "distribution_sentence_lengths": all_sent_lengths[:5000],  # Cap for JSON size
    }

    # POS distribution (requires spaCy)
    nlp = _get_spacy_model(lang)
    if nlp is not None:
        print(f"  Computing POS distribution ({lang})...")
        pos_counter = Counter()
        total_pos_tokens = 0

        # Process in batches for efficiency
        # Use first 500 chars of each doc to keep it fast
        sample_texts = [t[:2000] for t in texts]

        for doc in nlp.pipe(sample_texts, batch_size=50, n_process=1):
            for token in doc:
                if not token.is_space and not token.is_punct:
                    pos_counter[token.pos_] += 1
                    total_pos_tokens += 1

        pos_distribution = {}
        for pos, count in pos_counter.most_common():
            pos_distribution[pos] = {
                "count": count,
                "percentage": round(count / total_pos_tokens * 100, 2) if total_pos_tokens > 0 else 0,
            }

        result["pos_distribution"] = pos_distribution
        result["pos_total_tokens_analyzed"] = total_pos_tokens
    else:
        result["pos_distribution"] = None
        result["pos_note"] = "spaCy not available — install spaCy and language model to compute POS"

    return result
