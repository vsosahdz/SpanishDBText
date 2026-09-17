"""5. Quality and semantic metrics: duplicates, expansion ratio, NER, special chars, etc."""

import hashlib
import re
from collections import Counter

import numpy as np

from .loader import tokenize_simple

# Try to import spaCy for NER
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

_SPACY_MODELS = {}


def _get_spacy_model(lang: str):
    if not SPACY_AVAILABLE:
        return None
    model_name = "en_core_web_sm" if lang == "en" else "es_core_news_sm"
    if model_name not in _SPACY_MODELS:
        try:
            _SPACY_MODELS[model_name] = spacy.load(model_name)
        except OSError:
            _SPACY_MODELS[model_name] = None
    return _SPACY_MODELS[model_name]


def _text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def _shingle_set(text: str, k: int = 5) -> set:
    """Create k-shingle set for near-duplicate detection."""
    words = tokenize_simple(text)
    if len(words) < k:
        return set(tuple(words))
    return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0


def compute_quality_metrics(entries: list[dict]) -> dict:
    """Compute quality metrics across the bilingual corpus."""

    en_texts = [e["en_text"] for e in entries]
    es_texts = [e["es_text"] for e in entries]

    # ── Expansion ratio EN→ES (character-based: len(es_chars) / len(en_chars)) ──
    expansion_ratios = []
    for en, es in zip(en_texts, es_texts):
        if len(en) > 0:
            expansion_ratios.append(len(es) / len(en))

    exp_arr = np.array(expansion_ratios) if expansion_ratios else np.array([0])

    # ── Duplicates (exact) ──
    en_hashes = [_text_hash(t) for t in en_texts]
    es_hashes = [_text_hash(t) for t in es_texts]

    en_duplicates = len(en_hashes) - len(set(en_hashes))
    es_duplicates = len(es_hashes) - len(set(es_hashes))

    # ── Near-duplicates (Jaccard on shingles, sample-based for speed) ──
    near_dup_threshold = 0.85
    sample_size = min(500, len(en_texts))
    near_dup_pairs_en = 0
    near_dup_pairs_es = 0

    if sample_size > 1:
        indices = np.random.default_rng(42).choice(len(en_texts), sample_size, replace=False)
        en_shingles = [_shingle_set(en_texts[i]) for i in indices]
        es_shingles = [_shingle_set(es_texts[i]) for i in indices]

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if _jaccard(en_shingles[i], en_shingles[j]) >= near_dup_threshold:
                    near_dup_pairs_en += 1
                if _jaccard(es_shingles[i], es_shingles[j]) >= near_dup_threshold:
                    near_dup_pairs_es += 1

    # ── Special characters and uppercase ──
    en_special_pcts = []
    es_special_pcts = []
    en_upper_pcts = []
    es_upper_pcts = []

    for en, es in zip(en_texts, es_texts):
        if len(en) > 0:
            en_special = sum(1 for c in en if not c.isalnum() and not c.isspace())
            en_upper = sum(1 for c in en if c.isupper())
            en_special_pcts.append(en_special / len(en) * 100)
            en_upper_pcts.append(en_upper / len(en) * 100)
        if len(es) > 0:
            es_special = sum(1 for c in es if not c.isalnum() and not c.isspace())
            es_upper = sum(1 for c in es if c.isupper())
            es_special_pcts.append(es_special / len(es) * 100)
            es_upper_pcts.append(es_upper / len(es) * 100)

    # ── Stopword percentage ──
    en_stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                    "have", "has", "had", "do", "does", "did", "will", "would", "could",
                    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                    "on", "with", "at", "by", "from", "as", "into", "through", "during",
                    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
                    "it", "its", "this", "that", "these", "those", "he", "she", "they",
                    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
                    "his", "our", "their", "which", "who", "whom", "what", "where", "when"}
    es_stopwords = {"de", "la", "el", "en", "los", "las", "del", "que", "por", "con",
                    "una", "un", "para", "como", "fue", "son", "se", "es", "al", "lo",
                    "su", "más", "este", "esta", "pero", "sus", "le", "ya", "o", "y",
                    "no", "ha", "era", "muy", "también", "entre", "sin", "sobre", "todo",
                    "ser", "hay", "dos", "hasta", "desde", "mi", "nos", "les", "me",
                    "ni", "si", "donde", "quien", "cual", "cuando", "cada", "porque"}

    en_all_words = []
    es_all_words = []
    for en, es in zip(en_texts, es_texts):
        en_all_words.extend(tokenize_simple(en))
        es_all_words.extend(tokenize_simple(es))

    en_stop_count = sum(1 for w in en_all_words if w in en_stopwords)
    es_stop_count = sum(1 for w in es_all_words if w in es_stopwords)

    # ── Classification distribution ──
    classifications = Counter(e["classification"] for e in entries)

    # ── NER (if spaCy available) ──
    ner_en = _compute_ner(en_texts, "en")
    ner_es = _compute_ner(es_texts, "es")

    return {
        "expansion_ratio_en_es": {
            "mean": round(float(exp_arr.mean()), 3),
            "median": round(float(np.median(exp_arr)), 3),
            "std": round(float(exp_arr.std()), 3),
            "min": round(float(exp_arr.min()), 3),
            "max": round(float(exp_arr.max()), 3),
            "distribution": [round(r, 3) for r in expansion_ratios],
        },
        "duplicates": {
            "exact_en": en_duplicates,
            "exact_es": es_duplicates,
            "near_duplicate_pairs_en": near_dup_pairs_en,
            "near_duplicate_pairs_es": near_dup_pairs_es,
            "near_duplicate_sample_size": sample_size,
            "near_duplicate_threshold": near_dup_threshold,
        },
        "special_characters_pct": {
            "en_mean": round(float(np.mean(en_special_pcts)), 2) if en_special_pcts else 0,
            "es_mean": round(float(np.mean(es_special_pcts)), 2) if es_special_pcts else 0,
        },
        "uppercase_pct": {
            "en_mean": round(float(np.mean(en_upper_pcts)), 2) if en_upper_pcts else 0,
            "es_mean": round(float(np.mean(es_upper_pcts)), 2) if es_upper_pcts else 0,
        },
        "stopword_pct": {
            "en": round(en_stop_count / len(en_all_words) * 100, 2) if en_all_words else 0,
            "es": round(es_stop_count / len(es_all_words) * 100, 2) if es_all_words else 0,
        },
        "classification_distribution": dict(classifications.most_common()),
        "ner_en": ner_en,
        "ner_es": ner_es,
    }


def _compute_ner(texts: list[str], lang: str) -> dict | None:
    """Compute NER statistics if spaCy is available."""
    nlp = _get_spacy_model(lang)
    if nlp is None:
        return None

    print(f"  Computing NER ({lang})...")
    entity_counter = Counter()
    entity_type_counter = Counter()
    total_entities = 0

    # Use first 2000 chars per doc for speed
    sample_texts = [t[:2000] for t in texts]

    for doc in nlp.pipe(sample_texts, batch_size=50, n_process=1):
        for ent in doc.ents:
            entity_counter[ent.text.lower()] += 1
            entity_type_counter[ent.label_] += 1
            total_entities += 1

    unique_entities = len(entity_counter)

    return {
        "total_entities": total_entities,
        "unique_entities": unique_entities,
        "entity_diversity": round(unique_entities / total_entities, 4) if total_entities > 0 else 0,
        "entity_types": dict(entity_type_counter.most_common()),
        "top_entities": dict(entity_counter.most_common(30)),
    }
