"""Build data/inputs.jsonl with the 62 entries of the 7 June 2026 trial run.

Each entry holds the original as the evaluator received it (``english.text`` is the
``original_text`` that the pipeline stores) and the final translation (``spanish.text``).
It also holds the score given by the evaluator of that run (Mistral Large),
only as a historical reference.

Usage:
    python prepare_inputs.py
"""
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "corpus" / "pilot"
MAIN = PILOT / "wikitext-103-raw.wiki.test_260607.json.gz"
PIPELINE = PILOT / "wikitext-103-raw.wiki.test_pipeline_260607.json.gz"
OUT = HERE / "data" / "inputs.jsonl"

# Truncations found when reviewing the run (length ratio < 0.80)
KNOWN_TRUNCATED = {7, 24, 41}


def main() -> None:
    main_data = json.load(gzip.open(MAIN, "rt", encoding="utf-8"))["payload"][0]["data"]
    pipeline = json.load(gzip.open(PIPELINE, "rt", encoding="utf-8"))["pipeline"]
    historic = {e["article_id"]: e.get("evaluation") or {} for e in pipeline}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in sorted(main_data, key=lambda e: e["id"]):
        aid = entry["id"]
        original = entry["english"]["text"]
        translation = entry["spanish"]["text"]
        h = historic.get(aid, {})
        rows.append({
            "article_id": aid,
            "title": entry["english"].get("title", ""),
            "original": original,
            "translation": translation,
            "original_chars": len(original),
            "translation_chars": len(translation),
            "char_ratio": round(len(translation) / len(original), 3),
            "known_truncated": aid in KNOWN_TRUNCATED,
            "historic_mistral_overall": h.get("overall_score"),
            "historic_mistral_completeness": h.get("completeness"),
        })

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    trunc = [r for r in rows if r["known_truncated"]]
    print(f"{len(rows)} entries -> {OUT.relative_to(HERE)}")
    for r in trunc:
        print(f"  truncated id={r['article_id']:>2} ratio={r['char_ratio']:.2f} "
              f"({r['translation_chars']:,}/{r['original_chars']:,})  {r['title']}")


if __name__ == "__main__":
    main()
