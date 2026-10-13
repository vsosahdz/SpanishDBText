#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate again the WikiText entries whose Spanish side stayed in English.

In the production run, entries 6, 52, 264, 308 and 338 came out with the
`spanish` block in English: GPT-4.1 Mini and Mistral Large returned the source
text, the language check kept the last output after its retries, the reviewer
built on one of those candidates, and the evaluator, whose prompt does not state
the target language, scored the result 96-97.

This script runs those entries again through the full production pipeline
(`process_article` in agents/agente_wikitext.py): the same three translators,
reviewer, evaluator and metadata model, declared in the repository's .env. As a
safeguard the pipeline does not have, it checks that the final text is in
Spanish and, if not, runs the entry again (up to MAX_ATTEMPTS times).

Output: results/wikitext_<id>.json with the corpus entry (`result`), the trace
entry (`pipeline_entry`) and the run details. It can be resumed: entries that
already have a result file are skipped. The results of the October 2026
correction run are already in results/, so running it again does nothing unless
those files are removed.

Requires the expanded input:  python3 scripts/unpack_corpus.py --inputs
Usage (from the repository root):
    python3 scripts/correction/retranslate_wikitext.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
load_dotenv(REPO / ".env")
sys.path.insert(0, str(REPO / "agents"))
import agente_wikitext as A  # noqa: E402

IDS = [6, 52, 264, 308, 338]
SOURCE = REPO / "data" / "WikiText" / "wikitext-103-raw.wiki.train.selected_1000.raw"
OUT = HERE / "results"
MAX_ATTEMPTS = 3


def load_articles() -> list[str]:
    """Same as process_file: clean_text, split_articles, ID = position from 1."""
    text = A.clean_text(A.read_file(SOURCE))
    return A.split_articles(text)


def main() -> None:
    if not SOURCE.exists():
        sys.exit(f"Missing {SOURCE.relative_to(REPO)}; run: python3 scripts/unpack_corpus.py --inputs")
    OUT.mkdir(exist_ok=True)
    articles = load_articles()
    assert len(articles) == 1000, len(articles)
    config = A.load_pipeline_config()
    names = [m.name for m in config["translation_models"]]
    A.log.info(f"Translators: {names} | reviewer: {config['reviewer_model'].name} | "
               f"evaluator: {config['evaluator_model'].name}")
    if not A.verify_all_models(config):
        sys.exit("Could not verify the models")

    for aid in IDS:
        path = OUT / f"wikitext_{aid}.json"
        if path.exists():
            A.log.info(f"ID {aid}: already translated again, skipped")
            continue
        article = articles[aid - 1]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            A.log.info(f"ID {aid}: attempt {attempt}/{MAX_ATTEMPTS} — {article.splitlines()[0][:60]}")
            out = A.process_article(article, aid, config)
            if not out:
                A.log.warning(f"ID {aid}: the pipeline returned no result")
                continue
            result, pipeline_entry = out
            lang = A.detect_language(result["spanish"]["text"])
            if lang != "es":
                A.log.warning(f"ID {aid}: the final text is still '{lang}', trying again")
                continue
            path.write_text(json.dumps({
                "article_id": aid,
                "reason": "spanish side left untranslated in the production run",
                "rerun_at": datetime.now().isoformat(timespec="seconds"),
                "attempt": attempt,
                "result": result,
                "pipeline_entry": pipeline_entry,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            cost = pipeline_entry["token_usage"]["cost_usd"]["total_usd"]
            ev = pipeline_entry.get("evaluation") or {}
            A.log.info(f"ID {aid}: done — overall {ev.get('overall_score')}, cost US${cost:.4f}")
            break
        else:
            A.log.error(f"ID {aid}: no Spanish translation after {MAX_ATTEMPTS} attempts")


if __name__ == "__main__":
    main()
