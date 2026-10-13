# Correction run: five WikiText entries translated again

In the production run, WikiText entries **6, 52, 264, 308 and 338** were stored with
their Spanish side still in English. GPT-4.1 Mini and Mistral Large returned the
source text, the language check of Stage 3 kept the last output after its retries,
the reviewer built its translation on one of those candidates, and the evaluator,
whose prompt does not state the target language, scored the result 96–97. The
problem was found while auditing the corpus in October 2026.

The five entries were translated again with the same pipeline and models, and the
corpus, the trace and every metric were updated. The released files under
`corpus/` already contain the corrected entries; these scripts and results are the
record of how that was done.

| File | What it is |
|---|---|
| `retranslate_wikitext.py` | Runs the five entries through the full production pipeline (`agents/agente_wikitext.py`, models from `.env`) and checks that the final text is in Spanish; if not, it runs the entry again |
| `apply_retranslation.py` | Replaces the Spanish block of the five corpus entries (the English block and its metadata are kept) and their trace entries, in `corpus/` and in the expanded copies under `output/`. It adds a `rerun` block to each trace entry and a `corrections` block to the trace summary |
| `recompute_combined.py` | Recomputes `corpus/metrics/metrics_combined.json` |
| `results/wikitext_<id>.json` | The corpus entry and the trace entry produced for each ID |
| `results/replaced_entries.json` | The original entries that were replaced |

**Result:** the five entries are in Spanish, with overall scores of 94–97. GPT-4.1
Mini and Mistral Large again returned English on their first attempt, and this time
the language retry fixed it. The run cost US\$0.40, which brings the total cost of
the corpus to US\$454.30; the `total_cost_usd` field of the trace keeps the
production run alone (US\$453.90).

After the correction, all metrics were recomputed (`python3 -m metrics.run_metrics --all`,
notebook 06 and `recompute_combined.py`), and notebooks 04–07 were run again.

The pipeline code in `agents/` is the one that produced the corpus and was not
changed. A language check on the final translation, and stating the target language
in the evaluator prompt, would have caught these entries during the run; notebook 07
lists both as recommendations.
