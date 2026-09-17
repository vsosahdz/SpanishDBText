# The corpus

Everything here is gzipped. Expand it into the layout the code expects with:

```bash
python3 ../scripts/unpack_corpus.py --all      # or --final, --pipeline, --inputs, --pilot
```

Field-by-field documentation is in [../docs/corpus-format.md](../docs/corpus-format.md).

---

## At a glance

| Dataset | Domain | Entries | English words | Spanish words | Fidelity (LaBSE) | Omission |
|---|---|---:|---:|---:|---:|---:|
| WikiText-103 | Encyclopedic | 1,000 | 2,855,853 | 3,037,875 | 0.9170 | 1.43% |
| FEVER | Factual | 9,998 | 1,753,860 | 1,941,463 | 0.9228 | 1.16% |
| Europarl | Parliamentary | 1,005 | 6,950,546 | 6,954,048 | 0.9199 | 0.34% |
| **Total** | | **12,003** | **11,560,259** | **11,933,386** | | |

Spanish is 3.2% longer than English in word count — consistent with Spanish using
more function words, and with translations being more explicit than their
sources. Combined vocabulary is 117,323 unique English words and 151,264 Spanish.

Quality scores assigned by the Stage 5 evaluator, with no human reference:

| Dimension | WikiText | FEVER | Europarl |
|---|---:|---:|---:|
| Clarity | 97.7 | 96.9 | 97.7 |
| Fidelity | 97.6 | 96.7 | 98.1 |
| Completeness | 99.8 | 99.7 | 100.0 |
| Fluency | 97.5 | 96.7 | 97.4 |
| **Overall** | **95.9** | **95.1** | **97.0** |
| Share scoring ≥ 90 | 99.5% | 98.5% | 100.0% |

---

## What is in each directory

### `final/` — the corpus, 159 MB expanded

The clean bilingual result: one entry per record, with text, title, keywords,
summary and topic classification on both sides. This is what most users want.

```
wikitext-103-raw.wiki.train.selected_1000_260620_235838.json.gz    13 MB → 38 MB
fever_selected_260712_212751.json.gz                               11 MB → 36 MB
selected_260710_084528.json.gz                                     28 MB → 85 MB
```

### `pipeline/` — the traces, 278 MB expanded

One record per entry holding everything the pipeline decided: the three
candidate translations, the reviewer's per-candidate critique and its choice of
base, the evaluator's four dimension scores and error list, and token counts and
cost broken down per stage and per model.

This is what makes the cost analysis and the Stage 4 ablation possible without
retranslating anything. Join to the corpus on `article_id` ↔ `id`.

### `inputs/` — the English source subsets, 87 MB expanded

Exactly what the pipeline consumed: the stratified samples, not the full source
datasets. Shipped so that a run can be reproduced without downloading ~7 GB and
re-sampling.

```
wikitext-103-raw.wiki.train.selected_1000.raw.gz     1,000 articles
fever_selected.jsonl.gz                             10,000 passages
europarl_selected_1006.tar.gz                        1,006 debate files
```

### `selection/` — how the subsets were chosen

The tables produced by notebooks 01–03: which entries were selected, with the
topic and size bin each was stratified on. Useful for auditing the sampling
without rerunning it.

### `metrics/` — the computed results

The metrics files behind the paper's tables, kept so that a rerun can be
compared against them:

| File | Contents |
|---|---|
| `metrics_wikitext.json`, `metrics_fever.json`, `metrics_europarl.json` | General, frequency, lexical, syntactic and quality metrics, per dataset, both languages |
| `metrics_combined.json` | Cross-dataset totals, vocabulary overlap, duplicates, vocabulary growth curve |
| `semantic_full.json` | LaBSE cross-lingual fidelity and omission, all 12,003 entries |
| `ablation_consensus.json` | Consensus vs each single model and vs the per-entry oracle |

### `pilot/` — the June 2026 WikiText pilot

The trial run made before the production configuration was frozen, plus its log.
Notebook 05 reads it. Small, and the only surviving record of how the pipeline
behaved before the final settings.

### `samples/` — readable on GitHub

A few entries per dataset, uncompressed and pretty-printed, plus one pipeline
trace showing the three candidates, the critique and the scores for a single
FEVER entry. Selected at evenly spaced length percentiles rather than taken from
the top of the file, so they look like real records.

---

## Caveats

**Three entries are missing.** 12,006 were selected, 12,003 produced: 2 FEVER and
1 Europarl entries failed on API errors and were not retried.

**Four FEVER entries have no quality scores.** They were translated and reviewed
successfully but their evaluation call failed. They are complete as text. The
paper's evaluation statistics are therefore over 9,994 FEVER entries, not 9,998.

**Europarl keeps its XML markup on both sides.** `<CHAPTER>`, `<SPEAKER>` and
`<P>` tags are content in a parliamentary transcript, so they were preserved —
and they are counted as words by the totals above (4.2% of Europarl's words on
each side, 2.5% of the combined corpus). Strip them before computing surface
statistics, or you will be measuring tags.

**The English side keeps its source tokenisation.** WikiText and FEVER text
arrives with punctuation spaced out (`league .`, `5 -- 11`). The Spanish side is
normal prose. Normalise both before sentence segmentation — skipping that step
raises WikiText's measured omission rate from 1.4% to 4.5%, because the source
gets split at positions that are not sentence boundaries and the fragments have
no counterpart to match.

**Provenance fields are approximate.** `url` and `authors_name` in the corpus
headers were filled in by hand at run time; the WikiText file names the Kaggle
re-upload rather than Merity et al. (2017). Cite from
[../LICENSE-DATA.md](../LICENSE-DATA.md) instead.

---

## Licence

**Everything in this directory is CC BY-SA 4.0, not Apache 2.0.** GitHub labels
the repository "Apache-2.0" because it reads only the top-level `LICENSE` file;
that label covers the code, not the corpus.

The corpus licence is inherited, not chosen: two of the three sources are
CC BY-SA text from Wikipedia, and a translation is a derivative work, so the
ShareAlike obligation carries over. Redistributing the corpus means crediting
WikiText-103, FEVER and Europarl, and licensing your version compatibly. Full
terms and source attribution in [../LICENSE-DATA.md](../LICENSE-DATA.md).
