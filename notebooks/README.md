# Notebooks

Run them from this directory (`jupyter lab` at the repository root, then open
`notebooks/`). Every path inside is relative to the parent: `../data/`,
`../output/`, `../figures/`.

They are shipped **with their executed output**, so they can be read on GitHub
without running anything.

| # | Notebook | Needs | Produces |
|---|---|---|---|
| 01 | `01_selection_wikitext.ipynb` | `data/WikiText/wikitext-103-raw.wiki.train.raw` | `corpus/selection/wikitext_selection_1000.csv`, the selected `.raw` |
| 02 | `02_selection_fever.ipynb` | `data/FEVER/fever.jsonl` | `corpus/selection/fever_selection_10000.csv`, `data/FEVER/selected/` |
| 03 | `03_selection_europarl.ipynb` | `data/Europarl/en/` | `corpus/selection/europarl_selection_1006.csv`, `data/Europarl/selected/` |
| 04 | `04_pipeline_operational_analysis.ipynb` | the three pipeline traces | The operational findings of the paper's Model Selection and Development Process sections |
| 05 | `05_wikitext_translation_results.ipynb` | the WikiText corpus, trace and pilot | Translation results for the 1,000 selected WikiText articles |
| 06 | `06_corpus_metrics.ipynb` | all three corpora and traces, plus `metrics/` | The comparative metrics and the figures, into `figures/` |

## What each one does

**01–03, subset selection.** One per source dataset, all following the same
procedure: parse the full corpus, compute word counts, filter by length, assign a
topic by keyword matching, then sample stratified by topic and size bin with
`SEED = 42`. They are the implementation of Stage 1 in
[../docs/pipeline.md](../docs/pipeline.md#stage-1--selection).

Only run these if you want to rebuild the samples from the raw sources. The
samples they produced are shipped in `corpus/inputs/`:

```bash
python3 ../scripts/download_datasets.py --all    # ~2.2 GB, needed for 01-03
```

**04, operational analysis.** Reads the pipeline traces to extract what actually
happened during the run: which candidate the reviewer chose as its base and how
often, where the cost went, how the models compared. The source of the
development-process findings in the paper.

**05, WikiText results.** A closer look at one dataset: the translation results
for the 1,000 selected articles, against the June 2026 pilot run. Needs the pilot
files, so run `python3 ../scripts/unpack_corpus.py --pilot` first.

**06, corpus metrics.** The largest one. Imports the `metrics/` package, computes
and compares the measures across all three datasets, and writes the publication
figures. Needs the spaCy models (`en_core_web_sm`, `es_core_news_sm`) for the
syntactic and entity sections; everything else runs without them.

## Before running any of them

```bash
python3 ../scripts/unpack_corpus.py --all
```

Notebooks 04, 05 and 06 read the expanded corpus and traces from `../output/`.

## One change from the original

Notebook 06 wrote its figures straight into the paper's LaTeX source tree. It now
writes to `figures/`, which is the only edit made to any notebook for release.
