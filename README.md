# A Bilingual English–Spanish Parallel Corpus Built with Multi-Model LLM Consensus

*[Leer en español](README.es.md)*

This repository holds the corpus, the code and the analysis behind the paper
**"Building a Bilingual English–Spanish Parallel Corpus Using Multi-Model LLM
Translation with Consensus Review"**.

The corpus contains **12,003 aligned English–Spanish entries and 23.5 million
words** drawn from three domains — encyclopedic articles (WikiText-103), short
factual passages (FEVER) and parliamentary debate (Europarl). It was produced
entirely by general-purpose language models: three models translate every text
in parallel, a reviewer model merges their candidates into one consensus
translation, and an evaluator model scores the result against an anchored
scoring guide that needs no human reference translation. The whole corpus cost **\$454.30**,
or \$39 per million source words, on a single consumer laptop.

Everything needed to inspect the corpus, reproduce the analysis or rerun the
pipeline on new text is here.

---

## What is in the corpus

| Dataset | Domain | Entries | English words | Spanish words | Mean quality score |
|---|---|---:|---:|---:|---:|
| WikiText-103 | Encyclopedic | 1,000 | 2,855,853 | 3,038,960 | 95.9 |
| FEVER | Factual | 9,998 | 1,753,843 | 1,941,480 | 95.1 |
| Europarl | Parliamentary | 1,005 | 6,950,546 | 6,954,048 | 97.0 |
| **Total** | | **12,003** | **11,560,242** | **11,934,488** | **—** |

Cross-lingual semantic fidelity, measured with LaBSE against no reference
translation, is 0.92 and varies by about six thousandths across the three domains.
Sentence-level omission stays between 0.34% and 1.43%.

Every entry carries a title, keywords, a summary and a topic classification on
both sides, plus the quality scores the evaluator assigned. The exact record
layout is documented in [docs/corpus-format.md](docs/corpus-format.md).

**Look at the data before downloading it:** [`corpus/samples/`](corpus/samples/)
holds a handful of entries per dataset, pretty-printed and readable directly on
GitHub, including one pipeline trace showing the three candidate translations,
the reviewer's merge and the evaluator's scores for a single entry.

---

## Quick start

```bash
git clone https://github.com/vsosahdz/SpanishDBText.git
cd SpanishDBText

python3 -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python3 scripts/unpack_corpus.py --final
```

That expands the bilingual corpus into `output/`, which is where every script
and notebook in this repository looks for it:

```
output/wikitext/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json
output/FEVER/fever_selected_260712_212751.json
output/Europarl/selected_260710_084528.json
```

Read one of them with nothing but the standard library:

```python
import json

with open("output/FEVER/fever_selected_260712_212751.json", encoding="utf-8") as fh:
    corpus = json.load(fh)

for entry in corpus["payload"][0]["data"][:3]:
    print(entry["english"]["title"])
    print("  EN:", entry["english"]["text"][:120])
    print("  ES:", entry["spanish"]["text"][:120])
```

To go further — rerun the metrics, rerun the pipeline, or apply it to your own
text — follow [docs/reproduction.md](docs/reproduction.md).

---

## Why the data is compressed

The corpus is 437 MB of JSON. Shipped as `.json.gz` it fits in a GitHub
repository, with every file under the 100 MB per-file limit.

The code is the code that produced the corpus. The agents in `agents/` are
exactly as they ran in production, so their comments and log messages are still
in Spanish, the working language of the project. In the metrics package, the
ablation code and the notebooks, the comments and explanatory text were
translated into English for release, and no computation changed. All of them read
plain JSON from `output/` and `data/` exactly as they did during production, so
`scripts/unpack_corpus.py` restores that layout, filenames included.
[docs/reproduction.md](docs/reproduction.md#changes-made-for-release) lists every
change made for release.

```bash
python3 scripts/unpack_corpus.py --final      # the bilingual corpus       159 MB
python3 scripts/unpack_corpus.py --pipeline   # per-entry pipeline traces  278 MB
python3 scripts/unpack_corpus.py --inputs     # the selected English input  87 MB
python3 scripts/unpack_corpus.py --all        # everything                 530 MB
```

---

## Repository layout

```
.
├── corpus/                      the released data, compressed
│   ├── final/                   the bilingual corpus, one file per dataset
│   ├── pipeline/                per-entry traces: candidates, review, scores, cost
│   ├── inputs/                  the selected English subsets the pipeline consumed
│   ├── selection/               the stratified selection tables (CSV)
│   ├── metrics/                 the computed metrics behind the paper's tables
│   ├── pilot/                   the June 2026 WikiText trial runs
│   └── samples/                 a few entries, uncompressed and readable on GitHub
│
├── agents/                      the pipeline, one agent per source dataset
│   ├── agente_wikitext.py       also the shared core: providers, prompts, stages
│   ├── agente_fever.py
│   ├── agente_europarl.py
│   └── format_europarl.py       normalises Europarl markup to WikiText headings
│
├── metrics/                     the evaluation measures, as an importable package
│   ├── general_metrics.py       totals, lengths, word frequency, Zipf, hapax
│   ├── lexical_metrics.py       TTR, RTTR, CTTR, MATTR, MTLD
│   ├── syntactic_metrics.py     sentence length, POS distribution (spaCy)
│   ├── quality_metrics.py       named entities, duplicates, expansion ratio
│   ├── semantic_metrics.py      cross-lingual fidelity and omission (LaBSE)
│   ├── combined_metrics.py      cross-dataset totals, overlap, vocabulary growth
│   ├── ablation_consensus.py    the Stage 4 ablation: consensus vs its candidates
│   └── run_metrics.py           entry point: python3 -m metrics.run_metrics --all
│
├── prompts/                     every LLM instruction as plain text, one file each
├── ablation/                    the Stage 5 evaluator ablation: scoring guide × length evidence
├── notebooks/                   01–03 select the subsets, 04–06 analyse the output, 07 recommendations
├── scripts/                     download, unpack, pack, verify, sample, export prompts
│   └── correction/              the October 2026 correction run: five WikiText entries
├── docs/                        guides and reference, English and Spanish
├── data/  output/  figures/     empty; filled by the scripts
│
├── .env.example                 configuration template — copy to .env
├── docker-compose.yml           Ollama, for the locally hosted model
├── requirements.txt
├── LICENSE                      Apache 2.0, for the code
├── LICENSE-DATA.md              CC BY-SA 4.0, for the corpus, and source attribution
└── CITATION.cff
```

---

## Documentation

| Document | What it covers |
|---|---|
| [docs/reproduction.md](docs/reproduction.md) | Step by step: install, configure, rerun the metrics, rerun the pipeline, apply it to your own text |
| [docs/pipeline.md](docs/pipeline.md) | The six stages, what each one does and where it lives in the code |
| [docs/configuration.md](docs/configuration.md) | Every `.env` variable, every pipeline parameter and why it has the value it has |
| [docs/corpus-format.md](docs/corpus-format.md) | The JSON layout of the corpus files and of the pipeline traces, field by field |
| [docs/metrics-reference.md](docs/metrics-reference.md) | Every metric: what it measures, who introduced it, how it is computed here |
| [docs/metrics-results.md](docs/metrics-results.md) | The computed results, read critically, anomalies included |
| [docs/supplementary/](docs/supplementary/) | The paper's supplementary material (PDF and LaTeX source): the per-dataset tables and the Zipf figure |
| [prompts/README.md](prompts/README.md) | Every prompt the pipeline sends, what each one does and why it is worded that way |

Every guide has a Spanish version in [docs/es/](docs/es/); the supplementary material is in English only.

---

## The pipeline in one paragraph

Each entry passes through six stages. **Selection** takes a stratified sample
from the source dataset (seed 42). **Preprocessing** detects the language,
cleans the text and splits anything above ~6,000 tokens at a structural boundary
— section headers for WikiText, speaker turns for Europarl. **Translation** sends
the text to three models at once: GPT-4.1 Mini, Mistral Large and a locally
hosted Qwen 3.5 9B, chosen from three different families so that they do not
make the same mistakes. **Review** gives GPT-4.1 the original and all three
candidates, and asks it to critique them and merge them into one translation;
this single stage is 77% of the cost. **Evaluation** scores the result on
clarity, fidelity, completeness and fluency, with a programmatic character-ratio
check that overrides the model when it is too generous about truncation.
**Metadata** extracts the title, keywords, summary and classification on both
sides. Full detail in [docs/pipeline.md](docs/pipeline.md).

---

## Rerunning the pipeline

Reading the corpus needs none of this. Running it does, and it costs money — so
start with `--limit 10` and read the reported cost before scaling up.

**1. Configure.** Copy the template and fill in your endpoints, keys and
deployment names. `.env` is git-ignored and must stay that way.

```bash
cp .env.example .env
```

**2. Start Ollama**, if any role uses the `ollama` provider. The published
configuration runs Qwen 3.5 9B locally, which is what keeps a third of the
translation cost at zero.

```bash
docker compose up -d
docker exec -it ollama ollama pull qwen3.5:9b
curl http://localhost:11434/api/tags          # should list the model
```

Uncomment the `deploy:` block in `docker-compose.yml` to give the container your
NVIDIA GPU. Without it Ollama runs on the CPU — it works, but the local model is
already the slowest part of the pipeline and on CPU it dominates the total
runtime.

**3. Check that every configured model answers** before committing hours to a
run. This sends one short prompt per model and costs a fraction of a cent.

```bash
python3 scripts/verify_setup.py --models
```

**4. Run**, from the repository root — the agents resolve `data/` and `output/`
relative to the working directory.

```bash
python3 scripts/unpack_corpus.py --inputs
python3 agents/agente_fever.py --input data/FEVER/selected/fever_selected.jsonl --limit 10
```

An interrupted run is resumed by relaunching the same command: the pipeline
writes a progress file after every completed entry and skips what is already
done.

Every option, every parameter and the full procedure for the other two agents are
in [docs/reproduction.md](docs/reproduction.md#4-rerun-the-pipeline) and
[docs/configuration.md](docs/configuration.md).

---

## Reusing the pipeline for another language or domain

Nothing in the design is specific to English–Spanish. The models are declared in
`.env` as `provider:model` pairs and are read at startup, so swapping any role
needs no code change. Pointing the pipeline at a new language means editing the
prompts in `agents/agente_wikitext.py`, which are plain string constants grouped
together near the top of the file. A new source format means one new agent that
imports the shared core and supplies its own loader and chunking rule — the three
existing agents are the worked examples, and `agente_fever.py` is the smallest of
them. See [docs/reproduction.md](docs/reproduction.md#using-the-pipeline-on-your-own-text).

---

## Cost, if you rerun it

| Dataset | Entries | Cost (USD) | Per entry | Stage 4 share | Hours |
|---|---:|---:|---:|---:|---:|
| WikiText | 1,000 | 104.94 | 0.1049 | 73.5% | 120.8 |
| FEVER | 9,998 | 142.17 | 0.0142 | 80.7% | 266.3 |
| Europarl | 1,005 | 207.18 | 0.2062 | 75.5% | 227.6 |
| **Total** | **12,003** | **454.30** | **0.0378** | **76.7%** | **614.7** |

The WikiText cost includes the correction run of October 2026 (\$0.40, five entries
translated again; see [corpus/README.md](corpus/README.md)); the hours cover the production run only.

These are measured, not estimated: every API response reports its token counts,
and the pipeline writes the per-call cost into the trace files under
`corpus/pipeline/`. Rates are the Azure list prices in effect during the run
(\$0.40/\$1.60 per million tokens for GPT-4.1 Mini, \$2.00/\$8.00 for GPT-4.1,
\$0.50/\$1.50 for Mistral Large); the locally hosted Qwen model contributes zero.
Check current prices before budgeting, and see
[docs/reproduction.md](docs/reproduction.md#before-you-spend-anything) for how to
run a cheap pilot first.

---

## Licence

This repository is licensed in two parts, because the code and the data have
different origins.

- **Code** — `agents/`, `metrics/`, `scripts/`, `notebooks/` — Apache License 2.0
  ([LICENSE](LICENSE)).
- **Corpus** — everything under `corpus/` — Creative Commons Attribution-ShareAlike
  4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)).

The corpus licence is inherited, not chosen. Two of the three sources are
CC BY-SA text from Wikipedia, and a translation is a derivative work, so the
ShareAlike obligation carries over to the Spanish side. If you redistribute the
corpus you must credit WikiText-103, FEVER and Europarl — all three cited in
LICENSE-DATA.md — and license your version under compatible terms.

GitHub labels the whole repository "Apache-2.0" because it reads only the
`LICENSE` file. That label applies to the code; the corpus terms are the ones in
LICENSE-DATA.md.

## Citing

See [CITATION.cff](CITATION.cff). Please cite the three source datasets as well.

## Contact

Victor Adrian Sosa-Hernández — <vsosa@tec.mx>
School of Engineering and Sciences, Tecnológico de Monterrey, Monterrey, Mexico
