# Reproduction guide

*[Leer en español](es/reproduction.md)*

Four things can be reproduced from this repository, in increasing order of cost:

| | What it gives you | Cost | Time |
|---|---|---|---|
| [1. Read the corpus](#1-read-the-corpus) | The 12,003 bilingual entries | free | minutes |
| [2. Recompute the metrics](#2-recompute-the-metrics) | Every number in the paper's tables | free | 1–3 hours |
| [3. Redo the subset selection](#3-redo-the-subset-selection) | The stratified samples, from the raw sources | free | 2 GB download + 1 hour |
| [4. Rerun the pipeline](#4-rerun-the-pipeline) | A new corpus | **~\$454 for the full corpus** | ~615 h |

Most readers want 1 or 2. Read [Before you spend anything](#before-you-spend-anything)
before attempting 4.

---

## Install

Python 3.11 or newer is required: the code uses `X | None` type syntax.

```bash
git clone https://github.com/vsosahdz/SpanishDBText.git
cd SpanishDBText

python3 -m venv venv
source venv/bin/activate              # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Two models must be downloaded separately. They are needed only for the
syntactic and named-entity metrics; everything else runs without them.

```bash
python3 -m spacy download en_core_web_sm
python3 -m spacy download es_core_news_sm
```

**GPU note for the semantic metrics.** `metrics/semantic_metrics.py` uses LaBSE
through `sentence-transformers`, and the default PyPI build of `torch` targets
CUDA 13, which needs NVIDIA driver r580 or newer. With an older driver torch
silently falls back to the CPU — no error, just a warning — and a run that takes
minutes on a GPU takes hours. If your driver is older, install the matching
build first:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124   # driver 566.x = CUDA 12.7
python3 -c "import torch; print(torch.cuda.is_available())"            # must print True
```

Then check the whole environment at once:

```bash
python3 scripts/verify_setup.py
```

---

## 1. Read the corpus

```bash
python3 scripts/unpack_corpus.py --final
```

This expands three files into `output/`:

```
output/wikitext/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json    38 MB
output/FEVER/fever_selected_260712_212751.json                                  36 MB
output/Europarl/selected_260710_084528.json                                     85 MB
```

Their structure is documented field by field in [corpus-format.md](corpus-format.md).
In short, each file is one JSON object whose `payload[0]["data"]` is the list of
entries, and each entry has an `english` and a `spanish` block with `title`,
`keywords`, `summary`, `classification` and `text`.

```python
import json

path = "output/FEVER/fever_selected_260712_212751.json"
with open(path, encoding="utf-8") as fh:
    entries = json.load(fh)["payload"][0]["data"]

print(len(entries), "entries")
print(entries[0]["english"]["text"])
print(entries[0]["spanish"]["text"])
```

The files are large enough that `json.load` holds a few hundred megabytes in
memory. On a constrained machine, stream them with `ijson` instead, or work
from `corpus/samples/`.

For the per-entry record of what each model produced — the three candidate
translations, the reviewer's critique and merge, the evaluator's scores and the
token cost — unpack the traces as well:

```bash
python3 scripts/unpack_corpus.py --pipeline
```

---

## 2. Recompute the metrics

Everything in Sections 5 and 6 of the paper comes out of the `metrics/` package.
It reads the expanded corpus from `output/`, so do step 1 first.

### Lexical, syntactic and quality metrics

```bash
python3 -m metrics.run_metrics --dataset fever      # one dataset
python3 -m metrics.run_metrics --all                # all three
```

Results land in `metrics/output/metrics_<dataset>_<timestamp>.json`. The files
that produced the paper's tables are kept for comparison in `corpus/metrics/`.

Expect roughly 10 minutes for FEVER, longer for WikiText and Europarl: spaCy has
to parse every document on both sides, and Europarl documents average 6,900
words.

### Cross-lingual semantic fidelity

```bash
python3 -m metrics.semantic_metrics --sample 200                  # pilot, minutes
python3 -m metrics.semantic_metrics --full --out metrics/output/semantic_full.json
```

The first run downloads LaBSE (~1.8 GB). `--full` over all 12,003 entries takes
about an hour on a GPU and considerably longer on a CPU.

### The Stage 4 ablation

This is the one metrics run that costs money. It answers "consensus, better than
what?" by scoring each of the three candidate translations with the *same*
evaluator that scored the consensus — same model, same prompt, same scale, same
programmatic completeness adjustment. It does not retranslate anything: the
candidates are already in the pipeline traces. It does call the evaluator model,
so it needs a configured `.env` (see below).

```bash
python3 scripts/unpack_corpus.py --pipeline
python3 -m metrics.ablation_consensus --dry-run                   # count the calls first
python3 -m metrics.ablation_consensus --sample 200
```

`--progress <file>` lets an interrupted run resume instead of starting over.

---

## 3. Redo the subset selection

Only needed if you want to rebuild the stratified samples from the raw sources
rather than reuse the ones in `corpus/inputs/`.

```bash
python3 scripts/download_datasets.py --list       # see where each source comes from
python3 scripts/download_datasets.py --all        # ~2.2 GB download, ~7 GB expanded
jupyter lab                                       # then run notebooks 01, 02, 03
```

Selection uses `SEED = 42` throughout, so the samples are deterministic given
the same inputs. Two caveats about the inputs:

- **FEVER.** The FEVER Wikipedia dump ships as 109 shards of about 50,000 pages
  each. The run reported in the paper used a single 50,000-page shard, saved as
  `data/FEVER/fever.jsonl`. Starting from a different shard gives a different
  50,000-page pool and therefore a different sample, even with the same seed.
- **Europarl.** The run used 3,656 English chapter files covering the 2009–2011
  sessions (`ep-09-05-07-*.txt` through `ep-11-11-16-*.txt`), placed in
  `data/Europarl/en/`. The Europarl v7 release covers 1996–2011 and is larger;
  restrict it to that range to match.

To skip all of this and use exactly the subsets the pipeline consumed:

```bash
python3 scripts/unpack_corpus.py --inputs
```

---

## 4. Rerun the pipeline

### Before you spend anything

The full corpus cost **\$453.90** and 615 hours of wall-clock time. Three things
are worth doing first:

1. **Check current prices.** The figures in the paper are Azure list prices from
   mid-2026. The rates the code uses are hard-coded in `MODEL_PRICING` near the
   top of `agents/agente_wikitext.py`; update them or your cost accounting will
   be wrong.
2. **Run a pilot.** Every agent takes `--limit`, so start with ten entries and
   read the reported cost before scaling up.
3. **Know where the money goes.** Stage 4, the review, is 77% of the total. If
   the budget is tight, that is the one knob that matters — a cheaper reviewer
   changes the cost far more than anything else you can adjust.

### Configure

```bash
cp .env.example .env
```

Fill in your endpoints and keys, and point each role at your own deployments.
Every variable is explained in [configuration.md](configuration.md). `.env` is
git-ignored and must stay that way: it holds API keys.

If any role uses the `ollama` provider, start the local server and pull the model:

```bash
docker compose up -d
docker exec -it ollama ollama pull qwen3.5:9b
curl http://localhost:11434/api/tags          # should list the model
```

Uncomment the `deploy:` block in `docker-compose.yml` to give the container your
NVIDIA GPU. Without it, Ollama runs on the CPU, which works but is slow enough
to dominate the total runtime.

Then confirm that every configured model actually answers:

```bash
python3 scripts/verify_setup.py --models
```

### Run

Run from the repository root — the agents resolve `data/` and `output/`
relative to the working directory.

```bash
python3 scripts/unpack_corpus.py --inputs

python3 agents/agente_wikitext.py --input data/WikiText/wikitext-103-raw.wiki.train.selected_1000.raw --limit 10
python3 agents/agente_fever.py    --input data/FEVER/selected/fever_selected.jsonl --limit 10
python3 agents/agente_europarl.py --input data/Europarl/selected/ --limit 10
```

Shared options:

| Option | Meaning |
|---|---|
| `--input`, `-i` | Input file or, for Europarl, a directory |
| `--limit`, `-l` | Stop after N entries (`0` = all). **Always use this first.** |
| `--start-from`, `-s` | Begin at entry N |
| `--retries`, `-r` | Maximum attempts per API call (default 3) |
| `--id-offset` | Offset entry IDs, for processing several files into one corpus |

`agente_fever.py` adds `--min-words` (default 10) and `agente_europarl.py` adds
`--min-chars`, a minimum file size in bytes.

Each run writes three things into `output/<dataset>/`: the bilingual corpus
JSON, the pipeline trace JSON, and a timestamped `.log`.

### If a run is interrupted

It does not matter. The pipeline writes a progress file after every completed
entry, holding the finished IDs and their accumulated output. Relaunch the same
command and it skips what is already done and resumes at the first incomplete
entry. When the run finally completes, the progress file is consumed into the
two output JSONs. Network errors, rate limits and reboots are all recoverable
this way; what is *not* recoverable is deleting the progress file, which
discards everything not yet written to the final output.

---

## Using the pipeline on your own text

Nothing in the design is specific to English–Spanish or to these three datasets.

**A different model, or a different provider.** Edit `.env`. Roles are declared
as `provider:model` and read at startup, so no code changes. Supported providers
are `azure`, `foundry` and `ollama`.

**A different language pair.** The prompts are plain string constants grouped
near the top of `agents/agente_wikitext.py` — `TRANSLATION_SYSTEM`,
`REVIEWER_SYSTEM`, `EVALUATOR_SYSTEM`, `METADATA_SYSTEM_ES` / `_EN` and their
`build_*_prompt()` functions. Translate the instructions and change the target
language named in them. Note that Stage 3 validates the language of each
response and retries twice if the model answers in the wrong one; that check
also assumes Spanish.

**A different source format.** Write a new agent. `agents/agente_fever.py` is the
smallest worked example: it imports the shared core from `agente_wikitext`,
supplies its own loader (`load_fever_entries`), its own prompts, and its own
output directory — about 400 lines in total, most of it prompts. What you need
to provide is a loader that yields entries, and a chunking rule for texts above
the ~6,000-token threshold. WikiText splits at section headers, Europarl at
speaker boundaries, FEVER not at all.

**Tuning the thresholds.** Chunk size, the truncation ratio, the completeness
weight and the retry counts are all documented with their rationale in
[configuration.md](configuration.md). The truncation threshold of 0.92 in
particular encodes an assumption about *this* language pair — Spanish text runs
4–9% longer than its English source — and will need revisiting for a pair that
expands or contracts differently.

---

## Changes made for release

The code here is the code that produced the corpus. Two deliberate differences:

1. **The data ships compressed.** `corpus/*/*.gz` instead of plain JSON, so that
   every file stays under GitHub's 100 MB limit.
   `scripts/unpack_corpus.py` restores the original layout and filenames, which
   is why the hard-coded paths in `metrics/semantic_metrics.py` and
   `metrics/ablation_consensus.py` still resolve.
2. **Notebook 06 writes figures to `figures/`.** It previously wrote straight
   into the paper's LaTeX source tree, which is not part of this repository.

Files not carried over: the `.env` with live credentials, the LaTeX sources of
the paper, the pilot runs other than the WikiText one in `corpus/pilot/`, and the
intermediate output of the exploratory runs.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'metrics'`** — run from the repository
root. The metrics package is imported as `python3 -m metrics.<module>`.

**`ModuleNotFoundError: No module named 'agente_wikitext'`** — the FEVER and
Europarl agents import the shared core from the same directory. Invoke them by
path (`python3 agents/agente_fever.py`), which puts `agents/` on `sys.path`, or
run them from inside `agents/` with `data/` and `output/` reachable.

**`FileNotFoundError: No output JSON found in .../output/FEVER`** — the corpus
has not been expanded. Run `python3 scripts/unpack_corpus.py --final`.

**`[WARN] spaCy model 'es_core_news_sm' not installed`** — the syntactic and
entity metrics are skipped and everything else still runs. Install it with
`python3 -m spacy download es_core_news_sm`.

**The semantic metrics are unbearably slow** — torch is on the CPU. See the GPU
note under [Install](#install).

**A model answers in the wrong language** — Stage 3 detects this and retries
twice with a stronger instruction before giving up. Persistent failures usually
mean a small local model; try a larger one in that slot.

**An entry fails with an API error** — it is logged and skipped, and the run
continues. Three entries were lost this way in the published run (2 FEVER,
1 Europarl). Rerun with `--start-from` to retry a specific range.
