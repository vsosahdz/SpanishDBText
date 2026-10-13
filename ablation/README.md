# Evaluator ablation: scoring guide and length evidence

*[Leer en español](README.es.md)*

The Stage 5 evaluation prompt combines two measures: a **scoring guide** that describes what
each band of the 0–100 scale means, and **length evidence**, that is, the character counts of
both texts, their ratio and a warning when the translation is much shorter than the original.
This test measures what each one contributes. The paper summarises it in Section 4.3.3, in the
paragraph that follows the description of the evaluation prompt.

## Question

With the same evaluator and the same translations, what changes in the scores when the scoring
guide is removed, when the length evidence is removed, or when both are?

## Design

A 2 × 2 factorial design. Only the evaluator prompt changes:

| Config | Scoring guide (G) | Length evidence (L) | Note |
|---|---|---|---|
| **G1L1** | yes | yes | The production prompt, identical to the one that scored the corpus |
| G0L1 | no | yes | |
| G1L0 | yes | no | |
| G0L0 | no | no | Baseline: no anchoring and no length data |

- **G (scoring guide)** = the "ESTRICTO y CRÍTICO" (strict and critical) role, the sentence
  stating that a perfect score must be extremely rare, and the six bands of Table 1 of the
  paper. Without G, the role becomes "evaluador experto" (expert evaluator).
- **L (length evidence)** = the "CRÍTICO — VERIFICACIÓN DE COMPLETITUD" (completeness check)
  block of the system prompt and the METADATOS section of the user prompt: character counts,
  ratio, and the warnings below 0.92 and 0.85.
- **Common to all four:** the JSON output format and the definition of the four criteria.

`prompts.py` builds the variants **by removing blocks from the production prompt** in
`../agents/agente_wikitext.py`, with assertions that fail if any block changes. The resulting
prompts are in `prompts/`.

### What is held constant

- **Translations:** the 62 final translations of the 7 June 2026 trial run on the WikiText-103
  test split (`../corpus/pilot/wikitext-103-raw.wiki.test_260607.json.gz`). That run was chosen
  because it contains **three known truncations**:

  | ID | Article | Length ratio | Content lost |
  |---|---|---|---|
  | 7 | Second Battle of Naktong Bulge | 0.36 | 64% |
  | 24 | 2003 Pacific typhoon season | 0.26 | 74% |
  | 41 | Manila | 0.79 | 21% |

  The run used Mistral Large as evaluator and an earlier version of the evaluation prompt.
  Its scores are kept in `data/inputs.jsonl` (`historic_mistral_*`) as context only. They are
  not one of the experimental conditions.
- **Originals:** `english.text` from the same file, which is the `original_text` the evaluator
  received.
- **Evaluator:** `EVALUATOR_MODEL` from `../.env`, the production evaluator (GPT-4.1 Mini),
  temperature 0.2, `max_tokens = 1024`.
- **Code:** the call, retries, JSON parsing and programmatic completeness check are imported
  from the production agent. Nothing is reimplemented.

### Repetitions and order

Each (article, configuration) pair is evaluated 3 times (`--repeats`), since temperature 0.2
leaves some variation. Calls run in a fixed random order (seed 42) that interleaves the
configurations, so that drift in the service is not confused with the effect of a prompt.

## Results (10 October 2026)

744 of 744 evaluations (4 configurations × 62 articles × 3 repetitions), US$3.08. Full detail
in `results/summary.md` and `results/summary.json`; every call is in `results/raw/`.

**Score level**, overall score assigned by the model on the 59 complete translations (177
evaluations per configuration):

| Config | Mean | Median | SD | % ≥95 | Most frequent score |
|---|---|---|---|---|---|
| G1L1 (production) | 95.75 | 96 | 2.01 | 80.2 | 95 (28%) |
| G0L1 | 96.79 | 97 | 0.99 | 96.6 | 97 (82%) |
| G1L0 | 94.92 | 95 | 1.70 | 76.3 | 95 (37%) |
| G0L0 | 96.44 | 97 | 1.50 | 91.5 | 97 (79%) |

Paired difference per article against production, with a 95% bootstrap interval: G0L1 +1.04
[0.68, 1.42]; G1L0 −0.82 [−1.13, −0.52]; G0L0 +0.69 [0.23, 1.14].

**Known truncations**, completeness assigned by the model in the three repetitions:

| ID (ratio) | G1L1 | G0L1 | G1L0 | G0L0 |
|---|---|---|---|---|
| 7 (0.36) | 36, 36, 36 | 36, 36, 36 | 90, 95, 95 | 90, 90, 90 |
| 24 (0.26) | 26, 26, 26 | 30, 30, 30 | 95, 95, 95 | 90, 90, 90 |
| 41 (0.79) | 79, 70, 79 | 70, 70, 70 | 99, 99, 99 | 100, 99, 99 |

**Reading:**

1. **Truncation detection depends on the length evidence, not on the guide.** With the length
   evidence, the model sets completeness at the length ratio. Without it, the model gives 90–100
   to translations that lost 21–74% of their content, with or without the guide.
2. **The guide changes the level of the scores only slightly** (about one point) but keeps them
   from collapsing onto a single value: without it, about 80% of the evaluations receive the
   same score, 97, against 28–37% with it. It does not stop the evaluator from scoring high:
   with the guide, 76–80% of the complete translations still score 95 or above.
3. **The programmatic check does not replace the length evidence.** Without L, it lowers the
   two severe truncations to an overall score of 68–73, but leaves Manila (ratio 0.79) at 88–90.
4. On complete translations, removing the length evidence lowers the overall score by 0.8
   points: the ratio (about 1.04, with no warning) seems to reassure the evaluator.

## How to run it

From this folder, with `../.env` configured (see `../docs/configuration.md`):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_inputs.py               # data/inputs.jsonl (62 entries)
.venv/bin/python prompts.py                      # prompts/, to inspect the variants
.venv/bin/python run_ablation.py --dry-run       # token and cost estimate, no calls
.venv/bin/python run_ablation.py --limit 2 --repeats 1   # smoke test (includes a truncation)
.venv/bin/python run_ablation.py                 # full run (4 configs × 62 × 3)
.venv/bin/python analyze.py                      # results/summary.md and summary.json
```

`results/` already holds the published run. To reproduce it from scratch, move `results/raw/`
aside first; otherwise `run_ablation.py` finds every call done and skips them.

`run_ablation.py` can be resumed: if interrupted, rerunning it skips what is already evaluated
and retries what failed. `--workers` sets the number of parallel calls (2 by default). Each call
is wrapped in retry rounds with exponential backoff (20 to 320 s), because the production retry
loop does not wait and a 429 (tokens-per-minute quota) exhausts it at once. `analyze.py` keeps one
row per (article, repetition) and drops earlier failed attempts.

The script comments and messages are in English. The prompt texts in `prompts.py` stay in
Spanish, because they are the prompts the evaluator received.

## Files

| File | What it is |
|---|---|
| `prepare_inputs.py` | Extracts the 62 entries of the 7 June run into `data/inputs.jsonl` |
| `prompts.py` | Builds the prompt variants from the production prompt |
| `run_ablation.py` | Runs the calls and writes `results/raw/<config>.jsonl` |
| `analyze.py` | Summarises into `results/summary.md` and `results/summary.json` |
| `prompts/` | System prompt of each variant and an example user prompt |
| `results/raw/` | One line per call: model evaluation, adjusted evaluation, raw response, tokens and cost |

## Limitations

- 62 articles from a single domain (WikiText); the results do not carry over to FEVER or
  Europarl without further testing.
- There are only three truncated translations, so detection is reported case by case, not as a
  rate.
- The effect of the guide may depend on the evaluator. It is measured here with GPT-4.1 Mini,
  the production evaluator.
