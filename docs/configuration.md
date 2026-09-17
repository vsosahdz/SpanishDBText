# Configuration

*[Leer en español](es/configuration.md)*

Everything configurable at run time lives in `.env`. Everything else — thresholds,
retry counts, prices — is a constant near the top of `agents/agente_wikitext.py`.
This document covers both, and says why each value is what it is.

Start from the template:

```bash
cp .env.example .env
```

`.env` is git-ignored. It holds API keys; keep it that way.

---

## Declaring models

Every model role is written as `provider:model`:

```bash
TRANSLATION_MODEL_1=azure:my-gpt41mini-deployment
TRANSLATION_MODEL_2=foundry:my-mistral-large-deployment
TRANSLATION_MODEL_3=ollama:qwen3.5:9b
REVIEWER_MODEL=azure:my-gpt41-deployment
EVALUATOR_MODEL=azure:my-gpt41mini-deployment
```

The `model` part is whatever the provider calls it — for Azure that is your
**deployment name**, not the model name. Note that `ollama:qwen3.5:9b` contains
two colons; only the first one separates provider from model.

Roles are resolved at startup by `load_pipeline_config()`, so changing any of
them needs no code change. `verify_all_models()` then sends one short call to
each before any real work begins; `python3 scripts/verify_setup.py --models`
runs exactly that check on its own.

### Providers

| Provider | For | Credentials it reads |
|---|---|---|
| `azure` | Azure OpenAI deployments | `AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_API_VERSION` |
| `foundry` | Azure AI Foundry serverless models | `FOUNDRY_<FAMILY>_ENDPOINT`, `FOUNDRY_<FAMILY>_KEY` |
| `ollama` | A locally hosted model | `OLLAMA_BASE_URL` |

Foundry credentials are looked up by family, derived from the deployment name:
a deployment containing `mistral` reads `FOUNDRY_MISTRAL_ENDPOINT` and
`FOUNDRY_MISTRAL_KEY`. `CLAUDE`, `DEEPSEEK` and `LLAMA` are recognised the same
way. Only the Mistral pair was used in the published run.

### The roles

| Variable | Stage | What it does |
|---|---|---|
| `TRANSLATION_MODEL_1` … `_4` | 3 | Translate every entry in parallel. One to four; leave a slot empty to use fewer. |
| `REVIEWER_MODEL` | 4 | Critiques the candidates and merges them into the consensus. |
| `EVALUATOR_MODEL` | 5 | Scores the consensus. Also used for Stage 6 metadata. |
| `LLM_PROVIDER`, `OLLAMA_MODEL` | — | Legacy single-provider fallback, used only if no `TRANSLATION_MODEL_*` is set. Leave alone. |

Two things worth knowing before you change these:

**Pick translators from different families.** Three models from one family make
correlated mistakes, and a consensus over correlated errors adds nothing. The
published configuration deliberately mixes OpenAI, Mistral and Alibaba
architectures.

**The reviewer is the budget.** Stage 4 is 77% of total spend — it uses the most
capable model and processes the most text (original plus every candidate). If
cost is a constraint, this is the only variable that really moves it. Dropping a
translator saves far less; the local Ollama slot costs nothing at all.

---

## Model parameters

| Variable | Value used | Why |
|---|---|---|
| `TEMPERATURE` | `0.2` | Nearly deterministic while allowing slight variation. `0` makes runs identical; higher values add randomness that a fidelity task does not want. |
| `NUM_CTX` | `32768` | Context window. The limit of the models used. |
| `NUM_PREDICT` | `8192` | Maximum output tokens for translation and review. Evaluation and metadata calls use a fixed 1,024 and ignore this. |

---

## Pipeline constants

These are not in `.env`. They are module-level constants in
`agents/agente_wikitext.py`, inherited by the other two agents.

| Constant | Value | Why |
|---|---|---|
| `MAX_CHARS_PER_CHUNK` | `12000` | Deliberately **half** the arithmetic conversion of the token threshold (~4 chars/token would give 24,000). The margin absorbs both the variability of that ratio and the expansion Spanish adds. |
| Chunking threshold | ~6,000 est. tokens | Observed: entries above this consistently hit the 8,192-token output cap. |
| `TIMEOUT` | `600` seconds | Set for the local Qwen model, which is much slower than the cloud ones on long chunks. |
| `MAX_RETRIES` | `3` | Attempts per API call on transient failure. Overridable per run with `--retries`. |
| `TRANSLATION_LANGUAGE_RETRIES` | `2` | Retries when the model answered in the wrong language. Most are fixed on the first. |
| Truncation ratio threshold | `0.92` | Below this, the programmatic completeness check runs. |
| Truncation override margin | `10` points | The check only overrides the model when the gap is large, avoiding corrections for small differences. |
| Completeness weight | `40%` | Its share when the overall score is recomputed. Truncation is a worse failure than a clumsy sentence. |
| `SEED` | `42` | Every stratified sample, in the notebooks and in the metrics sampling. |

### The 0.92 threshold is language-pair specific

Spanish translations of English text run 4–9% longer than their source (mean
expansion 1.04–1.09, standard deviation 0.02–0.07). A translation shorter than
92% of its original therefore sits several standard deviations below the
expected range, which points to missing content rather than compact phrasing.

For a pair that expands or contracts differently, this number is wrong.
Measure the expansion ratio on a pilot before trusting it. The full piecewise
mapping from character ratio to completeness score is in
[pipeline.md](pipeline.md#the-programmatic-truncation-check).

---

## Prices

`MODEL_PRICING` in `agents/agente_wikitext.py` maps each model to its per-million-token
input and output rates, and `calculate_cost()` applies them to the token counts
the API returns. The table holds the Azure list prices in effect during the run:

| Key | Input / Output, USD per million tokens |
|---|---|
| `gpt-4.1-mini`, `gpt41mini` | 0.40 / 1.60 |
| `gpt-4.1`, `gpt41` | 2.00 / 8.00 |
| `gpt-4o-mini`, `gpt4omini` | 0.15 / 0.60 |
| `gpt-4o`, `gpt4o` | 2.50 / 10.00 |
| `mistral` | 0.50 / 1.50 |
| `deepseek` | 0.50 / 1.80 |
| `ollama` | 0 / 0 |

Lookup is by **substring of the model name**, longest key first, which is why
each family appears twice: Azure deployment names often drop the hyphens and
dots (`gpt41mini`). Two consequences worth knowing:

- Name your deployments so that a key matches. **A model whose name matches no
  key is silently priced at zero** — `calculate_cost()` returns `0.0` rather than
  raising — so an unrecognised deployment produces a run that looks free.
- **Update the rates before budgeting a new run.** Costs are written into the
  pipeline traces at run time from whatever the table said then. They are never
  recomputed, so stale rates produce confidently wrong accounting.

---

## Running Ollama

Only needed if a role uses the `ollama` provider.

```bash
docker compose up -d
docker exec -it ollama ollama pull qwen3.5:9b
curl http://localhost:11434/api/tags
```

`docker-compose.yml` sets `OLLAMA_KEEP_ALIVE=24h` so the model stays resident
between entries rather than reloading on every call.

For GPU access, uncomment the `deploy:` block:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Without it Ollama runs on the CPU. That works, but the local model is already
the slowest part of the pipeline and on CPU it dominates the total runtime — the
615 hours reported in the paper are mostly it.

If RAM is tight, a smaller tag in the same family works: the `qwen3.5` family
ranges from 0.8B to 122B. Quality in that slot matters less than it might seem,
since the reviewer chose the Qwen candidate as its base for only 8.2% of entries.

---

## The prompts

Not configuration in the strict sense, but the next thing you will want to change
for a different language or domain. They are plain string constants grouped near
the top of `agents/agente_wikitext.py`:

| Constant | Used by |
|---|---|
| `TRANSLATION_SYSTEM` | Stage 3 |
| `REVIEWER_SYSTEM` | Stage 4 |
| `EVALUATOR_SYSTEM` | Stage 5 |
| `METADATA_SYSTEM_ES` / `METADATA_SYSTEM_EN` | Stage 6 |

Each pairs with a `build_*_prompt()` function that assembles the user message.
The FEVER and Europarl agents override the translation and metadata prompts with
domain-specific variants — `FEVER_TRANSLATION_SYSTEM`,
`EUROPARL_TRANSLATION_SYSTEM` — and inherit the rest.

All prompts are written in Spanish, because the target language is Spanish. They
were developed iteratively, starting from minimal instructions and adding
constraints as problems appeared in early runs; the constraints that survived are
listed in [pipeline.md](pipeline.md#stage-3--candidate-generation).

Changing the target language means editing these strings **and** revisiting the
language-detection heuristic in Stage 2, which checks for Spanish function words
and is reused in Stage 3 to verify that the model actually translated.
