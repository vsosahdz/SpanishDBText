# The pipeline, stage by stage

*[Leer en español](es/pipeline.md)*

Every entry in the corpus passed through the six stages below. This document
says what each stage does, why it does it that way, and where to find it in the
code.

The pipeline is implemented as three agents, one per source dataset. They are
not three copies: `agents/agente_wikitext.py` holds the shared core — provider
clients, prompts, retry logic, all six stages — and the other two import from it,
overriding only what the input format requires.

```
agents/agente_wikitext.py     shared core + the WikiText agent
agents/agente_fever.py        imports the core; own loader, prompts, chunking (none)
agents/agente_europarl.py     imports the core; own parser, speaker-aware chunking
```

---

## Stage 1 — Selection

*Notebooks `01`–`03`. Output: `corpus/selection/*.csv`, `corpus/inputs/`.*

Translating a full source corpus would be neither affordable nor useful — much
of it is too short to translate meaningfully or too long to translate
economically. Each source is therefore sampled in four steps:

1. Parse the full corpus and compute word counts per entry.
2. Filter by word count, with per-corpus thresholds.
3. Assign a topic by keyword matching: each category is a list of domain terms,
   and an entry goes to the category with the most matches, or to *other* when
   nothing matches.
4. Sample stratified by topic and, within each topic, by size bin.

Step 3 uses a deterministic keyword rule rather than a model call on purpose.
Stratification runs over the entire source corpus — tens of thousands of entries —
*before* any budget is committed to translation, and a fixed rule keeps the
selection reproducible under the random seed. `SEED = 42` throughout.

| | Original | Filter | Discarded | Selected |
|---|---:|---|---:|---:|
| WikiText-103 | 29,423 | 1,000–10,000 words | 13.9% | 1,000 |
| FEVER | 50,000 | ≥ 100 words | 78.2% | 10,000 |
| Europarl | 3,656 | 200–25,000 words | 64.1% of 2,802 | 1,006 |

Europarl needs an extra step before filtering: 854 of the 3,656 files hold only
structural markup with no debate text and are dropped, leaving 2,802 with usable
content.

The topic schemes differ by source. WikiText and FEVER, both from Wikipedia,
share 14 categories (music, military history, sports, film and television,
geography, science, transport, biography, politics, nature, technology, history,
religion and philosophy, *other*). Europarl uses 15 categories adapted to
parliamentary business. The text matched against the keyword lists differs too:
title plus the first 2,000 characters of the body for WikiText and FEVER, but
the debate title alone for Europarl, where the session title already states the
subject.

---

## Stage 2 — Preprocessing

*Code: `clean_text()`, `detect_language()`, `split_articles()` and the
per-agent chunkers.*

### Language detection

The pipeline is bidirectional by design: direction is decided per entry from its
detected source language, so one run can process a mixed-language corpus. All
three sources here are English, so it ran English→Spanish throughout.

Detection is a stopword heuristic. It computes the ratio of 15 common Spanish
function words (*de, la, el, en, los, las, del, que, por, con, una, para, como,
fue, son*) to total words; above 8%, the text is Spanish, otherwise English.
Running Spanish text is well above 8%; English text reaches it only when quoting
Spanish. The known weakness is English entries dense in Spanish proper nouns,
where names alone can push the ratio over the threshold.

The same heuristic is reused in Stage 3 to verify that the model actually
translated. That reuse is the point of keeping it cheap.

### Cleaning

Formatting artefacts, unknown tokens and inherited encoding problems are
removed. Structural markers are normalised or kept depending on whether they
carry meaning: Europarl's XML speaker tags are kept on both language sides,
because who is speaking is part of the content of a parliamentary transcript,
not incidental formatting.

### Chunking

A long text sent in one call is prone to *silent* truncation: the model
translates the first portion, stops, and returns something that reads perfectly
while missing its last third. Nothing downstream would flag it.

Entries above ~6,000 estimated tokens are therefore split at structural
boundaries and regrouped into chunks of at most 12,000 characters. WikiText
splits at section headers, Europarl at speaker turns; FEVER entries are short
enough to send whole.

The 6,000-token threshold comes from observation: entries above it consistently
hit the 8,192-token output cap. The 12,000-character chunk limit is deliberately
half of the arithmetic conversion (~4 characters per token would give 24,000) —
a safety margin absorbing both the variability of that ratio and the expansion
Spanish adds.

---

## Stage 3 — Candidate generation

*Code: `stage_1_generate()` in each agent, `call_llm()` and `call_with_retries()`
in the core.*

Every chunk goes to all configured translation models at once, through a
`ThreadPoolExecutor`. The run used three, from three different families:

| Role | Model | Provider |
|---|---|---|
| Translator 1 | GPT-4.1 Mini | Azure OpenAI |
| Translator 2 | Mistral Large | Azure AI Foundry |
| Translator 3 | Qwen 3.5 9B | Ollama, local |

Different families is the design requirement, not a preference. Three models
from one family tend to make the same mistakes, and a consensus over correlated
errors adds nothing.

The exact text of every prompt in this and the following stages is published
under [`prompts/`](../prompts/), one file each.

The system prompt asks for semantic fidelity (invent nothing, omit nothing),
natural rather than literal phrasing, preservation of technical terms and proper
nouns, respect for tone and paragraph structure, and no summarising,
simplifying or censoring. Responses are then stripped of markdown, section
markers and any prefix the model added.

Three failure modes are handled explicitly:

- **Wrong output language.** A model that ignores the instruction and echoes the
  source produces fluent text that no later stage would catch. The Stage 2
  heuristic is rerun on the output, and a wrong language triggers up to two
  retries with a reinforced prompt naming the error. Most are fixed on the first
  retry.
- **Transient API failures.** HTTP errors and empty responses are retried up to
  three times. Content-filter refusals are logged and flagged but *not* retried —
  they will not succeed on a second attempt.
- **Output-limit truncation.** APIs report a finish reason; `"length"` instead of
  `"stop"` means the model ran out of output tokens. These are logged as
  warnings. This catches truncation from the token side; Stage 5 catches it from
  the content side.

If every retry fails for an entry, it is skipped and the run continues. Three
entries were lost this way in the published run.

---

## Stage 4 — Review and consensus

*Code: `stage_2_review()`, `_reviewer_critique()`, `_reviewer_merge_section()`,
`_split_text_aligned()`.*

The reviewer model — GPT-4.1, the most capable one in the configuration — receives
the original and all three candidates, and does two things.

**Critique.** For each candidate it names strengths and weaknesses and assigns a
0–100 score, then states which candidate should serve as the base and which
specific improvements to apply. The result is structured JSON, stored in the
trace files.

**Merge.** Guided by that critique, it writes a new translation. Short entries
are merged in one call. Long ones are merged section by section: original and
candidates are divided into aligned segments using the same structural
boundaries as Stage 2, each segment is merged independently, and the results are
concatenated.

Merging rather than selecting is the central design decision. Selecting caps
output quality at that of the best available model for that entry, and no model
is best everywhere — the reviewer chose Mistral Large as the base for 74.7% of
entries, GPT-4.1 Mini for 17.1% and Qwen 3.5 9B for 8.2%. Merging lets a passage
one model rendered better be placed into an output based on another. The
ablation in `metrics/ablation_consensus.py` tests this against the alternative:
the consensus scores above each of the three models used alone, on every
dataset, even though the strongest single model differs between them.

If no reviewer is configured, the pipeline falls back to the highest-scoring
Stage 3 candidate.

**This stage is 77% of total cost** (\$348 of \$454). It uses the most expensive
model and it processes the most text — the original plus every candidate. It is
the first place to look when optimising a budget.

---

## Stage 5 — Quality evaluation

*Code: `stage_3_evaluate()`, `_adjust_evaluation_by_completeness()`.*

This is the stage that removes the dependency on human reference translations.
The evaluator receives the original and the final translation and scores four
dimensions 0–100 — clarity, fidelity, completeness, fluency — plus an overall
score and a list of errors, against an anchored scoring guide.

### The programmatic truncation check

Language models are unreliable judges of missing content: they score what is in
front of them and rarely notice what is absent. So after the model scores an
entry, a second, mechanical check runs.

The character ratio is

    r = len(translation) / len(original)

and maps to a programmatic completeness score:

| Range of r | Programmatic score S_p |
|---|---|
| r < 0.85 | max(100·r, 10) |
| 0.85 ≤ r < 0.92 | 105·r |
| r ≥ 0.92 | not computed; the model's score stands |

Below 0.85 the score is the percentage of characters kept, with a minimum of 10.
Between 0.85 and 0.92 it is raised by 5% to allow for entries that are shorter
than their source for a valid reason.

The code (`_adjust_evaluation_by_completeness`) writes the first row as three
bands with floors of 10, 40 and 70, and the second with a floor of 85. Only the
floor of 10 can ever apply, because inside each band 100·r (or 105·r) is already
above its floor, so the table above gives the same scores.

If the model's completeness score exceeds S_p by more than 10 points, S_p wins,
a truncation error is recorded, and the overall score is recomputed as

    overall = mean(clarity, fidelity, fluency) × 0.6  +  completeness × 0.4

The 40% weight on completeness reflects a judgement about what matters:
truncation is a worse failure than a clumsy sentence.

The 0.92 threshold is specific to this language pair. Spanish translations of
English run 4–9% longer than their source (mean expansion 1.04–1.09, standard
deviation 0.02–0.07), so a translation below 92% of its original sits several
standard deviations low — missing content, not compact phrasing. **A different
language pair needs a different threshold.**

---

## Stage 6 — Metadata extraction

*Code: `build_metadata_prompt()`, `parse_metadata()`.*

For each entry, on both language sides: title, keywords, a summary and a topic
classification. The metadata model is GPT-4.1 Mini, the same one used for
evaluation, since extracting structured fields is an easy task.

If the model returns something unparseable, a fallback derives the fields
heuristically from the text rather than failing the entry.

---

## What a run produces

Two JSON files plus a log, per dataset, in `output/<dataset>/`:

- **The corpus file** — the clean bilingual result: one `english` and one
  `spanish` block per entry, each with text and metadata; the Spanish block also
  records the timestamp and the model that produced it.
- **The pipeline trace** — the full record of every decision: all candidates from
  Stage 3, the critique and merge from Stage 4, the scores from Stage 5, and
  token usage and cost broken down per entry, per stage and per model.

Both are documented field by field in [corpus-format.md](corpus-format.md).

### Cost accounting

Cost is tracked per call, per model, per stage, per entry:

    cost_usd = (tokens_in × price_in + tokens_out × price_out) / 1,000,000

Token counts come from the API responses, so the totals are measured rather than
estimated. Local models cost zero. Rates live in `MODEL_PRICING` near the top of
`agents/agente_wikitext.py` and must be updated when provider prices change.

### Incremental persistence

After every completed entry the pipeline writes a progress file holding the
finished IDs and their accumulated output from all six stages. A run interrupted
by a network error, a rate limit or a reboot is resumed by relaunching the same
command: completed entries are skipped and work continues at the first
incomplete one. When the run finishes, the progress file is consumed into the
two output JSONs.

At 615 hours of wall-clock time for the full corpus, this is not a convenience.
