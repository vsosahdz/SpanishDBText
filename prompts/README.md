# The prompts

Every instruction the pipeline sends to a language model, as plain text, one file
per prompt.

These are the substance of the method. The Python around them handles chunking,
retries, parsing and cost accounting, but what actually decides the quality of
the corpus is written here. They are published separately from the code so that
they can be read, diffed and adapted without opening a `.py` file — and so that a
reviewer can check what the models were really asked to do.

**They are in Spanish**, because Spanish is the target language. A prompt written
in the language you want back is the one thing every stage has in common.

---

## How a prompt is assembled

Every call is two layers:

- a **system prompt** — the model's role and the rules that never change, and
- a **user prompt** — the specific text for this entry, plus any per-entry context.

So a translation call is `system-wikitext.txt` + `user-wikitext.txt` with the
article substituted in. The files here are the templates; anything the pipeline
fills in at run time appears as `{{PLACEHOLDER}}`.

### Placeholders

| Placeholder | Filled with |
|---|---|
| `{{SOURCE_TEXT}}` | The entry, or the chunk of it being processed |
| `{{ORIGINAL_SECTION}}` | One aligned section, during a sectioned merge |
| `{{TRANSLATION}}` | The consensus translation being evaluated |
| `{{SOURCE_TITLE}}` | The source title (FEVER only — it has a page identifier) |
| `{{MODEL_1}}` … `{{MODEL_3}}` | Each translator's `provider:model` string |
| `{{CANDIDATE_n_TEXT}}` | Candidate *n*'s full translation |
| `{{CANDIDATE_n_SECTION}}` | Candidate *n*'s text for the section being merged |
| `{{CANDIDATE_n_CHARS}}` | Candidate *n*'s character count, shown to the reviewer |
| `{{BEST_MARKER_n}}` | `(MEJOR)` on the candidate the critique scored highest, empty otherwise |
| `{{REVIEWER_DECISION}}` | The `decision` field the critique produced |
| `{{SECTION_INDEX}}`, `{{TOTAL_SECTIONS}}` | Position of the section being merged |
| `{{ORIGINAL_CHARS}}`, `{{TRANSLATION_CHARS}}`, `{{RATIO}}` | Length metadata given to the evaluator |
| `{{LENGTH_WARNING}}` | One of three strings — see `5-evaluation/user-length-warnings.txt` |

---

## Every prompt

Folders are numbered by the pipeline stage in [docs/pipeline.md](../docs/pipeline.md).
Stages 1 and 2 (selection and preprocessing) use no LLM, so they have no prompts.

### Stage 3 — Translation

Three models translate every chunk in parallel. All three receive the same
prompt; the diversity comes from the models, not from asking them different
things.

| File | What it is |
|---|---|
| `3-translation/system-wikitext.txt` | Translator role, 10 rules. The base version. |
| `3-translation/system-fever.txt` | Same, adapted to encyclopedic excerpts. Adds rule 9 on sports anglicisms. |
| `3-translation/system-europarl.txt` | Same, adapted to parliamentary transcripts. Adds rule 8: preserve the XML tags intact. |
| `3-translation/user-wikitext.txt` | The article, with the instruction to translate and answer with nothing else. |
| `3-translation/user-fever.txt` | The same for a short excerpt. |
| `3-translation/user-europarl.txt` | The same, repeating the XML instruction where the model will see it last. |
| `3-translation/user-retry-es.txt` | Sent when the model answered in English instead of Spanish. |
| `3-translation/user-retry-en.txt` | The mirror case, for the Spanish→English direction. |

The three system prompts share the same skeleton — semantic fidelity, natural
rather than literal phrasing, keep technical terms and proper nouns, respect tone
and paragraph structure, never summarise or censor, no markdown that was not in
the original — and differ only where the domain demands it:

- **FEVER** adds: *use the established Spanish sports term when one exists
  (`ponchar`, not `strikeout`), and keep the English one when it does not.* FEVER
  is 67% sports content, and without this the models produced inconsistent
  anglicisms.
- **Europarl** adds: *preserve `<CHAPTER>`, `<SPEAKER>` and `<P>` intact — do not
  translate or modify them.* Those tags record who is speaking, which is content
  in a parliamentary transcript.

The **retry** prompts exist because of a failure mode that nothing downstream
would catch: a model that ignores the instruction and echoes the source produces
fluent text that reads perfectly and is simply not a translation. Stage 3 checks
the output language with the same stopword heuristic used in Stage 2 and, on a
mismatch, resends with the error named explicitly. Most are fixed on the first
retry; two are allowed.

### Stage 4 — Review and consensus

The reviewer works in two calls — critique, then merge — rather than one. The
critique is a cheap, short, JSON-only call over the first 2,000 characters of
each candidate; the merge is the expensive one and runs section by section for
long entries. Splitting them keeps the analysis out of the generation call, so
the reviewer is not spending its output budget explaining itself while it writes.

| File | What it is |
|---|---|
| `4-review/system-critique.txt` | Reviewer role for the analysis call. Demands strict JSON: per-candidate strengths, weaknesses and a 0–100 score, plus a `decision`. |
| `4-review/system-merge.txt` | Reviewer role for the generation call. Answer with the translated text only. |
| `4-review/user-critique-wikitext.txt` | Original plus all candidates, truncated to 2,000 characters each. |
| `4-review/user-critique-fever.txt` | The same; entries are short, so the header does not mention truncation. |
| `4-review/user-critique-europarl.txt` | The same, but with the XML tags stripped first, so the reviewer judges the prose rather than the markup. |
| `4-review/user-merge-wikitext.txt` | One aligned section from the original and from each candidate, carrying the critique's decision forward. |
| `4-review/user-merge-fever.txt` | The same for a single section — FEVER entries are never split. |
| `4-review/user-merge-europarl.txt` | The same, repeating the XML instruction at the point of generation. |
| `4-review/system-single-call.txt` | The earlier one-call reviewer: critique and final translation in one response, `ANALYSIS:` then `FINAL TRANSLATION:`. **Not used — see below.** |
| `4-review/user-single-call.txt` | Its user prompt: the original and all candidates, untruncated. **Not used.** |

The merge prompt marks the candidate the critique scored highest with `(MEJOR)`
and passes the critique's `decision` text verbatim. The reviewer is told which
one to build on and what to fix, rather than being asked to decide again.

**About the single-call pair.** It is the earlier design, superseded when long
entries started exhausting the output budget on the analysis before reaching the
translation. `REVIEWER_SYSTEM` and `build_reviewer_prompt()` are still defined in
`agents/agente_wikitext.py` but nothing calls them: all three agents take the
two-call path unconditionally. They are exported here because they are part of
the released code and because the design they represent is the obvious
alternative to the one the paper reports — but no entry in the corpus was
produced with them. Do not mistake them for a live fallback.

Note that "one call" here refers to fusing critique and merge. It is unrelated to
the merge itself being a single call for short entries and one call per section
for long ones, which is what the paper describes and what the pipeline does.

### Stage 5 — Evaluation

One evaluator scores the consensus translation against the original, with no
human reference translation. This is the stage that makes the method
self-contained, so its prompt is the most heavily iterated of the set.

| File | What it is |
|---|---|
| `5-evaluation/system.txt` | The evaluator role and the scoring guide. |
| `5-evaluation/user.txt` | Length metadata, the original, the translation, and the demand for JSON only. |
| `5-evaluation/user-length-warnings.txt` | The three values the `{{LENGTH_WARNING}}` slot can take, and what triggers each. |

Two things in the system prompt deserve attention, because both are corrections
for known LLM-judge behaviour:

**The score bands are written out explicitly** — 95–100 is *impeccable and must be
rare*, 90–94 excellent, 85–89 minor problems, below 70 serious ones — and the
model is told to be strict. LLM evaluators drift upward without this; naming the
bands anchors them.

**Completeness is singled out.** The model is given the character counts of both
texts and told that a translation significantly shorter than its original is
probably truncated, and that a truncated translation cannot score above 70 on
completeness no matter how good the translated part reads. This is the stage's
hardest problem: a model scores what is in front of it and rarely notices what is
absent.

That said, the prompt is not the safety net. A programmatic character-ratio check
runs after the model and can override its completeness score outright — see
[docs/pipeline.md](../docs/pipeline.md#the-programmatic-truncation-check). The
warning in the prompt just gives the model a chance to catch it first.

### Stage 6 — Metadata

Title, keywords, summary and topic classification, extracted for each entry on
both language sides. Six user prompts: three corpora × two languages.

| File | What it is |
|---|---|
| `6-metadata/system-es.txt` | Extractor role, Spanish. Two lines: answer in the exact format, nothing else. |
| `6-metadata/system-en.txt` | The same in English. |
| `6-metadata/user-wikitext-es.txt` / `-en.txt` | 3–8 keywords, 100-word summary, 10 topic categories, title stripped of `=` signs. |
| `6-metadata/user-fever-es.txt` / `-en.txt` | 3–6 keywords, 60-word summary, 12 categories. Entries are short, so the fields are. |
| `6-metadata/user-europarl-es.txt` / `-en.txt` | 3–8 keywords, 100-word summary, 10 categories adapted to parliamentary business. |

The two sides are generated independently rather than translated, so
`english.summary` and `spanish.summary` in the corpus are two summaries, not a
summary and its translation.

The FEVER Spanish prompt is the longest of the six, and the reason is worth
noting: it receives the original page title and is told to translate it when a
Spanish form exists (*Temporada 1999 de los Detroit Lions*) and to keep the
English only for proper nouns with no established equivalent. Without that
instruction the Spanish side kept English page titles verbatim.

Output is parsed as four labelled lines rather than JSON. When parsing fails, a
heuristic fallback derives the fields from the text rather than failing the
entry.

---

## Which agent uses which

| Prompt | WikiText | FEVER | Europarl |
|---|:-:|:-:|:-:|
| Translation system | own | own | own |
| Translation user | own | own | own |
| Translation retry | shared | shared | shared |
| Reviewer critique system | shared | shared | shared |
| Reviewer merge system | shared | shared | shared |
| Reviewer critique user | own | own | own |
| Reviewer merge user | own | own | own |
| Evaluator system and user | shared | shared | shared |
| Metadata system | shared | shared | shared |
| Metadata user | own | own | own |

"Shared" means defined once in `agents/agente_wikitext.py` and imported by the
other two. Everything domain-specific is a translation or metadata prompt; the
review and evaluation stages are identical across all three corpora, which is
what makes their scores comparable.

---

## How these files are produced

They are generated, not transcribed:

```bash
python3 scripts/export_prompts.py           # regenerate
python3 scripts/export_prompts.py --check   # verify against the code, write nothing
```

System prompts are imported straight from the agent modules. User prompts are
produced by calling the agents' own `build_*_prompt()` functions with the
placeholders as arguments, so what is on disk is exactly what the code emits.

Three of them — the reviewer's critique and merge user prompts — are assembled
inline inside the stage functions rather than in a callable of their own, so the
export script reproduces them as templates. It checks its reconstruction against
the real construction before writing, and fails rather than publishing a stale
prompt. The evaluator's user prompt is checked the same way, across all three
character-ratio bands.

**This README is maintained by hand.** If you change a prompt, regenerate the
files and update the description here.

---

## Adapting them to another language

The prompts assume Spanish as the target throughout, in three places that are
easy to miss:

1. **The prompts themselves** are written in Spanish, and so are the reviewer's
   critique and the evaluator's error list in the released traces.
2. **The retry prompts** name the languages explicitly ("your previous response
   was in English").
3. **The language check that triggers a retry** is a Spanish stopword heuristic
   in `agents/agente_wikitext.py`, not part of any prompt. Changing the prompts
   without changing it leaves the pipeline unable to tell whether the model
   actually translated.

Point 3 is the one that bites. See
[docs/reproduction.md](../docs/reproduction.md#using-the-pipeline-on-your-own-text).

Two further details carry assumptions about this specific language pair: the
0.92 truncation threshold in Stage 5 encodes that Spanish runs 4–9% longer than
English, and FEVER's sports-anglicism rule is Spanish-specific. Neither will
transfer unchanged.
