# Corpus format

*[Leer en español](es/corpus-format.md)*

Each pipeline run writes two JSON files. The **corpus file** holds the clean
bilingual result and is what most users want. The **pipeline trace** holds the
full record of how that result was produced. Both are UTF-8, and both are
shipped gzipped under `corpus/`.

To read along while you read this page, open
[`corpus/samples/`](../corpus/samples/) — a few entries from each dataset,
uncompressed and pretty-printed.

---

## The corpus file

`corpus/final/*.json.gz` → expands to `output/<dataset>/`.

One JSON object. The top level identifies the source, `payload[0]` describes the
run, and `payload[0]["data"]` is the list of entries.

```jsonc
{
  "source_name": "fever",
  "url": "https://fever.ai/dataset/fever.html",
  "creation_date": "2018",
  "authors_name": ["Thorne et al."],
  "payload": [
    {
      "file_name": "fever_selected.jsonl",
      "file_size": "25.85 MB",
      "file_format": "jsonl",
      "description": "FEVER (Fact Extraction and VERification) dataset. ...",
      "translation_date": "2026-07-10",
      "llm_used": [
        "azure:dataset-frontierscience-gpt41mini",
        "foundry:dataset-frontierscience-mistral-large",
        "ollama:qwen3.5:9b"
      ],
      "data": [ /* the entries */ ]
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `source_name` | `wikitext`, `fever` or `europarl` |
| `url`, `creation_date`, `authors_name` | Provenance of the source dataset, as recorded at run time |
| `payload[0].file_name` | The input file or directory the run consumed |
| `payload[0].file_size`, `.file_format` | Size and format of that input |
| `payload[0].description` | Free-text note on what the dataset is and why it was included |
| `payload[0].translation_date` | Date the run finished |
| `payload[0].llm_used` | The Stage 3 translation models, as `provider:model` |

> The provenance fields were filled in by hand at run time and are not always the
> canonical citation — the WikiText file, for instance, records the Kaggle
> re-upload it was downloaded from rather than Merity et al. (2017). For
> citation, use [LICENSE-DATA.md](../LICENSE-DATA.md), which gives the original
> source of each dataset.

`payload` is always a list of one element. It exists because the format was
designed to allow several input files in one corpus file; no released run uses
more than one.

### An entry

```jsonc
{
  "id": 1,
  "source_language": "en",
  "english": {
    "title": "1999 Detroit Lions season",
    "keywords": ["Detroit Lions", "1999 NFL season", "playoffs", "Barry Sanders"],
    "summary": "The 1999 Detroit Lions improved from a 5-11 record to make the playoffs ...",
    "classification": "sports",
    "text": "The 1999 Detroit Lions season was their 70th in the league . ..."
  },
  "spanish": {
    "title": "Temporada 1999 de los Detroit Lions",
    "keywords": ["Detroit Lions", "temporada 1999 de la NFL", "playoffs"],
    "summary": "Los Detroit Lions de 1999 mejoraron su marca de 5-11 y llegaron a los playoffs ...",
    "classification": "sports",
    "text": "La temporada 1999 de los Detroit Lions fue su 70.ª en la liga. ...",
    "translated_at": "2026-07-08T12:00:13.011",
    "translated_by": "azure:dataset-frontierscience-gpt41"
  }
}
```

| Field | Meaning |
|---|---|
| `id` | Sequential within the file, starting at 1 |
| `source_language` | `en` for every released entry; the pipeline is bidirectional but ran one way |
| `english` / `spanish` | The two aligned sides |
| `…title` | Extracted in Stage 6, not taken from the source |
| `…keywords` | 5–8 terms, extracted in Stage 6 |
| `…summary` | Two to four sentences, extracted in Stage 6 |
| `…classification` | One topic label, assigned in Stage 6 |
| `…text` | The full text. This is the corpus. |
| `spanish.translated_at` | ISO timestamp of the Stage 4 consensus |
| `spanish.translated_by` | The **reviewer** model that produced the consensus, not a translator |

The source-side block always carries the source language's metadata and the
target-side block its own, generated independently rather than translated — so
`english.summary` and `spanish.summary` are two summaries, not one summary and
its translation.

### What `text` contains, per dataset

- **WikiText** keeps the section structure of the original: headings appear as
  `= Title =`, `= = Section = =`, `= = = Subsection = = =`. The English side also
  keeps the dataset's original tokenisation, in which punctuation is spaced out
  (`league .`, `5 -- 11`); the Spanish side is written as normal prose.
- **FEVER** entries are flat passages of a few sentences, with the same spaced
  tokenisation on the English side.
- **Europarl** keeps the XML markup of the source on **both** sides:
  `<CHAPTER ID="002">`, `<SPEAKER ...>`, `<P>`. This is deliberate: who is
  speaking and where an intervention begins is part of the content of a
  parliamentary transcript. It is also counted as text by the word totals in the
  paper — markup is 4.2% of the words on each side of Europarl and 2.5% of the
  combined corpus. **Strip it before computing surface statistics**, or you will
  be measuring tags. `metrics/semantic_metrics.py` shows the normalisation used
  in the paper.

### Reading it

```python
import json

with open("output/FEVER/fever_selected_260712_212751.json", encoding="utf-8") as fh:
    entries = json.load(fh)["payload"][0]["data"]

pairs = [(e["english"]["text"], e["spanish"]["text"]) for e in entries]
```

The three files hold 1,000, 9,998 and 1,005 entries and are 38, 36 and 85 MB
expanded. `json.load` holds the whole thing in memory; stream with `ijson` on a
constrained machine.

### Known gaps

12,003 entries were produced from 12,006 selected. Three failed on API errors and
were not retried: 2 FEVER and 1 Europarl. A further **4 FEVER entries carry no
quality scores** — they were translated and reviewed successfully but their
Stage 5 evaluation call failed. They are present in the corpus and complete as
text. This is why the paper's evaluation statistics are computed over 9,994 FEVER
entries rather than 9,998.

---

## The pipeline trace

`corpus/pipeline/*.json.gz` → expands to `output/<dataset>/`.

A different shape: no `payload`, and `pipeline` is a flat list of per-entry
records. This file is what makes the cost analysis and the Stage 4 ablation
possible — the three candidate translations are stored here, so comparing the
consensus against them needs no retranslation.

```jsonc
{
  "source_file": "fever_selected.jsonl",
  "generated_at": "2026-07-10",
  "models_used": ["azure:...gpt41mini", "foundry:...mistral-large", "ollama:qwen3.5:9b"],
  "reviewer": "azure:dataset-frontierscience-gpt41",
  "evaluator": "azure:dataset-frontierscience-gpt41mini",
  "summary": {
    "articles_processed": 9998,
    "total_tokens": {
      "prompt_tokens": 63235451,
      "completion_tokens": 19903337,
      "total_tokens": 83138788
    },
    "total_cost_usd": {
      "stage_1_translation": {
        "foundry:...mistral-large": 7.692,
        "azure:...gpt41mini": 6.7591,
        "ollama:qwen3.5:9b": 0.0
      },
      "stage_2_review": 114.7305,
      "stage_3_evaluation": 6.3949,
      "metadata": 6.5962,
      "total": 142.1728
    }
  },
  "pipeline": [ /* one record per entry */ ]
}
```

> **The stage numbers in this file are off by two from the paper.** The trace
> writes `stage_1_translation`, `stage_2_review` and `stage_3_evaluation`; the
> paper calls the same things Stages 3, 4 and 5, counting selection and
> preprocessing as Stages 1 and 2. The keys are the original field names and were
> left untouched so that the released code reads the released data unmodified.

### An entry record

```jsonc
{
  "article_id": 1,
  "fever_id": "1999_Detroit_Lions_season",
  "source_language": "en",
  "mode": "consensus",
  "candidates": [ ... ],
  "review": { ... },
  "evaluation": { ... },
  "token_usage": { ... }
}
```

`article_id` matches `id` in the corpus file — that is the join key. `fever_id`
is the FEVER page identifier and appears only in the FEVER trace; the WikiText
and Europarl traces have no extra identifier field. `mode` is `consensus` when a
reviewer merged the candidates.

#### `candidates` — one per translation model

```jsonc
{
  "model": "foundry:dataset-frontierscience-mistral-large",
  "translated_at": "2026-07-01T19:08:50.667038",
  "char_count": 1069,
  "text": "La temporada 1999 de los Detroit Lions fue su 70.ª en la liga. ..."
}
```

These are the raw Stage 3 outputs, before review. Comparing them against the
consensus in the corpus file is exactly what `metrics/ablation_consensus.py`
does.

#### `review` — the Stage 4 critique

```jsonc
{
  "model": "azure:dataset-frontierscience-gpt41",
  "reviewed_at": "2026-07-01T19:10:12.482",
  "critique": {
    "candidates_analysis": [
      {
        "model": "foundry:...mistral-large",
        "strengths": ["Traducción fluida y natural", "Terminología deportiva precisa"],
        "weaknesses": ["Ligera omisión en la traducción de 'opening round' ..."],
        "score": 93
      }
      // one object per candidate
    ],
    "decision": "El candidato 1 es el más natural, preciso y fluido. Usaría el candidato 1 como base, corrigiendo ..."
  }
}
```

The critique is in Spanish: the reviewer prompt is written in the target
language. `score` is the reviewer's own 0–100 judgement of that candidate, which
is *not* the same scale as the Stage 5 evaluation and should not be compared
with it.

**The merged consensus text is not in this file.** It is `spanish.text` in the
corpus file. Joining on `article_id` gives you candidates and result together.

#### `evaluation` — the Stage 5 scores

```jsonc
{
  "clarity": 98,
  "fidelity": 98,
  "completeness": 100,
  "fluency": 98,
  "overall_score": 97,
  "errors": [
    "Uso inconsistente de comillas: el original usa comillas inglesas ...",
    "Traducción de 'Wild Card round' como 'ronda de comodines' es correcta ..."
  ],
  "char_ratio": 1.165,
  "model": "azure:dataset-frontierscience-gpt41mini",
  "evaluated_at": "2026-07-01T19:10:55.031950"
}
```

| Field | Meaning |
|---|---|
| `clarity`, `fidelity`, `completeness`, `fluency` | 0–100, assigned by the evaluator |
| `overall_score` | The evaluator's combined score, or the recomputed one if the truncation check fired |
| `errors` | Free-text observations. Mostly stylistic: the evaluator flagged something in 72% of WikiText and FEVER entries, but only 25.5% of Europarl, and these are overwhelmingly notes rather than failures. |
| `char_ratio` | `len(spanish) / len(english)`. Below 0.92, the programmatic truncation check of Stage 5 runs. |

When the programmatic check overrode the model, the record also carries the
details of that adjustment. The four FEVER entries whose evaluation call failed
have no `evaluation` block at all — handle its absence.

#### `token_usage` — measured cost

```jsonc
{
  "stage_1_translation": {
    "foundry:...mistral-large": {"prompt_tokens": 555, "completion_tokens": 263, "total_tokens": 818},
    "azure:...gpt41mini":       {"prompt_tokens": 535, "completion_tokens": 244, "total_tokens": 779},
    "ollama:qwen3.5:9b":        {"prompt_tokens": 573, "completion_tokens": 271, "total_tokens": 844}
  },
  "stage_2_review":     {"prompt_tokens": 2472, "completion_tokens": 669, "total_tokens": 3141},
  "stage_3_evaluation": {"prompt_tokens": 1038, "completion_tokens": 120, "total_tokens": 1158},
  "metadata":           {"prompt_tokens": 830,  "completion_tokens": 192, "total_tokens": 1022},
  "cost_usd": { /* the same breakdown, in dollars */ }
}
```

Token counts come from the API responses, so costs are measured, not estimated.
Locally hosted models report tokens but cost zero. Rates are applied from
`MODEL_PRICING` in `agents/agente_wikitext.py` at run time, which means **the
recorded costs reflect the prices in effect during the run** and do not update
if you change that table later.

---

## The other released files

| Path | What it is |
|---|---|
| `corpus/selection/*.csv` | The stratified selection tables from notebooks 01–03: which entries were chosen, with their topic and size bin |
| `corpus/metrics/metrics_<dataset>.json` | Output of `python3 -m metrics.run_metrics` — the general, frequency, lexical, syntactic and quality metrics behind the paper's per-dataset tables |
| `corpus/metrics/metrics_combined.json` | Cross-dataset totals, vocabulary overlap, duplicates, vocabulary growth |
| `corpus/metrics/semantic_full.json` | LaBSE fidelity and omission over all 12,003 entries |
| `corpus/metrics/ablation_consensus.json` | The Stage 4 ablation: consensus against each single model and against the per-entry oracle |
| `corpus/pilot/*` | The June 2026 WikiText pilot run, read by notebook 05 |
| `corpus/samples/*` | A few entries per dataset, uncompressed |
