# Formato del corpus

*[Read in English](../corpus-format.md)*

Cada corrida del pipeline escribe dos archivos JSON. El **archivo del corpus**
contiene el resultado bilingüe limpio y es lo que quiere la mayoría de los
usuarios. La **traza del pipeline** contiene el registro completo de cómo se
produjo ese resultado. Ambos son UTF-8 y ambos se distribuyen comprimidos bajo
`corpus/`.

Para leer esta página con los datos delante, abra
[`corpus/samples/`](../../corpus/samples/): algunas entradas de cada dataset, sin
comprimir y con formato legible.

---

## El archivo del corpus

`corpus/final/*.json.gz` → se expande a `output/<dataset>/`.

Un solo objeto JSON. El nivel superior identifica la fuente, `payload[0]`
describe la corrida, y `payload[0]["data"]` es la lista de entradas.

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
      "data": [ /* las entradas */ ]
    }
  ]
}
```

| Campo | Significado |
|---|---|
| `source_name` | `wikitext`, `fever` o `europarl` |
| `url`, `creation_date`, `authors_name` | Procedencia del dataset de origen, tal como se registró en la corrida |
| `payload[0].file_name` | El archivo o directorio de entrada que consumió la corrida |
| `payload[0].file_size`, `.file_format` | Tamaño y formato de esa entrada |
| `payload[0].description` | Nota libre sobre qué es el dataset y por qué se incluyó |
| `payload[0].translation_date` | Fecha en que terminó la corrida |
| `payload[0].llm_used` | Los modelos de traducción de la Etapa 3, como `proveedor:modelo` |

> Los campos de procedencia se llenaron a mano durante la corrida y no siempre
> son la cita canónica: el archivo de WikiText, por ejemplo, registra la
> re-subida a Kaggle desde la que se descargó y no a Merity et al. (2017). Para
> citar, use [LICENSE-DATA.md](../../LICENSE-DATA.md), que da la fuente original
> de cada dataset.

`payload` siempre es una lista de un elemento. Existe porque el formato se diseñó
para admitir varios archivos de entrada en un mismo corpus; ninguna corrida
publicada usa más de uno.

### Una entrada

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

| Campo | Significado |
|---|---|
| `id` | Secuencial dentro del archivo, empieza en 1 |
| `source_language` | `en` en todas las entradas publicadas; el pipeline es bidireccional pero corrió en un solo sentido |
| `english` / `spanish` | Los dos lados alineados |
| `…title` | Extraído en la Etapa 6, no tomado de la fuente |
| `…keywords` | 5–8 términos, extraídos en la Etapa 6 |
| `…summary` | De dos a cuatro oraciones, extraídas en la Etapa 6 |
| `…classification` | Una etiqueta temática, asignada en la Etapa 6 |
| `…text` | El texto completo. Esto es el corpus. |
| `spanish.translated_at` | Marca de tiempo ISO del consenso de la Etapa 4 |
| `spanish.translated_by` | El modelo **revisor** que produjo el consenso, no un traductor |

El bloque del idioma fuente siempre lleva sus propios metadatos y el del idioma
destino los suyos, generados de forma independiente y no traducidos: así que
`english.summary` y `spanish.summary` son dos resúmenes, no un resumen y su
traducción.

### Qué contiene `text`, por dataset

- **WikiText** conserva la estructura de secciones del original: los encabezados
  aparecen como `= Título =`, `= = Sección = =`, `= = = Subsección = = =`. El
  lado en inglés conserva además la tokenización original del dataset, en la que
  la puntuación va separada (`league .`, `5 -- 11`); el lado en español es prosa
  normal.
- **FEVER** son pasajes planos de unas pocas oraciones, con la misma tokenización
  separada en el lado inglés.
- **Europarl** conserva el marcado XML de la fuente en **ambos** lados:
  `<CHAPTER ID="002">`, `<SPEAKER ...>`, `<P>`. Es deliberado: quién habla y
  dónde empieza una intervención es parte del contenido de un acta parlamentaria.
  También cuenta como texto en los totales de palabras del artículo: el marcado
  es el 4.2% de las palabras de cada lado de Europarl y el 2.5% del corpus
  combinado. **Elimínelo antes de calcular estadísticas de superficie**, o estará
  midiendo etiquetas. `metrics/semantic_metrics.py` muestra la normalización
  usada en el artículo.

### Cómo leerlo

```python
import json

with open("output/FEVER/fever_selected_260712_212751.json", encoding="utf-8") as fh:
    entries = json.load(fh)["payload"][0]["data"]

pares = [(e["english"]["text"], e["spanish"]["text"]) for e in entries]
```

Los tres archivos tienen 1,000, 9,998 y 1,005 entradas y ocupan 38, 36 y 85 MB
expandidos. `json.load` mantiene todo en memoria; use `ijson` en streaming si la
máquina es limitada.

### Huecos conocidos

Se produjeron 12,003 entradas de 12,006 seleccionadas. Tres fallaron por errores
de API y no se reintentaron: 2 de FEVER y 1 de Europarl. Además, **4 entradas de
FEVER no llevan calificaciones de calidad**: se tradujeron y revisaron con éxito
pero su llamada de evaluación de la Etapa 5 falló. Están presentes en el corpus y
completas como texto. Por eso las estadísticas de evaluación del artículo se
calculan sobre 9,994 entradas de FEVER y no 9,998.

---

## La traza del pipeline

`corpus/pipeline/*.json.gz` → se expande a `output/<dataset>/`.

Otra forma: sin `payload`, y `pipeline` es una lista plana de registros por
entrada. Este archivo es lo que hace posible el análisis de costos y la ablación
de la Etapa 4: las tres traducciones candidatas están guardadas aquí, así que
comparar el consenso contra ellas no requiere retraducir nada.

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
  "pipeline": [ /* un registro por entrada */ ]
}
```

> **Los números de etapa de este archivo están desfasados en dos respecto al
> artículo.** La traza escribe `stage_1_translation`, `stage_2_review` y
> `stage_3_evaluation`; el artículo llama a lo mismo Etapas 3, 4 y 5, contando
> selección y preprocesamiento como Etapas 1 y 2. Las claves son los nombres de
> campo originales y se dejaron intactas para que el código publicado lea los
> datos publicados sin modificaciones.

### Un registro de entrada

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

`article_id` coincide con `id` en el archivo del corpus: esa es la clave de
unión. `fever_id` es el identificador de página de FEVER y aparece solo en la
traza de FEVER; las trazas de WikiText y Europarl no tienen campo identificador
adicional. `mode` es `consensus` cuando un revisor fusionó los candidatos.

#### `candidates` — uno por modelo de traducción

```jsonc
{
  "model": "foundry:dataset-frontierscience-mistral-large",
  "translated_at": "2026-07-01T19:08:50.667038",
  "char_count": 1069,
  "text": "La temporada 1999 de los Detroit Lions fue su 70.ª en la liga. ..."
}
```

Son las salidas crudas de la Etapa 3, antes de la revisión. Compararlas contra el
consenso del archivo del corpus es exactamente lo que hace
`metrics/ablation_consensus.py`.

#### `review` — la crítica de la Etapa 4

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
      // un objeto por candidato
    ],
    "decision": "El candidato 1 es el más natural, preciso y fluido. Usaría el candidato 1 como base, corrigiendo ..."
  }
}
```

La crítica está en español: el prompt del revisor está escrito en el idioma
destino. `score` es el juicio propio del revisor sobre ese candidato, de 0 a 100,
que *no* está en la misma escala que la evaluación de la Etapa 5 y no debe
compararse con ella.

**El texto fusionado del consenso no está en este archivo.** Es `spanish.text` en
el archivo del corpus. Unir por `article_id` da candidatos y resultado juntos.

#### `evaluation` — las calificaciones de la Etapa 5

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

| Campo | Significado |
|---|---|
| `clarity`, `fidelity`, `completeness`, `fluency` | 0–100, asignados por el evaluador |
| `overall_score` | La puntuación combinada del evaluador, o la recalculada si se disparó la verificación de truncamiento |
| `errors` | Observaciones en texto libre. Mayormente estilísticas: el evaluador señaló algo en el 72% de las entradas de WikiText y FEVER, pero solo en el 25.5% de Europarl, y son en su abrumadora mayoría notas y no fallos. |
| `char_ratio` | `len(español) / len(inglés)`. Por debajo de 0.92 corre la verificación programática de truncamiento de la Etapa 5. |

Cuando la verificación programática sobrescribió al modelo, el registro lleva
además los detalles de ese ajuste. Las cuatro entradas de FEVER cuya llamada de
evaluación falló no tienen bloque `evaluation` en absoluto: contemple su
ausencia.

#### `token_usage` — costo medido

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
  "cost_usd": { /* el mismo desglose, en dólares */ }
}
```

Los conteos de tokens vienen de las respuestas de la API, así que los costos son
medidos y no estimados. Los modelos locales reportan tokens pero cuestan cero.
Las tarifas se aplican desde `MODEL_PRICING` en `agents/agente_wikitext.py` en
tiempo de ejecución, lo que significa que **los costos registrados reflejan los
precios vigentes durante la corrida** y no se actualizan si después cambia esa
tabla.

---

## Los demás archivos publicados

| Ruta | Qué es |
|---|---|
| `corpus/selection/*.csv` | Las tablas de selección estratificada de los notebooks 01–03: qué entradas se eligieron, con su tema y rango de tamaño |
| `corpus/metrics/metrics_<dataset>.json` | Salida de `python3 -m metrics.run_metrics`: las métricas generales, de frecuencia, léxicas, sintácticas y de calidad detrás de las tablas por dataset del artículo |
| `corpus/metrics/metrics_combined.json` | Totales entre datasets, solapamiento de vocabulario, duplicados, curva de crecimiento léxico |
| `corpus/metrics/semantic_full.json` | Fidelidad y omisión con LaBSE sobre las 12,003 entradas |
| `corpus/metrics/ablation_consensus.json` | La ablación de la Etapa 4: consenso contra cada modelo individual y contra el oráculo por entrada |
| `corpus/pilot/*` | Las corridas de prueba de WikiText de junio de 2026, que leen el notebook 05 y `ablation/` |
| `corpus/samples/*` | Algunas entradas por dataset, sin comprimir |
