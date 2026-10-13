# Ablation del evaluador: guía de puntuación y evidencia de longitud

*[Read in English](README.md)*

El prompt de evaluación de la etapa 5 combina dos medidas. La **guía de puntuación** describe
qué significa cada banda de la escala 0–100. La **evidencia de longitud** consiste en los conteos
de caracteres de ambos textos, su ratio y una advertencia cuando la traducción es mucho más corta
que el original. Esta prueba mide qué aporta cada una. El artículo la resume en la sección 4.3.3,
en el párrafo que sigue a la descripción del prompt de evaluación.

## Pregunta

Con el mismo evaluador y las mismas traducciones, ¿qué cambia en los puntajes cuando se quita la
guía, la evidencia de longitud, o ambas?

## Diseño

Diseño factorial 2 × 2. Solo cambia el prompt del evaluador:

| Config | Guía de puntuación (G) | Evidencia de longitud (L) | Nota |
|---|---|---|---|
| **G1L1** | sí | sí | Prompt de producción, idéntico al que puntuó el corpus |
| G0L1 | no | sí | |
| G1L0 | sí | no | |
| G0L0 | no | no | Línea base: sin anclaje ni datos de longitud |

- **G (guía)** = el rol "ESTRICTO y CRÍTICO", la frase "una puntuación perfecta (100) debe ser
  extremadamente rara" y las seis bandas de la Tabla 1 del artículo. Sin G, el rol pasa a ser
  "evaluador experto".
- **L (evidencia de longitud)** = el bloque "CRÍTICO — VERIFICACIÓN DE COMPLETITUD" del system
  prompt y la sección METADATOS del user prompt: conteos de caracteres, ratio y las advertencias
  por debajo de 0.92 y 0.85.
- **Común a las cuatro:** el formato JSON de salida y la definición de los cuatro criterios.

`prompts.py` genera las variantes **quitando bloques del prompt de producción** de
`../agents/agente_wikitext.py`, con `assert` que fallan si algún bloque cambia. Los prompts
resultantes quedan en `prompts/`.

### Qué se mantiene constante

- **Traducciones:** las 62 traducciones finales de la corrida de prueba del 7 de junio de 2026
  sobre el split de test de WikiText-103
  (`../corpus/pilot/wikitext-103-raw.wiki.test_260607.json.gz`). Se eligió porque contiene **tres
  truncados conocidos**:

  | ID | Artículo | Ratio de longitud | Contenido perdido |
  |---|---|---|---|
  | 7 | Second Battle of Naktong Bulge | 0.36 | 64% |
  | 24 | 2003 Pacific typhoon season | 0.26 | 74% |
  | 41 | Manila | 0.79 | 21% |

  Aquella corrida usó Mistral Large como evaluador y una versión anterior del prompt de
  evaluación. Sus puntajes quedan en `data/inputs.jsonl` (`historic_mistral_*`) solo como
  contexto; no son una de las condiciones del experimento.
- **Originales:** `english.text` del mismo archivo, que es el `original_text` que recibió el
  evaluador.
- **Evaluador:** `EVALUATOR_MODEL` de `../.env`, el de producción (GPT-4.1 Mini), temperatura 0.2
  y `max_tokens = 1024`.
- **Código:** la llamada, los reintentos, el parseo del JSON y el ajuste programático de
  completitud se importan del agente de producción. No hay reimplementaciones.

### Repeticiones y orden

Cada par (artículo, configuración) se evalúa 3 veces (`--repeats`), porque la temperatura 0.2
deja algo de variación. Las llamadas se ejecutan en un orden aleatorio fijo (semilla 42) que
intercala las configuraciones, para que una deriva del servicio no se confunda con el efecto de
un prompt.

## Resultados (10 de octubre de 2026)

744 de 744 evaluaciones (4 configuraciones × 62 artículos × 3 repeticiones), US$3.08. El detalle
está en `results/summary.md` y `results/summary.json`; cada llamada está en `results/raw/`.

**Nivel de los puntajes**, overall asignado por el modelo en las 59 traducciones completas (177
evaluaciones por configuración):

| Config | Media | Mediana | DE | % ≥95 | Valor más frecuente |
|---|---|---|---|---|---|
| G1L1 (producción) | 95.75 | 96 | 2.01 | 80.2 | 95 (28%) |
| G0L1 | 96.79 | 97 | 0.99 | 96.6 | 97 (82%) |
| G1L0 | 94.92 | 95 | 1.70 | 76.3 | 95 (37%) |
| G0L0 | 96.44 | 97 | 1.50 | 91.5 | 97 (79%) |

Diferencia pareada por artículo contra producción, con IC 95% bootstrap: G0L1 +1.04
[0.68, 1.42]; G1L0 −0.82 [−1.13, −0.52]; G0L0 +0.69 [0.23, 1.14].

**Truncados conocidos**, completitud asignada por el modelo en las tres repeticiones:

| ID (ratio) | G1L1 | G0L1 | G1L0 | G0L0 |
|---|---|---|---|---|
| 7 (0.36) | 36, 36, 36 | 36, 36, 36 | 90, 95, 95 | 90, 90, 90 |
| 24 (0.26) | 26, 26, 26 | 30, 30, 30 | 95, 95, 95 | 90, 90, 90 |
| 41 (0.79) | 79, 70, 79 | 70, 70, 70 | 99, 99, 99 | 100, 99, 99 |

**Lectura:**

1. **La detección de truncamiento depende de la evidencia de longitud, no de la guía.** Con la
   evidencia de longitud, el modelo fija la completitud en el ratio. Sin ella, da 90–100 a
   traducciones que perdieron entre el 21 y el 74% del contenido, con o sin guía.
2. **La guía cambia poco el nivel de los puntajes** (alrededor de un punto), pero evita que se
   concentren en un solo valor: sin ella, cerca del 80% de las evaluaciones recibe el mismo
   número, 97; con ella, entre el 28 y el 37%. No impide que el evaluador puntúe alto: con la
   guía, el 76–80% de las traducciones completas sigue en 95 o más.
3. **El ajuste programático no sustituye a la evidencia de longitud.** Sin L, baja los dos
   truncados graves a un overall de 68–73, pero deja a Manila (ratio 0.79) en 88–90.
4. En las traducciones completas, quitar la evidencia de longitud baja el overall 0.8 puntos: el
   ratio (alrededor de 1.04, sin advertencia) parece tranquilizar al evaluador.

## Cómo correrlo

Desde esta carpeta, con `../.env` configurado (ver `../docs/es/configuration.md`):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_inputs.py               # data/inputs.jsonl (62 entradas)
.venv/bin/python prompts.py                      # prompts/, para revisar las variantes
.venv/bin/python run_ablation.py --dry-run       # estimación de tokens y costo, sin llamadas
.venv/bin/python run_ablation.py --limit 2 --repeats 1   # prueba de humo (incluye un truncado)
.venv/bin/python run_ablation.py                 # corrida completa (4 configs × 62 × 3)
.venv/bin/python analyze.py                      # results/summary.md y summary.json
```

`results/` ya contiene la corrida publicada. Para reproducirla desde cero, mueve antes
`results/raw/` a otro lugar; si no, `run_ablation.py` encuentra todas las llamadas hechas y las
salta.

`run_ablation.py` es reanudable: si se interrumpe, al volver a correrlo salta lo ya evaluado y
reintenta lo que falló. `--workers` fija las llamadas en paralelo (2 por defecto). Cada llamada
se envuelve en rondas de reintento con espera exponencial (20 a 320 s), porque el bucle de
reintentos de producción no espera y un 429 (cuota de tokens por minuto) lo agota de inmediato.
`analyze.py` conserva una fila por (artículo, repetición) y descarta los intentos fallidos
previos.

## Archivos

| Archivo | Qué es |
|---|---|
| `prepare_inputs.py` | Extrae las 62 entradas de la corrida del 7 de junio a `data/inputs.jsonl` |
| `prompts.py` | Genera las variantes del prompt a partir del de producción |
| `run_ablation.py` | Ejecuta las llamadas y escribe `results/raw/<config>.jsonl` |
| `analyze.py` | Resume en `results/summary.md` y `results/summary.json` |
| `prompts/` | System prompt de cada variante y un user prompt de ejemplo |
| `results/raw/` | Una línea por llamada: evaluación del modelo, evaluación ajustada, respuesta cruda, tokens y costo |

## Limitaciones

- 62 artículos de un solo dominio (WikiText); los resultados no se trasladan a FEVER ni a
  Europarl sin probarlos.
- Solo hay tres truncados, así que la detección se reporta caso por caso, no como una tasa.
- El efecto de la guía puede depender del evaluador. Aquí se mide con GPT-4.1 Mini, el de
  producción.
