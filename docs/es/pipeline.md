# El pipeline, etapa por etapa

*[Read in English](../pipeline.md)*

Cada entrada del corpus pasó por las seis etapas siguientes. Este documento dice
qué hace cada una, por qué lo hace así y dónde encontrarla en el código.

El pipeline está implementado como tres agentes, uno por dataset de origen. No
son tres copias: `agents/agente_wikitext.py` contiene el núcleo compartido
—clientes de proveedores, prompts, lógica de reintentos, las seis etapas— y los
otros dos importan de él, sobrescribiendo solo lo que exige el formato de
entrada.

```
agents/agente_wikitext.py     núcleo compartido + el agente de WikiText
agents/agente_fever.py        importa el núcleo; cargador, prompts y fragmentación propios (ninguna)
agents/agente_europarl.py     importa el núcleo; parser propio, fragmentación por orador
```

---

## Etapa 1 — Selección

*Notebooks `01`–`03`. Salida: `corpus/selection/*.csv`, `corpus/inputs/`.*

Traducir un corpus de origen completo no sería asequible ni útil: buena parte es
demasiado corta para traducirse con sentido o demasiado larga para traducirse de
forma económica. Cada fuente se muestrea en cuatro pasos:

1. Analizar el corpus completo y calcular el conteo de palabras por entrada.
2. Filtrar por conteo de palabras, con umbrales propios de cada corpus.
3. Asignar un tema por coincidencia de palabras clave: cada categoría es una
   lista de términos del dominio, y una entrada va a la categoría con más
   coincidencias, o a *other* cuando no coincide nada.
4. Muestrear estratificando por tema y, dentro de cada tema, por rango de tamaño.

El paso 3 usa una regla determinista de palabras clave en vez de una llamada a un
modelo a propósito. La estratificación recorre el corpus de origen entero —decenas
de miles de entradas— *antes* de comprometer presupuesto en traducción, y una
regla fija mantiene la selección reproducible bajo la semilla aleatoria.
`SEED = 42` en todo el proyecto.

| | Original | Filtro | Descartado | Seleccionado |
|---|---:|---|---:|---:|
| WikiText-103 | 29,423 | 1,000–10,000 palabras | 13.9% | 1,000 |
| FEVER | 50,000 | ≥ 100 palabras | 78.2% | 10,000 |
| Europarl | 3,656 | 200–25,000 palabras | 64.1% de 2,802 | 1,006 |

Europarl necesita un paso extra antes del filtro: 854 de los 3,656 archivos
contienen solo marcado estructural sin texto de debate y se descartan, dejando
2,802 con contenido aprovechable.

Los esquemas temáticos difieren según la fuente. WikiText y FEVER, ambos de
Wikipedia, comparten 14 categorías (música, historia militar, deportes, cine y
televisión, geografía, ciencia, transporte, biografía, política, naturaleza,
tecnología, historia, religión y filosofía, *other*). Europarl usa 15 categorías
adaptadas a la actividad parlamentaria. También difiere el texto que se compara
contra las listas de palabras clave: título más los primeros 2,000 caracteres del
cuerpo para WikiText y FEVER, pero solo el título del debate para Europarl, donde
el título de la sesión ya declara el tema.

---

## Etapa 2 — Preprocesamiento

*Código: `clean_text()`, `detect_language()`, `split_articles()` y los
fragmentadores de cada agente.*

### Detección de idioma

El pipeline es bidireccional por diseño: la dirección se decide por entrada a
partir de su idioma detectado, así que una sola corrida puede procesar un corpus
mixto. Las tres fuentes de aquí están en inglés, de modo que corrió
inglés→español de principio a fin.

La detección es una heurística de palabras funcionales. Calcula la proporción de
15 palabras funcionales frecuentes del español (*de, la, el, en, los, las, del,
que, por, con, una, para, como, fue, son*) sobre el total de palabras; por
encima del 8%, el texto es español, y si no, inglés. El español corrido está
muy por encima del 8%; el inglés solo llega ahí cuando cita material en español.
La debilidad conocida son las entradas en inglés densas en nombres propios
españoles, donde los nombres por sí solos pueden empujar la proporción sobre el
umbral.

La misma heurística se reutiliza en la Etapa 3 para verificar que el modelo
efectivamente tradujo. Esa reutilización es la razón de mantenerla barata.

### Limpieza

Se eliminan artefactos de formato, tokens desconocidos y problemas de
codificación heredados. Los marcadores estructurales se normalizan o se
conservan según carguen significado o no: las etiquetas XML de orador de
Europarl se conservan en ambos idiomas, porque quién habla y dónde empieza una
intervención son parte del contenido de un acta parlamentaria, no formato
incidental.

### Fragmentación

Un texto largo enviado en una sola llamada es propenso al truncamiento
*silencioso*: el modelo traduce la primera porción, se detiene, y devuelve algo
que se lee perfectamente mientras le falta el último tercio. Nada aguas abajo lo
detectaría.

Por eso las entradas por encima de ~6,000 tokens estimados se dividen en
fronteras estructurales y se reagrupan en fragmentos de máximo 12,000
caracteres. WikiText divide en encabezados de sección, Europarl en turnos de
orador; las entradas de FEVER son lo bastante cortas para enviarse enteras.

El umbral de 6,000 tokens viene de la observación: las entradas por encima
alcanzaban sistemáticamente el tope de 8,192 tokens de salida. El límite de
12,000 caracteres por fragmento es deliberadamente **la mitad** de la conversión
aritmética (~4 caracteres por token darían 24,000): un margen de seguridad que
absorbe tanto la variabilidad de esa razón como la expansión que añade el
español.

---

## Etapa 3 — Generación de candidatos

*Código: `stage_1_generate()` en cada agente, `call_llm()` y
`call_with_retries()` en el núcleo.*

Cada fragmento va a todos los modelos de traducción configurados a la vez, con un
`ThreadPoolExecutor`. La corrida usó tres, de tres familias distintas:

| Rol | Modelo | Proveedor |
|---|---|---|
| Traductor 1 | GPT-4.1 Mini | Azure OpenAI |
| Traductor 2 | Mistral Large | Azure AI Foundry |
| Traductor 3 | Qwen 3.5 9B | Ollama, local |

Que sean familias distintas es un requisito de diseño, no una preferencia. Tres
modelos de una misma familia tienden a cometer los mismos errores, y un consenso
sobre errores correlacionados no aporta nada.

El texto exacto de cada prompt de esta etapa y las siguientes está publicado en
[`prompts/`](../../prompts/), un archivo por prompt.

El prompt de sistema pide fidelidad semántica (no inventar, no omitir), fraseo
natural y no literal, preservación de términos técnicos y nombres propios,
respeto al tono y a la estructura de párrafos, y nada de resumir, simplificar ni
censurar. Después se limpia la respuesta de markdown, marcadores de sección y
cualquier prefijo que el modelo haya añadido.

Tres modos de fallo se tratan explícitamente:

- **Idioma de salida equivocado.** Un modelo que ignora la instrucción y devuelve
  la fuente produce texto fluido que ninguna etapa posterior detectaría. La
  heurística de la Etapa 2 se vuelve a aplicar sobre la salida, y un idioma
  equivocado dispara hasta dos reintentos con un prompt reforzado que nombra el
  error. La mayoría se corrige al primer reintento.
- **Fallos transitorios de la API.** Errores HTTP y respuestas vacías se
  reintentan hasta tres veces. Los rechazos por filtro de contenido se registran
  y marcan pero **no** se reintentan: no van a funcionar en un segundo intento.
- **Truncamiento por límite de salida.** Las APIs reportan una razón de
  finalización; `"length"` en vez de `"stop"` significa que el modelo se quedó sin
  tokens de salida. Se registra como advertencia. Esto detecta el truncamiento
  por el lado de los tokens; la Etapa 5 lo detecta por el lado del contenido.

Si todos los reintentos fallan para una entrada, se omite y la corrida continúa.
Así se perdieron tres entradas en la corrida publicada.

---

## Etapa 4 — Revisión y consenso

*Código: `stage_2_review()`, `_reviewer_critique()`, `_reviewer_merge_section()`,
`_split_text_aligned()`.*

El modelo revisor —GPT-4.1, el más capaz de la configuración— recibe el original
y los tres candidatos, y hace dos cosas.

**Crítica.** De cada candidato nombra fortalezas y debilidades y le asigna una
puntuación de 0 a 100, y luego declara qué candidato debe servir de base y qué
mejoras concretas aplicar. El resultado es JSON estructurado, guardado en las
trazas.

**Fusión.** Guiado por esa crítica, escribe una traducción nueva. Las entradas
cortas se fusionan en una sola llamada. Las largas se fusionan sección por
sección: original y candidatos se dividen en segmentos alineados usando las
mismas fronteras estructurales de la Etapa 2, cada segmento se fusiona de forma
independiente, y los resultados se concatenan.

Fusionar en lugar de seleccionar es la decisión de diseño central. Seleccionar
limita la calidad de salida a la del mejor modelo disponible para esa entrada, y
ningún modelo es el mejor en todas: el revisor eligió a Mistral Large como base
para el 74.7% de las entradas, a GPT-4.1 Mini para el 17.1% y a Qwen 3.5 9B para
el 8.2%. Fusionar permite que un pasaje que un modelo resolvió mejor acabe en una
salida basada en otro. La ablación de `metrics/ablation_consensus.py` lo contrasta
con la alternativa: el consenso puntúa por encima de cada uno de los tres modelos
usados por separado, en todos los datasets, aunque el modelo individual más
fuerte no sea el mismo en todos.

Si no hay revisor configurado, el pipeline recurre al candidato de la Etapa 3 con
mayor puntuación.

**Esta etapa es el 77% del costo total** (\$348 de \$454). Usa el modelo más caro
y procesa la mayor cantidad de texto: el original más todos los candidatos. Es el
primer lugar donde mirar al optimizar un presupuesto.

---

## Etapa 5 — Evaluación de calidad

*Código: `stage_3_evaluate()`, `_adjust_evaluation_by_completeness()`.*

Es la etapa que elimina la dependencia de traducciones humanas de referencia. El
evaluador recibe el original y la traducción final y califica cuatro dimensiones
de 0 a 100 —claridad, fidelidad, completitud, fluidez— más una puntuación global
y una lista de errores, contra una guía calibrada.

### La verificación programática de truncamiento

Los modelos de lenguaje son jueces poco fiables del contenido faltante:
califican lo que tienen delante y rara vez notan lo ausente. Así que después de
que el modelo califica una entrada, corre una segunda verificación, mecánica.

La razón de caracteres es

    r = len(traducción) / len(original)

y se mapea a una puntuación programática de completitud:

| Rango de r | Puntuación programática S_p |
|---|---|
| r < 0.50 | max(100·r, 10) |
| 0.50 ≤ r < 0.75 | max(100·r, 40) |
| 0.75 ≤ r < 0.85 | max(100·r, 70) |
| 0.85 ≤ r < 0.92 | max(105·r, 85) |
| r ≥ 0.92 | no se calcula; vale la puntuación del modelo |

Cada banda aplica un piso, de modo que una traducción corta no se penaliza más
allá de lo que su longitud justifica por sí sola; la tolerancia del 5% de la
banda superior cubre entradas legítimamente más compactas que su fuente.

Si la puntuación de completitud del modelo supera a S_p por más de 10 puntos,
gana S_p, se registra un error de truncamiento y la puntuación global se
recalcula como

    global = media(claridad, fidelidad, fluidez) × 0.6  +  completitud × 0.4

El peso del 40% en completitud refleja un juicio sobre qué importa: el
truncamiento es un fallo peor que una frase torpe.

El umbral de 0.92 es específico de este par de idiomas. Las traducciones al
español de texto en inglés corren 4–9% más largas que su fuente (expansión media
1.04–1.09, desviación estándar 0.02–0.07), así que una traducción por debajo del
92% de su original queda varias desviaciones estándar abajo: falta contenido, no
es fraseo compacto. **Otro par de idiomas necesita otro umbral.**

---

## Etapa 6 — Extracción de metadatos

*Código: `build_metadata_prompt()`, `parse_metadata()`.*

Para cada entrada, en ambos idiomas: título, palabras clave, un resumen y una
clasificación temática. El modelo de metadatos es GPT-4.1 Mini, el mismo usado
para evaluación, ya que extraer campos estructurados es una tarea fácil.

Si el modelo devuelve algo no parseable, un mecanismo de respaldo deriva los
campos heurísticamente del texto en vez de hacer fallar la entrada.

---

## Qué produce una corrida

Dos archivos JSON más un log, por dataset, en `output/<dataset>/`:

- **El archivo del corpus** — el resultado bilingüe limpio: un bloque `english` y
  uno `spanish` por entrada, cada uno con texto y metadatos; el bloque en español
  registra además la marca de tiempo y el modelo que lo produjo.
- **La traza del pipeline** — el registro completo de cada decisión: todos los
  candidatos de la Etapa 3, la crítica y fusión de la Etapa 4, las puntuaciones
  de la Etapa 5, y el uso de tokens y el costo desglosados por entrada, por etapa
  y por modelo.

Ambos están documentados campo por campo en [corpus-format.md](corpus-format.md).

### Contabilidad de costos

El costo se registra por llamada, por modelo, por etapa y por entrada:

    costo_usd = (tokens_in × precio_in + tokens_out × precio_out) / 1,000,000

Los conteos de tokens vienen de las respuestas de la API, así que los totales son
medidos y no estimados. Los modelos locales cuestan cero. Las tarifas viven en
`MODEL_PRICING`, cerca del inicio de `agents/agente_wikitext.py`, y hay que
actualizarlas cuando cambien los precios del proveedor.

### Persistencia incremental

Después de cada entrada completada el pipeline escribe un archivo de progreso con
los IDs terminados y su salida acumulada de las seis etapas. Una corrida
interrumpida por un error de red, un límite de tasa o un reinicio se reanuda
relanzando el mismo comando: las entradas completadas se omiten y el trabajo
continúa en la primera incompleta. Al terminar, el archivo de progreso se consume
para generar los dos JSON de salida.

Con 615 horas de reloj para el corpus completo, esto no es una comodidad.
