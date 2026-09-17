# Un corpus paralelo bilingüe inglés–español construido con consenso multi-modelo

*[Read in English](README.md)*

Este repositorio contiene el corpus, el código y el análisis del artículo
**«Building a Bilingual English–Spanish Parallel Corpus Using Multi-Model LLM
Translation with Consensus Review»**.

El corpus reúne **12,003 entradas alineadas inglés–español y 23.5 millones de
palabras** de tres dominios: artículos enciclopédicos (WikiText-103), pasajes
factuales breves (FEVER) y debate parlamentario (Europarl). Lo produjeron
únicamente modelos de lenguaje de propósito general: tres modelos traducen cada
texto en paralelo, un modelo revisor fusiona sus candidatos en una sola
traducción por consenso, y un modelo evaluador califica el resultado contra una
guía calibrada que no necesita traducción humana de referencia. El corpus
completo costó **\$453.90**, es decir \$39 por millón de palabras fuente, en una
sola laptop de consumo.

Aquí está todo lo necesario para inspeccionar el corpus, reproducir el análisis o
volver a ejecutar el pipeline sobre texto nuevo.

---

## Qué contiene el corpus

| Dataset | Dominio | Entradas | Palabras EN | Palabras ES | Calificación media |
|---|---|---:|---:|---:|---:|
| WikiText-103 | Enciclopédico | 1,000 | 2,855,853 | 3,037,875 | 95.9 |
| FEVER | Factual | 9,998 | 1,753,860 | 1,941,463 | 95.1 |
| Europarl | Parlamentario | 1,005 | 6,950,546 | 6,954,048 | 97.0 |
| **Total** | | **12,003** | **11,560,259** | **11,933,386** | **—** |

La fidelidad semántica translingüe, medida con LaBSE sin traducción de
referencia, es de 0.92 y se mantiene dentro de seis milésimas en los tres
dominios. La omisión a nivel de oración queda entre 0.34% y 1.43%.

Cada entrada lleva título, palabras clave, resumen y clasificación temática en
ambos idiomas, además de las calificaciones que asignó el evaluador. La
estructura exacta de cada registro está documentada en
[docs/es/corpus-format.md](docs/es/corpus-format.md).

**Vea los datos antes de descargarlos:** [`corpus/samples/`](corpus/samples/)
contiene algunas entradas por dataset, sin comprimir y legibles directamente en
GitHub, incluida una traza del pipeline que muestra las tres traducciones
candidatas, la fusión del revisor y las calificaciones del evaluador para una
sola entrada.

---

## Inicio rápido

```bash
git clone https://github.com/vsosahdz/SpanishDBText.git
cd SpanishDBText

python3 -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python3 scripts/unpack_corpus.py --final
```

Eso expande el corpus bilingüe en `output/`, que es donde lo buscan todos los
scripts y notebooks del repositorio:

```
output/wikitext/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json
output/FEVER/fever_selected_260712_212751.json
output/Europarl/selected_260710_084528.json
```

Para leer uno de ellos basta la biblioteca estándar:

```python
import json

with open("output/FEVER/fever_selected_260712_212751.json", encoding="utf-8") as fh:
    corpus = json.load(fh)

for entry in corpus["payload"][0]["data"][:3]:
    print(entry["english"]["title"])
    print("  EN:", entry["english"]["text"][:120])
    print("  ES:", entry["spanish"]["text"][:120])
```

Para ir más lejos —recalcular las métricas, volver a correr el pipeline o
aplicarlo a texto propio— siga [docs/es/reproduction.md](docs/es/reproduction.md).

---

## Por qué los datos están comprimidos

El corpus son 437 MB de JSON. Distribuido como `.json.gz` cabe en un repositorio
de GitHub, con cada archivo por debajo del límite de 100 MB por archivo.

Esa compresión es la **única** diferencia entre lo que se publica y lo que se
ejecutó. Los agentes, el paquete de métricas y los notebooks no se modificaron
para la publicación: leen JSON plano de `output/` y `data/` exactamente como
durante la producción, así que `scripts/unpack_corpus.py` restaura esa
estructura, nombres de archivo incluidos. (La única excepción está anotada en
[docs/es/reproduction.md](docs/es/reproduction.md#cambios-hechos-para-la-publicación):
el notebook 06 ahora escribe sus figuras en `figures/` y no dentro del árbol
LaTeX del artículo.)

```bash
python3 scripts/unpack_corpus.py --final      # el corpus bilingüe            159 MB
python3 scripts/unpack_corpus.py --pipeline   # trazas por entrada            278 MB
python3 scripts/unpack_corpus.py --inputs     # los subconjuntos de entrada    87 MB
python3 scripts/unpack_corpus.py --all        # todo                          530 MB
```

---

## Estructura del repositorio

```
.
├── corpus/                      los datos publicados, comprimidos
│   ├── final/                   el corpus bilingüe, un archivo por dataset
│   ├── pipeline/                trazas por entrada: candidatos, revisión, scores, costo
│   ├── inputs/                  los subconjuntos en inglés que consumió el pipeline
│   ├── selection/               las tablas de selección estratificada (CSV)
│   ├── metrics/                 las métricas calculadas detrás de las tablas del artículo
│   ├── pilot/                   la corrida piloto de WikiText (junio 2026)
│   └── samples/                 algunas entradas, sin comprimir y legibles en GitHub
│
├── agents/                      el pipeline, un agente por dataset de origen
│   ├── agente_wikitext.py       también el núcleo compartido: proveedores, prompts, etapas
│   ├── agente_fever.py
│   ├── agente_europarl.py
│   └── format_europarl.py       normaliza el marcado de Europarl a encabezados WikiText
│
├── metrics/                     las medidas de evaluación, como paquete importable
│   ├── general_metrics.py       totales, longitudes, frecuencia de palabras, Zipf, hapax
│   ├── lexical_metrics.py       TTR, RTTR, CTTR, MATTR, MTLD
│   ├── syntactic_metrics.py     longitud de oración, distribución POS (spaCy)
│   ├── quality_metrics.py       entidades nombradas, duplicados, razón de expansión
│   ├── semantic_metrics.py      fidelidad y omisión translingüe (LaBSE)
│   ├── combined_metrics.py      totales entre datasets, solapamiento, crecimiento léxico
│   ├── ablation_consensus.py    la ablación de la Etapa 4: consenso vs sus candidatos
│   └── run_metrics.py           punto de entrada: python3 -m metrics.run_metrics --all
│
├── notebooks/                   01–03 seleccionan los subconjuntos, 04–06 analizan la salida
├── scripts/                     descargar, descomprimir, comprimir, verificar, muestrear
├── docs/                        guías y referencia, en inglés y español
├── data/  output/  figures/     vacíos; los llenan los scripts
│
├── .env.example                 plantilla de configuración — copiar a .env
├── docker-compose.yml           Ollama, para el modelo alojado localmente
├── requirements.txt
├── LICENSE                      Apache 2.0, para el código
├── LICENSE-DATA.md              CC BY-SA 4.0, para el corpus, y atribución de fuentes
└── CITATION.cff
```

---

## Documentación

| Documento | Qué cubre |
|---|---|
| [docs/es/reproduction.md](docs/es/reproduction.md) | Paso a paso: instalar, configurar, recalcular métricas, correr el pipeline, aplicarlo a texto propio |
| [docs/es/pipeline.md](docs/es/pipeline.md) | Las seis etapas, qué hace cada una y dónde vive en el código |
| [docs/es/configuration.md](docs/es/configuration.md) | Cada variable de `.env`, cada parámetro del pipeline y por qué tiene el valor que tiene |
| [docs/es/corpus-format.md](docs/es/corpus-format.md) | La estructura JSON del corpus y de las trazas, campo por campo |
| [docs/es/metrics-reference.md](docs/es/metrics-reference.md) | Cada métrica: qué mide, quién la introdujo, cómo se calcula aquí |
| [docs/es/metrics-results.md](docs/es/metrics-results.md) | Los resultados calculados, leídos críticamente, anomalías incluidas |

---

## El pipeline en un párrafo

Cada entrada pasa por seis etapas. **Selección** toma una muestra estratificada
del dataset de origen (semilla 42). **Preprocesamiento** detecta el idioma,
limpia el texto y divide todo lo que supere ~6,000 tokens en una frontera
estructural: encabezados de sección en WikiText, turnos de orador en Europarl.
**Traducción** envía el texto a tres modelos a la vez: GPT-4.1 Mini, Mistral
Large y un Qwen 3.5 9B alojado localmente, elegidos de tres familias distintas
para que no cometan los mismos errores. **Revisión** entrega a GPT-4.1 el
original y los tres candidatos, y le pide criticarlos y fusionarlos en una sola
traducción; esta etapa sola es el 77% del costo. **Evaluación** califica el
resultado en claridad, fidelidad, completitud y fluidez, con una verificación
programática de la razón de caracteres que sobrescribe al modelo cuando es
demasiado generoso respecto al truncamiento. **Metadatos** extrae título,
palabras clave, resumen y clasificación en ambos idiomas. El detalle completo
está en [docs/es/pipeline.md](docs/es/pipeline.md).

---

## Volver a correr el pipeline

Leer el corpus no necesita nada de esto. Ejecutarlo sí, y cuesta dinero: empiece
con `--limit 10` y lea el costo reportado antes de escalar.

**1. Configurar.** Copie la plantilla y complete sus endpoints, llaves y nombres
de deployment. `.env` está en `.gitignore` y debe seguir así.

```bash
cp .env.example .env
```

**2. Levantar Ollama**, si algún rol usa el proveedor `ollama`. La configuración
publicada corre Qwen 3.5 9B localmente, que es lo que mantiene en cero un tercio
del costo de traducción.

```bash
docker compose up -d
docker exec -it ollama ollama pull qwen3.5:9b
curl http://localhost:11434/api/tags          # debe listar el modelo
```

Descomente el bloque `deploy:` de `docker-compose.yml` para darle al contenedor
su GPU NVIDIA. Sin él, Ollama corre en CPU: funciona, pero el modelo local ya es
la parte más lenta del pipeline y en CPU domina el tiempo total.

**3. Verificar que cada modelo configurado responde** antes de comprometer horas
de corrida. Esto envía un prompt corto por modelo y cuesta una fracción de
centavo.

```bash
python3 scripts/verify_setup.py --models
```

**4. Ejecutar**, desde la raíz del repositorio: los agentes resuelven `data/` y
`output/` relativo al directorio de trabajo.

```bash
python3 scripts/unpack_corpus.py --inputs
python3 agents/agente_fever.py --input data/FEVER/selected/fever_selected.jsonl --limit 10
```

Una corrida interrumpida se reanuda relanzando el mismo comando: el pipeline
escribe un archivo de progreso tras cada entrada completada y se salta lo ya
hecho.

Cada opción, cada parámetro y el procedimiento completo para los otros dos
agentes están en
[docs/es/reproduction.md](docs/es/reproduction.md#4-volver-a-correr-el-pipeline)
y [docs/es/configuration.md](docs/es/configuration.md).

---

## Reutilizar el pipeline para otro idioma o dominio

Nada en el diseño es específico del par inglés–español. Los modelos se declaran
en `.env` como pares `proveedor:modelo` y se leen al arrancar, así que cambiar
cualquier rol no requiere tocar código. Apuntar el pipeline a un idioma nuevo
significa editar los prompts en `agents/agente_wikitext.py`, que son constantes
de texto agrupadas cerca del inicio del archivo. Un formato de origen nuevo
significa un agente nuevo que importe el núcleo compartido y aporte su propio
cargador y su regla de fragmentación: los tres agentes existentes son los
ejemplos resueltos, y `agente_fever.py` es el más pequeño. Vea
[docs/es/reproduction.md](docs/es/reproduction.md#usar-el-pipeline-con-texto-propio).

---

## Costo, si vuelve a ejecutarlo

| Dataset | Entradas | Costo (USD) | Por entrada | Etapa 4 | Horas |
|---|---:|---:|---:|---:|---:|
| WikiText | 1,000 | 104.54 | 0.1045 | 73.5% | 120.8 |
| FEVER | 9,998 | 142.17 | 0.0142 | 80.7% | 266.3 |
| Europarl | 1,005 | 207.18 | 0.2062 | 75.5% | 227.6 |
| **Total** | **12,003** | **453.90** | **0.0378** | **76.6%** | **614.7** |

Son cifras medidas, no estimadas: cada respuesta de la API reporta sus conteos de
tokens, y el pipeline escribe el costo por llamada en las trazas de
`corpus/pipeline/`. Las tarifas son los precios de lista de Azure vigentes
durante la corrida (\$0.40/\$1.60 por millón de tokens para GPT-4.1 Mini,
\$2.00/\$8.00 para GPT-4.1, \$0.50/\$1.50 para Mistral Large); el modelo Qwen
local aporta cero. Consulte los precios actuales antes de presupuestar, y vea
[docs/es/reproduction.md](docs/es/reproduction.md#antes-de-gastar-nada) para
correr primero un piloto barato.

---

## Licencia

Este repositorio se licencia en dos partes, porque el código y los datos tienen
orígenes distintos.

- **Código** — `agents/`, `metrics/`, `scripts/`, `notebooks/` — Apache License 2.0
  ([LICENSE](LICENSE)).
- **Corpus** — todo lo que está bajo `corpus/` — Creative Commons
  Atribución-CompartirIgual 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)).

La licencia del corpus es heredada, no elegida. Dos de las tres fuentes son texto
CC BY-SA de Wikipedia, y una traducción es obra derivada, así que la obligación
de CompartirIgual se traslada al lado en español. Si redistribuye el corpus debe
acreditar a WikiText-103, FEVER y Europarl —los tres citados en
LICENSE-DATA.md— y licenciar su versión en términos compatibles.

GitHub etiqueta todo el repositorio como «Apache-2.0» porque solo lee el archivo
`LICENSE`. Esa etiqueta aplica al código; los términos del corpus son los de
LICENSE-DATA.md.

## Citar

Vea [CITATION.cff](CITATION.cff). Por favor cite también los tres datasets de
origen.

## Contacto

Victor Adrian Sosa-Hernández — <vsosa@tec.mx>
School of Engineering and Sciences, Tecnológico de Monterrey, Monterrey, México
