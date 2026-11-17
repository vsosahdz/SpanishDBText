# Guía de reproducción

*[Read in English](../reproduction.md)*

De este repositorio se pueden reproducir cuatro cosas, en orden creciente de
costo:

| | Qué obtiene | Costo | Tiempo |
|---|---|---|---|
| [1. Leer el corpus](#1-leer-el-corpus) | Las 12,003 entradas bilingües | gratis | minutos |
| [2. Recalcular las métricas](#2-recalcular-las-métricas) | Cada número de las tablas del artículo | gratis | 1–3 horas |
| [3. Rehacer la selección](#3-rehacer-la-selección-de-subconjuntos) | Las muestras estratificadas, desde las fuentes | gratis | 2 GB de descarga + 1 hora |
| [4. Volver a correr el pipeline](#4-volver-a-correr-el-pipeline) | Un corpus nuevo | **~\$454 el corpus completo** | ~615 h |

La mayoría de los lectores quiere 1 o 2. Lea
[Antes de gastar nada](#antes-de-gastar-nada) antes de intentar 4.

---

## Instalación

Se requiere Python 3.11 o superior: el código usa la sintaxis de tipos `X | None`.

```bash
git clone https://github.com/vsosahdz/SpanishDBText.git
cd SpanishDBText

python3 -m venv venv
source venv/bin/activate              # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la ejecución del script de activación:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Dos modelos se descargan aparte. Solo hacen falta para las métricas sintácticas
y de entidades nombradas; todo lo demás corre sin ellos.

```bash
python3 -m spacy download en_core_web_sm
python3 -m spacy download es_core_news_sm
```

**Nota sobre GPU para las métricas semánticas.** `metrics/semantic_metrics.py`
usa LaBSE a través de `sentence-transformers`, y la build por defecto de PyPI de
`torch` apunta a CUDA 13, que exige driver NVIDIA r580 o superior. Con un driver
anterior torch cae silenciosamente a CPU —sin error, solo un warning— y una
corrida de minutos en GPU tarda horas. Si su driver es más antiguo, instale
primero la build correspondiente:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124   # driver 566.x = CUDA 12.7
python3 -c "import torch; print(torch.cuda.is_available())"            # debe imprimir True
```

Después verifique todo el entorno de una sola vez:

```bash
python3 scripts/verify_setup.py
```

---

## 1. Leer el corpus

```bash
python3 scripts/unpack_corpus.py --final
```

Esto expande tres archivos en `output/`:

```
output/wikitext/wikitext-103-raw.wiki.train.selected_1000_260620_235838.json    38 MB
output/FEVER/fever_selected_260712_212751.json                                  36 MB
output/Europarl/selected_260710_084528.json                                     85 MB
```

Su estructura está documentada campo por campo en
[corpus-format.md](corpus-format.md). En resumen: cada archivo es un objeto JSON
cuyo `payload[0]["data"]` es la lista de entradas, y cada entrada tiene un bloque
`english` y uno `spanish` con `title`, `keywords`, `summary`, `classification` y
`text`.

```python
import json

path = "output/FEVER/fever_selected_260712_212751.json"
with open(path, encoding="utf-8") as fh:
    entries = json.load(fh)["payload"][0]["data"]

print(len(entries), "entradas")
print(entries[0]["english"]["text"])
print(entries[0]["spanish"]["text"])
```

Los archivos son lo bastante grandes como para que `json.load` mantenga unos
cientos de megabytes en memoria. En una máquina limitada, léalos en streaming con
`ijson`, o trabaje sobre `corpus/samples/`.

Para el registro por entrada de lo que produjo cada modelo —las tres
traducciones candidatas, la crítica y fusión del revisor, las calificaciones del
evaluador y el costo en tokens— descomprima también las trazas:

```bash
python3 scripts/unpack_corpus.py --pipeline
```

---

## 2. Recalcular las métricas

Todo lo de las secciones 5 y 6 del artículo sale del paquete `metrics/`. Lee el
corpus expandido desde `output/`, así que haga primero el paso 1.

### Métricas léxicas, sintácticas y de calidad

```bash
python3 -m metrics.run_metrics --dataset fever      # un dataset
python3 -m metrics.run_metrics --all                # los tres
```

Los resultados quedan en `metrics/output/metrics_<dataset>_<timestamp>.json`. Los
archivos que produjeron las tablas del artículo se conservan en
`corpus/metrics/` para comparar.

Cuente unos 10 minutos para FEVER y más para WikiText y Europarl: spaCy tiene que
analizar cada documento en ambos idiomas, y los documentos de Europarl promedian
6,900 palabras.

### Fidelidad semántica translingüe

```bash
python3 -m metrics.semantic_metrics --sample 200                  # piloto, minutos
python3 -m metrics.semantic_metrics --full --out metrics/output/semantic_full.json
```

La primera ejecución descarga LaBSE (~1.8 GB). `--full` sobre las 12,003 entradas
toma alrededor de una hora en GPU y bastante más en CPU.

### La ablación de la Etapa 4

Es la única corrida de métricas que cuesta dinero. Responde a «consenso, ¿mejor
que qué?» calificando cada una de las tres traducciones candidatas con el
*mismo* evaluador que calificó el consenso: mismo modelo, mismo prompt, misma
escala, mismo ajuste programático de completitud. No retraduce nada, porque los
candidatos ya están en las trazas. Sí llama al modelo evaluador, así que necesita
un `.env` configurado (ver más abajo).

```bash
python3 scripts/unpack_corpus.py --pipeline
python3 -m metrics.ablation_consensus --dry-run                   # contar las llamadas primero
python3 -m metrics.ablation_consensus --sample 200
```

`--progress <archivo>` permite reanudar una corrida interrumpida en vez de
empezar de cero.

---

## 3. Rehacer la selección de subconjuntos

Solo hace falta si quiere reconstruir las muestras estratificadas desde las
fuentes crudas en vez de reutilizar las de `corpus/inputs/`.

```bash
python3 scripts/download_datasets.py --list       # ver de dónde viene cada fuente
python3 scripts/download_datasets.py --all        # ~2.2 GB de descarga, ~7 GB expandidos
jupyter lab                                       # después correr los notebooks 01, 02, 03
```

La selección usa `SEED = 42` en todo, de modo que las muestras son
deterministas dadas las mismas entradas. Dos advertencias sobre esas entradas:

- **FEVER.** El volcado de Wikipedia de FEVER viene en 109 fragmentos de unas
  50,000 páginas cada uno. La corrida del artículo usó un solo fragmento de
  50,000 páginas, guardado como `data/FEVER/fever.jsonl`. Partir de otro
  fragmento da un universo distinto y por tanto una muestra distinta, aun con la
  misma semilla.
- **Europarl.** La corrida usó 3,656 archivos de capítulo en inglés de las
  sesiones 2009–2011 (`ep-09-05-07-*.txt` hasta `ep-11-11-16-*.txt`), colocados
  en `data/Europarl/en/`. La versión v7 de Europarl cubre 1996–2011 y es mayor;
  restrínjala a ese rango para coincidir.

Para saltarse todo esto y usar exactamente los subconjuntos que consumió el
pipeline:

```bash
python3 scripts/unpack_corpus.py --inputs
```

---

## 4. Volver a correr el pipeline

### Antes de gastar nada

El corpus completo costó **\$454.30** y 615 horas de reloj. Conviene hacer tres
cosas primero:

1. **Consultar los precios actuales.** Las cifras del artículo son precios de
   lista de Azure de mediados de 2026. Las tarifas que usa el código están fijas
   en `MODEL_PRICING`, cerca del inicio de `agents/agente_wikitext.py`;
   actualícelas o su contabilidad de costos será incorrecta.
2. **Correr un piloto.** Todos los agentes aceptan `--limit`: empiece con diez
   entradas y lea el costo reportado antes de escalar.
3. **Saber a dónde se va el dinero.** La Etapa 4, la revisión, es el 77% del
   total. Si el presupuesto aprieta, esa es la única perilla que importa de
   verdad: un revisor más barato mueve el costo mucho más que cualquier otra
   cosa que pueda ajustar.

### Configurar

```bash
cp .env.example .env
```

Complete sus endpoints y llaves, y apunte cada rol a sus propios *deployments*.
Cada variable está explicada en [configuration.md](configuration.md). `.env` está
en `.gitignore` y debe seguir así: contiene llaves de API.

Si algún rol usa el proveedor `ollama`, levante el servidor local y descargue el
modelo:

```bash
docker compose up -d
docker exec -it ollama ollama pull qwen3.5:9b
curl http://localhost:11434/api/tags          # debe listar el modelo
```

Descomente el bloque `deploy:` de `docker-compose.yml` para darle al contenedor
su GPU NVIDIA. Sin él, Ollama corre en CPU: funciona, pero con la lentitud
suficiente para dominar el tiempo total.

Después confirme que cada modelo configurado realmente responde:

```bash
python3 scripts/verify_setup.py --models
```

### Ejecutar

Ejecute desde la raíz del repositorio: los agentes resuelven `data/` y `output/`
relativo al directorio de trabajo.

```bash
python3 scripts/unpack_corpus.py --inputs

python3 agents/agente_wikitext.py --input data/WikiText/wikitext-103-raw.wiki.train.selected_1000.raw --limit 10
python3 agents/agente_fever.py    --input data/FEVER/selected/fever_selected.jsonl --limit 10
python3 agents/agente_europarl.py --input data/Europarl/selected/ --limit 10
```

Opciones comunes:

| Opción | Significado |
|---|---|
| `--input`, `-i` | Archivo de entrada o, para Europarl, un directorio |
| `--limit`, `-l` | Detenerse tras N entradas (`0` = todas). **Úsela siempre la primera vez.** |
| `--start-from`, `-s` | Empezar en la entrada N |
| `--retries`, `-r` | Máximo de intentos por llamada a la API (por defecto 3) |
| `--id-offset` | Desplaza los IDs, para procesar varios archivos en un mismo corpus |

`agente_fever.py` añade `--min-words` (por defecto 10) y `agente_europarl.py`
añade `--min-chars`, un tamaño mínimo de archivo en bytes.

Cada corrida escribe tres cosas en `output/<dataset>/`: el JSON del corpus
bilingüe, el JSON de la traza del pipeline y un `.log` con marca de tiempo.

### Si una corrida se interrumpe

No importa. El pipeline escribe un archivo de progreso después de cada entrada
completada, con los IDs terminados y su salida acumulada. Relance el mismo
comando y se salta lo ya hecho y reanuda en la primera entrada incompleta. Al
terminar, el archivo de progreso se consume para generar los dos JSON de salida.
Errores de red, límites de tasa y reinicios son todos recuperables así; lo que
**no** se recupera es borrar el archivo de progreso, que descarta todo lo que aún
no se escribió a la salida final.

---

## Usar el pipeline con texto propio

Nada del diseño es específico del par inglés–español ni de estos tres datasets.

**Otro modelo, u otro proveedor.** Edite `.env`. Los roles se declaran como
`proveedor:modelo` y se leen al arrancar, así que no hay cambios de código. Los
proveedores soportados son `azure`, `foundry` y `ollama`.

**Otro par de idiomas.** Los prompts son constantes de texto agrupadas cerca del
inicio de `agents/agente_wikitext.py`: `TRANSLATION_SYSTEM`, `REVIEWER_SYSTEM`,
`EVALUATOR_SYSTEM`, `METADATA_SYSTEM_ES` / `_EN` y sus funciones
`build_*_prompt()`. Traduzca las instrucciones y cambie el idioma destino que
nombran. Tenga en cuenta que la Etapa 3 valida el idioma de cada respuesta y
reintenta dos veces si el modelo contesta en el equivocado; esa verificación
también asume español.

**Otro formato de origen.** Escriba un agente nuevo. `agents/agente_fever.py` es
el ejemplo más pequeño: importa el núcleo compartido de `agente_wikitext`, aporta
su propio cargador (`load_fever_entries`), sus propios prompts y su directorio de
salida — unas 400 líneas, la mayoría prompts. Lo que hay que proveer es un
cargador que produzca entradas y una regla de fragmentación para textos por
encima del umbral de ~6,000 tokens. WikiText divide en encabezados de sección,
Europarl en fronteras de orador, FEVER no divide.

**Ajustar los umbrales.** El tamaño de fragmento, la razón de truncamiento, el
peso de completitud y los conteos de reintentos están documentados con su
justificación en [configuration.md](configuration.md). El umbral de truncamiento
de 0.92 en particular codifica una suposición sobre *este* par de idiomas —el
español corre 4–9% más largo que su fuente en inglés— y habrá que revisarlo para
un par que se expanda o contraiga de otra manera.

---

## Cambios hechos para la publicación

El código aquí es el que produjo el corpus. Tres diferencias deliberadas:

1. **Los datos se distribuyen comprimidos.** `corpus/*/*.gz` en vez de JSON
   plano, para que cada archivo quede bajo el límite de 100 MB de GitHub.
   `scripts/unpack_corpus.py` restaura la estructura y los nombres originales,
   que es la razón por la que las rutas fijas de `metrics/semantic_metrics.py` y
   `metrics/ablation_consensus.py` siguen resolviendo.
2. **El notebook 06 escribe sus figuras en `figures/`.** Antes escribía
   directamente en el árbol LaTeX del artículo, que no forma parte de este
   repositorio.
3. **Los comentarios y el texto explicativo están en inglés.** Los comentarios,
   docstrings y mensajes de `metrics/`, `ablation/` y los notebooks 01–03 y 05
   se tradujeron del español, el idioma de trabajo del proyecto. Ningún cálculo
   cambió, y los notebooks no se volvieron a ejecutar. Tres tipos de texto siguen
   en español: los comentarios y mensajes de log de los agentes de `agents/`, que
   se dejaron exactamente como se ejecutaron; las etiquetas de los gráficos de los
   notebooks 01–03 y 05, que forman parte de sus figuras guardadas; y los nombres
   de columna y las etiquetas de tamaño de las tablas de selección de
   `corpus/selection/`, que son datos.

Lo que no se trasladó: el `.env` con credenciales vivas, las fuentes LaTeX del
artículo, las corridas piloto distintas de las de WikiText en `corpus/pilot/`, y
la salida intermedia de las corridas exploratorias.

---

## Problemas frecuentes

**`ModuleNotFoundError: No module named 'metrics'`** — ejecute desde la raíz del
repositorio. El paquete de métricas se importa como `python3 -m metrics.<módulo>`.

**`ModuleNotFoundError: No module named 'agente_wikitext'`** — los agentes de
FEVER y Europarl importan el núcleo compartido del mismo directorio. Invóquelos
por ruta (`python3 agents/agente_fever.py`), lo que pone `agents/` en `sys.path`,
o ejecútelos desde dentro de `agents/` con `data/` y `output/` alcanzables.

**`FileNotFoundError: No output JSON found in .../output/FEVER`** — el corpus no
se ha expandido. Ejecute `python3 scripts/unpack_corpus.py --final`.

**`[WARN] spaCy model 'es_core_news_sm' not installed`** — se omiten las métricas
sintácticas y de entidades y todo lo demás sigue corriendo. Instálelo con
`python3 -m spacy download es_core_news_sm`.

**Las métricas semánticas van insoportablemente lentas** — torch está en CPU. Vea
la nota sobre GPU en [Instalación](#instalación).

**Un modelo responde en el idioma equivocado** — la Etapa 3 lo detecta y
reintenta dos veces con una instrucción más fuerte antes de rendirse. Los fallos
persistentes suelen indicar un modelo local pequeño; pruebe uno mayor en esa
ranura.

**Una entrada falla con un error de API** — se registra y se omite, y la corrida
continúa. Así se perdieron tres entradas en la corrida publicada (2 de FEVER, 1
de Europarl). Relance con `--start-from` para reintentar un rango concreto.
