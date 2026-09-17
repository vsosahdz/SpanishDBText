# Configuración

*[Read in English](../configuration.md)*

Todo lo configurable en tiempo de ejecución vive en `.env`. Lo demás —umbrales,
conteos de reintentos, precios— es una constante cerca del inicio de
`agents/agente_wikitext.py`. Este documento cubre ambas cosas y explica por qué
cada valor es el que es.

Parta de la plantilla:

```bash
cp .env.example .env
```

`.env` está en `.gitignore`. Contiene llaves de API; que siga así.

---

## Declarar modelos

Cada rol de modelo se escribe como `proveedor:modelo`:

```bash
TRANSLATION_MODEL_1=azure:mi-deployment-gpt41mini
TRANSLATION_MODEL_2=foundry:mi-deployment-mistral-large
TRANSLATION_MODEL_3=ollama:qwen3.5:9b
REVIEWER_MODEL=azure:mi-deployment-gpt41
EVALUATOR_MODEL=azure:mi-deployment-gpt41mini
```

La parte `modelo` es como lo llame el proveedor: en Azure es el **nombre del
deployment**, no el del modelo. Note que `ollama:qwen3.5:9b` lleva dos signos de
dos puntos; solo el primero separa proveedor de modelo.

Los roles se resuelven al arrancar en `load_pipeline_config()`, así que cambiar
cualquiera no requiere tocar código. Después `verify_all_models()` envía una
llamada corta a cada uno antes de empezar el trabajo real;
`python3 scripts/verify_setup.py --models` ejecuta exactamente esa verificación
por separado.

### Proveedores

| Proveedor | Para | Credenciales que lee |
|---|---|---|
| `azure` | Deployments de Azure OpenAI | `AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_API_VERSION` |
| `foundry` | Modelos serverless de Azure AI Foundry | `FOUNDRY_<FAMILIA>_ENDPOINT`, `FOUNDRY_<FAMILIA>_KEY` |
| `ollama` | Un modelo alojado localmente | `OLLAMA_BASE_URL` |

Las credenciales de Foundry se buscan por familia, derivada del nombre del
deployment: un deployment que contenga `mistral` lee `FOUNDRY_MISTRAL_ENDPOINT` y
`FOUNDRY_MISTRAL_KEY`. `CLAUDE`, `DEEPSEEK` y `LLAMA` se reconocen igual. En la
corrida publicada solo se usó el par de Mistral.

### Los roles

| Variable | Etapa | Qué hace |
|---|---|---|
| `TRANSLATION_MODEL_1` … `_4` | 3 | Traducen cada entrada en paralelo. De uno a cuatro; deje una ranura vacía para usar menos. |
| `REVIEWER_MODEL` | 4 | Critica los candidatos y los fusiona en el consenso. |
| `EVALUATOR_MODEL` | 5 | Califica el consenso. También se usa para los metadatos de la Etapa 6. |
| `LLM_PROVIDER`, `OLLAMA_MODEL` | — | Respaldo de proveedor único, usado solo si no hay ningún `TRANSLATION_MODEL_*`. No lo toque. |

Dos cosas que conviene saber antes de cambiarlos:

**Elija traductores de familias distintas.** Tres modelos de una misma familia
cometen errores correlacionados, y un consenso sobre errores correlacionados no
aporta nada. La configuración publicada mezcla deliberadamente arquitecturas de
OpenAI, Mistral y Alibaba.

**El revisor es el presupuesto.** La Etapa 4 es el 77% del gasto total: usa el
modelo más capaz y procesa la mayor cantidad de texto (el original más todos los
candidatos). Si el costo es una restricción, esta es la única variable que
realmente lo mueve. Quitar un traductor ahorra mucho menos; la ranura local de
Ollama no cuesta nada.

---

## Parámetros del modelo

| Variable | Valor usado | Por qué |
|---|---|---|
| `TEMPERATURE` | `0.2` | Casi determinista, permitiendo ligera variación. `0` haría las corridas idénticas; valores mayores añaden aleatoriedad que una tarea de fidelidad no quiere. |
| `NUM_CTX` | `32768` | Ventana de contexto. El límite de los modelos usados. |
| `NUM_PREDICT` | `8192` | Tokens máximos de salida para traducción y revisión. Las llamadas de evaluación y metadatos usan 1,024 fijos e ignoran este valor. |

---

## Constantes del pipeline

No están en `.env`. Son constantes a nivel de módulo en
`agents/agente_wikitext.py`, heredadas por los otros dos agentes.

| Constante | Valor | Por qué |
|---|---|---|
| `MAX_CHARS_PER_CHUNK` | `12000` | Deliberadamente **la mitad** de la conversión aritmética del umbral de tokens (~4 car/token darían 24,000). El margen absorbe tanto la variabilidad de esa razón como la expansión que añade el español. |
| Umbral de fragmentación | ~6,000 tokens est. | Observado: las entradas por encima alcanzaban sistemáticamente el tope de 8,192 tokens de salida. |
| `TIMEOUT` | `600` segundos | Puesto para el modelo Qwen local, mucho más lento que los de nube en fragmentos largos. |
| `MAX_RETRIES` | `3` | Intentos por llamada a la API ante fallo transitorio. Se puede sobrescribir por corrida con `--retries`. |
| `TRANSLATION_LANGUAGE_RETRIES` | `2` | Reintentos cuando el modelo respondió en el idioma equivocado. La mayoría se corrige en el primero. |
| Umbral de razón de truncamiento | `0.92` | Por debajo, corre la verificación programática de completitud. |
| Margen de sobrescritura | `10` puntos | La verificación solo sobrescribe al modelo cuando la brecha es grande, evitando correcciones por diferencias pequeñas. |
| Peso de completitud | `40%` | Su participación al recalcular la puntuación global. El truncamiento es un fallo peor que una frase torpe. |
| `SEED` | `42` | Toda muestra estratificada, en los notebooks y en el muestreo de métricas. |

### El umbral de 0.92 es específico del par de idiomas

Las traducciones al español de texto en inglés corren 4–9% más largas que su
fuente (expansión media 1.04–1.09, desviación estándar 0.02–0.07). Por eso una
traducción más corta que el 92% de su original queda varias desviaciones estándar
por debajo del rango esperado, lo que apunta a contenido faltante y no a fraseo
compacto.

Para un par que se expanda o contraiga de otra manera, este número es incorrecto.
Mida la razón de expansión en un piloto antes de confiar en él. El mapeo completo
por tramos de razón de caracteres a puntuación de completitud está en
[pipeline.md](pipeline.md#la-verificación-programática-de-truncamiento).

---

## Precios

`MODEL_PRICING` en `agents/agente_wikitext.py` mapea cada modelo a sus tarifas de
entrada y salida por millón de tokens, y `calculate_cost()` las aplica a los
conteos que devuelve la API. La tabla contiene los precios de lista de Azure
vigentes durante la corrida:

| Clave | Entrada / Salida, USD por millón de tokens |
|---|---|
| `gpt-4.1-mini`, `gpt41mini` | 0.40 / 1.60 |
| `gpt-4.1`, `gpt41` | 2.00 / 8.00 |
| `gpt-4o-mini`, `gpt4omini` | 0.15 / 0.60 |
| `gpt-4o`, `gpt4o` | 2.50 / 10.00 |
| `mistral` | 0.50 / 1.50 |
| `deepseek` | 0.50 / 1.80 |
| `ollama` | 0 / 0 |

La búsqueda es por **subcadena del nombre del modelo**, clave más larga primero,
que es la razón de que cada familia aparezca dos veces: los nombres de deployment
de Azure suelen omitir guiones y puntos (`gpt41mini`). Dos consecuencias que
conviene conocer:

- Nombre sus deployments de forma que alguna clave coincida. **Un modelo cuyo
  nombre no coincida con ninguna clave se tarifa silenciosamente en cero**
  —`calculate_cost()` devuelve `0.0` en vez de lanzar un error—, así que un
  deployment no reconocido produce una corrida que parece gratuita.
- **Actualice las tarifas antes de presupuestar una corrida nueva.** Los costos
  se escriben en las trazas en tiempo de ejecución con lo que dijera la tabla
  entonces. Nunca se recalculan, así que tarifas obsoletas producen una
  contabilidad equivocada con toda confianza.

---

## Ejecutar Ollama

Solo hace falta si algún rol usa el proveedor `ollama`.

```bash
docker compose up -d
docker exec -it ollama ollama pull qwen3.5:9b
curl http://localhost:11434/api/tags
```

`docker-compose.yml` fija `OLLAMA_KEEP_ALIVE=24h` para que el modelo permanezca
residente entre entradas en vez de recargarse en cada llamada.

Para acceso a GPU, descomente el bloque `deploy:`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Sin él Ollama corre en CPU. Funciona, pero el modelo local ya es la parte más
lenta del pipeline y en CPU domina el tiempo total: las 615 horas reportadas en
el artículo son mayormente eso.

Si la RAM es escasa, una etiqueta menor de la misma familia sirve: la familia
`qwen3.5` va de 0.8B a 122B. La calidad en esa ranura importa menos de lo que
parece, ya que el revisor eligió al candidato de Qwen como base en solo el 8.2%
de las entradas.

---

## Los prompts

No son configuración en sentido estricto, pero sí lo siguiente que querrá cambiar
para otro idioma o dominio.

Cada prompt está publicado además como texto plano en [`prompts/`](../../prompts/),
un archivo por prompt, con [`prompts/README.md`](../../prompts/README.md) que
explica qué hace cada uno y por qué está redactado así. Léalos ahí; edítelos en el
código.

Son constantes de texto agrupadas cerca del inicio de
`agents/agente_wikitext.py`:

| Constante | Usada por |
|---|---|
| `TRANSLATION_SYSTEM` | Etapa 3 |
| `REVIEWER_SYSTEM` | Etapa 4 |
| `EVALUATOR_SYSTEM` | Etapa 5 |
| `METADATA_SYSTEM_ES` / `METADATA_SYSTEM_EN` | Etapa 6 |

Cada una se acompaña de una función `build_*_prompt()` que arma el mensaje de
usuario. Los agentes de FEVER y Europarl sobrescriben los prompts de traducción y
metadatos con variantes propias del dominio —`FEVER_TRANSLATION_SYSTEM`,
`EUROPARL_TRANSLATION_SYSTEM`— y heredan el resto.

Todos los prompts están escritos en español, porque el idioma destino es el
español. Se desarrollaron de forma iterativa, partiendo de instrucciones mínimas
y añadiendo restricciones conforme aparecían problemas en las primeras corridas;
las restricciones que sobrevivieron están listadas en
[pipeline.md](pipeline.md#etapa-3--generación-de-candidatos).

Cambiar el idioma destino implica editar estas cadenas **y** revisar la
heurística de detección de idioma de la Etapa 2, que busca palabras funcionales
del español y se reutiliza en la Etapa 3 para verificar que el modelo realmente
tradujo.
