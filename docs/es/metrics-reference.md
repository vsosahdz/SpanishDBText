# Documentación de las métricas del corpus

*[Read in English](../metrics-reference.md)*

Este documento describe cada métrica que calcula el pipeline de métricas,
organizada por categoría. De cada una explicamos qué mide, quién la introdujo,
dónde se ha usado, cómo se calcula en general, cómo la calculamos en nuestra
implementación y qué resultados cabe esperar.

Todas las métricas se calculan de forma independiente para los textos en inglés y
en español, lo que permite comparar directamente el lado fuente y el lado
traducido del corpus.

---

## 1. Métricas generales del corpus

Son estadísticas descriptivas básicas que caracterizan el tamaño, la forma y la
distribución del corpus. Constituyen el fundamento de cualquier análisis de
corpus y aparecen prácticamente en todo artículo de datasets de PLN.

### 1.1 Total de tokens

- **Qué mide**: el número total de ocurrencias de palabras en todos los documentos del corpus.
- **Origen**: lingüística de corpus básica. La distinción type/token la formalizó Charles Sanders Peirce a finales del siglo XIX y se adoptó ampliamente en lingüística y PLN.
- **Campos que la usan**: lingüística de corpus, PLN, traducción automática, modelado del lenguaje.
- **Fórmula general**: suma de los conteos de palabras de todos los documentos: $\sum_{d \in D} |words(d)|$.
- **Nuestra implementación**: tokenizamos cada texto con un separador de palabras basado en expresiones regulares (`\b\w+\b` sobre el texto en minúsculas) y sumamos los conteos. Implementado en `general_metrics.py`, función `compute_general_metrics()`.
- **Resultados esperados**: WikiText debería tener el mayor conteo de tokens por documento (artículos enciclopédicos largos). FEVER, el menor (pasajes factuales cortos). Los totales en español deberían ser entre 5% y 15% mayores que en inglés por la expansión morfológica.

### 1.2 Types únicos (tamaño del vocabulario)

- **Qué mide**: el número de formas de palabra distintas en el corpus.
- **Origen**: el mismo que los tokens: la distinción type/token de Peirce.
- **Campos que la usan**: lingüística de corpus, lexicografía, estudios de vocabulario.
- **Fórmula general**: $|V| = |\{w : w \in D\}|$, la cardinalidad del conjunto de palabras únicas.
- **Nuestra implementación**: `len(set(all_words))` tras pasar a minúsculas y tokenizar.
- **Resultados esperados**: el español suele tener un vocabulario mayor que el inglés para el mismo contenido, porque tiene más conjugaciones verbales, adjetivos con género y formas flexionadas. Por ejemplo, «was/were/been/being» en inglés se corresponde con muchas más formas en español (fue, era, sido, siendo, estaba, estuvo, etc.).

### 1.3 Total de caracteres

- **Qué mide**: el conteo total de caracteres de todos los documentos, espacios y puntuación incluidos.
- **Origen**: medición básica de texto.
- **Campos que la usan**: traducción automática (modelos a nivel de carácter), estimación de almacenamiento.
- **Fórmula general**: $\sum_{d \in D} len(d)$.
- **Nuestra implementación**: `sum(len(text) for text in texts)`.
- **Resultados esperados**: los conteos de caracteres en español suelen ser entre 10% y 20% mayores que en inglés para contenido equivalente. Esta razón es importante para detectar truncamiento en las traducciones.

### 1.4 Palabras por documento

- **Qué mide**: la distribución de longitudes de documento en palabras: media, mediana, desviación estándar, mínimo y máximo.
- **Origen**: estadística descriptiva básica aplicada al análisis de corpus.
- **Campos que la usan**: cualquier campo que trabaje con colecciones de documentos.
- **Fórmula general**: medidas estadísticas estándar sobre el vector de conteos de palabras por documento.
- **Nuestra implementación**: calculamos `mean`, `median`, `std`, `min` y `max` con NumPy sobre el arreglo de conteos por documento. La distribución completa también se guarda para graficar histogramas.
- **Resultados esperados**: WikiText: media ~3,300, varianza alta. FEVER: media ~190, varianza baja. Europarl: media ~3,300, varianza alta con distribución bimodal (actas procedimentales cortas + debates sustantivos largos).

### 1.5 Oraciones por documento

- **Qué mide**: cuántas oraciones contiene cada documento en promedio; indica complejidad estructural.
- **Origen**: análisis de texto, estudios de legibilidad.
- **Campos que la usan**: evaluación de legibilidad, resumen automático, análisis de estructura documental.
- **Fórmula general**: contar las fronteras de oración por documento mediante patrones de puntuación.
- **Nuestra implementación**: dividimos por puntuación de fin de oración (`[.!?]` seguido de espacio en blanco) con `sentence_split()` en `loader.py`. Es una heurística simple que funciona bien con texto bien formado, pero puede dividir de más en abreviaturas.
- **Resultados esperados**: proporcional a la longitud del documento. WikiText y Europarl deberían tener los conteos más altos. Las entradas de FEVER suelen tener de 1 a 5 oraciones.

### 1.6 Rango de longitud de documento (mín./máx.)

- **Qué mide**: los extremos de longitud de documento; valida que el filtrado de la Etapa 1 funcionó correctamente.
- **Origen**: validación de datos.
- **Campos que la usan**: aseguramiento de calidad en la construcción de datasets.
- **Nuestra implementación**: `min()` y `max()` de los conteos de palabras por documento.
- **Resultados esperados**: deberían coincidir con los umbrales de filtrado definidos en la Etapa 1. Por ejemplo, el mínimo de WikiText debería rondar las 1,000 palabras y el máximo las 10,000.

### 1.7 Desviación estándar de la longitud de documento

- **Qué mide**: cuánta variación hay en el tamaño de los documentos; indica homogeneidad.
- **Origen**: estadística básica.
- **Campos que la usan**: diseño de corpus, optimización de procesamiento por lotes.
- **Fórmula general**: $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2}$.
- **Nuestra implementación**: `np.std(word_counts)`.
- **Resultados esperados**: alta para Europarl (distribución bimodal), moderada para WikiText, baja para FEVER.

---

## 2. Métricas de frecuencia de palabras

Estas métricas analizan cómo se distribuyen las palabras a lo largo del corpus.
Ayudan a validar que el texto traducido se comporta como lenguaje natural y a
detectar anomalías.

### 2.1 Distribución de Zipf

- **Qué mide**: si las frecuencias de palabras siguen la ley de Zipf, la observación de que en lenguaje natural la frecuencia de una palabra es inversamente proporcional a su rango. La k-ésima palabra más común aparece aproximadamente 1/k veces lo que aparece la más común.
- **Origen**: George Kingsley Zipf (1935), *The Psycho-biology of Language*. Formalizada matemáticamente más tarde por Benoit Mandelbrot.
- **Campos que la usan**: lingüística de corpus, recuperación de información, PLN, ciencia de la complejidad, economía (distribución de la riqueza), estudios urbanos (tamaño de ciudades).
- **Fórmula general**: $f(r) \propto r^{-\alpha}$, donde $f(r)$ es la frecuencia de la palabra en el rango $r$ y $\alpha$ es el exponente de Zipf (aproximadamente 1.0 para lenguaje natural). En espacio log-log esto se convierte en una recta: $\log f(r) = -\alpha \log r + c$.
- **Nuestra implementación**: ordenamos todas las frecuencias de palabras de forma descendente, calculamos `log10(rango)` y `log10(frecuencia)` para las 500 palabras más frecuentes, y estimamos el exponente $\alpha$ por regresión lineal sobre los valores log-log con `np.polyfit()`. Los puntos crudos se guardan para graficar. Implementado en `general_metrics.py`, función `compute_frequency_metrics()`.
- **Resultados esperados**: el texto natural suele producir $\alpha \approx 0.9$–$1.1$. Si el texto traducido muestra un exponente de Zipf similar al del original, confirma que las traducciones generadas por el LLM mantienen la distribución del lenguaje natural. Un exponente notablemente distinto sugeriría que el proceso de traducción introdujo artefactos distribucionales. Nuestros resultados muestran exponentes entre 0.89 y 0.99 para los tres corpus en ambos idiomas, lo que confirma un comportamiento natural.

### 2.2 Hapax legomena

- **Qué mide**: palabras que aparecen exactamente una vez en todo el corpus. Un conteo alto de hapax indica un vocabulario rico y diverso con muchos términos raros.
- **Origen**: el término viene del griego y significa «dicho solo una vez». Se ha usado durante siglos en filología clásica para estudiar palabras raras en textos antiguos. Lo formalizó en lingüística cuantitativa Gustav Herdan (1960).
- **Campos que la usan**: lingüística de corpus, atribución de autoría, lexicografía, filología.
- **Fórmula general**: $H = |\{w : freq(w) = 1\}|$. Suele reportarse como porcentaje del total de types: $H\% = H / |V| \times 100$.
- **Nuestra implementación**: `sum(1 for w, c in Counter(words).items() if c == 1)`. Reportamos tanto el conteo absoluto como el porcentaje sobre el total de types.
- **Resultados esperados**: típicamente entre 40% y 60% de todos los types son hapax legomena en un corpus de lenguaje natural. Un porcentaje mucho menor podría indicar lenguaje repetitivo o formulaico. Uno mayor indica contenido muy diverso. Los porcentajes de hapax en español deberían ser comparables a los del inglés para contenido equivalente.

### 2.3 Dis legomena

- **Qué mide**: palabras que aparecen exactamente dos veces en el corpus. Junto con los hapax legomena, los dis legomena caracterizan la «cola» de la distribución de frecuencias.
- **Origen**: también de la filología griega. Se usa junto con los hapax en la estimación de frecuencias de Good-Turing.
- **Campos que la usan**: PLN estadístico (técnicas de suavizado), lingüística de corpus.
- **Fórmula general**: $D = |\{w : freq(w) = 2\}|$.
- **Nuestra implementación**: `sum(1 for w, c in Counter(words).items() if c == 2)`.
- **Resultados esperados**: normalmente entre 10% y 20% de los types. La razón entre hapax y dis legomena se usa en algunos modelos de crecimiento del vocabulario.

### 2.4 Cobertura del vocabulario

- **Qué mide**: qué porcentaje de todos los tokens del corpus está cubierto por las N palabras más frecuentes. Muestra qué tan concentrado o distribuido está el uso del vocabulario.
- **Origen**: lingüística de corpus, recuperación de información. Relacionada con el principio de Pareto (regla 80/20).
- **Campos que la usan**: enseñanza de idiomas (vocabulario nuclear), compresión de texto, motores de búsqueda.
- **Fórmula general**: $C(N) = \frac{\sum_{i=1}^{N} freq(w_i)}{\sum_{i=1}^{|V|} freq(w_i)} \times 100$, con las palabras ordenadas por frecuencia.
- **Nuestra implementación**: calculamos la cobertura en hitos: las 100, 500, 1,000, 5,000 y 10,000 palabras más frecuentes. Implementado acumulando frecuencias desde `Counter.most_common()`.
- **Resultados esperados**: en inglés, las 100 palabras más frecuentes suelen cubrir ~50% de todos los tokens (dominadas por palabras funcionales). Las 1,000 más frecuentes cubren ~75–80%. Una cobertura similar en inglés y español indica que la traducción mantiene una distribución de frecuencias comparable.

### 2.5 Curva de crecimiento del vocabulario

- **Qué mide**: cómo crece el número de types únicos conforme se observan más tokens. También se conoce como curva de acumulación type-token.
- **Origen**: Herdan (1960), *Type-Token Mathematics*. Relacionada con la ley de Heaps: $|V| \propto N^{\beta}$, donde $N$ es el número de tokens y $\beta \approx 0.4$–$0.6$.
- **Campos que la usan**: lingüística de corpus, lexicografía computacional, estudios de diversidad de especies (ecología).
- **Fórmula general**: graficar $|V(n)|$ contra $n$, donde $V(n)$ es el conjunto de types únicos vistos en los primeros $n$ tokens.
- **Nuestra implementación**: recorremos todos los tokens manteniendo un conjunto de types vistos. Muestreamos en ~200 intervalos equiespaciados para mantener manejable el tamaño de la salida. Se guarda como una lista de puntos `{tokens, types}` para graficar.
- **Resultados esperados**: la curva debería subir con fuerza al principio y luego aplanarse conforme el vocabulario se satura. Un corpus que nunca se aplana tiene diversidad muy alta (o ruido). Las curvas del español suelen subir algo más rápido que las del inglés por la riqueza morfológica, pero se aplanan de forma parecida.

---

## 3. Métricas de diversidad léxica

Estas métricas cuantifican qué tan variado es el uso del vocabulario en el texto.
Son importantes para evaluar la calidad de la traducción, porque una buena
traducción debería mantener una diversidad léxica similar a la del original: ni
simplificarla en exceso ni inflar el vocabulario artificialmente.

### 3.1 Type-Token Ratio (TTR)

- **Qué mide**: la razón entre palabras únicas (types) y palabras totales (tokens). Una medida simple de diversidad de vocabulario.
- **Origen**: Johnson (1944), una de las primeras medidas cuantitativas de complejidad lingüística. Ampliamente usada desde los años cuarenta.
- **Campos que la usan**: investigación en adquisición del lenguaje, lingüística clínica (evaluación de afasia), legibilidad, estudios de ESL/EFL.
- **Fórmula general**: $TTR = \frac{|V|}{N}$, donde $|V|$ es el número de types y $N$ el de tokens.
- **Nuestra implementación**: calculada tanto a nivel de corpus como por documento. `len(set(words)) / len(words)`.
- **Resultados esperados**: el TTR depende fuertemente de la longitud del texto: los textos largos siempre tienen TTR menor porque las palabras comunes se reutilizan. El TTR a nivel de corpus será muy bajo (0.01–0.05 en corpus grandes). El TTR por documento es más informativo (típicamente 0.3–0.7 según la longitud). Esa sensibilidad a la longitud es la razón por la que el TTR se complementa con las métricas corregidas de abajo.

### 3.2 Root Type-Token Ratio (RTTR)

- **Qué mide**: el TTR normalizado por la raíz cuadrada del número de tokens, lo que corrige parcialmente la dependencia de la longitud del texto.
- **Origen**: también conocido como índice de Guiraud. Propuesto por Pierre Guiraud (1954), *Les Caracteres Statistiques du Vocabulaire*.
- **Campos que la usan**: lingüística de corpus, estilometría.
- **Fórmula general**: $RTTR = \frac{|V|}{\sqrt{N}}$.
- **Nuestra implementación**: `types / math.sqrt(tokens)`.
- **Resultados esperados**: más estable que el TTR entre textos de distinta longitud, aunque tampoco del todo independiente de ella. Los valores suelen ir de 5 a 15 en análisis a nivel de documento. Nota: el índice de Guiraud es matemáticamente idéntico al RTTR.

### 3.3 Corrected Type-Token Ratio (CTTR)

- **Qué mide**: otra variante del TTR corregida por longitud, usando $\sqrt{2N}$ como denominador.
- **Origen**: Carroll (1964), propuesta como mejora del TTR simple.
- **Campos que la usan**: lingüística de corpus, investigación en adquisición de segundas lenguas.
- **Fórmula general**: $CTTR = \frac{|V|}{\sqrt{2N}}$.
- **Nuestra implementación**: `types / math.sqrt(2 * tokens)`.
- **Resultados esperados**: comportamiento similar al RTTR pero escalado por un factor de $\sqrt{2}$. Los valores son aproximadamente un 29% menores que los del RTTR.

### 3.4 Moving Average Type-Token Ratio (MATTR)

- **Qué mide**: el TTR promedio calculado sobre una ventana deslizante de tamaño fijo. Esto lo hace independiente de la longitud del texto, ya que el tamaño de ventana es constante.
- **Origen**: Covington y McFall (2010), *Cutting the Gordian Knot: The Moving-Average Type-Token Ratio (MATTR)*. Publicado en el Journal of Quantitative Linguistics.
- **Campos que la usan**: lingüística clínica, evaluación del lenguaje, comparación de corpus, evaluación de calidad de traducción.
- **Fórmula general**: para una ventana de tamaño $W$: $MATTR = \frac{1}{N-W+1} \sum_{i=1}^{N-W+1} TTR(w_i, ..., w_{i+W-1})$.
- **Nuestra implementación**: tamaño de ventana $W = 100$ palabras. En cada posición calculamos el TTR de la ventana de 100 palabras y promediamos todos los TTR de ventana. En documentos de menos de 100 palabras usamos el documento completo. Implementado en `lexical_metrics.py`, función `_mattr()`.
- **Resultados esperados**: los valores suelen ir de 0.6 a 0.8. Valores mayores indican un uso de vocabulario más diverso. Inglés y español deberían mostrar valores de MATTR similares si la traducción mantiene una diversidad equivalente. Nuestros resultados muestran un MATTR de alrededor de 0.69–0.72 en todos los corpus, lo que indica texto natural y diverso.
- **Por qué la elegimos**: el MATTR es una de las métricas de diversidad léxica más recomendadas en la literatura reciente, porque es verdaderamente independiente de la longitud del texto y fácil de interpretar.

### 3.5 Measure of Textual Lexical Diversity (MTLD)

- **Qué mide**: el número promedio de palabras consecutivas necesarias antes de que el TTR caiga por debajo de un umbral. Secuencias más largas antes de la caída indican mayor diversidad léxica.
- **Origen**: McCarthy y Jarvis (2010), *MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment*. Publicado en Behavior Research Methods.
- **Campos que la usan**: evaluación del lenguaje, adquisición de segundas lenguas, evaluación de calidad de escritura, lingüística clínica.
- **Fórmula general**: desde el inicio del texto, se calcula el TTR de forma incremental palabra por palabra. Cuando el TTR cae por debajo de un umbral (típicamente 0.72), se cuenta un «factor» y se reinicia. MTLD = tokens totales / número de factores. El cálculo se hace en dirección directa e inversa, y se promedian los resultados.
- **Nuestra implementación**: umbral = 0.72. Calculamos el MTLD hacia adelante y hacia atrás, y promediamos. Los factores parciales al final se cuentan proporcionalmente: $(1 - TTR_{restante}) / (1 - umbral)$. Implementado en `lexical_metrics.py`, función `_mtld()`.
- **Resultados esperados**: valores mayores indican vocabulario más diverso. Los valores típicos van de 50 a 150 en texto natural. Nuestros resultados muestran un MTLD de 63 a 91, dentro del rango esperado. El MTLD del español a veces es menor que el del inglés porque el español tiene más palabras funcionales y terminaciones morfológicas repetidas.
- **Por qué la elegimos**: el MTLD se considera el estándar de referencia para diversidad léxica en lingüística computacional. Es robusto frente a la longitud del texto, está ampliamente validado contra juicios humanos y es muy citado.

---

## 4. Métricas sintácticas

Estas métricas analizan la estructura gramatical del texto. Ayudan a verificar
que las traducciones mantienen la complejidad estructural del original en lugar
de simplificar los patrones oracionales.

### 4.1 Longitud de oración (en palabras)

- **Qué mide**: el número promedio de palabras por oración, más la distribución sobre todas las oraciones.
- **Origen**: investigación básica en legibilidad. Usada en fórmulas de legibilidad desde Flesch (1948).
- **Campos que la usan**: evaluación de legibilidad, estudios de traducción, lingüística computacional.
- **Fórmula general**: $\bar{L} = \frac{\text{palabras totales}}{\text{oraciones totales}}$.
- **Nuestra implementación**: dividimos el texto en oraciones con `sentence_split()` (expresión regular sobre `.!?`), tokenizamos cada oración y calculamos estadísticas (media, mediana, desviación estándar, mínimo, máximo) sobre todas las longitudes de oración. También calculamos la longitud media de oración por documento para el análisis de distribución. Implementado en `syntactic_metrics.py`.
- **Resultados esperados**: el inglés suele tener oraciones más cortas que el español para contenido equivalente. El texto académico o enciclopédico (WikiText) debería tener una longitud de oración moderada (15–25 palabras). El texto parlamentario (Europarl) tiende a oraciones más largas y complejas (20–30+ palabras). Los pasajes de FEVER deberían tener longitud moderada.

### 4.2 Distribución POS (categorías gramaticales)

- **Qué mide**: la proporción de cada categoría gramatical (sustantivos, verbos, adjetivos, adverbios, etc.) en el corpus.
- **Origen**: el etiquetado de categorías gramaticales ha sido una tarea central del PLN desde los años sesenta. El etiquetado POS automatizado fue pionero en el proyecto Brown Corpus (Francis y Kucera, 1967).
- **Campos que la usan**: lingüística computacional, estilometría, tipología lingüística, evaluación de traducción automática.
- **Fórmula general**: para cada etiqueta POS $p$: $\text{proporción}(p) = \frac{count(p)}{\sum_q count(q)}$.
- **Nuestra implementación**: requiere spaCy con los modelos de idioma (`en_core_web_sm` y `es_core_news_sm`). Procesamos los primeros 2,000 caracteres de cada documento por eficiencia y luego contamos las etiquetas POS (excluyendo espacios y puntuación). Los resultados se reportan como conteos y como porcentajes. Implementado en `syntactic_metrics.py`.
- **Resultados esperados**: el inglés y el español tienen distribuciones POS distintas por naturaleza: el español usa más determinantes y pronombres (por los artículos con género), más formas verbales (por la conjugación) y un orden adjetivo-sustantivo diferente. Una buena traducción debería mostrar las diferencias POS translingües esperadas, en lugar de reflejar exactamente la distribución del inglés.
- **Nota**: esta métrica requiere que spaCy esté instalado. Si no lo está, la métrica se omite y se registra una nota en la salida.

---

## 5. Métricas de calidad y semánticas

Estas métricas evalúan la calidad, la consistencia y las características
semánticas del corpus. Son especialmente importantes para validar que el pipeline
de traducción produjo una salida fiable.

### 5.1 Razón de expansión (EN a ES)

- **Qué mide**: cuánto más larga es la traducción al español respecto al original en inglés, medido en caracteres.
- **Origen**: bien conocida en la industria de la traducción y la localización. Documentada en guías de localización (por ejemplo, los estándares de localización de Microsoft e IBM).
- **Campos que la usan**: evaluación de traducción automática, localización de software, composición tipográfica.
- **Fórmula general**: $R = \frac{len(\text{traducción})}{len(\text{original})}$ por documento.
- **Nuestra implementación**: se calcula por documento y luego se agrega (media, mediana, desviación estándar, mínimo, máximo). La distribución completa se guarda para graficar. Implementado en `quality_metrics.py`.
- **Resultados esperados**: del inglés al español suele expandirse entre 10% y 20% (razón 1.10–1.20). Una razón notablemente inferior a 1.0 sugiere truncamiento o contenido faltante. Una razón superior a 1.3 podría indicar que el traductor añadió texto ajeno. Nuestros resultados muestran razones de 1.04 a 1.09 entre corpus, lo cual es razonable. Europarl tiene una razón algo menor porque el texto parlamentario contiene fórmulas fijas y nombres propios que no se expanden.

### 5.2 Duplicados exactos

- **Qué mide**: documentos con contenido idéntico (tras pasar a minúsculas y quitar espacios).
- **Origen**: verificación estándar de calidad de datos en la construcción de datasets.
- **Campos que la usan**: aseguramiento de calidad de datasets de PLN, rastreo web, deduplicación en motores de búsqueda.
- **Fórmula general**: calcular un hash de cada documento y contar colisiones.
- **Nuestra implementación**: calculamos hashes MD5 de `text.strip().lower()` para todos los documentos y contamos hashes duplicados. Se calcula por separado para el lado inglés y el español. Implementado en `quality_metrics.py`.
- **Resultados esperados**: idealmente 0 duplicados. Podrían existir algunos en el lado inglés si el corpus original tenía artículos casi idénticos que pasaron el filtro de selección. El lado español debería tener 0 duplicados porque cada artículo se traduce de forma independiente.

### 5.3 Casi duplicados

- **Qué mide**: documentos muy similares pero no idénticos; detecta paráfrasis, ediciones menores o artículos sobre temas muy parecidos.
- **Origen**: Broder (1997), *On the Resemblance and Containment of Documents*, introdujo MinHash y la similitud de Jaccard para deduplicación a escala web.
- **Campos que la usan**: búsqueda web, detección de plagio, deduplicación de datasets (por ejemplo, la limpieza de datos de entrenamiento de GPT).
- **Fórmula general**: crear k-shingles (secuencias de k palabras) de cada documento. Calcular la similitud de Jaccard: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$. Los documentos con $J \geq 0.85$ se consideran casi duplicados.
- **Nuestra implementación**: usamos shingles de 5 palabras y calculamos la similitud de Jaccard entre todos los pares de una muestra aleatoria de 500 documentos (la comparación completa por pares es O(N^2) y demasiado lenta para 10,000 documentos). La muestra usa una semilla fija (42) para reproducibilidad. Implementado en `quality_metrics.py`.
- **Resultados esperados**: debería ser 0 o muy pocos pares. Si se encuentran casi duplicados, merecen inspección manual.

### 5.4 Porcentaje de stopwords

- **Qué mide**: la proporción de tokens que son palabras funcionales (artículos, preposiciones, conjunciones, pronombres, etc.).
- **Origen**: Luhn (1958) introdujo el concepto de stopwords para recuperación de información. El término lo acuñó Hans Peter Luhn en IBM.
- **Campos que la usan**: recuperación de información, clasificación de texto, evaluación de calidad de corpus.
- **Fórmula general**: $S\% = \frac{|\{w \in texto : w \in stopwords\}|}{|texto|} \times 100$.
- **Nuestra implementación**: usamos una lista curada de ~60 palabras funcionales de alta frecuencia por idioma. Las listas están escritas en el código en lugar de usar una biblioteca externa, para evitar dependencias. Implementado en `quality_metrics.py`.
- **Resultados esperados**: el inglés suele tener entre 30% y 40% de stopwords; el español, entre 35% y 45% (el español usa más artículos y preposiciones por la concordancia de género y número). Un corpus traducido que muestre niveles naturales de stopwords confirma que la traducción es fluida y no excesivamente literal.

### 5.5 Porcentaje de caracteres especiales

- **Qué mide**: la proporción de caracteres que no son alfanuméricos ni espacios en blanco (puntuación, símbolos, corchetes, etc.).
- **Origen**: evaluación de calidad de datos.
- **Campos que la usan**: preprocesamiento de texto, evaluación de calidad de OCR, limpieza de corpus web.
- **Fórmula general**: $\frac{|\{c : c \notin alnum \wedge c \notin space\}|}{len(texto)} \times 100$.
- **Nuestra implementación**: clasificación carácter a carácter, promediada entre documentos. Se reporta por separado para EN y ES.
- **Resultados esperados**: típicamente entre 2% y 5% en texto limpio. Europarl puede ser mayor por las etiquetas tipo XML. Valores por encima del 10% podrían indicar artefactos de formato o tokens especiales sin limpiar.

### 5.6 Porcentaje de mayúsculas

- **Qué mide**: la proporción de caracteres en mayúscula; detecta anomalías como texto EN MAYÚSCULAS o títulos que no se limpiaron correctamente.
- **Origen**: evaluación de calidad de datos.
- **Campos que la usan**: normalización de texto, detección de spam, corrección de OCR.
- **Fórmula general**: $\frac{|\{c : c.isupper()\}|}{len(texto)} \times 100$.
- **Nuestra implementación**: clasificación carácter a carácter, promediada entre documentos.
- **Resultados esperados**: típicamente entre 2% y 5% en texto corrido normal. Valores mayores indican títulos, encabezados o siglas. El español debería tener un porcentaje de mayúsculas algo menor que el inglés, porque el español capitaliza menos palabras (por ejemplo, los nombres de idiomas y los días de la semana van en minúscula).

### 5.7 Reconocimiento de entidades nombradas (NER)

- **Qué mide**: el número y la diversidad de entidades nombradas (personas, organizaciones, lugares, fechas, etc.) en el corpus.
- **Origen**: las conferencias MUC (1987–1998) establecieron el NER como tarea estándar de PLN. El NER moderno se basa en etiquetado neuronal de secuencias.
- **Campos que la usan**: extracción de información, respuesta a preguntas, construcción de grafos de conocimiento, evaluación de traducción.
- **Fórmula general**: aplicar un modelo de NER, contar entidades por tipo y calcular la diversidad como $\frac{entidades\_únicas}{entidades\_totales}$.
- **Nuestra implementación**: requiere spaCy con los modelos de idioma. Procesamos los primeros 2,000 caracteres de cada documento, extraemos las entidades, las contamos por tipo (PERSON, ORG, GPE, DATE, etc.) y reportamos las 30 entidades más frecuentes. La diversidad de entidades se calcula como la razón entre entidades únicas y menciones totales. Implementado en `quality_metrics.py`.
- **Resultados esperados**: inglés y español deberían tener conteos de entidades similares si la traducción preserva los nombres propios correctamente. La diversidad de entidades debería ser comparable. Las diferencias en los conteos pueden revelar casos en los que el traductor tradujo indebidamente u omitió entidades nombradas.
- **Nota**: esta métrica requiere que spaCy esté instalado.

### 5.8 Distribución de clasificaciones

- **Qué mide**: la distribución de categorías temáticas en el corpus, según las asigna la extracción de metadatos de la Etapa 6 del pipeline.
- **Origen**: métrica interna; usa las clasificaciones ya calculadas durante la traducción.
- **Campos que la usan**: evaluación del balance del dataset, análisis temático.
- **Nuestra implementación**: contamos las ocurrencias de cada valor de `classification` en las entradas del dataset.
- **Resultados esperados**: debería reflejar el muestreo estratificado de la Etapa 1. Por ejemplo, FEVER debería mostrar ~66% de contenido deportivo. WikiText debería mostrar una distribución más equilibrada entre 14 categorías.

---

## Métricas no implementadas (y por qué)

Las siguientes métricas de la lista original se consideraron pero no se
incluyeron en la implementación actual:

| Métrica | Motivo de exclusión |
|--------|---------------------|
| **HD-D** (Hypergeometric Distribution D) | Redundante con MTLD y MATTR: aporta información similar con mayor complejidad computacional. |
| **C de Herdan** | Matemáticamente equivalente a $\log(types) / \log(tokens)$: no aporta nada sobre el RTTR. |
| **Índice de Guiraud** | Idéntico al RTTR ($types / \sqrt{tokens}$): mantener ambos sería redundante. |
| **Profundidad del árbol sintáctico** | Requiere análisis de dependencias completo de cada oración: demasiado costoso computacionalmente para más de 12,000 documentos y sin valor añadido significativo sobre la longitud de oración. |
| **Complejidad sintáctica** | Requiere análisis de oraciones subordinadas mediante parsing profundo: el mismo problema de costo que la profundidad de árbol. |
| **Modelado de tópicos (LDA/BERTopic)** | Ya tenemos las clasificaciones temáticas de los metadatos de la Etapa 6. Ejecutar un modelo de tópicos aparte sería redundante. |
| **Coherencia de tópicos** | Depende del modelado de tópicos, que no ejecutamos. |
| **Similitud semántica entre documentos** | Requiere calcular embeddings de todos los documentos y la similitud coseno por pares: cómputo O(N^2), caro para corpus grandes. |
| **Densidad semántica** | Similar a la similitud semántica: requiere infraestructura de embeddings. |
| **OOV (fuera de vocabulario)** | Depende fuertemente del vocabulario de referencia elegido. El texto generado por LLM tiene muy pocas palabras realmente fuera de vocabulario, lo que vuelve la métrica poco informativa. |
| **Errores ortográficos** | Los LLM producen muy pocos errores ortográficos. Los correctores generan muchos falsos positivos con nombres propios, términos técnicos y contenido multilingüe, lo que hace los resultados ruidosos y poco fiables. |

---

## Métricas sugeridas para trabajo futuro

| Métrica | Descripción | Qué hace falta | Esfuerzo | Valor |
|--------|-------------|----------------|--------|-------|
| **BLEU** | Métrica estándar de TA que compara contra traducciones humanas de referencia. Biblioteca: `sacrebleu`. | Un subconjunto (50–100 artículos) con traducciones humanas profesionales como referencia. La métrica en sí es trivial de calcular: el costo está por completo en producir la referencia. | Alto (hace falta un traductor humano) | Alto: es el estándar que los revisores de artículos esperan ver. |
| **chrF / chrF++** | F-score de n-gramas de caracteres para evaluación de traducción. Incluido en `sacrebleu`. | La misma referencia humana que BLEU: una vez que la tienes, chrF se calcula junto a BLEU sin costo extra. | El mismo que BLEU | Alto: más robusto que BLEU para lenguas morfológicamente ricas como el español, donde las formas de palabra varían más. |
| **COMET** | Métrica neuronal de evaluación de TA de Unbabel. Biblioteca: `unbabel-comet` (~2 GB de descarga de modelo). | La misma referencia humana más el modelo COMET. No requiere GPU, pero es más lento que BLEU/chrF. | Medio (instalar el modelo + tener la referencia) | Muy alto: estado del arte, correlaciona mejor con el juicio humano que BLEU o chrF. |
| **Acuerdo entre modelos** | Similitud entre los 3 candidatos de traducción antes del consenso. | Ya disponible: los JSON del pipeline contienen los 3 candidatos por artículo. Solo hace falta código nuevo para calcular la similitud por pares (por ejemplo, BLEU o similitud coseno entre candidatos). | Bajo: solo código, sin datos nuevos | Medio: interesante y original. Mostraría si los 3 modelos producen candidatos genuinamente diversos (lo que justifica la etapa de consenso) o salidas muy similares (lo que sugeriría que el consenso aporta poco). Pocos artículos reportan esta métrica. |

### Recomendaciones prácticas

1. **El acuerdo entre modelos** es lo más fácil de implementar ahora mismo, ya que todos los datos existen en los JSON del pipeline. Sería una contribución original que valida la arquitectura multi-modelo.

2. **BLEU/chrF sobre un micro-subconjunto**: traducir a mano de 30 a 50 artículos cortos de FEVER sería la vía más rápida para tener una referencia humana. Los artículos de FEVER son cortos (mediana de 152 palabras), así que un traductor profesional podría hacer 50 artículos en unas horas. Eso daría puntuaciones BLEU, chrF y COMET para al menos un corpus.

3. **COMET** es lo que más valor añade por unidad de esfuerzo si ya se tiene la referencia humana, ya que correlaciona ~0.95 con el juicio humano (frente a ~0.85 de BLEU).

## Reevaluación de las métricas no implementadas

Tras revisar las métricas excluidas de la implementación actual, esta es una
valoración actualizada de si alguna añadiría valor significativo en caso de
reconsiderarse:

| Métrica | ¿Vale la pena implementarla? | Razonamiento |
|--------|-------------------|-----------|
| **Acuerdo entre modelos** | **Sí** | Puede calcularse ahora mismo con los datos existentes de los JSON del pipeline. Validaría que usar 3 arquitecturas de modelo distintas produce candidatos genuinamente diversos, que es una suposición central del pipeline. |
| **Similitud semántica entre documentos** | **Quizá** | Podría detectar si el corpus es repetitivo o tiene grupos de contenido muy similar. Sin embargo, requiere un modelo de embeddings y cómputo por pares O(N^2) para más de 12,000 documentos, lo que es caro. Una versión muestreada (por ejemplo, 500 pares aleatorios) podría ser práctica. |
| **OOV (fuera de vocabulario)** | **No** | Los LLM casi nunca generan palabras fuera de vocabulario. La métrica devolvería ~0% con muchos falsos positivos en nombres propios, términos técnicos y texto con cambio de código. No es informativa para traducciones generadas por LLM. |
| **Errores ortográficos** | **No** | El mismo problema que OOV. Los LLM producen muy pocos errores ortográficos, y los correctores generan falsos positivos en exceso con contenido multilingüe, nombres propios y terminología de dominio. La razón ruido-señal hace la métrica poco fiable. |
| **Coherencia de tópicos** | **No** | Ya tenemos las clasificaciones temáticas de la extracción de metadatos de la Etapa 6. Ejecutar un modelo de tópicos aparte (LDA o BERTopic) sobre datos ya clasificados sería redundante. La métrica de distribución de clasificaciones ya captura el balance temático. |
| **Profundidad del árbol sintáctico** | **No** | Requiere análisis de dependencias completo de cada oración en más de 12,000 documentos: computacionalmente muy caro. La métrica de longitud de oración ya captura la señal principal de complejidad sintáctica. El conocimiento adicional que aporta la profundidad de árbol no justifica el costo. |

---

## Formato de salida

Cada corrida de métricas produce un archivo JSON con la siguiente estructura de
nivel superior:

```json
{
  "metadata": {
    "dataset": "fever",
    "source_file": "fever_selected_260712_212751.json",
    "total_entries": 9998,
    "generated_at": "2026-07-23T14:56:42",
    "elapsed_seconds": 29.5
  },
  "general_metrics": {
    "english": { ... },
    "spanish": { ... }
  },
  "frequency_metrics": {
    "english": { ... },
    "spanish": { ... }
  },
  "lexical_diversity": {
    "english": { ... },
    "spanish": { ... }
  },
  "syntactic_metrics": {
    "english": { ... },
    "spanish": { ... }
  },
  "quality_metrics": { ... }
}
```

Los campos que terminan en `distribution_*` contienen arreglos de datos crudos
aptos para graficar histogramas, diagramas de dispersión o gráficas de líneas.
Los campos que terminan en `_curve` contienen puntos `{x, y}` para gráficas de
líneas.

---

## Ámbito de aplicación: métricas por dataset frente a combinadas

No todas las métricas son igual de útiles en ambos niveles. Algunas están
diseñadas para caracterizar un solo corpus y pierden sentido cuando se mezclan
corpus de dominios distintos. Otras solo resultan interesantes al aplicarse sobre
el conjunto combinado completo. Esta sección explica qué métricas se calculan en
qué nivel y por qué.

### Métricas calculadas solo por dataset

Estas métricas se calculan de forma independiente para cada uno de los tres
datasets (WikiText, FEVER, Europarl). Caracterizan las propiedades de cada corpus
por separado y se comparan lado a lado para entender en qué se diferencian.

| Métrica | Por qué solo por dataset |
|--------|---------------------|
| **Exponente de Zipf** | Mezclar corpus de dominios distintos (enciclopédico, factual, parlamentario) siempre produce un exponente de Zipf cercano a 1.0, lo cual es trivialmente esperable. Los exponentes por dataset son más informativos porque revelan patrones propios del dominio: Europarl es muy zipfiano (0.985) mientras que WikiText muestra más diversidad de vocabulario (0.897). |
| **MATTR / MTLD** | Son métricas por documento que luego se promedian. Combinar 10,000 artículos cortos de FEVER con 1,000 artículos largos de WikiText produciría una media ponderada dominada por el conteo de FEVER, que ocultaría las diferencias significativas entre datasets. |
| **TTR / RTTR / CTTR** | El TTR a nivel de corpus depende fuertemente del tamaño del corpus. Combinar corpus de tamaños distintos produciría un TTR que refleja más el tamaño total que la diversidad del contenido. |
| **Longitud de oración** | Mezclar oraciones parlamentarias (media de 25 palabras) con pasajes factuales (media de 21) produce un promedio que no describe bien ninguno de los dos dominios. |
| **Distribución POS** | Cada dominio tiene patrones POS característicos (Europarl tiene más pronombres y auxiliares, FEVER más nombres propios). Mezclarlos diluye las señales propias de dominio que hacen interesante el análisis POS. |
| **Hapax / dis legomena** | En el corpus combinado, muchas palabras que son hapax en un dataset aparecerán en otro, lo que baja artificialmente el conteo de hapax. Los porcentajes por dataset son más significativos para evaluar la riqueza del vocabulario. |
| **Razón de expansión** | La razón de expansión depende del tipo de contenido. Los artículos con muchos nombres propios (contenido deportivo de FEVER) se expanden menos que los discursos parlamentarios. Las razones por dataset revelan esas diferencias; una media combinada las ocultaría. |
| **Stopwords / caracteres especiales / mayúsculas** | Estos porcentajes varían según el dominio (Europarl tiene 43% de stopwords frente al 33% de FEVER). Un valor combinado sería una media ponderada que no describe ningún dominio concreto. |

### Métricas calculadas sobre el corpus combinado

Estas métricas se calculan sobre los tres datasets juntos. Aportan información
que solo tiene sentido en el nivel combinado, ya sea porque miden relaciones
entre datasets o porque caracterizan el corpus como un todo.

| Métrica | Por qué aplica al corpus combinado | Qué revela |
|--------|--------------------------------------|-----------------|
| **Totales agregados** (tokens, types, caracteres, documentos) | El propósito principal es dimensionar el corpus completo para reportarlo. Un artículo necesita afirmar «el corpus contiene X millones de tokens en Y documentos». | El tamaño y la escala totales del recurso bilingüe. |
| **Tamaño de vocabulario combinado** | El vocabulario combinado no es la suma de los vocabularios individuales: hay solapamiento. Medir el tamaño combinado real revela cuánto vocabulario comparten los tres datasets. | Si el vocabulario combinado es mucho menor que la suma de las partes, los datasets son redundantes. Si se acerca a la suma, son complementarios. |
| **Solapamiento de vocabulario** (Jaccard por pares, types compartidos, types exclusivos) | Esta métrica solo existe al comparar dos o más datasets. Mide cuánto vocabulario comparte cada par de datasets y cuánto aporta cada uno en exclusiva al vocabulario combinado. | Un solapamiento alto entre WikiText y FEVER es esperable (ambos vienen de Wikipedia). Un solapamiento bajo con Europarl confirmaría que el texto parlamentario aporta un dominio de vocabulario distinto. |
| **Detección de duplicados entre datasets** | Los duplicados dentro de un mismo dataset se detectan por dataset. Pero los duplicados entre datasets —por ejemplo, un artículo de Wikipedia que aparece tanto en WikiText como en FEVER— solo pueden detectarse comparando todos los datasets a la vez. | Como WikiText y FEVER provienen ambos de Wikipedia, hay un riesgo real de solapamiento entre datasets. Esta métrica cuantifica si el proceso de selección produjo subconjuntos independientes. |
| **Curva de crecimiento de vocabulario combinada** | Al procesar los tres datasets en secuencia y seguir cómo crece el vocabulario, se puede visualizar si cada dataset nuevo aporta vocabulario fresco o solo repite lo ya visto. | Una curva que sigue subiendo con fuerza al empezar un dataset nuevo indica alta complementariedad. Una curva que se aplana sugiere saturación o redundancia. La gráfica con código de color muestra exactamente dónde empieza y termina la aportación de cada dataset. |
| **Distribución de clasificaciones combinada** | Las distribuciones por dataset muestran el balance interno. La distribución combinada muestra el perfil temático global de todo el corpus, con desgloses por dataset que indican cuál aporta a cada tema. | Importante para usuarios posteriores que necesitan conocer el balance temático global. La vista combinada deja claro que el contenido deportivo (de FEVER) domina el corpus, mientras que el contenido político (de Europarl) y el enciclopédico (de WikiText) aportan amplitud. |

### Por qué algunas métricas por dataset NO se aplicaron al corpus combinado

La decisión de excluir ciertas métricas del análisis combinado se basa en un
principio simple: **una métrica combinada solo es útil si dice algo que las
métricas por dataset no dicen.**

Por ejemplo:
- El **exponente de Zipf combinado** sería aproximadamente 0.95 (la media de los tres exponentes individuales ponderada por conteo de tokens). Ese número no añade nada: ya sabemos que cada corpus individual sigue la ley de Zipf.
- El **MATTR combinado** sería aproximadamente 0.69 (dominado por los 10,000 documentos de FEVER). Esa media oculta que Europarl tiene una diversidad léxica notablemente mayor (0.72) que FEVER (0.67).
- La **longitud de oración combinada** sería de unas 23 palabras: una media sin sentido entre discursos parlamentarios, artículos enciclopédicos y pasajes factuales.

En cambio, las métricas combinadas listadas arriba sí revelan información
genuinamente nueva: cuánto se complementan los datasets entre sí, si hay
contaminación cruzada y cómo se ve el corpus completo como recurso unificado.

### Formato de salida combinado

Las métricas combinadas se guardan en `metrics_combined_{timestamp}.json` con la
siguiente estructura:

```json
{
  "metadata": {
    "datasets": ["wikitext", "fever", "europarl"],
    "total_entries": 12003,
    "generated_at": "2026-07-23T..."
  },
  "combined_totals": {
    "english": { "total_documents": ..., "total_tokens": ..., "total_types": ..., "per_dataset": { ... } },
    "spanish": { ... }
  },
  "vocabulary_overlap": {
    "english": {
      "combined_vocab_size": ...,
      "pairwise_overlap": { "wikitext_vs_fever": { "shared_types": ..., "jaccard_similarity": ... }, ... },
      "exclusive_contributions": { "wikitext": { "exclusive_types": ..., "contribution_to_combined_pct": ... }, ... }
    },
    "spanish": { ... }
  },
  "cross_dataset_duplicates": {
    "english": { "cross_dataset_duplicate_pairs": ..., "details": [...] },
    "spanish": { ... }
  },
  "combined_growth_curve": {
    "english": { "processing_order": [...], "per_dataset_contribution": [...], "growth_curve": [...] },
    "spanish": { ... }
  },
  "combined_classification": {
    "combined": { "sports": ..., "politics": ..., ... },
    "total_entries": ...,
    "per_dataset": { "wikitext": { ... }, "fever": { ... }, "europarl": { ... } }
  }
}
```

---

## Referencias

- Broder, A. Z. (1997). On the resemblance and containment of documents. *Proceedings of the Compression and Complexity of Sequences*, 21-29.
- Carroll, J. B. (1964). *Language and Thought*. Prentice-Hall.
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian Knot: The Moving-Average Type-Token Ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94-100.
- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology*, 32(3), 221-233.
- Francis, W. N., & Kucera, H. (1967). *Computational Analysis of Present-Day American English*. Brown University Press.
- Guiraud, P. (1954). *Les Caracteres Statistiques du Vocabulaire*. Presses Universitaires de France.
- Herdan, G. (1960). *Type-Token Mathematics*. Mouton.
- Johnson, W. (1944). Studies in language behavior: A program of research. *Psychological Monographs*, 56(2), 1-15.
- Luhn, H. P. (1958). The automatic creation of literature abstracts. *IBM Journal of Research and Development*, 2(2), 159-165.
- McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381-392.
- Zipf, G. K. (1935). *The Psycho-biology of Language*. Houghton Mifflin.
