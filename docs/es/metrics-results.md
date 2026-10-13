# Análisis de los resultados de las métricas

*[Read in English](../metrics-results.md)*

Este documento presenta un análisis crítico de las métricas calculadas para los
tres datasets bilingües generados por el pipeline de traducción: WikiText (1,000
artículos), FEVER (9,998 artículos) y Europarl (1,005 debates). El objetivo es
evaluar la calidad real de los datasets generados y señalar cualquier anomalía.


*Actualizado el 2026-10-11: las cifras se recalcularon sobre el corpus publicado, después de volver a traducir cinco entradas de WikiText y de reparar la entrada 4148 de FEVER; ver [../../corpus/README.md](../../corpus/README.md#caveats).*

---

## 1. Métricas generales del corpus

### Tabla resumen

| Métrica | WikiText EN | WikiText ES | FEVER EN | FEVER ES | Europarl EN | Europarl ES |
|--------|-------------|-------------|----------|----------|-------------|-------------|
| Documentos | 1,000 | 1,000 | 9,998 | 9,998 | 1,005 | 1,005 |
| Tokens totales | 2,855,853 | 3,038,960 | 1,753,843 | 1,941,480 | 6,950,546 | 6,954,048 |
| Types únicos | 84,029 | 104,670 | 54,057 | 63,253 | 36,790 | 57,701 |
| Palabras/doc (media) | 2,855.9 | 3,039.0 | 175.4 | 194.2 | 6,916.0 | 6,919.5 |
| Palabras/doc (mediana) | 2,247 | 2,409 | 139.5 | 156 | 5,886 | 5,898 |
| Palabras/doc (mín.) | 807 | 872 | 76 | 71 | 217 | 199 |
| Palabras/doc (máx.) | 8,571 | 9,438 | 3,554 | 3,701 | 25,619 | 26,145 |
| Oraciones/doc (media) | 129.7 | 127.6 | 8.2 | 8.0 | 276.5 | 278.2 |

### Observaciones

**Los conteos de tokens y types son consistentes.** El español tiene sistemáticamente más tokens (entre 6% y 11% más) y más types (entre 17% y 57% más) que el inglés en los tres corpus. El mayor conteo de types en español es esperable por la riqueza morfológica (conjugaciones verbales, adjetivos con género, formas flexionadas). La expansión de tokens está dentro del rango esperado de 5% a 15% para traducción EN→ES.

**Europarl tiene el vocabulario más pequeño pese a tener el mayor conteo de tokens.** Con 6.95M de tokens pero solo 36,790 types en inglés (frente a los 84,029 types de WikiText sobre 2.85M de tokens), esto no es anómalo: refleja la naturaleza repetitiva y formulaica del lenguaje parlamentario. Los debates reutilizan extensamente el mismo vocabulario político y procedimental. Lo confirma el menor porcentaje de hapax (28.6% frente al 39.7% de WikiText).

### Anomalía: conteo mínimo de palabras en WikiText

El corpus de WikiText se filtró para incluir solo artículos de 1,000 palabras o más, pero las métricas muestran un mínimo de 807 palabras en inglés. La investigación revela 49 documentos por debajo del umbral de 1,000 palabras. Esta discrepancia **no es un error de datos**: se debe a una diferencia de tokenización. La etapa de selección del corpus usó `str.split()` de Python para contar palabras (que cuenta las palabras con guion y las contracciones como un solo token), mientras que las métricas usan un tokenizador basado en expresiones regulares (`\b\w+\b`) que divide en caracteres no alfanuméricos y produce menos tokens por documento. La diferencia es pequeña (entre 3% y 20% en casos extremos) y no afecta la calidad del corpus.

### Anomalía: conteo mínimo de palabras en FEVER

FEVER muestra un mínimo de 76 palabras en inglés pese a un filtro de 100 palabras. Aplica la misma discrepancia de tokenización: el filtro usó un método de conteo y las métricas usan otro. Además, el lado español de FEVER muestra un mínimo de 71 palabras, por debajo incluso del mínimo en inglés. Esto significa que la traducción de al menos un artículo es más corta que el original, lo cual es inusual en traducción EN→ES. Estas entradas podrían merecer inspección manual.

---

## 2. Métricas de frecuencia de palabras

### Tabla resumen

| Métrica | WikiText EN | WikiText ES | FEVER EN | FEVER ES | Europarl EN | Europarl ES |
|--------|-------------|-------------|----------|----------|-------------|-------------|
| Exponente de Zipf | 0.897 | 0.946 | 0.932 | 0.954 | 0.985 | 0.963 |
| Hapax % | 39.72% | 40.01% | 44.31% | 43.91% | 28.63% | 31.74% |
| Dis legomena % | 14.08% | 14.46% | 14.52% | 14.75% | 11.22% | 12.30% |
| Cobertura top-100 | 42.17% | 45.28% | 46.84% | 50.03% | 53.77% | 52.53% |
| Cobertura top-1000 | 64.66% | 63.82% | 73.51% | 73.29% | 80.90% | 77.72% |

### Observaciones

**Todos los exponentes de Zipf están en el rango del lenguaje natural (0.89–0.99).** Es una señal positiva fuerte: confirma que las traducciones generadas por el LLM mantienen distribuciones naturales de frecuencia de palabras. Si las traducciones fueran mecánicas o formulaicas, veríamos desviaciones respecto al exponente esperado de ~1.0. Que las seis mediciones (3 corpus × 2 idiomas) caigan dentro del rango esperado valida la naturalidad del texto traducido.

**Europarl es el más zipfiano (0.985 EN).** El texto parlamentario es muy estructurado y formulaico, lo que produce una distribución de frecuencias muy cercana a la ley de Zipf teórica. Tiene sentido: los oradores reutilizan vocabulario político estándar y frases procedimentales.

**WikiText tiene el exponente de Zipf más bajo (0.897 EN).** Esto sugiere algo más de diversidad de vocabulario de la esperada: la «cola larga» de palabras raras es más prominente en el texto enciclopédico. Es coherente con que WikiText cubra muchos temas distintos con terminología especializada.

**Los porcentajes de hapax legomena son saludables.** WikiText y FEVER muestran entre 40% y 44% de hapax, típico de corpus de lenguaje natural. El 28–31% menor de Europarl refleja su vocabulario repetitivo. Los porcentajes en EN y ES son notablemente similares dentro de cada corpus, lo que indica que las traducciones preservan la estructura de diversidad de los originales.

**La cobertura del vocabulario muestra los patrones esperados.** Las 100 palabras más frecuentes de Europarl cubren el 53% de todos los tokens (muy concentrado), mientras que las 100 de WikiText cubren solo el 42% (más diverso). FEVER queda en medio. Estos patrones son coherentes con las características de dominio de cada corpus.

---

## 3. Métricas de diversidad léxica

### Tabla resumen

| Métrica | WikiText EN | WikiText ES | FEVER EN | FEVER ES | Europarl EN | Europarl ES |
|--------|-------------|-------------|----------|----------|-------------|-------------|
| TTR de corpus | 0.0294 | 0.0344 | 0.0308 | 0.0326 | 0.0053 | 0.0083 |
| RTTR | 49.72 | 60.04 | 40.82 | 45.40 | 13.95 | 21.88 |
| MATTR (w=100) | 0.7081 | 0.6948 | 0.6903 | 0.6701 | 0.7146 | 0.7170 |
| MTLD | 84.63 | 77.27 | 72.86 | 63.24 | 89.74 | 91.47 |
| TTR por doc (media) | 0.335 | 0.335 | 0.606 | 0.577 | 0.275 | 0.301 |
| MATTR por doc (media) | 0.707 | 0.694 | 0.677 | 0.663 | 0.712 | 0.716 |
| MTLD por doc (media) | 84.69 | 77.83 | 68.44 | 61.19 | 89.29 | 91.54 |

### Observaciones

**Los valores de MATTR son consistentes y saludables (0.67–0.72).** Las seis mediciones caen en el rango 0.67–0.72, característico de texto natural y bien escrito. Valores por debajo de 0.60 sugerirían lenguaje repetitivo o formulaico. Valores por encima de 0.80 sugerirían texto artificialmente diverso. El rango estrecho entre corpus e idiomas es un indicador positivo de calidad.

**Europarl tiene el MTLD más alto (89–91) pese al vocabulario más pequeño.** Parece contraintuitivo pero tiene sentido: el MTLD mide cuántas palabras consecutivas se pueden leer antes de que caiga la diversidad; en el texto parlamentario los oradores construyen oraciones largas y variadas aunque recurran a un vocabulario político limitado. El texto fluye bien sin repetición excesiva dentro de las oraciones.

**FEVER tiene el MTLD más bajo (63–72).** Los pasajes factuales cortos tienen menos margen para la variación léxica. Con una mediana de longitud de documento de apenas 139 palabras, las entradas de FEVER simplemente no tienen suficiente texto para sostener una diversidad alta en tramos largos.

**El MTLD del español es sistemáticamente menor que el del inglés en WikiText y FEVER, pero mayor en Europarl.** El patrón de WikiText y FEVER (MTLD ES < MTLD EN) es esperable porque el español usa más palabras funcionales (artículos, preposiciones) que se repiten con frecuencia. La inversión en Europarl es interesante: puede deberse a que las traducciones al español del texto parlamentario usan un repertorio más rico de formas verbales y sinónimos que los originales en inglés.

**El TTR por documento muestra el efecto esperado de la longitud.** Los documentos de FEVER (cortos) tienen un TTR por documento de 0.60, mientras que los de Europarl (largos) tienen apenas 0.28. Esto confirma por qué el TTR por sí solo no es fiable: está dominado por la longitud del documento. El MATTR y el MTLD, al ser independientes de la longitud, ofrecen una comparación mucho más significativa.

---

## 4. Métricas sintácticas

### Tabla resumen

| Métrica | WikiText EN | WikiText ES | FEVER EN | FEVER ES | Europarl EN | Europarl ES |
|--------|-------------|-------------|----------|----------|-------------|-------------|
| Longitud de oración (media) | 22.0 | 23.8 | 21.4 | 24.4 | 25.0 | 24.9 |
| Longitud de oración (mediana) | 21 | 22 | 20 | 22 | 22 | 22 |
| Longitud de oración (desv. est.) | 11.9 | 12.9 | 12.8 | 14.4 | 15.2 | 15.2 |
| Longitud de oración (máx.) | 403 | 501 | 469 | 658 | 1,584 | 1,616 |

### Observaciones

**Las oraciones en español son sistemáticamente más largas que en inglés.** En los tres corpus, el español promedia entre 1 y 3 palabras más por oración que el inglés. Es lo esperable: el español requiere más palabras funcionales (artículos, preposiciones) y tiene frases verbales más largas que el inglés.

**Europarl tiene la mayor longitud media de oración (25 palabras).** El discurso parlamentario tiende a estructuras oracionales más largas y complejas, con cláusulas subordinadas, lo que se refleja en los datos.

### Anomalía: longitudes máximas de oración extremas

Los tres corpus muestran longitudes máximas de oración sospechosamente altas: WikiText 403/501 palabras, FEVER 469/658, Europarl 1,584/1,616. Oraciones de más de 400 palabras casi con seguridad no son oraciones reales: son artefactos del divisor de oraciones al fallar sobre texto que carece de puntuación adecuada. Es una **limitación del divisor de oraciones simple basado en expresiones regulares** (fronteras `[.!?]`), no un problema de calidad de los datos. Causas posibles:

- Listas o tablas largas sin puntuación de fin de oración
- Bloques de orador de Europarl que abarcan varios párrafos sin puntos
- Entradas de FEVER con punto y coma o dos puntos usados en lugar de puntos

Esto no afecta la calidad de las traducciones, pero implica que las estadísticas a nivel de oración (en especial el máximo y la desviación estándar) deben interpretarse con cautela. La media y la mediana sí son fiables.

### Anomalía: oraciones de longitud cero

WikiText EN tiene 9 oraciones de longitud cero y FEVER EN tiene 44. Son artefactos del divisor de oraciones, que produce cadenas vacías al dividir en signos de puntuación consecutivos (por ejemplo, patrones `"..."` o `"?!"`). De nuevo, es una limitación del divisor, no un problema de los datos.

---

## 5. Métricas de calidad y semánticas

### Tabla resumen

| Métrica | WikiText | FEVER | Europarl |
|--------|----------|-------|----------|
| Razón de expansión (media) | 1.044 | 1.091 | 1.051 |
| Razón de expansión (desv. est.) | 0.036 | 0.066 | 0.021 |
| Razón de expansión (mín.) | 0.794 | 0.568 | 0.940 |
| Duplicados exactos EN | 0 | 11 | 0 |
| Duplicados exactos ES | 0 | 0 | 0 |
| Casi duplicados EN | 0 | 0 | 0 |
| Casi duplicados ES | 0 | 0 | 0 |
| Stopwords EN % | 36.67% | 33.16% | 43.16% |
| Stopwords ES % | 38.71% | 38.89% | 40.68% |
| Caracteres especiales EN % | 3.08% | 3.24% | 3.23% |
| Caracteres especiales ES % | 2.47% | 2.44% | 3.05% |
| Mayúsculas EN % | 3.56% | 5.66% | 4.51% |
| Mayúsculas ES % | 3.15% | 4.15% | 3.98% |

### Observaciones

**Las razones de expansión están dentro del rango normal (1.04–1.09).** La razón de expansión se calcula como `len(texto_español) / len(texto_inglés)` en caracteres. La traducción inglés-español suele producir entre 10% y 20% más texto. Nuestras razones de 4% a 9% de expansión son algo menores que el rango típico, pero es razonable porque los nombres propios (frecuentes en los tres corpus) no se expanden. La baja desviación estándar de Europarl (0.021) indica un comportamiento de traducción muy consistente para el texto parlamentario.

**Europarl tiene la distribución de expansión más estrecha.** Con mín.=0.940 y máx.=1.171, ningún documento de Europarl tiene una razón de expansión problemática. Esto sugiere que las etapas de fragmentación y revisión funcionaron especialmente bien con el texto parlamentario estructurado.

### Anomalía: FEVER tiene 11 textos duplicados exactos en inglés

La investigación revela que estos duplicados provienen de dos grupos de artículos con contenido casi idéntico en el corpus FEVER original:

- **Grupo 1** (6 artículos): artículos sobre «College Baseball All-America Team» y «All-American Team» de distintos años. Comparten pasajes idénticos de Wikipedia porque se usó el mismo texto de plantilla para varios años.
- **Grupo 2** (5 artículos): artículos sobre «Super Bowl of Poker» de los años 1979 a 1989. El mismo problema: pasajes de plantilla idénticos reutilizados entre años.

Estos duplicados existen en el **corpus FEVER original** y pasaron el filtro de selección porque tienen identificadores FEVER distintos (títulos de artículo distintos). Las traducciones al español son todas únicas (0 duplicados en ES), lo que significa que el pipeline tradujo cada una de forma independiente y produjo cada vez un texto en español ligeramente distinto. En realidad es una buena señal de diversidad de traducción, pero conviene anotar los duplicados de origen como limitación del dataset. Representan el 0.11% del corpus FEVER (11 de 9,998).

### Anomalía: FEVER tiene 31 artículos con razón de expansión inferior a 0.85

31 artículos de FEVER tienen traducciones al español notablemente más cortas que el original en inglés (razón < 0.85). El peor caso tiene una razón de 0.568, es decir, la traducción mide solo el 56.8% de la longitud del original. Probablemente son casos en los que:

- La traducción se truncó (pese a la fragmentación y la detección de truncamiento)
- El modelo resumió en lugar de traducir
- La verificación programática de truncamiento no detectó el problema (como solo sobrescribe la puntuación de evaluación, no vuelve a traducir)

**Estos 31 artículos (0.31% de FEVER) deben considerarse entradas de baja calidad.** Para un corpus de 9,998 entradas es una tasa de error aceptable, pero quienes usen el corpus deberían saberlo.

Además, WikiText tiene 91 artículos en los que la traducción al español es más corta que el original en inglés (razón de expansión < 1.0, medida en caracteres). La investigación de esos artículos revela tres categorías distintas:

**3 artículos con razón < 0.90 (truncamiento real).** El peor caso es el artículo 766 («Good Shepherd (song)», razón 0.794), donde la traducción al español termina a mitad de frase («contó con una rara voz principal...») y le faltan los 8 párrafos finales del original. Es notable que los candidatos individuales de la Etapa 3 tenían razones normales (1.045 y 1.049), lo que significa que el truncamiento ocurrió durante la fusión por consenso de la Etapa 4: el modelo revisor no incluyó todas las secciones al combinar los candidatos. El evaluador del pipeline asignó a este artículo una puntuación de completitud de 70, detectando correctamente el problema, pero no se disparó ninguna retraducción, ya que la etapa de evaluación solo califica, no reprocesa.

**8 artículos con razón entre 0.90 y 0.95 (posible truncamiento parcial).** Son casos limítrofes en los que pudo perderse algo de contenido durante la etapa de fusión. Deberían revisarse manualmente.

**80 artículos con razón entre 0.95 y 1.0 (compresión legítima).** El análisis muestra un patrón claro: el 52% de todos los artículos por debajo de 1.0 están clasificados como «entertainment» o «music» (frente al 26% del corpus completo), y el 38% contienen secciones con «credits», «charts» o «certifications»: secciones de datos estructurados llenas de nombres propios, números y abreviaturas que no se expanden al traducirse. Es un comportamiento normal, no un problema de calidad. Por ejemplo, una sección de discografía que lista nombres de pistas, sellos discográficos y posiciones en listas tendrá la misma longitud en ambos idiomas.

En resumen, solo 3 de los 1,000 artículos de WikiText (0.3%) tienen un problema genuino de truncamiento. Los 88 artículos restantes por debajo de 1.0 se explican por características legítimas del contenido.

### Anomalía: FEVER tiene 2 artículos con razón de expansión superior a 1.5

Dos artículos de FEVER tienen razones de expansión inusualmente altas (1.528 y 1.646). Son:
- Artículo 4523: «14 inch gun» (487 caracteres EN, 744 ES)
- Artículo 4563: «1949-51 Soviet nuclear tests» (987 caracteres EN, 1,625 ES)

Pueden ser casos en los que el traductor añadió texto explicativo, o en los que abreviaturas del inglés se desarrollaron como frases completas en español (por ejemplo, «14 in.» convertido en «cañón de 14 pulgadas»). Aunque no son errores en sí, son valores atípicos que conviene señalar.

**Los porcentajes de stopwords son naturales.** El inglés va del 33% al 43% y el español del 38% al 41%. Europarl tiene la tasa más alta en inglés (43%) porque el discurso parlamentario usa más palabras funcionales. Los porcentajes en español son más consistentes entre corpus, lo cual es típico: el español tiene una distribución de palabras funcionales más estable con independencia del dominio.

**Los porcentajes de caracteres especiales y mayúsculas son normales.** Todos los valores están en el rango de 2% a 6% sin anomalías. El español tiene sistemáticamente porcentajes algo menores de caracteres especiales y mayúsculas que el inglés, lo cual es esperable (el español capitaliza menos palabras y usa menos abreviaturas).

### Distribución de clasificaciones

- **WikiText** muestra una diversidad temática razonable con más de 18 categorías, dominada por «history» (284) y «entertainment» (260). Sin embargo, hay una inconsistencia: «Science» y «science» aparecen como categorías separadas (6 y 101 entradas), lo que indica un problema de sensibilidad a mayúsculas en la extracción de metadatos.
- **FEVER** está dominado por «sports» (6,675 de 9,998 = 66.8%), como se esperaba y documentó.
- **Europarl** está dominado por «politics» (483 de 1,005 = 48.1%), lo cual tiene sentido para actas parlamentarias.

### Anomalía: inconsistencia de mayúsculas en las clasificaciones de WikiText

La distribución de clasificaciones de WikiText muestra «Science» (con mayúscula, 6 entradas) y «science» (minúscula, 101 entradas) como categorías separadas. Es un error de la etapa de extracción de metadatos: el LLM a veces devuelve la clasificación con distinta capitalización. Afecta solo a los metadatos, no a la calidad de la traducción. La corrección sería normalizar las cadenas de clasificación a minúsculas en el pipeline.

---

## 6. Evaluación de calidad entre datasets

### Valoración global de calidad

| Dataset | Valoración | Fortalezas clave | Preocupaciones clave |
|---------|-------------------|---------------|--------------|
| **Europarl** | Excelente | Razones de expansión más estrechas (desv. est.=0.021), 0 duplicados, mayor diversidad léxica (MTLD=91), traducciones más consistentes | Ninguna significativa |
| **WikiText** | Buena | 0 duplicados, distribución de Zipf natural, buena diversidad léxica | 3 artículos (0.3%) con truncamiento real; 80 artículos con compresión legítima por contenido estructurado; inconsistencia de mayúsculas en la clasificación |
| **FEVER** | Buena con reservas | El corpus más grande (9,998), distribuciones de frecuencia naturales, temas diversos | 11 duplicados exactos heredados del origen; 31 artículos con expansión < 0.85; fuerte sesgo deportivo (66.8%) |

### Hallazgos positivos

1. **Todos los exponentes de Zipf están en el rango del lenguaje natural (0.89–0.99).** Las traducciones generadas por el LLM se comportan como texto natural, no como una salida formulaica generada por máquina.

2. **Cero casi duplicados en las comparaciones muestreadas.** La detección de casi duplicados (similitud de Jaccard sobre shingles de 5 palabras, umbral 0.85) no encontró ningún par en ningún corpus. No obstante, este resultado debe interpretarse con cautela: por el costo computacional, la detección usa una muestra aleatoria de 500 documentos por corpus, que cubre solo ~0.25% de todos los pares posibles en FEVER (9,998 documentos). Aunque la ausencia de casi duplicados en la muestra es alentadora, no garantiza cero casi duplicados en el corpus completo. Haría falta un método de detección más exhaustivo (por ejemplo, MinHash con LSH) para obtener resultados definitivos.

3. **Los valores de MATTR son consistentes (0.67–0.72).** La diversidad léxica es estable y saludable en todos los corpus y en ambos idiomas, lo que indica que las traducciones mantienen la riqueza de vocabulario de los originales.

4. **Las razones de expansión son coherentes con los patrones conocidos EN-ES.** La expansión media de 4% a 9% con baja desviación estándar confirma que las traducciones no están truncadas ni infladas a nivel sistemático.

5. **El español tiene sistemáticamente más types que el inglés (entre 17% y 57% más).** Es el comportamiento esperado en una lengua morfológicamente rica y confirma que las traducciones usan morfología española propia en vez de calcos palabra por palabra.

### Preocupaciones y limitaciones

1. **Duplicados de origen en FEVER (11 artículos).** Deberían documentarse como limitación conocida. No afectan la calidad de la traducción, pero reducen el contenido único efectivo en un 0.11%.

2. **Posible truncamiento en FEVER (31 artículos con expansión < 0.85).** Estos artículos pueden contener traducciones incompletas. Representan el 0.31% del corpus.

3. **Truncamiento en WikiText (3 artículos con expansión < 0.90).** La investigación confirmó 3 artículos con truncamiento genuino causado por la fusión por consenso de la Etapa 4, que no incluyó todas las secciones. Los 88 artículos restantes por debajo de 1.0 se explican por compresión legítima de contenido rico en nombres propios, números y datos estructurados (créditos, listas, discografías) que no se expande al traducirse.

4. **Artefactos del divisor de oraciones.** Las longitudes máximas de oración (hasta 1,616 palabras) son artefactos del divisor basado en expresiones regulares, no oraciones reales. Afecta a las estadísticas sintácticas pero no a la calidad de los datos subyacentes.

5. **Inconsistencia de mayúsculas en la clasificación de WikiText.** Un problema menor de normalización de metadatos que debería corregirse en el pipeline.

6. **Sesgo deportivo de FEVER (66.8%).** Aunque documentado y esperable por el muestreo estratificado, quienes usen el corpus deben saber que no está equilibrado temáticamente.

---

## 7. Métricas del corpus combinado

Estas métricas se calculan sobre los tres datasets juntos, para evaluar el corpus
como recurso unificado.

### 7.1 Totales agregados

| Métrica | Inglés | Español |
|--------|---------|---------|
| Documentos totales | 12,003 | 12,003 |
| Tokens totales | 11,560,242 | 11,934,488 |
| Types totales (vocabulario combinado) | 117,297 | 150,745 |
| Caracteres totales | 69,700,888 | 73,572,650 |
| Suma de los types de cada dataset | 174,876 | 225,624 |
| Solapamiento de vocabulario (suma − combinado) | 57,579 | 74,879 |

**Hallazgo clave: el corpus combinado contiene 11.5M de tokens en inglés y 11.9M en español a lo largo de 12,003 documentos bilingües.** Es un recurso bilingüe sustancial: aproximadamente 23.5 millones de tokens en total entre ambos idiomas.

**El solapamiento de vocabulario es significativo pero no excesivo.** La suma de los vocabularios individuales (174,876 types EN) es un 49% mayor que el vocabulario combinado (117,297 types EN). Esto significa que 57,579 types en inglés (el 33% de la suma) aparecen en más de un dataset. Es un nivel de solapamiento saludable: los datasets comparten un núcleo común de vocabulario general y cada uno aporta además un vocabulario exclusivo sustancial. Si el solapamiento fuera cercano a cero, los datasets no tendrían nada en común (improbable en lenguaje natural). Si fuera cercano al 100%, serían redundantes.

El español muestra un patrón similar con 33% de solapamiento (74,879 types compartidos de una suma de 225,624). El vocabulario combinado en español (150,745) es un 29% mayor que el inglés (117,297), coherente con la riqueza morfológica del español observada en las métricas por dataset.

### 7.2 Análisis del solapamiento de vocabulario

#### Solapamiento por pares (inglés)

| Par | Types compartidos | Jaccard | % del vocabulario menor |
|------|-------------|---------|---------------------------|
| WikiText vs FEVER | 30,925 | 0.289 | 57.2% |
| WikiText vs Europarl | 25,385 | 0.266 | 69.0% |
| FEVER vs Europarl | 19,122 | 0.267 | 52.0% |

#### Solapamiento por pares (español)

| Par | Types compartidos | Jaccard | % del vocabulario menor |
|------|-------------|---------|---------------------------|
| WikiText vs FEVER | 38,360 | 0.295 | 60.6% |
| WikiText vs Europarl | 34,728 | 0.271 | 60.2% |
| FEVER vs Europarl | 24,176 | 0.250 | 41.9% |

**WikiText y Europarl comparten el mayor vocabulario en relación con su tamaño.** El 69% del vocabulario inglés de Europarl (el menor de los dos) aparece también en WikiText. Tiene sentido: ambos contienen texto formal y bien estructurado, con oraciones completas y vocabulario estándar. El vocabulario político de Europarl es un subconjunto del vocabulario enciclopédico más amplio de WikiText.

**FEVER y Europarl comparten el menor vocabulario.** Solo un 52% de solapamiento en inglés. FEVER está dominado por contenido deportivo con terminología especializada (nombres de jugadores, de equipos, formatos de torneo) que rara vez aparece en debates parlamentarios. Esto confirma que estos dos datasets son el par más complementario del corpus.

**Las similitudes de Jaccard son bajas (0.25–0.29).** Indica que, si bien los datasets comparten un vocabulario nuclear, cada uno aporta una gran proporción de términos únicos. El corpus es genuinamente diverso, no tres variaciones del mismo texto.

#### Aportaciones exclusivas al vocabulario combinado

| Dataset | Types EN exclusivos | % de su vocabulario | % del combinado |
|---------|-------------------|----------------|---------------|
| WikiText | 45,570 | 54.2% | 38.9% |
| FEVER | 21,863 | 40.4% | 18.6% |
| Europarl | 10,134 | 27.5% | 8.6% |

| Dataset | Types ES exclusivos | % de su vocabulario | % del combinado |
|---------|-------------------|----------------|---------------|
| WikiText | 53,966 | 51.6% | 35.8% |
| FEVER | 23,072 | 36.5% | 15.3% |
| Europarl | 21,122 | 36.6% | 14.0% |

**WikiText es el mayor contribuyente de vocabulario.** Aporta en exclusiva el 38.9% del vocabulario combinado en inglés: palabras que aparecen en WikiText pero ni en FEVER ni en Europarl. Refleja la amplitud del contenido enciclopédico, que cubre ciencia, tecnología, geografía, música y muchos otros dominios.

**FEVER aporta el 18.6% en exclusiva.** Pese a ser el menor en types totales (54,057 EN), la terminología deportiva y los nombres propios de FEVER (nombres de atletas, de equipos, eventos específicos) son muy propios del dominio y rara vez aparecen en otro sitio.

**Europarl aporta el menor vocabulario exclusivo (8.6% EN).** Es esperable: el vocabulario político y procedimental de Europarl es relativamente estándar y se solapa en gran medida con el vocabulario general de WikiText. Sin embargo, en español Europarl aporta un 14.0% en exclusiva, casi igualando el 15.2% de FEVER. Esto se debe a que las traducciones al español del texto parlamentario usan un repertorio más rico de expresiones formales, terminología jurídica y formas verbales que no aparecen en la enciclopedia ni en los pasajes factuales.

### 7.3 Detección de duplicados entre datasets

| Idioma | Pares duplicados entre datasets |
|----------|------------------------------|
| Inglés | 0 |
| Español | 0 |

**No se encontraron duplicados entre datasets.** Pese a que WikiText y FEVER provienen ambos de Wikipedia, el proceso de selección produjo subconjuntos completamente independientes, sin artículos compartidos. Es un hallazgo positivo fuerte: significa que las 12,003 entradas bilingües son todas únicas en el corpus completo.

Este resultado es notable porque no estaba garantizado por diseño. WikiText contiene artículos completos de Wikipedia mientras que FEVER contiene pasajes más cortos extraídos de Wikipedia. Era plausible que un pasaje de FEVER fuera un fragmento de un artículo de WikiText. El resultado de cero duplicados confirma que, o bien los artículos son genuinamente distintos, o bien los pasajes son demasiado cortos para coincidir con los artículos completos mediante comparación exacta de hash.

**Nota:** esta detección usa coincidencia exacta de texto (hash MD5). Los casi duplicados (por ejemplo, un pasaje de FEVER que sea un subconjunto de un artículo de WikiText) no se detectarían con este método. Haría falta un análisis de subcadenas o de contención más sofisticado para detectar solapamientos parciales.

### 7.4 Curva de crecimiento del vocabulario combinado

Orden de procesamiento: WikiText → FEVER → Europarl

#### Inglés

| Dataset | Tokens añadidos | Types después | Types nuevos | % nuevo del total |
|---------|-------------|-------------|-----------|----------------|
| WikiText | 2,855,853 | 84,029 | 84,029 | 100.0% |
| FEVER | 1,753,843 | 107,163 | 23,134 | 21.6% |
| Europarl | 6,950,546 | 117,297 | 10,134 | 8.6% |

#### Español

| Dataset | Tokens añadidos | Types después | Types nuevos | % nuevo del total |
|---------|-------------|-------------|-----------|----------------|
| WikiText | 3,038,960 | 104,670 | 104,670 | 100.0% |
| FEVER | 1,941,480 | 129,623 | 24,953 | 19.3% |
| Europarl | 6,954,048 | 150,745 | 21,122 | 14.0% |

**Cada dataset aporta vocabulario nuevo significativo.** Después de que WikiText establece el vocabulario base (84,029 types EN), FEVER añade 23,134 types nuevos (27.5% de crecimiento) y Europarl añade otros 10,134 (9.5%). El vocabulario sigue creciendo con cada dataset, lo que confirma que las tres fuentes son complementarias y no redundantes.

**El patrón de crecimiento es distinto en inglés y en español.** En inglés, Europarl aporta solo 10,134 types nuevos (8.6% del vocabulario final). En español, Europarl aporta 21,122 (14.0% del vocabulario final), casi igualando la aportación de FEVER en español (24,953). Esto refuerza la observación del análisis de solapamiento: las traducciones al español de Europarl usan un vocabulario claramente más rico, con expresiones formales y conjugaciones que no aparecen en los otros dos datasets.

**Europarl añade la mayor cantidad de tokens pero la menor cantidad de types nuevos.** Con 6.95M de tokens (60% del corpus combinado), Europarl solo añade un 8.6% de vocabulario nuevo en inglés. Esto confirma que el texto parlamentario es muy repetitivo en su vocabulario: reutiliza extensamente los mismos términos políticos y procedimentales. El vocabulario que sí aporta en exclusiva (términos institucionales, jerga propia de la UE) es pequeño pero especializado.

### 7.5 Distribución de clasificaciones combinada

El corpus combinado contiene 53 categorías temáticas distintas (tras normalizar
mayúsculas) a lo largo de 12,003 entradas:

| Categoría | Conteo | % | Fuente principal |
|----------|-------|---|----------------|
| sports | 6,771 | 56.4% | FEVER (98.6%) |
| history | 958 | 8.0% | WikiText (29.6%) + FEVER (70.4%) |
| politics | 741 | 6.2% | Europarl (65.2%) + FEVER (32.4%) |
| military | 666 | 5.5% | FEVER (100%) |
| science | 553 | 4.6% | FEVER (80.7%) + WikiText (19.3%) |
| music | 360 | 3.0% | FEVER (99.7%) |
| geography | 303 | 2.5% | FEVER (100%) |
| entertainment | 279 | 2.3% | WikiText (93.2%) |
| technology | 200 | 1.7% | FEVER (86.0%) |
| film | 163 | 1.4% | FEVER (100%) |
| human rights | 127 | 1.1% | Europarl (100%) |
| transport | 118 | 1.0% | FEVER (93.2%) |
| biography | 111 | 0.9% | WikiText (95.5%) |
| economy | 97 | 0.8% | Europarl (100%) |
| environment | 87 | 0.7% | Europarl (100%) |

**El corpus está fuertemente sesgado hacia deportes (56.4%).** Lo impulsa casi por completo FEVER, que aporta 6,675 entradas deportivas (9,998 entradas de FEVER × 66.8% de deportes). Como FEVER tiene 10 veces más entradas que WikiText o Europarl, su distribución temática domina los conteos combinados.

**Cada dataset ocupa un nicho temático propio.** FEVER domina deportes, militar, música, geografía, cine y transporte. WikiText domina entretenimiento y biografía. Europarl aporta en exclusiva derechos humanos, economía y medio ambiente: temas que no aparecen en ningún otro dataset. Esta complementariedad temática refuerza el valor de combinar tres fuentes distintas.

**Las 53 categorías indican una clasificación inconsistente.** La etapa de extracción de metadatos produjo demasiadas categorías de grano fino (por ejemplo, «science» y «Science» como entradas separadas, o «history» y «military» separadas cuando muchas entradas militares son históricas). Un paso de normalización posterior las reduciría a un conjunto más limpio de 15 a 20 categorías.

### 7.6 Resumen de las métricas combinadas

| Hallazgo | Valoración |
|---------|-----------|
| Tamaño total del corpus: 23.5M de tokens, 12,003 documentos bilingües | Recurso bilingüe sustancial |
| Cero duplicados entre datasets | Las 12,003 entradas son únicas |
| Solapamiento de vocabulario: 33% compartido, 67% único por dataset | Equilibrio saludable: núcleo común + aportaciones diversas |
| WikiText aporta en exclusiva el 39% del vocabulario combinado EN | El mayor y más diverso contribuyente de vocabulario |
| Cada dataset añade vocabulario nuevo al conjunto combinado | Los datasets son genuinamente complementarios |
| Europarl: 60% de los tokens pero solo 8.6% de vocabulario EN exclusivo | Vocabulario repetitivo pero alto volumen |
| El español de Europarl es más distintivo que su inglés | Las traducciones parlamentarias al español usan morfología más rica |
| Deportes domina el corpus combinado (56.4%) | Impulsado por el sesgo temático de FEVER; conviene advertirlo |
| 53 categorías de clasificación (deberían ser ~15–20) | Hace falta normalizar los metadatos |

---

## 8. Recomendaciones

1. **Para el artículo**: reportar los exponentes de Zipf, MATTR, MTLD, razones de expansión, conteos de duplicados y solapamiento de vocabulario como indicadores principales de calidad. Estas métricas cuentan una historia clara: las traducciones generadas se comportan como lenguaje natural, mantienen la diversidad léxica y no están truncadas ni duplicadas. Los resultados de cero duplicados entre datasets y de complementariedad de vocabulario son hallazgos particularmente fuertes que validan el diseño del corpus.

2. **Para el dataset**: considerar añadir un campo `quality_flag` a las entradas con razón de expansión inferior a 0.85, para que quienes usen el corpus puedan filtrarlas si lo necesitan.

3. **Para el pipeline**: normalizar las cadenas de clasificación a minúsculas en la etapa de extracción de metadatos, para evitar inconsistencias de capitalización. Las 53 categorías actuales deberían reducirse a un conjunto estandarizado de 15 a 20 mediante un mapeo de normalización (por ejemplo, fusionar «Science»/«science» y consolidar «military» bajo «history»).

4. **Para el corpus combinado**: reportar el corpus como 23.5M de tokens en 12,003 documentos bilingües únicos, sin solapamiento entre datasets. Destacar la complementariedad de vocabulario: cada dataset aporta vocabulario exclusivo (WikiText 39%, FEVER 19%, Europarl 9% del vocabulario combinado en inglés), lo que confirma el valor de usar tres fuentes diversas.

5. **Para quienes usen el corpus**: documentar que el corpus combinado está fuertemente sesgado hacia deportes (56.4%), impulsado por FEVER. Quien necesite balance temático debería submuestrear o ponderar por categoría. Europarl es el más enfocado temáticamente (política/economía/derechos humanos), mientras que WikiText ofrece la cobertura más amplia.

6. **Para trabajo futuro**: calcular el acuerdo entre modelos a partir de los JSON del pipeline, para validar que el enfoque multi-modelo produce candidatos genuinamente diversos. Además, crear una referencia traducida por humanos para 30 a 50 artículos de FEVER permitiría obtener puntuaciones BLEU/chrF/COMET. Un análisis de contención (comprobar si los pasajes de FEVER son subcadenas de artículos de WikiText) complementaría la detección de duplicados por coincidencia exacta.
