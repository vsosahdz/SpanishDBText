# Corpus Metrics Documentation

*[Leer en español](es/metrics-reference.md)*

This document describes every metric computed by the metrics pipeline, organized by category. For each metric we explain what it measures, who introduced it, where it has been used, how it is calculated in general, how we calculate it in our implementation, and what results to expect.

All metrics are computed independently for English and Spanish texts, enabling direct comparison between the source and translated sides of the corpus.

---

## 1. General Corpus Metrics

These are basic descriptive statistics that characterize the size, shape, and distribution of the corpus. They are the foundation for any corpus analysis and appear in virtually every NLP dataset paper.

### 1.1 Total Tokens

- **What it measures**: The total number of word occurrences across all documents in the corpus.
- **Origin**: Basic corpus linguistics. The type/token distinction was formalized by Charles Sanders Peirce in the late 19th century and adopted widely in linguistics and NLP.
- **Fields that use it**: Corpus linguistics, NLP, machine translation, language modeling.
- **General formula**: Sum of word counts across all documents: $\sum_{d \in D} |words(d)|$.
- **Our implementation**: We tokenize each text using a regex-based word splitter (`\b\w+\b` on lowercased text) and sum the counts. Implemented in `general_metrics.py`, function `compute_general_metrics()`.
- **Expected results**: WikiText should have the highest token count per document (long encyclopedia articles). FEVER should have the lowest (short factual passages). Spanish totals should be 5-15% higher than English due to morphological expansion.

### 1.2 Unique Types (Vocabulary Size)

- **What it measures**: The number of distinct word forms in the corpus.
- **Origin**: Same as tokens — the type/token distinction from Peirce.
- **Fields that use it**: Corpus linguistics, lexicography, vocabulary studies.
- **General formula**: $|V| = |\{w : w \in D\}|$, the cardinality of the set of unique words.
- **Our implementation**: `len(set(all_words))` after lowercasing and tokenization.
- **Expected results**: Spanish typically has a larger vocabulary than English for the same content, because Spanish has more verb conjugations, gendered adjectives, and inflected forms. For example, "was/were/been/being" in English maps to many more forms in Spanish (fue, era, sido, siendo, estaba, estuvo, etc.).

### 1.3 Total Characters

- **What it measures**: The total character count across all documents, including spaces and punctuation.
- **Origin**: Basic text measurement.
- **Fields that use it**: Machine translation (character-level models), storage estimation.
- **General formula**: $\sum_{d \in D} len(d)$.
- **Our implementation**: `sum(len(text) for text in texts)`.
- **Expected results**: Spanish character counts are typically 10-20% higher than English for equivalent content. This ratio is important for detecting truncation in translations.

### 1.4 Words per Document

- **What it measures**: Distribution of document lengths in words — mean, median, standard deviation, min, and max.
- **Origin**: Basic descriptive statistics applied to corpus analysis.
- **Fields that use it**: Any field that works with document collections.
- **General formula**: Standard statistical measures over the vector of per-document word counts.
- **Our implementation**: We compute `mean`, `median`, `std`, `min`, `max` using NumPy over the array of per-document word counts. The full distribution is also stored for plotting histograms.
- **Expected results**: WikiText: mean ~3,300, high variance. FEVER: mean ~190, low variance. Europarl: mean ~3,300, high variance with bimodal distribution (short procedural + long substantive debates).

### 1.5 Sentences per Document

- **What it measures**: How many sentences each document contains on average — indicates structural complexity.
- **Origin**: Text analysis, readability studies.
- **Fields that use it**: Readability assessment, text summarization, document structure analysis.
- **General formula**: Count sentence boundaries per document using punctuation patterns.
- **Our implementation**: We split on sentence-ending punctuation (`[.!?]` followed by whitespace) using `sentence_split()` in `loader.py`. This is a simple heuristic that works well for well-formed text but may over-split on abbreviations.
- **Expected results**: Proportional to document length. WikiText and Europarl should have the highest counts. FEVER entries are typically 1-5 sentences.

### 1.6 Document Length Range (Min/Max)

- **What it measures**: The extremes of document length — validates that filtering in Stage 1 worked correctly.
- **Origin**: Data validation.
- **Fields that use it**: Quality assurance in dataset construction.
- **Our implementation**: `min()` and `max()` of the per-document word counts.
- **Expected results**: Should match the filtering thresholds defined in Stage 1. For example, WikiText min should be around 1,000 words and max around 10,000.

### 1.7 Standard Deviation of Document Length

- **What it measures**: How much variation there is in document sizes — indicates homogeneity.
- **Origin**: Basic statistics.
- **Fields that use it**: Corpus design, batch processing optimization.
- **General formula**: $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2}$.
- **Our implementation**: `np.std(word_counts)`.
- **Expected results**: High for Europarl (bimodal distribution), moderate for WikiText, low for FEVER.

---

## 2. Word Frequency Metrics

These metrics analyze how words are distributed across the corpus. They help validate that the translated text behaves like natural language and detect anomalies.

### 2.1 Zipf Distribution

- **What it measures**: Whether word frequencies follow Zipf's law — the observation that in natural language, the frequency of a word is inversely proportional to its rank. The k-th most common word appears approximately 1/k times as often as the most common word.
- **Origin**: George Kingsley Zipf (1935), *The Psycho-biology of Language*. Later formalized mathematically by Benoit Mandelbrot.
- **Fields that use it**: Corpus linguistics, information retrieval, NLP, complexity science, economics (wealth distribution), urban studies (city sizes).
- **General formula**: $f(r) \propto r^{-\alpha}$, where $f(r)$ is the frequency of the word at rank $r$, and $\alpha$ is the Zipf exponent (approximately 1.0 for natural language). In log-log space this becomes a straight line: $\log f(r) = -\alpha \log r + c$.
- **Our implementation**: We sort all word frequencies in descending order, compute `log10(rank)` and `log10(frequency)` for the top 500 words, and estimate the Zipf exponent $\alpha$ via linear regression on the log-log values using `np.polyfit()`. The raw data points are stored for plotting. Implemented in `general_metrics.py`, function `compute_frequency_metrics()`.
- **Expected results**: Natural text typically produces $\alpha \approx 0.9$-$1.1$. If the translated text shows a similar Zipf exponent to the original, it confirms that the LLM-generated translations maintain natural language distribution. A significantly different exponent would suggest that the translation process introduced distributional artifacts. Our results show exponents between 0.89 and 0.99 for all three corpora in both languages, confirming natural behavior.

### 2.2 Hapax Legomena

- **What it measures**: Words that appear exactly once in the entire corpus. A high hapax count indicates a rich, diverse vocabulary with many rare terms.
- **Origin**: The term comes from Greek, meaning "said only once." Used in classical philology for centuries to study rare words in ancient texts. Formalized in quantitative linguistics by Gustav Herdan (1960).
- **Fields that use it**: Corpus linguistics, authorship attribution, lexicography, philology.
- **General formula**: $H = |\{w : freq(w) = 1\}|$. Often reported as a percentage of total types: $H\% = H / |V| \times 100$.
- **Our implementation**: `sum(1 for w, c in Counter(words).items() if c == 1)`. We report both the absolute count and the percentage of total types.
- **Expected results**: Typically 40-60% of all types are hapax legomena in a natural language corpus. A significantly lower percentage might indicate repetitive or formulaic language. A higher percentage indicates very diverse content. Spanish hapax percentages should be comparable to English for equivalent content.

### 2.3 Dis Legomena

- **What it measures**: Words that appear exactly twice in the corpus. Together with hapax legomena, dis legomena characterize the "tail" of the frequency distribution.
- **Origin**: Also from Greek philology. Used alongside hapax in Good-Turing frequency estimation.
- **Fields that use it**: Statistical NLP (smoothing techniques), corpus linguistics.
- **General formula**: $D = |\{w : freq(w) = 2\}|$.
- **Our implementation**: `sum(1 for w, c in Counter(words).items() if c == 2)`.
- **Expected results**: Usually 10-20% of types. The ratio of hapax to dis legomena is used in some vocabulary growth models.

### 2.4 Vocabulary Coverage

- **What it measures**: What percentage of all tokens in the corpus are accounted for by the N most frequent words. Shows how concentrated or distributed the vocabulary usage is.
- **Origin**: Corpus linguistics, information retrieval. Related to the Pareto principle (80/20 rule).
- **Fields that use it**: Language teaching (core vocabulary), text compression, search engines.
- **General formula**: $C(N) = \frac{\sum_{i=1}^{N} freq(w_i)}{\sum_{i=1}^{|V|} freq(w_i)} \times 100$, where words are sorted by frequency.
- **Our implementation**: We compute coverage at milestones: top 100, 500, 1,000, 5,000, and 10,000 words. Implemented by accumulating frequencies from `Counter.most_common()`.
- **Expected results**: In English, the top 100 words typically cover ~50% of all tokens (dominated by function words). The top 1,000 cover ~75-80%. Similar coverage in English and Spanish indicates that the translation maintains a comparable frequency distribution.

### 2.5 Vocabulary Growth Curve

- **What it measures**: How the number of unique word types grows as more tokens are observed. Also known as the type-token accumulation curve.
- **Origin**: Herdan (1960), *Type-Token Mathematics*. Related to Heaps' law: $|V| \propto N^{\beta}$, where $N$ is the number of tokens and $\beta \approx 0.4$-$0.6$.
- **Fields that use it**: Corpus linguistics, computational lexicography, species diversity studies (ecology).
- **General formula**: Plot $|V(n)|$ vs $n$, where $V(n)$ is the set of unique types seen in the first $n$ tokens.
- **Our implementation**: We iterate through all tokens, maintaining a set of seen types. We sample at ~200 evenly spaced intervals to keep the output data manageable. Stored as a list of `{tokens, types}` points for plotting.
- **Expected results**: The curve should rise steeply at first and then flatten as the vocabulary saturates. A corpus that never flattens has very high diversity (or noise). Spanish curves typically rise slightly faster than English due to morphological richness but flatten similarly.

---

## 3. Lexical Diversity Metrics

These metrics quantify how varied the vocabulary usage is in the text. They are important for assessing translation quality because a good translation should maintain similar lexical diversity to the original — neither oversimplifying nor artificially inflating vocabulary.

### 3.1 Type-Token Ratio (TTR)

- **What it measures**: The ratio of unique words (types) to total words (tokens). A simple measure of vocabulary diversity.
- **Origin**: Johnson (1944), one of the earliest quantitative measures of language complexity. Widely used since the 1940s.
- **Fields that use it**: Language acquisition research, clinical linguistics (aphasia assessment), readability, ESL/EFL studies.
- **General formula**: $TTR = \frac{|V|}{N}$, where $|V|$ is the number of types and $N$ is the number of tokens.
- **Our implementation**: Computed both at corpus level and per document. `len(set(words)) / len(words)`.
- **Expected results**: TTR is heavily dependent on text length — longer texts always have lower TTR because common words are reused. Corpus-level TTR will be very low (0.01-0.05 for large corpora). Per-document TTR is more meaningful (typically 0.3-0.7 depending on document length). This sensitivity to length is why TTR is supplemented by the corrected metrics below.

### 3.2 Root Type-Token Ratio (RTTR)

- **What it measures**: TTR normalized by the square root of the number of tokens, which partially corrects for text length dependence.
- **Origin**: Also known as Guiraud's Index. Proposed by Pierre Guiraud (1954), *Les Caracteres Statistiques du Vocabulaire*.
- **Fields that use it**: Corpus linguistics, stylometry.
- **General formula**: $RTTR = \frac{|V|}{\sqrt{N}}$.
- **Our implementation**: `types / math.sqrt(tokens)`.
- **Expected results**: More stable across different text lengths than TTR, but still not fully independent of length. Values typically range from 5 to 15 for document-level analysis. Note: Guiraud's Index is mathematically identical to RTTR.

### 3.3 Corrected Type-Token Ratio (CTTR)

- **What it measures**: Another length-corrected variant of TTR, using $\sqrt{2N}$ as the denominator.
- **Origin**: Carroll (1964), proposed as an improvement over simple TTR.
- **Fields that use it**: Corpus linguistics, second language acquisition research.
- **General formula**: $CTTR = \frac{|V|}{\sqrt{2N}}$.
- **Our implementation**: `types / math.sqrt(2 * tokens)`.
- **Expected results**: Similar behavior to RTTR but scaled down by a factor of $\sqrt{2}$. Values are lower than RTTR by approximately 29%.

### 3.4 Moving Average Type-Token Ratio (MATTR)

- **What it measures**: The average TTR computed over a sliding window of fixed size. This makes it independent of text length, since the window size is constant.
- **Origin**: Covington and McFall (2010), *Cutting the Gordian Knot: The Moving-Average Type-Token Ratio (MATTR)*. Published in the Journal of Quantitative Linguistics.
- **Fields that use it**: Clinical linguistics, language assessment, corpus comparison, translation quality evaluation.
- **General formula**: For a window of size $W$: $MATTR = \frac{1}{N-W+1} \sum_{i=1}^{N-W+1} TTR(w_i, ..., w_{i+W-1})$.
- **Our implementation**: Window size $W = 100$ words. For each position, compute TTR of the 100-word window. Average all window TTRs. For documents shorter than 100 words, use the full document. Implemented in `lexical_metrics.py`, function `_mattr()`.
- **Expected results**: Values typically range from 0.6 to 0.8. Higher values indicate more diverse vocabulary usage. English and Spanish should show similar MATTR values if the translation maintains equivalent diversity. Our results show MATTR around 0.69-0.72 for all corpora, indicating natural, diverse text.
- **Why we chose it**: MATTR is one of the most recommended lexical diversity metrics in recent literature because it is truly independent of text length and easy to interpret.

### 3.5 Measure of Textual Lexical Diversity (MTLD)

- **What it measures**: The average number of consecutive words needed before the TTR drops below a threshold. Longer sequences before the TTR drops indicate higher lexical diversity.
- **Origin**: McCarthy and Jarvis (2010), *MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment*. Published in Behavior Research Methods.
- **Fields that use it**: Language assessment, second language acquisition, writing quality evaluation, clinical linguistics.
- **General formula**: Starting from the beginning of the text, compute TTR incrementally word by word. When TTR drops below a threshold (typically 0.72), count one "factor" and restart. MTLD = total tokens / number of factors. The computation is done in both forward and backward directions, and the results are averaged.
- **Our implementation**: Threshold = 0.72. We compute MTLD forward and backward, then average. Partial factors at the end are counted proportionally: $(1 - TTR_{remaining}) / (1 - threshold)$. Implemented in `lexical_metrics.py`, function `_mtld()`.
- **Expected results**: Higher values indicate more diverse vocabulary. Typical values range from 50 to 150 for natural text. Our results show MTLD of 63-91, which is in the expected range. Spanish MTLD is sometimes lower than English because Spanish has more function words and repeated morphological endings.
- **Why we chose it**: MTLD is considered the gold standard for lexical diversity in computational linguistics. It is robust to text length, validated extensively against human judgments, and widely cited.

---

## 4. Syntactic Metrics

These metrics analyze the grammatical structure of the text. They help verify that translations maintain the structural complexity of the original rather than simplifying sentence patterns.

### 4.1 Sentence Length (in words)

- **What it measures**: The average number of words per sentence, plus the distribution across all sentences.
- **Origin**: Basic readability research. Used in readability formulas since Flesch (1948).
- **Fields that use it**: Readability assessment, translation studies, computational linguistics.
- **General formula**: $\bar{L} = \frac{\text{total words}}{\text{total sentences}}$.
- **Our implementation**: We split text into sentences using `sentence_split()` (regex on `.!?`), tokenize each sentence, and compute statistics (mean, median, std, min, max) over all sentence lengths. We also compute the average sentence length per document for distribution analysis. Implemented in `syntactic_metrics.py`.
- **Expected results**: English typically has shorter sentences than Spanish for equivalent content. Academic/encyclopedia text (WikiText) should have moderate sentence length (15-25 words). Parliamentary text (Europarl) tends to have longer, more complex sentences (20-30+ words). FEVER passages should have moderate sentence length.

### 4.2 POS Distribution (Part-of-Speech)

- **What it measures**: The proportion of each grammatical category (nouns, verbs, adjectives, adverbs, etc.) in the corpus.
- **Origin**: Part-of-speech tagging has been a core NLP task since the 1960s. Automated POS tagging was pioneered by the Brown Corpus project (Francis and Kucera, 1967).
- **Fields that use it**: Computational linguistics, stylometry, language typology, machine translation evaluation.
- **General formula**: For each POS tag $p$: $\text{proportion}(p) = \frac{count(p)}{\sum_q count(q)}$.
- **Our implementation**: Requires spaCy with language models (`en_core_web_sm` and `es_core_news_sm`). We process the first 2,000 characters of each document for efficiency, then count POS tags (excluding spaces and punctuation). Results are reported as both counts and percentages. Implemented in `syntactic_metrics.py`.
- **Expected results**: English and Spanish have different POS distributions by nature: Spanish uses more determiners and pronouns (due to articles with gender), more verb forms (due to conjugation), and a different adjective-noun ordering. A good translation should show the expected cross-lingual POS differences rather than mirroring the English distribution exactly.
- **Note**: This metric requires spaCy to be installed. If spaCy is not available, the metric is skipped and a note is recorded in the output.

---

## 5. Quality and Semantic Metrics

These metrics assess the quality, consistency, and semantic characteristics of the corpus. They are especially important for validating that the translation pipeline produced reliable output.

### 5.1 Expansion Ratio (EN to ES)

- **What it measures**: How much longer the Spanish translation is compared to the English original, measured in characters.
- **Origin**: Well-known in translation and localization industry. Documented in localization guides (e.g., Microsoft, IBM localization standards).
- **Fields that use it**: Machine translation evaluation, software localization, typesetting.
- **General formula**: $R = \frac{len(\text{translation})}{len(\text{original})}$ per document.
- **Our implementation**: Computed per document, then aggregated (mean, median, std, min, max). The full distribution is stored for plotting. Implemented in `quality_metrics.py`.
- **Expected results**: English to Spanish typically expands by 10-20% (ratio 1.10-1.20). A ratio significantly below 1.0 suggests truncation or missing content. A ratio above 1.3 might indicate the translator added extraneous text. Our results show ratios of 1.04-1.09 across corpora, which is reasonable. Europarl has a slightly lower ratio because parliamentary text has fixed formulas and proper nouns that don't expand.

### 5.2 Exact Duplicates

- **What it measures**: Documents with identical content (after lowercasing and stripping whitespace).
- **Origin**: Standard data quality check in dataset construction.
- **Fields that use it**: NLP dataset quality assurance, web crawling, search engine deduplication.
- **General formula**: Hash each document and count collisions.
- **Our implementation**: We compute MD5 hashes of `text.strip().lower()` for all documents and count duplicate hashes. Computed separately for English and Spanish sides. Implemented in `quality_metrics.py`.
- **Expected results**: Ideally 0 duplicates. A few duplicates in the English side might exist if the original corpus had near-identical articles that passed the selection filter. The Spanish side should have 0 duplicates because each article is translated independently.

### 5.3 Near-Duplicates

- **What it measures**: Documents that are very similar but not identical — detects paraphrases, minor edits, or articles about very similar topics.
- **Origin**: Broder (1997), *On the Resemblance and Containment of Documents*, introduced MinHash and Jaccard similarity for web-scale deduplication.
- **Fields that use it**: Web search, plagiarism detection, dataset deduplication (e.g., GPT training data cleaning).
- **General formula**: Create k-shingles (k-word sequences) for each document. Compute Jaccard similarity: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$. Documents with $J \geq 0.85$ are considered near-duplicates.
- **Our implementation**: We use 5-word shingles and compute Jaccard similarity between all pairs in a random sample of 500 documents (full pairwise comparison is O(N^2) and too slow for 10,000 documents). The sample uses a fixed seed (42) for reproducibility. Implemented in `quality_metrics.py`.
- **Expected results**: Should be 0 or very few pairs. If near-duplicates are found, they warrant manual inspection.

### 5.4 Stopword Percentage

- **What it measures**: The proportion of tokens that are function words (articles, prepositions, conjunctions, pronouns, etc.).
- **Origin**: Luhn (1958) introduced the concept of stopwords for information retrieval. The term was coined by Hans Peter Luhn at IBM.
- **Fields that use it**: Information retrieval, text classification, corpus quality assessment.
- **General formula**: $S\% = \frac{|\{w \in text : w \in stopwords\}|}{|text|} \times 100$.
- **Our implementation**: We use a curated list of ~60 high-frequency function words for each language. The lists are hardcoded rather than using an external library to avoid dependencies. Implemented in `quality_metrics.py`.
- **Expected results**: English typically has 30-40% stopwords, Spanish 35-45% (Spanish uses more articles and prepositions due to gender/number agreement). A translated corpus that shows natural stopword levels confirms that the translation is fluent and not overly literal.

### 5.5 Special Characters Percentage

- **What it measures**: The proportion of characters that are neither alphanumeric nor whitespace (punctuation, symbols, brackets, etc.).
- **Origin**: Data quality assessment.
- **Fields that use it**: Text preprocessing, OCR quality evaluation, web corpus cleaning.
- **General formula**: $\frac{|\{c : c \notin alnum \wedge c \notin space\}|}{len(text)} \times 100$.
- **Our implementation**: Per-character classification, averaged across documents. Reported separately for EN and ES.
- **Expected results**: Typically 2-5% for clean text. Europarl may be higher due to XML-like tags. Values above 10% might indicate formatting artifacts or uncleaned special tokens.

### 5.6 Uppercase Percentage

- **What it measures**: The proportion of characters that are uppercase — detects anomalies like ALL CAPS text or titles that weren't properly cleaned.
- **Origin**: Data quality assessment.
- **Fields that use it**: Text normalization, spam detection, OCR correction.
- **General formula**: $\frac{|\{c : c.isupper()\}|}{len(text)} \times 100$.
- **Our implementation**: Per-character classification, averaged across documents.
- **Expected results**: Typically 2-5% in normal running text. Higher values indicate titles, headings, or acronyms. Spanish should have slightly lower uppercase percentage than English because Spanish capitalizes fewer words (e.g., language names, days of the week are lowercase in Spanish).

### 5.7 Named Entity Recognition (NER)

- **What it measures**: The number and diversity of named entities (people, organizations, locations, dates, etc.) in the corpus.
- **Origin**: The MUC conferences (1987-1998) established NER as a standard NLP task. Modern NER is based on neural sequence labeling.
- **Fields that use it**: Information extraction, question answering, knowledge graph construction, translation evaluation.
- **General formula**: Apply a NER model, count entities by type, compute diversity as $\frac{unique\_entities}{total\_entities}$.
- **Our implementation**: Requires spaCy with language models. We process the first 2,000 characters of each document, extract entities, count them by type (PERSON, ORG, GPE, DATE, etc.), and report the top 30 most frequent entities. Entity diversity is computed as the ratio of unique entities to total entity mentions. Implemented in `quality_metrics.py`.
- **Expected results**: English and Spanish should have similar entity counts if the translation preserves proper nouns correctly. Entity diversity should be comparable. Differences in entity counts may reveal cases where the translator incorrectly translated or omitted named entities.
- **Note**: This metric requires spaCy to be installed.

### 5.8 Classification Distribution

- **What it measures**: The distribution of topic categories across the corpus, as assigned by the metadata extraction in Stage 6 of the pipeline.
- **Origin**: Internal metric — uses the classifications already computed during translation.
- **Fields that use it**: Dataset balance assessment, topic analysis.
- **Our implementation**: We count the occurrences of each `classification` value from the dataset entries.
- **Expected results**: Should reflect the stratified sampling in Stage 1. For example, FEVER should show ~66% sports content. WikiText should show a more balanced distribution across 14 categories.

---

## Metrics Not Implemented (and why)

The following metrics from the original list were considered but not included in the current implementation:

| Metric | Reason for exclusion |
|--------|---------------------|
| **HD-D** (Hypergeometric Distribution D) | Redundant with MTLD and MATTR — provides similar information with higher computational complexity. |
| **Herdan's C** | Mathematically equivalent to $\log(types) / \log(tokens)$ — provides no additional insight over RTTR. |
| **Guiraud's Index** | Identical to RTTR ($types / \sqrt{tokens}$) — keeping both would be redundant. |
| **Syntactic tree depth** | Requires full dependency parsing of every sentence — too computationally expensive for 12,000+ documents without significant added value over sentence length. |
| **Syntactic complexity** | Requires subordinate clause analysis via deep parsing — same cost concern as tree depth. |
| **Topic Modeling (LDA/BERTopic)** | Already have topic classifications from Stage 6 metadata. Running a separate topic model would be redundant. |
| **Topic Coherence** | Depends on topic modeling, which we don't run. |
| **Semantic similarity between documents** | Requires computing embeddings for all documents and pairwise cosine similarity — O(N^2) computation, expensive for large corpora. |
| **Semantic density** | Similar to semantic similarity — requires embeddings infrastructure. |
| **OOV (Out-of-Vocabulary)** | Highly dependent on the reference vocabulary chosen. LLM-generated text has very few true OOV words, making this metric uninformative. |
| **Spelling errors** | LLMs produce very few spelling errors. Spell-checkers generate many false positives on proper nouns, technical terms, and multilingual content, making the results noisy and unreliable. |

---

## Metrics Suggested for Future Work

| Metric | Description | What is needed | Effort | Value |
|--------|-------------|----------------|--------|-------|
| **BLEU** | Standard MT metric comparing against human reference translations. Library: `sacrebleu`. | A subset (50-100 articles) with professional human translations as reference. The metric itself is trivial to compute — the cost is entirely in producing the reference. | High (human translator needed) | High — it is the standard that paper reviewers expect to see. |
| **chrF / chrF++** | Character n-gram F-score for translation evaluation. Included in `sacrebleu`. | Same human reference as BLEU — once you have it, chrF is computed alongside BLEU at no extra cost. | Same as BLEU | High — more robust than BLEU for morphologically rich languages like Spanish, where word forms vary more. |
| **COMET** | Neural MT evaluation metric from Unbabel. Library: `unbabel-comet` (~2GB model download). | Same human reference + the COMET model. Does not require GPU but is slower than BLEU/chrF. | Medium (install model + have the reference) | Very high — state of the art, correlates better with human judgment than BLEU or chrF. |
| **Inter-model agreement** | Similarity between the 3 translation candidates before consensus. | Already available — the pipeline JSON files contain all 3 candidates per article. Only new code is needed to compute pairwise similarity (e.g., BLEU or cosine similarity between candidates). | Low — only code, no new data needed | Medium — interesting and original. Would show whether the 3 models produce genuinely diverse candidates (justifying the consensus stage) or very similar output (suggesting the consensus adds little). Few papers report this metric. |

### Practical recommendations

1. **Inter-model agreement** is the easiest to implement right now since all the data already exists in the pipeline JSON files. It would be an original contribution that validates the multi-model architecture.

2. **BLEU/chrF on a micro-subset** — Translating 30-50 short FEVER articles manually would be the fastest path to having a human reference. FEVER articles are short (median 152 words), so a professional translator could do 50 articles in a few hours. This would give you BLEU, chrF, and COMET scores for at least one corpus.

3. **COMET** adds the most value per effort if you already have the human reference, since it correlates ~0.95 with human judgment (vs ~0.85 for BLEU).

## Reassessment of Non-Implemented Metrics

After reviewing the metrics that were excluded from the current implementation, here is an updated assessment of whether any would add significant value if reconsidered:

| Metric | Worth implementing? | Reasoning |
|--------|-------------------|-----------|
| **Inter-model agreement** | **Yes** | Can be computed now from existing pipeline JSON data. Would validate that using 3 different model architectures produces genuinely diverse candidates, which is a core assumption of the pipeline. |
| **Semantic similarity between documents** | **Maybe** | Could detect if the corpus is repetitive or has clusters of very similar content. However, it requires an embedding model and O(N^2) pairwise computation for 12,000+ documents, which is expensive. A sampled version (e.g., 500 random pairs) might be practical. |
| **OOV (Out-of-Vocabulary)** | **No** | LLMs almost never generate out-of-vocabulary words. The metric would return ~0% with many false positives on proper nouns, technical terms, and code-switched text. Not informative for LLM-generated translations. |
| **Spelling errors** | **No** | Same problem as OOV. LLMs produce very few spelling errors, and spell-checkers generate excessive false positives on multilingual content, proper nouns, and domain-specific terminology. The noise-to-signal ratio makes this metric unreliable. |
| **Topic Coherence** | **No** | Already have topic classifications from Stage 6 metadata extraction. Running a separate topic model (LDA or BERTopic) on top of already-classified data would be redundant. The classification distribution metric already captures topic balance. |
| **Syntactic tree depth** | **No** | Requires full dependency parsing of every sentence in 12,000+ documents — computationally very expensive. The sentence length metric already captures the main signal of syntactic complexity. The additional insight from tree depth does not justify the cost. |

---

## Output Format

Each metrics run produces a JSON file with the following top-level structure:

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

Fields ending in `distribution_*` contain raw data arrays suitable for plotting histograms, scatter plots, or line charts. Fields ending in `_curve` contain `{x, y}` data points for line plots.

---

## Scope of Application: Per-Dataset vs Combined Metrics

Not all metrics are equally useful at both levels. Some are designed to characterize a single corpus and lose meaning when corpora from different domains are mixed together. Others only become interesting when applied across the entire combined dataset. This section explains which metrics are computed at which level and why.

### Metrics computed per dataset only

These metrics are computed independently for each of the three datasets (WikiText, FEVER, Europarl). They characterize the properties of each corpus individually and are compared side-by-side to understand how the datasets differ.

| Metric | Why per-dataset only |
|--------|---------------------|
| **Zipf exponent** | Mixing corpora from different domains (encyclopedic, factual, parliamentary) always produces a Zipf exponent close to 1.0, which is trivially expected. The per-dataset exponents are more informative because they reveal domain-specific patterns — Europarl is highly Zipfian (0.985) while WikiText shows more vocabulary diversity (0.897). |
| **MATTR / MTLD** | These are per-document metrics that are then averaged. Combining 10,000 short FEVER articles with 1,000 long WikiText articles would produce a weighted average dominated by FEVER's count, obscuring the meaningful per-dataset differences. |
| **TTR / RTTR / CTTR** | Corpus-level TTR is heavily dependent on corpus size. Combining corpora of different sizes would produce a TTR that reflects the total size more than the content diversity. |
| **Sentence length** | Mixing parliamentary sentences (mean 25 words) with factual passages (mean 21 words) produces an average that describes neither domain well. |
| **POS distribution** | Each domain has characteristic POS patterns (Europarl has more pronouns and auxiliaries, FEVER has more proper nouns). Mixing them dilutes the domain-specific signals that make POS analysis interesting. |
| **Hapax / Dis legomena** | In the combined corpus, many words that are hapax in one dataset will appear in another, making the hapax count artificially lower. The per-dataset hapax percentages are more meaningful for assessing vocabulary richness. |
| **Expansion ratio** | The expansion ratio depends on content type. Articles with many proper nouns (FEVER sports content) expand less than parliamentary speeches. The per-dataset ratios reveal these differences; a combined average would hide them. |
| **Stopwords / Special chars / Uppercase** | These percentages vary by domain (Europarl has 43% stopwords vs FEVER's 33%). A combined value would be a weighted average that describes no specific domain. |

### Metrics computed for the combined corpus

These metrics are computed across all three datasets together. They provide information that is only meaningful at the combined level — either because they measure relationships between datasets or because they characterize the corpus as a whole.

| Metric | Why it applies to the combined corpus | What it reveals |
|--------|--------------------------------------|-----------------|
| **Aggregate totals** (tokens, types, chars, docs) | The primary purpose is to dimension the full corpus for reporting. A paper needs to state "the corpus contains X million tokens across Y documents." | The total size and scale of the bilingual resource. |
| **Combined vocabulary size** | The combined vocabulary is not the sum of individual vocabularies — there is overlap. Measuring the actual combined size reveals how much vocabulary the three datasets share. | If the combined vocabulary is much smaller than the sum of parts, the datasets are redundant. If it is close to the sum, they are complementary. |
| **Vocabulary overlap** (pairwise Jaccard, shared types, exclusive types) | This metric only exists when comparing two or more datasets. It measures how much vocabulary is shared between each pair of datasets and how much each dataset contributes exclusively to the combined vocabulary. | High overlap between WikiText and FEVER is expected (both come from Wikipedia). Low overlap with Europarl would confirm that parliamentary text adds a different vocabulary domain. |
| **Cross-dataset duplicate detection** | Duplicates within a single dataset are detected per-dataset. But duplicates across datasets — for example, a Wikipedia article that appears in both WikiText and FEVER — can only be detected by comparing all datasets together. | Since WikiText and FEVER both source from Wikipedia, there is a real risk of cross-dataset overlap. This metric quantifies whether the selection process produced independent subsets. |
| **Combined vocabulary growth curve** | By processing the three datasets in sequence and tracking how the vocabulary grows, we can visualize whether each new dataset contributes fresh vocabulary or just repeats what was already seen. | A curve that keeps rising steeply when a new dataset starts indicates high complementarity. A curve that flattens suggests saturation or redundancy. The color-coded plot shows exactly where each dataset's contribution begins and ends. |
| **Combined classification distribution** | The per-dataset distributions show internal balance. The combined distribution shows the overall topical profile of the entire corpus, with per-dataset breakdowns showing which dataset contributes to each topic. | Important for downstream users who need to know the overall topic balance. The combined view makes it clear that sports content (from FEVER) dominates the corpus, while political content (from Europarl) and encyclopedic content (from WikiText) provide breadth. |

### Why some per-dataset metrics were NOT applied to the combined corpus

The decision to exclude certain metrics from the combined analysis is based on a simple principle: **a combined metric is only useful if it tells you something that the per-dataset metrics do not.**

For example:
- The **combined Zipf exponent** would be approximately 0.95 (the average of the three individual exponents weighted by token count). This number adds nothing — we already know each individual corpus follows Zipf's law.
- The **combined MATTR** would be approximately 0.69 (dominated by FEVER's 10,000 documents). This average hides the fact that Europarl has notably higher lexical diversity (0.72) than FEVER (0.67).
- The **combined sentence length** would be approximately 23 words — a meaningless average of parliamentary speeches, encyclopedia articles, and factual passages.

In contrast, the combined metrics listed above reveal genuinely new information: how much the datasets complement each other, whether there is cross-contamination, and what the full corpus looks like as a unified resource.

### Combined output format

The combined metrics are saved to `metrics_combined_{timestamp}.json` with the following structure:

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

## References

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
