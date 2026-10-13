# Supplementary material

The supplementary material of the paper, as submitted to the journal. Section S1 holds the
per-dataset measures that the paper summarizes in its text; Section S2 holds the configuration
of the production run (implementation and hardware, sampling categories, parameters, and prompts).

| Item | Content |
|---|---|
| Figure S1 | Zipf distribution (log-log) per dataset and language |
| Table S1 | General corpus statistics: words, unique words, words per document |
| Table S2 | Lexical diversity: TTR, MATTR, MTLD |
| Table S3 | Sentence length |
| Table S4 | Quality indicators: expansion ratio, duplicates, stopwords, special characters |
| Table S5 | Vocabulary growth as the datasets are added in sequence |
| Table S6 | Pipeline parameters |
| Table S7 | Prompt inventory: structure and corpus-specific variants |
| Table S8 | Topic categories offered to the metadata model, per corpus |

`supplementary.pdf` is the file uploaded with the paper. `supplementary.tex` is its source;
it compiles on its own with `pdflatex supplementary.tex` and reads its figure from
`graphics/`.

The measures of Section S1, with more measures and the anomalies behind them, are discussed in
[../metrics-results.md](../metrics-results.md). They are produced by notebook 06
(`../../notebooks/06_corpus_metrics.ipynb`) from the files in `../../corpus/metrics/`.
