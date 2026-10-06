# Financial News to Market Events with Lightweight NLP

An end-to-end pipeline that turns raw financial disclosures into structured `(date, ticker, impact summary)` events that can be joined with market data.
The whole pipeline is **frugal**: a Flan-T5 model fine-tuned in about 2 GPU-hours, instead of the Mixtral 8×7B teacher that produced the reference labels.

M1 Data Science NLP project, Université Paris-Saclay (February 2026).
Team: Baptiste Pras, Martin Leiva, Vladimir Herrera-Nativi, Javier Peña-Castaño.
Full write-up: [docs/report.pdf](docs/report.pdf) · Slides: [docs/slides.pdf](docs/slides.pdf)

## Pipeline

```text
raw disclosures ─> cleaning ─┬─> map-reduce summarization (Flan-T5) ─> impact summary ─┐
                             │                                                          ├─> (date, ticker, summary) events ─> market returns
                             └─> NER (BERT) ─> ticker linking (yfinance + fuzzy match) ─┘
```

1. **Data audit and cleaning**: missing and empty fields, date checks, duplicates, outliers, a summary-leakage (extractiveness) check, FastText language ID to keep English only, regex removal of boilerplate (safe-harbor, forward-looking statements, exhibits), and a chronological train/test split to avoid leakage.
2. **Impact summarization** (`train_chunk_reduce.py`): long filings are split into overlapping 768-token chunks.
   - *Map*: a notes model extracts factual bullet points per chunk (who, what, figures, counterparties).
   - *Reduce*: notes are recursively merged into a 2–4 sentence impact summary, so the input never overflows the context.
   - Training is a student-teacher distillation of the Mixtral summaries, with a higher loss weight on the first chunk (key figures are usually front-loaded) and prompts that require the issuer to be named and numbers to be kept verbatim.
3. **LLM-as-a-judge evaluation** (`evaluate.py`): ROUGE rewards wording overlap, not factual correctness. So a local Qwen2.5-7B-Instruct audits each (source, summary) pair on six 0–5 criteria.
4. **Entity extraction and linking** (`data_audit_and_entity_linking.ipynb`): a BERT NER model finds organizations, which are matched to listed tickers via yfinance metadata and fuzzy string matching (rapidfuzz). Tickers below a coverage threshold are dropped, keeping about 30% of the articles with reliable resolution.

## Results

On the 267 held-out (most recent) articles:

| Criterion (LLM judge, 0–5) | Mean |
|---|---|
| Accuracy | 3.00 |
| Issuer grounding | 3.17 |
| Numeric fidelity | 3.01 |
| Coverage | 3.07 |
| Conciseness | 2.08 |
| Anti-filler professionalism | 2.38 |

ROUGE-L F1: **0.276**. An earlier model scored slightly higher on ROUGE (0.29) but dropped key figures, so model selection used the judge scores instead.
Per-article predictions and judge rationales are in [`results/`](results/).

## Setup

```bash
pip install -r requirements.txt
```

- **Data**: [High-Quality Financial News Dataset for NLP Tasks](https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks/data) (Kaggle). Save it as `dataset.csv` in the repo root.
- **Language ID model**: download FastText's [`lid.176.bin`](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin) to the repo root.

## Usage

```bash
# Train the summarizer and write predictions with ROUGE-L (GPU recommended)
python train_chunk_reduce.py --data dataset.csv --pred_csv results/pred.csv --fp16

# Score the predictions with the LLM judge
python evaluate.py --input_csv results/pred.csv --out_json results/judge_results.json
```

The data audit and the NER and ticker-linking steps are in `data_audit_and_entity_linking.ipynb`.

## Repository structure

```text
├── train_chunk_reduce.py                # chunking, map-reduce summarizer, training, ROUGE-L
├── evaluate.py                          # LLM-as-a-judge evaluation (Qwen2.5-7B-Instruct)
├── data_audit_and_entity_linking.ipynb  # dataset audit, cleaning, NER, ticker linking, split
├── results/                             # predictions and judge scores on the test set
└── docs/                                # report and slides
```

## Status

Cleaning, summarization, evaluation, ticker linking and market-data retrieval work end to end. The regression from summary features to short-horizon abnormal returns is the next step and is not implemented here.
