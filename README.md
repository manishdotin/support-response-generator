# Support Response Generator

A command-line tool that suggests next words and short response continuations using a smoothed n-gram model trained on customer-support interactions.

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/37hp7c6nde/badge.svg)](https://msrit.ask.thili.ai/b/37hp7c6nde)

## Requirements

Python 3.12 (or later) and the packages in `requirements.txt`.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli hello
python -m app.cli suggest "thank you for contacting" --n 2 --words 10 --top-k 3
pytest -q
```

Use `--n 3` for a trigram model. `--words` controls the continuation length, and `--top-k` controls how many next-word options are shown. Use `--corpus PATH` to train from another plain-text corpus.

## Corpus and privacy

`data/support_corpus.txt` is a small **synthetic example corpus** included so the tool runs immediately; it is not represented as real historical customer data. `data/support_interactions_raw.txt` is its source. The preprocessing function in `app/core.py` replaces common email addresses, phone numbers, order/ticket references, and URLs before writing a prepared corpus. For real interactions, obtain authorization, remove direct and indirect identifiers, and review the output before using it. Do not include passwords, payment-card data, or other sensitive information.

To preprocess a source file:

```bash
python -c "from app.core import milestone_1; milestone_1('path/to/source.txt', 'data/support_corpus.txt')"
```

## Model

The model counts bigrams or trigrams, converts counts to probabilities with Add-k smoothing, ranks next-word suggestions, and generates continuations by sampling from the learned counts.
