# Startup Dataset Q&A Bot

A simple AI bot that answers questions about Indian startups using a Kaggle dataset, TF-IDF + cosine similarity retrieval, and OpenAI for grounded answer synthesis.

This is a simplified, single-purpose rebuild of the original startup-diligence-copilot concept, scoped to Unit 1 information retrieval fundamentals (TF-IDF, vector space model, cosine similarity) plus one LLM call for natural-language answers.

## How it works

1. Data: data/startups.csv - the Startups in India dataset from Kaggle.
2. Retrieval (retrieval.py): Builds a TF-IDF matrix over each startup's sector + profile text. A user question is vectorized with the same vocabulary, and cosine similarity ranks all startups against it. Works fully offline, no API key required.
3. Answer synthesis (llm.py): Top-k retrieved rows are passed as context into an OpenAI chat completion call, which answers strictly from that context and cites which row(s) it used.
4. CLI (main.py): Ties both stages together into a question-and-answer loop.

## Setup

```
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your OPENAI_API_KEY
python main.py
```

## Testing retrieval alone (no API key needed)

```
python retrieval.py
```

## Dataset source

Startups in India Dataset - Kaggle (anshtanwar/list-of-startups)
