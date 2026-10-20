# Startup Diligence Copilot - Dataset Research

Developed by Samisha Boywar.

A simple Python/Streamlit dataset-analysis prototype supporting the broader Startup Diligence Copilot problem statement: help investors ingest documents, assess claims against evidence, ask questions, compare startups and prepare research summaries.

## What this repository implements now

- Loads the Indian Startups Top 300 CSV with explicit schema validation.
- Cleans whitespace, missing markers, currency strings and invalid numeric values; reports missingness and removed records.
- Preserves missing numbers rather than replacing them with zero; keeps employee ranges as text.
- Removes unnamed companies and case-insensitive duplicate names, retaining the first record. The audit reports these removals.
- Shows city/industry distributions, funding summaries, filters, top-funded records and CSV exports.
- Compares 2-4 explicitly selected companies using their actual dataset fields.
- Uses TF-IDF unigrams/bigrams and cosine similarity for lexical search, with optional OpenAI answer synthesis and source record identifiers.
- Works for exploration, comparisons and retrieval without an OpenAI API key.
- Displays project credit in both Streamlit and the CLI.

This is not the complete diligence system. LangGraph orchestration, document uploads/parsing, external claim verification, founder credibility assessment and market-size estimation are not implemented here. The Next.js frontend remains in the separate startup-diligence-copilot repository; this Python application is not yet wired into it.

## Dataset and provenance

Author: Ashish Raut. Dataset: Indian startups / Indian Startups Top 300.

Source: https://www.kaggle.com/datasets/ashishraut64/indian-startups-top-300

Kaggle describes 300 startups and lists the dataset under CC0: Public Domain. The description refers to 1984-2022 coverage; do not treat that as proof that every record is current through 2022 or today. Actual row counts and year values are read from the CSV at runtime.

Expected file: `data/Startups1.csv`.

Fields: Company, City, Starting Year, Founders, Industries, Description, No. of Employees, Funding Amount in $, Funding Round, No. of Investors. The source may also contain Sr.No.; the application uses its own stable source record position instead.

The old `data/startups.csv`, if present, is legacy data and is no longer loaded. No fabricated sample rows are substituted. The replacement CSV is downloaded explicitly rather than bundled in this update.

## Run locally

Use Python 3.10 or newer. In a terminal in this repository:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
python retrieval.py --download
python -m streamlit run app.py
```

The download command requests the public Kaggle dataset archive, extracts only Startups1.csv, checks its schema and saves it under data/. It needs internet access. If Kaggle returns a login page, blocks the request or changes its API, download Startups1.csv manually from the source link and put it in data/, or upload it through the Streamlit sidebar. Download errors are shown explicitly; there is no silent fallback to the old dataset.

Optional AI answers: copy `.env.example` to `.env` and set your own OPENAI_API_KEY. Do not commit API keys. OPENAI_MODEL is configurable and defaults to gpt-4o-mini. OpenAI requests can incur charges. Retrieved records and the question are sent to OpenAI only when a key is configured and an AI-answer action is used. Without a key, the app displays the dataset evidence instead.

CLI:

```bash
python main.py
```

Data-quality report and retrieval smoke test:

```bash
python retrieval.py
```

## Demonstrate dataset work to faculty

1. Open Data preparation: show raw/clean counts, missingness, invalid numeric values, duplicate handling and exported cleaning report.
2. Open Explore and analyse: filter cities/industries, show multi-label sector counts and funding summaries, and export filtered records.
3. Open Compare startups: select real company names and compare founding year, founders, employee ranges, funding, rounds and investors.
4. Open Investor Q&A: show retrieved records and similarity scores before any optional AI answer; trace citations back to the cleaned dataset.
5. Show unavailable evidence honestly: revenue, valuation, credibility and TAM/SAM/SOM require additional sources.

CSV record identifiers count logical records with the header as record 1; quoted multiline cells mean these are not physical line numbers. Identifiers remain tied to original input positions after cleaning. Changing the input file can change identifiers.

## Interpretation limits

- This is a historical, selected top-300 dataset, not a representative sample of all Indian startups.
- Funding totals describe the selected records only. They are not market size, revenue or valuation. Missing funding is excluded; recorded zeros are retained.
- Industry counts are multi-label: one company can contribute to several categories.
- TF-IDF similarity reflects text overlap, not factual truth, credibility or investment merit. It does not reliably execute numeric comparisons or exhaustive aggregation; use the structured filters and analysis tab for those.
- AI is instructed to cite dataset records and acknowledge gaps, but citations and prompts do not guarantee factual correctness. Review answers against the displayed evidence.
- Exact comparisons bypass approximate retrieval and include all selected companies.

## Hosting

For this Python prototype, use a Python/Streamlit-capable host, with app.py as the entry point and requirements.txt installed. Provide data/Startups1.csv during deployment and configure OPENAI_API_KEY only in the host's secret/environment settings if AI synthesis is needed. Sidebar upload is available for a temporary demonstration but is not persistent storage. Deployment has not been performed by this commit.

## Validation status for this update

Nine local fixture checks passed for currency conversion, missing funding, duplicate/unnamed-company removal, named-company retrieval, empty/no-match queries, exact comparison, source record IDs and rejection of the old schema. These used small synthetic fixtures, not fabricated production dataset contents. The actual Kaggle download, full dataset statistics, Streamlit runtime and live OpenAI API were not exercised in the editing environment. Test locally after pulling and downloading the real dataset.
