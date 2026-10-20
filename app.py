import json
import pandas as pd
import streamlit as st
from retrieval import StartupRetriever, DATA_PATH, SOURCE, REQUIRED
from llm import answer_question

st.set_page_config(page_title="Startup Diligence Copilot - Dataset Research", layout="wide")
st.title("Startup Diligence Copilot: Dataset Research")
st.caption("Developed by Samisha Boywar | Indian Startups Top 300 | Dataset analysis prototype")
st.info("This module demonstrates dataset preparation, exploration, comparison and evidence-backed Q&A. Document ingestion, LangGraph orchestration and independent claim verification are future work, not implemented here.")
st.markdown(f"Dataset: [Ashish Raut - Indian Startups Top 300]({SOURCE}). Historical snapshot, not live company data.")
st.warning("Funding is not market size or valuation. Dataset completeness and retrieval similarity do not establish startup credibility.")

uploaded = st.sidebar.file_uploader("Optional: upload the original Startups1.csv", type=["csv"])
try:
    retriever = StartupRetriever(uploaded if uploaded is not None else DATA_PATH)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.code("python retrieval.py --download\nstreamlit run app.py", language="bash")
    st.write("Alternatively download Startups1.csv from the dataset link and upload it using the sidebar. The old startups.csv is not used.")
    st.stop()

df = retriever.df
columns = ["source_row"] + REQUIRED
quality, explore, compare, qa = st.tabs(["Data preparation", "Explore and analyse", "Compare startups", "Investor Q&A"])

with quality:
    st.subheader("Dataset preparation and audit")
    a, b, c = st.columns(3)
    a.metric("Raw records", retriever.report["raw_rows"])
    b.metric("Clean records", len(df))
    c.metric("Fields analysed", len(REQUIRED))
    st.write("Whitespace and missing-value markers are normalised. Currency symbols and commas are removed before numeric conversion. Missing or invalid numbers stay missing, not zero. Empty company names are removed; case-insensitive duplicate company names keep their first record.")
    st.write("Employee ranges remain text. Industry labels are split only for frequency analysis. No model training or credibility scoring is performed.")
    st.json(retriever.report)
    st.caption("CSV record IDs count logical records including the header as record 1; they are not physical line numbers when quoted cells contain line breaks. The original Sr.No. column, when present, is not used as a unique identifier.")
    st.download_button("Download cleaning report", json.dumps(retriever.report, indent=2, default=int), "cleaning_report.json", "application/json")
    st.download_button("Download cleaned dataset", df[columns].to_csv(index=False), "cleaned_startups.csv", "text/csv")

with explore:
    st.subheader("Explore the actual dataset")
    cities = st.multiselect("City", sorted(df.loc[df.City.ne(""), "City"].unique()))
    industry = st.text_input("Industry contains", placeholder="FinTech, Education, Health Care...")
    view = df.copy()
    if cities:
        view = view.loc[view.City.isin(cities)]
    if industry.strip():
        view = view.loc[view.Industries.str.contains(industry.strip(), case=False, regex=False)]
    st.write(f"Matching startups: {len(view)}")
    st.dataframe(view[columns], use_container_width=True, hide_index=True)
    st.download_button("Export filtered startups", view[columns].to_csv(index=False), "filtered_startups.csv", "text/csv")
    if not view.empty:
        known = view["Funding Amount in $"].dropna()
        st.write(f"Known funding entries: {len(known)} / {len(view)}")
        if not known.empty:
            st.metric("Total reported funding in selected records (USD)", f"${known.sum():,.0f}")
            st.metric("Median reported funding (USD)", f"${known.median():,.0f}")
        st.caption("These describe only the selected dataset records, not the Indian startup population or a market-size estimate. Missing funding is excluded; recorded zero values are retained.")
        st.write("Top cities (selected records)")
        st.bar_chart(view.loc[view.City.ne(""), "City"].value_counts().head(10))
        labels = view.Industries.str.split(",").explode().str.strip()
        st.write("Top industry labels (multi-label: one company can appear in several categories)")
        st.bar_chart(labels.loc[labels.ne("")].value_counts().head(10))
        st.write("Highest reported funding (selected records)")
        st.dataframe(view.dropna(subset=["Funding Amount in $"]).nlargest(10, "Funding Amount in $")[["source_row", "Company", "Funding Amount in $", "Funding Round", "No. of Investors"]], hide_index=True)

with compare:
    st.subheader("Exact company comparison")
    chosen = st.multiselect("Choose 2-4 companies", sorted(df.Company.unique()), max_selections=4)
    if len(chosen) >= 2:
        comparison = retriever.compare(chosen)
        st.dataframe(comparison.set_index("Company").T, use_container_width=True)
        st.download_button("Export comparison", comparison.to_csv(index=False), "startup_comparison.csv", "text/csv")
        if st.button("Summarise this comparison with AI"):
            with st.spinner("Preparing dataset-backed comparison..."):
                st.write(answer_question("Compare these selected companies using only their supplied fields. Identify missing evidence for diligence; do not recommend an investment.", comparison))
    else:
        st.write("Choose at least two companies. Comparison uses exact selections, not approximate text search.")

with qa:
    st.subheader("Evidence-backed investor research")
    st.caption("Try: fintech payments; healthcare in Bengaluru; education startups. Retrieval is lexical TF-IDF, not a full-dataset SQL engine. Use the Explore tab for exhaustive filters, counts and funding aggregates.")
    question = st.text_input("Question")
    k = st.slider("Evidence records", 1, 10, 5)
    if st.button("Retrieve evidence and answer"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            results = retriever.retrieve(question, k=k)
            st.dataframe(results[columns + ["similarity"]], use_container_width=True, hide_index=True)
            st.caption("Similarity measures text overlap, not investment quality. Answers cover the retrieved subset, not necessarily every relevant startup. Record citations show provenance but do not guarantee factual correctness.")
            with st.spinner("Preparing answer..."):
                st.write(answer_question(question, results))
            st.write("Evidence still required for full diligence: pitch decks, audited financials, revenue/customer evidence, founder and legal records, independent funding sources, and dated market research for TAM/SAM/SOM.")
