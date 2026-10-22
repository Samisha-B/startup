import json
from pathlib import Path
import streamlit as st
from retrieval import REQUIRED
from llm import answer_question

COLUMNS = ["source_row"] + REQUIRED
FUNDING = "Funding Amount in $"


def apply_styles():
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def header():
    st.caption("STARTUP ATLAS / RESEARCH WORKSPACE")
    st.title("Discover the story behind the startup.")
    st.write("Explore Indian startups, compare companies, and ask questions with dataset evidence.")


def section(title, description):
    st.subheader(title)
    st.caption(description)


def metric_cards(items):
    for column, (label, value) in zip(st.columns(len(items)), items):
        with column:
            with st.container(border=True):
                st.metric(label, value)


def overview(df):
    metric_cards([("Startups indexed", f"{len(df):,}"), ("Cities represented", f"{df.loc[df.City.ne(''), 'City'].nunique():,}"), ("Funding coverage", f"{df[FUNDING].notna().sum():,} / {len(df):,}")])


def table(df):
    st.dataframe(df, use_container_width=True, hide_index=True)


def export_csv(label, df, filename, key):
    st.download_button(label, df.to_csv(index=False), filename, "text/csv", key=key)


def generate_answer(question, evidence, key):
    try:
        with st.spinner("Preparing an evidence-backed response…"):
            text = answer_question(question, evidence)
        st.session_state[key] = {"text": text}
    except Exception:
        st.session_state[key] = {"error": True}


def show_answer(key):
    result = st.session_state.get(key)
    if not result:
        return
    with st.container(border=True):
        if result.get("error"):
            st.error("AI response unavailable. Check your local API configuration and try again. Dataset exploration and evidence retrieval still work without AI.")
        else:
            st.markdown(result["text"])


def render_explore(df):
    section("Find your next research lead", "Filter the dataset, inspect records, and export your shortlist.")
    with st.container(border=True):
        left, middle, right = st.columns([2, 2, 3])
        cities = left.multiselect("City", sorted(df.loc[df.City.ne(""), "City"].unique()), placeholder="All cities")
        industry = middle.text_input("Industry contains", placeholder="FinTech, education…")
        company = right.text_input("Company name contains", placeholder="Search companies…")
    view = df.copy()
    if cities:
        view = view.loc[view.City.isin(cities)]
    for field, term in [("Industries", industry), ("Company", company)]:
        if term.strip():
            view = view.loc[view[field].str.contains(term.strip(), case=False, regex=False)]
    st.caption(f"{len(view):,} of {len(df):,} startups match your filters")
    if view.empty:
        st.info("No startups match these filters. Clear a city or try a broader industry or company name.")
        return
    known = view[FUNDING].dropna()
    metric_cards([("Matching startups", f"{len(view):,}"), ("Reported funding · USD", f"${known.sum():,.0f}" if not known.empty else "Not available"), ("Median funding · USD", f"${known.median():,.0f}" if not known.empty else "Not available")])
    st.caption(f"Funding known for {len(known)} / {len(view)} selected records. Missing values are excluded; recorded zeros are retained. These are dataset aggregates, not market-size estimates.")
    records, insights = st.tabs(["Startup directory", "Dataset insights"])
    with records:
        table(view[COLUMNS])
        export_csv("Download shortlist CSV", view[COLUMNS], "filtered_startups.csv", "export_explore")
    with insights:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("### Top cities")
                counts = view.loc[view.City.ne(""), "City"].value_counts().head(10)
                if counts.empty:
                    st.info("No city data in this selection.")
                else:
                    st.bar_chart(counts, color="#6366f1")
        with right:
            with st.container(border=True):
                st.markdown("### Top industry labels")
                labels = view.Industries.str.split(",").explode().str.strip()
                counts = labels.loc[labels.ne("")].value_counts().head(10)
                if counts.empty:
                    st.info("No industry labels in this selection.")
                else:
                    st.bar_chart(counts, color="#14b8a6")
                st.caption("Multi-label: a company can appear in several categories.")
        st.markdown("### Highest reported funding")
        table(view.dropna(subset=[FUNDING]).nlargest(10, FUNDING)[["source_row", "Company", FUNDING, "Funding Round", "No. of Investors"]])


def render_compare(retriever):
    section("Compare companies side by side", "Choose 2–4 exact company records. No approximate matching or investment ranking.")
    chosen = st.multiselect("Companies to compare", sorted(retriever.df.Company.unique()), max_selections=4, placeholder="Select at least two companies")
    signature = tuple(sorted(chosen))
    if st.session_state.get("comparison_selection") != signature:
        st.session_state.pop("comparison_answer", None)
        st.session_state["comparison_selection"] = signature
    if len(chosen) < 2:
        st.info("Build a comparison by selecting at least two companies above.")
        return
    comparison = retriever.compare(chosen)
    with st.container(border=True):
        st.dataframe(comparison.set_index("Company").T, use_container_width=True)
    export_csv("Download comparison CSV", comparison, "startup_comparison.csv", "export_compare")
    if st.button("Generate comparison summary", type="primary"):
        generate_answer("Compare these selected companies using only their supplied fields. Identify missing evidence for diligence; do not recommend an investment.", comparison, "comparison_answer")
    show_answer("comparison_answer")


def render_qa(retriever):
    section("Ask a better research question", "Retrieve relevant records using lexical TF-IDF, then generate a dataset-grounded answer.")
    st.caption("Try: fintech payments · healthcare in Bengaluru · education startups. Use Explore for exhaustive filters, counts, and aggregates.")
    with st.form("research_question"):
        question = st.text_area("Research question", placeholder="Which startups work on digital payments?", height=100)
        k = st.slider("Evidence records", 1, 10, 5)
        submitted = st.form_submit_button("Find evidence and answer", type="primary")
    if submitted:
        st.session_state.pop("research_answer", None)
        st.session_state.pop("research_evidence", None)
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            results = retriever.retrieve(question, k=k)
            st.session_state["research_evidence"] = {"question": question, "records": results}
            if not results.empty:
                generate_answer(question, results, "research_answer")
    saved = st.session_state.get("research_evidence")
    if saved is not None:
        st.markdown("### Submitted question")
        st.write(saved["question"])
        if saved["records"].empty:
            st.info("No text matches found. Try an industry, city, or company name from the dataset.")
        else:
            st.markdown("### Research answer")
            show_answer("research_answer")
            with st.expander("Inspect source records", expanded=True):
                table(saved["records"][COLUMNS + ["similarity"]])
            st.caption("Similarity measures text overlap, not investment quality. Answers cover only retrieved records. Citations show provenance, not a guarantee of correctness.")
    with st.expander("What this research does not verify"):
        st.write("Full diligence still needs pitch decks, audited financials, customer and revenue evidence, founder and legal records, independent funding sources, and dated TAM/SAM/SOM research.")


def render_audit(retriever):
    section("Understand your evidence", "Inspect cleaning decisions, missing values, and the provenance of your records.")
    metric_cards([("Raw records", retriever.report["raw_rows"]), ("Clean records", len(retriever.df)), ("Fields analysed", len(REQUIRED))])
    with st.container(border=True):
        st.markdown("### Cleaning methodology")
        st.write("Whitespace and missing-value markers are normalised. Currency symbols and commas are removed before numeric conversion. Missing or invalid numbers stay missing, not zero. Empty company names are removed; case-insensitive duplicates keep their first record.")
        st.write("Employee ranges remain text. Industry labels are split only for frequency analysis. No model training or credibility scoring is performed.")
    with st.expander("View full cleaning report"):
        st.json(retriever.report)
    st.caption("Source record IDs count logical CSV records, with the header as record 1, not physical lines inside quoted cells. Original Sr.No. values are not used as unique identifiers.")
    st.download_button("Download cleaning report", json.dumps(retriever.report, indent=2, default=int), "cleaning_report.json", "application/json")
    export_csv("Download cleaned dataset", retriever.df[COLUMNS], "cleaned_startups.csv", "export_clean")
