import base64
import html
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from retrieval import REQUIRED, SOURCE
from llm import answer_question

COLUMNS = ["source_row"] + REQUIRED
FUNDING = "Funding Amount in $"
ASSETS = Path(__file__).resolve().parent.parent / "assets"
QUERY_MODES = {
    "Free-form question": None,
    "Find startups by sector": "Which startups in the dataset work in the {sector} sector?",
    "Compare two startups": "Compare the business models of {a} and {b} based on their profiles.",
    "Shortlist top-k for investment thesis": "Shortlist the top startups that best match this investment thesis: {thesis}",
}


def esc(value):
    return html.escape("" if value is None or pd.isna(value) else str(value))


def render(markup):
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def background_css():
    for name in ("background.jpg", "background.jpeg", "background.png"):
        path = ASSETS / name
        if path.exists():
            mime = "image/png" if path.suffix == ".png" else "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode()
            return (
                ".stApp{background:linear-gradient(rgba(10,18,28,.55),rgba(10,18,28,.78)),"
                f"url('data:{mime};base64,{data}') left top / 125% auto no-repeat fixed #0e1620;}}"
            )
    return ".stApp{background:linear-gradient(160deg,#16202b,#0e1620);}"


def apply_styles():
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown("<style>" + css + background_css() + "</style>", unsafe_allow_html=True)


def sidebar_info(retriever):
    with st.sidebar:
        render('<div class="side-title">Startup Diligence Copilot</div>')
        render('<div class="side-note">Dataset research on Indian startups.</div>')
        render('<div class="side-label">Dataset</div>')
        st.markdown(f"[Indian Startups Top 300]({SOURCE}) by Ashish Raut")
        render(f'<div class="side-note">{len(retriever.df)} clean records loaded. Historical snapshot, not live company data.</div>')
        render('<div class="side-label">Author</div>')
        render('<div class="side-note">Samisha Boywar</div>')


def hero():
    render(
        '<div class="hero"><div class="hero-eyebrow">Dataset research</div>'
        "<h1>Startup Diligence Copilot</h1>"
        "<p>Explore, compare and question the Indian Startups Top 300 dataset. Each answer cites the dataset record it relies on.</p></div>"
    )


def notice(kind, text):
    render(f'<div class="notice {kind}">{text}</div>')


def section(title, subtitle):
    render(f'<div class="sec-title">{esc(title)}</div><div class="sec-sub">{esc(subtitle)}</div>')


def stat_cards(items):
    cards = "".join(f'<div class="stat"><div class="stat-label">{esc(label)}</div><div class="stat-value">{esc(value)}</div></div>' for label, value in items)
    render(f'<div class="stats">{cards}</div>')


def empty_state(title, text):
    render(f'<div class="empty"><b>{esc(title)}</b>{esc(text)}</div>')


def money(value):
    return "N/A" if pd.isna(value) else f"${value:,.0f}"


def whole(value):
    return "N/A" if pd.isna(value) else f"{int(value)}"


def startup_card(row, rank=None, similarity=None):
    employees = esc(row["No. of Employees"])
    parts = [esc(row["City"])]
    if pd.notna(row["Starting Year"]):
        parts.append(f"Founded {whole(row['Starting Year'])}")
    if employees:
        parts.append(f"{employees} employees")
    meta = " &middot; ".join(p for p in parts if p)
    tags = [t.strip() for t in str(row["Industries"]).split(",") if t.strip()][:4]
    pills = "".join(f'<span class="tag">{esc(t)}</span>' for t in tags)
    text = "" if pd.isna(row["Description"]) else str(row["Description"])
    description = esc(text[:330] + ("..." if len(text) > 330 else "")) or "No description in dataset."
    label = f'Record {int(row["source_row"])}' if rank is None else f'#{rank} &middot; Record {int(row["source_row"])}'
    bar = ""
    if similarity is not None:
        width = max(4, min(100, similarity * 100))
        bar = f'<div class="sim"><div style="width:{width:.0f}%"></div></div><div class="sc-meta">Similarity {similarity:.3f}</div>'
    founders = esc(row["Founders"]) or "N/A"
    facts = (
        f'<div class="fact"><b>{money(row[FUNDING])}</b><span>Funding</span></div>'
        f'<div class="fact"><b>{whole(row["Funding Round"])}</b><span>Round</span></div>'
        f'<div class="fact"><b>{whole(row["No. of Investors"])}</b><span>Investors</span></div>'
        f'<div class="fact"><b>{founders}</b><span>Founders</span></div>'
    )
    render(
        f'<div class="sc"><div class="sc-top"><div><div class="sc-name">{esc(row["Company"])}</div><div class="sc-meta">{meta}</div></div>'
        f'<div style="text-align:right"><div class="rank">{label}</div>{bar}</div></div>'
        f'<div class="sc-desc">{description}</div><div>{pills}</div><div class="facts">{facts}</div></div>'
    )


def table(df):
    st.dataframe(df, use_container_width=True, hide_index=True)


def export_csv(label, df, filename, key):
    st.download_button(label, df.to_csv(index=False), filename, "text/csv", key=key)


def answer_panel(text):
    with st.container(border=True):
        render('<div class="ans-head">Grounded answer</div>')
        st.markdown(text)


def overview(df):
    cities = df.loc[df.City.ne(""), "City"].nunique()
    stat_cards([("Startups", f"{len(df):,}"), ("Cities", f"{cities:,}"), ("Funding known", f"{df[FUNDING].notna().sum():,} / {len(df):,}")])


def render_audit(retriever):
    df = retriever.df
    section("Data preparation and audit", "How the dataset was cleaned before analysis.")
    stat_cards([("Raw records", retriever.report["raw_rows"]), ("Clean records", len(df)), ("Fields analysed", len(REQUIRED))])
    st.write("Whitespace and missing-value markers are normalised. Currency symbols and commas are removed before numeric conversion. Missing or invalid numbers stay missing, not zero. Empty company names are removed; case-insensitive duplicate company names keep their first record.")
    st.write("Employee ranges remain text. Industry labels are split only for frequency analysis. No model training or credibility scoring is performed.")
    with st.expander("Full cleaning report"):
        st.json(retriever.report)
    st.caption("CSV record IDs count logical records including the header as record 1; they are not physical line numbers when quoted cells contain line breaks. The original Sr.No. column, when present, is not used as a unique identifier.")
    left, right = st.columns(2)
    left.download_button("Download cleaning report", json.dumps(retriever.report, indent=2, default=int), "cleaning_report.json", "application/json", key="dl_report")
    right.download_button("Download cleaned dataset", df[COLUMNS].to_csv(index=False), "cleaned_startups.csv", "text/csv", key="dl_clean")


def render_explore(df):
    section("Explore the dataset", "Filter by city and industry, then look at funding and categories.")
    a, b = st.columns(2)
    cities = a.multiselect("City", sorted(df.loc[df.City.ne(""), "City"].unique()), placeholder="All cities")
    industry = b.text_input("Industry contains", placeholder="FinTech, Education, Health Care...")
    view = df.copy()
    if cities:
        view = view.loc[view.City.isin(cities)]
    if industry.strip():
        view = view.loc[view.Industries.str.contains(industry.strip(), case=False, regex=False)]
    known = view[FUNDING].dropna()
    stat_cards([
        ("Matching startups", f"{len(view):,}"),
        ("Total reported funding", f"${known.sum():,.0f}" if not known.empty else "N/A"),
        ("Median funding", f"${known.median():,.0f}" if not known.empty else "N/A"),
        ("Known funding entries", f"{len(known)} / {len(view)}"),
    ])
    if view.empty:
        empty_state("No startups match these filters", "Clear a city or try a broader industry term.")
        return
    directory, cards, charts, top = st.tabs(["Table", "Cards", "Charts", "Top funded"])
    with directory:
        table(view[COLUMNS])
        export_csv("Export filtered startups", view[COLUMNS], "filtered_startups.csv", "dl_filtered")
    with cards:
        limit = 12
        for _, row in view.head(limit).iterrows():
            startup_card(row)
        if len(view) > limit:
            st.caption(f"Showing the first {limit} of {len(view)} matches. Use the table for all records.")
    with charts:
        left, right = st.columns(2)
        with left:
            st.markdown("##### Top cities")
            counts = view.loc[view.City.ne(""), "City"].value_counts().head(10)
            if counts.empty:
                st.caption("No city data in this selection.")
            else:
                st.bar_chart(counts, color="#1f3a4d")
        with right:
            st.markdown("##### Top industry labels")
            labels = view.Industries.str.split(",").explode().str.strip()
            counts = labels.loc[labels.ne("")].value_counts().head(10)
            if counts.empty:
                st.caption("No industry data in this selection.")
            else:
                st.bar_chart(counts, color="#9a7b4f")
            st.caption("Multi-label: one company can appear in several categories.")
    with top:
        table(view.dropna(subset=[FUNDING]).nlargest(10, FUNDING)[["source_row", "Company", FUNDING, "Funding Round", "No. of Investors"]])
    st.caption("These describe only the selected dataset records, not the Indian startup population or a market-size estimate. Missing funding is excluded; recorded zero values are retained.")


def render_compare(retriever):
    section("Compare startups", "Exact company selections, side by side.")
    chosen = st.multiselect("Choose 2-4 companies", sorted(retriever.df.Company.unique()), max_selections=4, placeholder="Search and select companies")
    if st.session_state.get("cmp_key") != tuple(sorted(chosen)):
        st.session_state["cmp_key"] = tuple(sorted(chosen))
        st.session_state.pop("cmp_answer", None)
    if len(chosen) < 2:
        empty_state("Pick at least two companies", "Comparison uses exact selections, not approximate text search.")
        return
    comparison = retriever.compare(chosen)
    for column, (_, row) in zip(st.columns(len(comparison)), comparison.iterrows()):
        with column:
            startup_card(row)
    with st.expander("Detailed comparison table"):
        st.dataframe(comparison.set_index("Company").T, use_container_width=True)
    export_csv("Export comparison", comparison, "startup_comparison.csv", "dl_compare")
    if st.button("Summarise this comparison with AI", type="primary"):
        with st.spinner("Preparing dataset-backed comparison..."):
            st.session_state["cmp_answer"] = answer_question("Compare these selected companies using only their supplied fields. Identify missing evidence for diligence; do not recommend an investment.", comparison)
    if st.session_state.get("cmp_answer"):
        answer_panel(st.session_state["cmp_answer"])


def render_qa(retriever):
    section("Investor Q&A", "Evidence-backed research. Retrieval is lexical TF-IDF, not a full-dataset SQL engine.")
    mode = st.selectbox("Query mode", list(QUERY_MODES.keys()))
    question = None
    if mode == "Free-form question":
        question = st.text_input("Ask a question about the startups in this dataset", placeholder="fintech payments; healthcare in Bengaluru; education startups")
    elif mode == "Find startups by sector":
        sector = st.text_input("Sector", placeholder="Healthcare, Agritech, Fintech")
        question = QUERY_MODES[mode].format(sector=sector) if sector.strip() else None
    elif mode == "Compare two startups":
        c1, c2 = st.columns(2)
        a, b = c1.text_input("Startup A name"), c2.text_input("Startup B name")
        question = QUERY_MODES[mode].format(a=a, b=b) if a.strip() and b.strip() else None
    else:
        thesis = st.text_area("Describe your investment thesis", placeholder="B2B SaaS in Bengaluru with strong investor interest")
        question = QUERY_MODES[mode].format(thesis=thesis) if thesis.strip() else None
    k = st.slider("Number of rows to retrieve (top-k)", 1, 10, 5)
    go = st.button("Get Answer", type="primary")
    if not go:
        empty_state("Ask a question", "Choose a query mode, fill in the fields and press Get Answer.")
        return
    if not question:
        notice("warn", "Fill in the fields above first.")
        return
    results = retriever.retrieve(question, k=k)
    st.markdown("### Retrieved evidence")
    st.caption("Similarity measures text overlap, not investment quality. Answers cover the retrieved subset, not necessarily every relevant startup. Record citations show provenance but do not guarantee factual correctness.")
    if results.empty:
        empty_state("No matching records", "Try an industry, city or company name from the dataset.")
    else:
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            startup_card(row, rank=rank, similarity=float(row["similarity"]))
        with st.expander("View evidence as table"):
            table(results[COLUMNS + ["similarity"]])
    with st.spinner("Preparing answer..."):
        answer = answer_question(question, results)
    answer_panel(answer)
    st.caption("Evidence still required for full diligence: pitch decks, audited financials, revenue/customer evidence, founder and legal records, independent funding sources, and dated market research for TAM/SAM/SOM.")
