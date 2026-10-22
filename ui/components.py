import html
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from retrieval import REQUIRED
from llm import answer_question

COLUMNS = ["source_row"] + REQUIRED
FUNDING = "Funding Amount in $"
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


def apply_styles():
    st.markdown("<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def sidebar_brand():
    render('<div class="brand"><div class="brand-logo">🚀</div><div class="brand-name">Startup Diligence Copilot</div></div>')
    render('<div class="side-note">Dataset research for Indian startups. Evidence-first, never invented.</div>')


def hero():
    render(
        '<div class="hero"><div class="hero-badge">Dataset research prototype</div>'
        "<h1>Research Indian startups with evidence you can trace.</h1>"
        "<p>Explore, compare and question the Indian Startups Top 300 dataset. Every answer points back to the exact record it came from.</p>"
        '<div class="hero-chips"><span class="chip-w">🔎 TF-IDF retrieval</span><span class="chip-w">📌 Record-level citations</span>'
        '<span class="chip-w">📊 Funding insights</span><span class="chip-w">🧹 Transparent cleaning</span></div></div>'
    )


def notice(kind, text):
    icon = {"info": "ℹ️", "warn": "⚠️", "ok": "✅"}[kind]
    render(f'<div class="notice {kind}"><div>{icon}</div><div>{text}</div></div>')


def section(title, subtitle):
    render(f'<div class="sec-title">{esc(title)}</div><div class="sec-sub">{esc(subtitle)}</div>')


def stat_cards(items):
    cards = "".join(f'<div class="stat"><div class="stat-ic">{icon}</div><div class="stat-label">{esc(label)}</div><div class="stat-value">{esc(value)}</div></div>' for icon, label, value in items)
    render(f'<div class="stats">{cards}</div>')


def empty_state(icon, title, text):
    render(f'<div class="empty"><div class="em-ic">{icon}</div><b>{esc(title)}</b>{esc(text)}</div>')


def money(value):
    return "N/A" if pd.isna(value) else f"${value:,.0f}"


def whole(value):
    return "N/A" if pd.isna(value) else f"{int(value)}"


def startup_card(row, rank=None, similarity=None):
    name = esc(row["Company"])
    initial = esc(str(row["Company"])[:1].upper() or "?")
    employees = esc(row["No. of Employees"])
    parts = [esc(row["City"])]
    if pd.notna(row["Starting Year"]):
        parts.append(f"Founded {whole(row['Starting Year'])}")
    if employees:
        parts.append(f"{employees} employees")
    meta = " · ".join(p for p in parts if p)
    tags = [t.strip() for t in str(row["Industries"]).split(",") if t.strip()][:4]
    pills = "".join(f'<span class="pill {"p1" if i % 2 == 0 else "p2"}">{esc(t)}</span>' for i, t in enumerate(tags))
    text = "" if pd.isna(row["Description"]) else str(row["Description"])
    description = esc(text[:330] + ("…" if len(text) > 330 else "")) or "No description in dataset."
    label = f'Record {int(row["source_row"])}' if rank is None else f'#{rank} · Record {int(row["source_row"])}'
    bar = ""
    if similarity is not None:
        width = max(4, min(100, similarity * 100))
        bar = f'<div class="sim" title="Cosine similarity"><div style="width:{width:.0f}%"></div></div><div class="sc-meta">Similarity {similarity:.3f}</div>'
    founders = esc(row["Founders"]) or "N/A"
    facts = (
        f'<div class="fact"><b>{money(row[FUNDING])}</b><span>Funding</span></div>'
        f'<div class="fact"><b>{whole(row["Funding Round"])}</b><span>Round</span></div>'
        f'<div class="fact"><b>{whole(row["No. of Investors"])}</b><span>Investors</span></div>'
        f'<div class="fact"><b>{founders}</b><span>Founders</span></div>'
    )
    render(
        f'<div class="sc"><div class="sc-top"><div class="sc-left"><div class="avatar">{initial}</div><div>'
        f'<div class="sc-name">{name}</div><div class="sc-meta">{meta}</div></div></div>'
        f'<div style="text-align:right"><span class="rank">{label}</span>{bar}</div></div>'
        f'<div class="sc-desc">{description}</div><div>{pills}</div><div class="facts">{facts}</div></div>'
    )


def table(df):
    st.dataframe(df, use_container_width=True, hide_index=True)


def export_csv(label, df, filename, key):
    st.download_button(label, df.to_csv(index=False), filename, "text/csv", key=key)


def answer_panel(text):
    with st.container(border=True):
        render('<div class="ans-head"><span class="ans-dot"></span>Grounded answer</div>')
        st.markdown(text)


def overview(df):
    cities = df.loc[df.City.ne(""), "City"].nunique()
    stat_cards([("🏢", "Startups indexed", f"{len(df):,}"), ("📍", "Cities", f"{cities:,}"), ("💰", "Funding known", f"{df[FUNDING].notna().sum():,} / {len(df):,}")])


def render_audit(retriever):
    df = retriever.df
    section("Data preparation and audit", "See exactly how the dataset was cleaned before analysis.")
    stat_cards([("📥", "Raw records", retriever.report["raw_rows"]), ("✨", "Clean records", len(df)), ("🧩", "Fields analysed", len(REQUIRED))])
    with st.container(border=True):
        st.markdown("##### How the data is cleaned")
        st.write("Whitespace and missing-value markers are normalised. Currency symbols and commas are removed before numeric conversion. Missing or invalid numbers stay missing, not zero. Empty company names are removed; case-insensitive duplicate company names keep their first record.")
        st.write("Employee ranges remain text. Industry labels are split only for frequency analysis. No model training or credibility scoring is performed.")
    with st.expander("Full cleaning report"):
        st.json(retriever.report)
    st.caption("CSV record IDs count logical records including the header as record 1; they are not physical line numbers when quoted cells contain line breaks. The original Sr.No. column, when present, is not used as a unique identifier.")
    left, right = st.columns(2)
    left.download_button("Download cleaning report", json.dumps(retriever.report, indent=2, default=int), "cleaning_report.json", "application/json", key="dl_report")
    right.download_button("Download cleaned dataset", df[COLUMNS].to_csv(index=False), "cleaned_startups.csv", "text/csv", key="dl_clean")


def render_explore(df):
    section("Explore the dataset", "Filter by city and industry, then dig into funding and categories.")
    with st.container(border=True):
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
        ("🎯", "Matching startups", f"{len(view):,}"),
        ("💵", "Total reported funding", f"${known.sum():,.0f}" if not known.empty else "N/A"),
        ("📈", "Median funding", f"${known.median():,.0f}" if not known.empty else "N/A"),
        ("🧾", "Known funding entries", f"{len(known)} / {len(view)}"),
    ])
    if view.empty:
        empty_state("🔍", "No startups match these filters", "Clear a city or try a broader industry term.")
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
            with st.container(border=True):
                st.markdown("##### Top cities")
                counts = view.loc[view.City.ne(""), "City"].value_counts().head(10)
                if counts.empty:
                    st.caption("No city data in this selection.")
                else:
                    st.bar_chart(counts, color="#6366f1")
        with right:
            with st.container(border=True):
                st.markdown("##### Top industry labels")
                labels = view.Industries.str.split(",").explode().str.strip()
                counts = labels.loc[labels.ne("")].value_counts().head(10)
                if counts.empty:
                    st.caption("No industry data in this selection.")
                else:
                    st.bar_chart(counts, color="#14b8a6")
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
        empty_state("⚖️", "Pick at least two companies", "Comparison uses exact selections, not approximate text search.")
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
    with st.container(border=True):
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
        k = st.slider("Evidence records to retrieve (top-k)", 1, 10, 5)
        go = st.button("Retrieve evidence and answer", type="primary")
    if not go:
        empty_state("💬", "Ask something to get started", "Pick a query mode, fill in the fields and press the button.")
        return
    if not question:
        notice("warn", "Fill in the fields above first.")
        return
    results = retriever.retrieve(question, k=k)
    st.markdown("### Retrieved evidence")
    st.caption("Similarity measures text overlap, not investment quality. Answers cover the retrieved subset, not necessarily every relevant startup. Record citations show provenance but do not guarantee factual correctness.")
    if results.empty:
        empty_state("🔍", "No matching records", "Try an industry, city or company name from the dataset.")
    else:
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            startup_card(row, rank=rank, similarity=float(row["similarity"]))
        with st.expander("View evidence as table"):
            table(results[COLUMNS + ["similarity"]])
    with st.spinner("Preparing answer..."):
        answer = answer_question(question, results)
    answer_panel(answer)
    notice("info", "Evidence still required for full diligence: pitch decks, audited financials, revenue/customer evidence, founder and legal records, independent funding sources, and dated market research for TAM/SAM/SOM.")
