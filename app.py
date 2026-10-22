import pandas as pd
import streamlit as st
from retrieval import StartupRetriever, DATA_PATH, SOURCE
from ui.components import apply_styles, header, overview, render_audit, render_explore, render_compare, render_qa

st.set_page_config(page_title="Startup Atlas | Research workspace", page_icon="🔎", layout="wide")
apply_styles()
with st.sidebar:
    st.title("Startup Atlas")
    st.caption("INDIAN STARTUP RESEARCH")
    st.divider()
    st.subheader("Your dataset")
    uploaded = st.file_uploader("Upload Startups1.csv", type=["csv"], help="Optional: replace the bundled dataset for this session.")
    st.caption("Uses the bundled dataset when no file is uploaded.")
    st.divider()
    st.markdown(f"[View original dataset]({SOURCE})")
    st.caption("Historical snapshot · Not live company data")
    st.caption("Developed by Samisha Boywar")
header()
try:
    retriever = StartupRetriever(uploaded if uploaded is not None else DATA_PATH)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.info("Upload the original Startups1.csv using the sidebar, or download the dataset locally.")
    st.code("python retrieval.py --download\nstreamlit run app.py", language="bash")
    st.stop()
overview(retriever.df)
st.caption("Research prototype: funding is not valuation or market size. Data completeness and text similarity do not establish credibility.")
explore, compare, qa, audit = st.tabs(["Explore startups", "Compare companies", "Ask the dataset", "Data quality"])
with explore:
    render_explore(retriever.df)
with compare:
    render_compare(retriever)
with qa:
    render_qa(retriever)
with audit:
    render_audit(retriever)
st.divider()
st.caption("Dataset research only · Document ingestion, independent verification and LangGraph orchestration are not implemented.")
