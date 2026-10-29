from pathlib import Path
import pandas as pd
import streamlit as st
from retrieval import StartupRetriever, DATA_PATH, SOURCE, download_dataset
from ui.components import apply_styles, sidebar_info, hero, notice, overview, render_audit, render_explore, render_compare, render_qa
from ui.evaluate import render_evaluate

st.set_page_config(page_title="Startup Diligence Copilot", layout="wide")
apply_styles()


@st.cache_resource(show_spinner=False)
def load_retriever():
    if not Path(DATA_PATH).exists():
        download_dataset()
    return StartupRetriever(DATA_PATH)


hero()
notice("info", "This module demonstrates dataset preparation, exploration, comparison and evidence-backed Q&A. Document ingestion, LangGraph orchestration and independent claim verification are future work, not implemented here.")
notice("warn", "Funding is not market size or valuation. Dataset completeness and retrieval similarity do not establish startup credibility.")

try:
    with st.spinner("Loading the startup dataset..."):
        retriever = load_retriever()
except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.write(f"Download Startups1.csv from the [Kaggle dataset]({SOURCE}) and save it as `{DATA_PATH}`, then reload this page.")
    st.stop()

sidebar_info(retriever)
overview(retriever.df)
quality, explore, compare, qa, evaluate = st.tabs(["Data preparation", "Explore and analyse", "Compare startups", "Investor Q&A", "Evaluate a startup"])
with quality:
    render_audit(retriever)
with explore:
    render_explore(retriever.df)
with compare:
    render_compare(retriever)
with qa:
    render_qa(retriever)
with evaluate:
    render_evaluate(retriever)
st.markdown('<div class="footer">Startup Diligence Copilot &middot; Dataset research &middot; Developed by Samisha Boywar</div>', unsafe_allow_html=True)
