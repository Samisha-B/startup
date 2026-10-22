import pandas as pd
import streamlit as st
from retrieval import StartupRetriever, DATA_PATH, SOURCE
from ui.components import apply_styles, sidebar_brand, hero, notice, overview, render_audit, render_explore, render_compare, render_qa

st.set_page_config(page_title="Startup Diligence Copilot", page_icon="🚀", layout="wide")
apply_styles()

with st.sidebar:
    sidebar_brand()
    st.divider()
    uploaded = st.file_uploader("Optional: upload the original Startups1.csv", type=["csv"])
    st.divider()
    st.markdown(f"[Dataset: Ashish Raut - Indian Startups Top 300]({SOURCE})")
    st.markdown('<div class="side-note">Historical snapshot, not live company data.<br><br>Developed by Samisha Boywar</div>', unsafe_allow_html=True)

hero()
notice("info", "This module demonstrates dataset preparation, exploration, comparison and evidence-backed Q&A. Document ingestion, LangGraph orchestration and independent claim verification are future work, not implemented here.")
notice("warn", "Funding is not market size or valuation. Dataset completeness and retrieval similarity do not establish startup credibility.")

try:
    retriever = StartupRetriever(uploaded if uploaded is not None else DATA_PATH)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.code("python retrieval.py --download\nstreamlit run app.py", language="bash")
    st.write("Alternatively download Startups1.csv from the dataset link and upload it using the sidebar. The old startups.csv is not used.")
    st.stop()

overview(retriever.df)
quality, explore, compare, qa = st.tabs(["Data preparation", "Explore and analyse", "Compare startups", "Investor Q&A"])
with quality:
    render_audit(retriever)
with explore:
    render_explore(retriever.df)
with compare:
    render_compare(retriever)
with qa:
    render_qa(retriever)
st.markdown('<div class="footer">Startup Diligence Copilot · Dataset Research · Developed by Samisha Boywar</div>', unsafe_allow_html=True)
