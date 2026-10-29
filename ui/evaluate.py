import pandas as pd
import streamlit as st
from evaluation import (STAGES, BUSINESS_MODELS, EMPLOYEE_BANDS, sector_options, find_peers,
                        compare_to_peers, score_startup, ai_summary)


def render_evaluate(retriever):
    df = retriever.df
    st.subheader("Evaluate a new startup")
    st.caption("Fill in the details. The startup is compared with dataset peers in the same sector. Inputs are self-reported and unverified.")
    with st.form("startup_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Startup name")
        sector = c2.selectbox("Sector / industry", sector_options(df))
        city = c3.selectbox("City", sorted(c for c in df["City"].unique() if c) + ["Other"])
        year = c1.number_input("Founded in", 1990, pd.Timestamp.now().year, pd.Timestamp.now().year - 2)
        stage = c2.selectbox("Funding stage", STAGES)
        model = c3.selectbox("Business model", BUSINESS_MODELS)
        funding = c1.number_input("Total funding raised (USD)", 0, step=100000)
        investors = c2.number_input("Number of investors", 0, 100, 1)
        employees = c3.selectbox("Employees", EMPLOYEE_BANDS)
        revenue_status = c1.selectbox("Revenue status", ["No revenue", "Early revenue", "Growing revenue", "Profitable"])
        market_size = c2.selectbox("Target market size", ["Small", "Medium", "Large", "Very large"])
        runway = c3.slider("Runway (months)", 0, 36, 12)
        founder_experience = c1.slider("Founder experience (years)", 0, 30, 3)
        team_size = c2.number_input("Core team size", 1, 500, 5)
        moat = c3.checkbox("Has a defensible moat (IP, network effects, regulation)")
        description = st.text_area("What does the startup do?")
        submitted = st.form_submit_button("Analyse startup")
    if not submitted:
        return
    if not name.strip():
        st.error("Please enter a startup name.")
        return
    profile = {
        "name": name.strip(), "sector": sector, "city": city, "year": int(year), "stage": stage,
        "business_model": model, "funding": int(funding), "investors": int(investors), "employees": employees,
        "revenue_status": revenue_status, "market_size": market_size, "runway_months": int(runway),
        "founder_experience": int(founder_experience), "team_size": int(team_size),
        "has_competitive_moat": bool(moat), "description": description.strip(),
    }
    st.markdown(f"### 1. Startup profile: {profile['name']}")
    table = pd.DataFrame({"Field": [k.replace("_", " ").title() for k in profile], "Value": [str(v) for v in profile.values()]})
    st.dataframe(table, hide_index=True, use_container_width=True)

    peers = find_peers(df, sector)
    st.markdown(f"### 2. Comparison with {len(peers)} peers in {sector}")
    if peers.empty:
        st.warning("No peers found in the dataset for this sector.")
    comparison = compare_to_peers(profile, peers)
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    valid = comparison.dropna(subset=["Percentile"])
    if not valid.empty:
        st.bar_chart(valid.set_index("Metric")["Percentile"])
    if not peers.empty:
        top = peers.sort_values("Funding Amount in $", ascending=False).head(5)
        st.caption("Top funded peers")
        st.dataframe(top[["Company", "City", "Funding Amount in $", "Funding Round", "No. of Investors"]], hide_index=True, use_container_width=True)

    score, verdict, reasons, risks = score_startup(profile, comparison)
    st.markdown("### 3. AI investment summary")
    m1, m2 = st.columns(2)
    m1.metric("Score", f"{score}/100")
    m2.metric("Verdict", verdict)
    with st.spinner("Generating summary..."):
        st.write(ai_summary(profile, comparison, score, verdict, reasons, risks, len(peers)))
    st.caption("Decision support only, not financial advice. Based on self-reported data and a historical dataset.")
