import streamlit as st
from retrieval import StartupRetriever
from llm import answer_question

st.set_page_config(page_title="Startup Investor Q&A Bot", page_icon="\U0001F50E", layout="wide")

QUERY_MODES = {
    "Free-form question": None,
    "Find startups by sector": "Which startups in the dataset work in the {sector} sector?",
    "Compare two startups": "Compare the business models of {a} and {b} based on their profiles.",
    "Shortlist top-k for investment thesis": (
        "Shortlist the top startups that best match this investment thesis: {thesis}"
    ),
}


@st.cache_resource
def load_retriever():
    return StartupRetriever()


retriever = load_retriever()

st.title("Startup Investor Q&A Bot")
st.caption(
    "TF-IDF + cosine similarity retrieval over a Kaggle dataset of Indian startups, "
    "with OpenAI used only to phrase answers grounded in the retrieved rows. "
    "Unlike ChatGPT, every claim here is traceable to a specific dataset row."
)

mode = st.selectbox("Query mode", list(QUERY_MODES.keys()))

question = None

if mode == "Free-form question":
    question = st.text_input("Ask a question about the startups in this dataset:")

elif mode == "Find startups by sector":
    sector = st.text_input("Sector (e.g. Healthcare, Agritech, Fintech):")
    if sector:
        question = QUERY_MODES[mode].format(sector=sector)

elif mode == "Compare two startups":
    col1, col2 = st.columns(2)
    a = col1.text_input("Startup A name:")
    b = col2.text_input("Startup B name:")
    if a and b:
        question = QUERY_MODES[mode].format(a=a, b=b)

elif mode == "Shortlist top-k for investment thesis":
    thesis = st.text_area("Describe your investment thesis:")
    if thesis:
        question = QUERY_MODES[mode].format(thesis=thesis)

top_k = st.slider("Number of rows to retrieve (top-k)", min_value=1, max_value=10, value=5)

if st.button("Get Answer", type="primary") and question:
    with st.spinner("Retrieving relevant startups (TF-IDF + cosine similarity)..."):
        results = retriever.retrieve(question, k=top_k)

    st.subheader("Retrieved Evidence (before any LLM call)")
    st.caption(
        "These are the exact rows the algorithm matched, ranked by cosine similarity. "
        "The LLM below is only allowed to answer using these rows."
    )
    if results.empty:
        st.warning("No matching rows found in the dataset for this query.")
    else:
        display_df = results[[
            "Name of the startup", "Sector", "Location of company", "similarity"
        ]].reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df = display_df.rename(columns={
            "Name of the startup": "Startup",
            "Location of company": "Location",
            "similarity": "Cosine similarity",
        })
        st.dataframe(display_df, use_container_width=True)

        with st.expander("Show full retrieved company profiles"):
            for i, (_, row) in enumerate(results.iterrows(), start=1):
                st.markdown(f"**[Row {i}] {row['Name of the startup']}** \u2014 {row['Sector']}, {row['Location of company']}")
                st.write(row["Company profile"])
                st.divider()

    st.subheader("Grounded Answer")
    with st.spinner("Synthesizing answer from retrieved rows..."):
        answer = answer_question(question, results)
    st.write(answer)

elif not question:
    st.info("Fill in the fields above and click **Get Answer**.")
