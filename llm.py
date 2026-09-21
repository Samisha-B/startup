import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def build_context(results: pd.DataFrame) -> str:
    lines = []
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        lines.append(
            f"[Row {i}] Startup: {row['Name of the startup']} | "
            f"Sector: {row['Sector']} | "
            f"Location: {row['Location of company']} | "
            f"Profile: {row['Company profile']}"
        )
    return "\n".join(lines)


def answer_question(question: str, results: pd.DataFrame) -> str:
    if results.empty:
        return ("NOT FOUND IN DATASET. No rows in the startup dataset are "
                "relevant to this question. This system only answers from "
                "the provided dataset and does not use general knowledge.")

    context = build_context(results)

    prompt = f"""You are an investor-research assistant. You must answer questions
about Indian startups using ONLY the DATA block below. You are strictly
forbidden from using any outside knowledge, training data, or general
facts about companies not listed in DATA.

Rules:
1. If DATA does not contain information relevant to the QUESTION, reply
   exactly: "NOT FOUND IN DATASET" followed by a one-line explanation.
2. Every factual claim in your answer must cite the Row number(s) it
   came from, e.g. (Row 2).
3. Do not guess, infer, or fill gaps with outside knowledge about real
   companies, even if you recognize the name.
4. Keep the answer concise and structured for an investor skimming
   quickly.

DATA:
{context}

QUESTION:
{question}

ANSWER:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content
