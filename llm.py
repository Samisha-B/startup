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
        return "No relevant startups were found in the dataset for this question."

    context = build_context(results)

    prompt = f"""You are a research assistant answering questions about Indian startups
using ONLY the data provided below. Do not use outside knowledge.
If the data below does not contain the answer, say so clearly.
Always mention which Row number(s) you used to answer.

DATA:
{context}

QUESTION:
{question}

ANSWER:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
