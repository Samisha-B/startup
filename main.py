from retrieval import StartupRetriever, DATA_PATH
from llm import answer_question


def main():
    print("Startup Diligence Copilot - Dataset Research")
    print("Developed by Samisha Boywar")
    print("Dataset: Ashish Raut / Indian Startups Top 300 (historical snapshot)")
    print("Dataset research only; no independent verification or market sizing. Type exit to quit.")
    try:
        retriever = StartupRetriever()
    except (OSError, ValueError) as exc:
        print(f"{exc}\nRun: python retrieval.py --download\nExpected file: {DATA_PATH}")
        return
    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        results = retriever.retrieve(question, k=5)
        print(results[["source_row", "Company", "Industries", "City", "similarity"]].to_string(index=False))
        print(answer_question(question, results))


if __name__ == "__main__":
    main()
