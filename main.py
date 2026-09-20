from retrieval import StartupRetriever
from llm import answer_question

TOP_K = 5


def main():
    print("Startup Dataset Q&A Bot (TF-IDF retrieval + OpenAI synthesis)")
    print("Dataset: Startups in India (Kaggle)")
    print("Type 'exit' to quit.\n")

    retriever = StartupRetriever()

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        results = retriever.retrieve(question, k=TOP_K)

        print("\n--- Retrieved rows (TF-IDF + cosine similarity) ---")
        if results.empty:
            print("No matching rows found.")
        else:
            for _, row in results.iterrows():
                print(f"[{row['similarity']:.3f}] {row['Name of the startup']} ({row['Sector']}) - {row['Location of company']}")

        print("\n--- LLM Answer ---")
        print(answer_question(question, results))
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
