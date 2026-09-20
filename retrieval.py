import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/startups.csv"


class StartupRetriever:
    def __init__(self, csv_path: str = DATA_PATH):
        self.df = self._load_and_clean(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

    @staticmethod
    def _load_and_clean(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["Company profile"] = df["Company profile"].fillna("")
        df["Sector"] = df["Sector"].fillna("")
        df["Location of company"] = df["Location of company"].fillna("Unknown")
        df["combined_text"] = (df["Sector"] + ". " + df["Company profile"]).str.strip()
        df = df[df["combined_text"].str.len() > 3].reset_index(drop=True)
        return df

    def retrieve(self, query: str, k: int = 5) -> pd.DataFrame:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:k]
        results = self.df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        return results[results["similarity"] > 0]


if __name__ == "__main__":
    retriever = StartupRetriever()
    query = "startups working on drones and UAV technology"
    results = retriever.retrieve(query, k=5)
    print(f"Query: {query}\n")
    for _, row in results.iterrows():
        print(f"[{row['similarity']:.3f}] {row['Name of the startup']} ({row['Sector']}) - {row['Location of company']}")
