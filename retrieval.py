from pathlib import Path
import io
import json
import sys
import urllib.request
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Startups1.csv"
SOURCE = "https://www.kaggle.com/datasets/ashishraut64/indian-startups-top-300"
TEXT_COLUMNS = ["Company", "City", "Founders", "Industries", "Description", "No. of Employees"]
NUMERIC_COLUMNS = ["Starting Year", "Funding Amount in $", "Funding Round", "No. of Investors"]
REQUIRED = TEXT_COLUMNS + NUMERIC_COLUMNS


def download_dataset():
    url = "https://www.kaggle.com/api/v1/datasets/download/ashishraut64/indian-startups-top-300"
    request = urllib.request.Request(url, headers={"User-Agent": "startup-dataset-project/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if Path(n).name.lower() == "startups1.csv"]
            if len(names) != 1:
                raise ValueError("Expected exactly one Startups1.csv in Kaggle archive")
            content = archive.read(names[0])
        candidate = pd.read_csv(io.BytesIO(content))
        candidate.columns = candidate.columns.str.strip()
        missing = set(REQUIRED) - set(candidate.columns)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = DATA_PATH.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(DATA_PATH)
        return DATA_PATH
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}. Download Startups1.csv manually from {SOURCE} and put it at {DATA_PATH}.") from exc


class StartupRetriever:
    def __init__(self, csv_path=DATA_PATH):
        raw = pd.read_csv(csv_path)
        raw.columns = raw.columns.str.strip()
        missing = set(REQUIRED) - set(raw.columns)
        if missing:
            raise ValueError(f"Wrong dataset schema. Missing: {sorted(missing)}. Use Ashish Raut's Startups1.csv, not the old startups.csv.")
        self.report = {"source": SOURCE, "raw_rows": len(raw), "missing_before_cleaning": raw[REQUIRED].isna().sum().to_dict()}
        df = raw.copy()
        df["source_row"] = range(2, len(df) + 2)
        for column in TEXT_COLUMNS:
            df[column] = df[column].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
            df.loc[df[column].str.lower().isin(["not available", "n/a", "na", "nan", "unknown"]), column] = ""
        self.report["invalid_numeric"] = {}
        for column in NUMERIC_COLUMNS:
            original = df[column].astype("string").str.strip()
            cleaned = original.str.replace(r"[$,\s]", "", regex=True)
            values = pd.to_numeric(cleaned, errors="coerce")
            invalid = values.notna() & ((values < 0) | (values % 1 != 0))
            if column == "Starting Year":
                invalid |= values.notna() & ((values < 1800) | (values > pd.Timestamp.now().year))
            self.report["invalid_numeric"][column] = int((invalid | (values.isna() & original.notna() & original.ne(""))).sum())
            df[column] = values.mask(invalid)
        empty = df["Company"].eq("")
        self.report["dropped_unnamed"] = int(empty.sum())
        df = df.loc[~empty].copy()
        duplicate = df["Company"].str.casefold().duplicated()
        self.report["dropped_duplicate_companies"] = int(duplicate.sum())
        df = df.loc[~duplicate].copy().reset_index(drop=True)
        self.report["clean_rows"] = len(df)
        self.report["missing_after_cleaning"] = {c: int(df[c].eq("").sum()) if c in TEXT_COLUMNS else int(df[c].isna().sum()) for c in REQUIRED}
        if df.empty:
            raise ValueError("Dataset has no usable named startups")
        # Compatibility with the existing command-line client.
        for old, new in {"Name of the startup": "Company", "Sector": "Industries", "Location of company": "City", "Company profile": "Description"}.items():
            df[old] = df[new]
        df["combined_text"] = df[TEXT_COLUMNS].fillna("").agg(". ".join, axis=1)
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(df["combined_text"])

    def retrieve(self, query, k=5):
        if not query.strip() or k < 1:
            return self.df.iloc[:0].assign(similarity=pd.Series(dtype=float))
        scores = cosine_similarity(self.vectorizer.transform([query]), self.tfidf_matrix).ravel()
        indices = scores.argsort()[::-1][:int(k)]
        results = self.df.iloc[indices].copy()
        results["similarity"] = scores[indices]
        return results.loc[results["similarity"] > 0]

    def compare(self, companies):
        return self.df.loc[self.df["Company"].isin(companies), ["source_row"] + REQUIRED].copy()


if __name__ == "__main__":
    if "--download" in sys.argv:
        print(download_dataset())
    retriever = StartupRetriever()
    print(json.dumps(retriever.report, indent=2, default=int))
    print(retriever.retrieve("fintech payments")[ ["source_row", "Company", "Industries", "similarity"] ].to_string(index=False))
