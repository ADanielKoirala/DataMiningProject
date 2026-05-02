import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


LABEL_MAP = {"ham": 0, "spam": 1}
INVERSE_LABEL_MAP = {0: "ham", 1: "spam"}


def clean_text(text: object) -> str:
    """Normalize raw SMS text for vectorization."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\b\d+\b", " number ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS and len(word) > 1]
    return " ".join(tokens)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the spam dataset and return standardized label/text columns.

    The included dataset follows the common SMS Spam Collection format where
    the first column is the class label and the second column is message text.
    Extra unnamed columns are ignored safely.
    """
    path = Path(path)
    data = pd.read_csv(
        path,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
        header=0,
        on_bad_lines="skip",
    )
    data = data.dropna(subset=["label", "text"])
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["label"].isin(LABEL_MAP)]
    data["text"] = data["text"].astype(str)
    data["clean_text"] = data["text"].apply(clean_text)
    return data.reset_index(drop=True)


def build_vectorizer(max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """Create the TF-IDF vectorizer used by all models."""
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)
