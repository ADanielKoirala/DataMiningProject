from pathlib import Path

import joblib

from .config import MODEL_PATH, VECTORIZER_PATH
from .preprocessing import clean_text


class SpamPredictor:
    """Load trained artifacts and classify new SMS messages."""

    def __init__(self, model_path: str | Path = MODEL_PATH, vectorizer_path: str | Path = VECTORIZER_PATH):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str, threshold: float = 0.50) -> dict[str, object]:
        if not text or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        cleaned = clean_text(text)
        features = self.vectorizer.transform([cleaned])
        spam_probability = float(self.model.predict_proba(features)[0][1])
        label = "spam" if spam_probability >= threshold else "ham"
        return {
            "label": label,
            "is_spam": label == "spam",
            "spam_probability": round(spam_probability, 4),
            "clean_text": cleaned,
        }
