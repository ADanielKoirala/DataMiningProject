from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .config import MAX_FEATURES, NGRAM_RANGE, RANDOM_STATE, TEST_SIZE
from .preprocessing import LABEL_MAP, build_vectorizer


def build_models() -> dict[str, object]:
    """Return the baseline and ensemble classifiers used in this project."""
    naive_bayes = MultinomialNB()
    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("naive_bayes", naive_bayes), ("random_forest", random_forest)],
        voting="soft",
    )
    return {
        "Naive Bayes": naive_bayes,
        "Random Forest": random_forest,
        "Soft Voting Ensemble": ensemble,
    }


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Calculate standard binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_and_evaluate(data: pd.DataFrame, model_dir: str | Path, report_dir: str | Path) -> pd.DataFrame:
    """Train models, save artifacts, and write a metrics report."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = build_vectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X = vectorizer.fit_transform(data["clean_text"])
    y = data["label"].map(LABEL_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rows = []
    models = build_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions)
        rows.append({"model": name, **metrics})

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, model_dir / f"{safe_name}.pkl")

    joblib.dump(models["Soft Voting Ensemble"], model_dir / "ensemble_model.pkl")
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(report_dir / "metrics.csv", index=False)
    (report_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    return metrics_df
