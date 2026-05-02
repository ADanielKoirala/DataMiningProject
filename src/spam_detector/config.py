from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "spam_dataset.csv"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_spam_dataset.csv"
MODEL_PATH = MODEL_DIR / "ensemble_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METRICS_PATH = REPORT_DIR / "metrics.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
