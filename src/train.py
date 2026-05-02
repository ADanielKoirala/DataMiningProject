from spam_detector.config import DATA_DIR, MODEL_DIR, RAW_DATA_PATH, REPORT_DIR
from spam_detector.modeling import train_and_evaluate
from spam_detector.preprocessing import load_dataset


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    data = load_dataset(RAW_DATA_PATH)
    metrics = train_and_evaluate(data, MODEL_DIR, REPORT_DIR)
    print("Training complete. Metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
