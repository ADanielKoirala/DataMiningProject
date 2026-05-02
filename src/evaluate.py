from pathlib import Path

import pandas as pd

from spam_detector.config import METRICS_PATH


def main() -> None:
    if not Path(METRICS_PATH).exists():
        raise FileNotFoundError("No metrics report found. Run `python src/train.py` first.")
    metrics = pd.read_csv(METRICS_PATH)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
