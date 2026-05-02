import argparse

from spam_detector.predictor import SpamPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an SMS message as spam or ham.")
    parser.add_argument("text", help="SMS text to classify")
    parser.add_argument("--threshold", type=float, default=0.50, help="Spam probability threshold")
    args = parser.parse_args()

    predictor = SpamPredictor()
    result = predictor.predict(args.text, threshold=args.threshold)
    print(f"Prediction: {result['label']}")
    print(f"Spam probability: {result['spam_probability']:.4f}")


if __name__ == "__main__":
    main()
