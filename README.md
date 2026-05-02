# SMS Spam Detection System

A machine learning project for classifying SMS messages as **spam** or **ham** using text preprocessing, TF-IDF feature extraction, and supervised classification models.

## Overview

This project builds an end-to-end spam detection pipeline. It starts with raw SMS text, cleans and vectorizes the messages, trains multiple classification models, evaluates them with standard metrics, and serves predictions through both a command-line interface and a Flask web app.

The goal is to demonstrate a complete data mining workflow: preprocessing, feature engineering, model training, model evaluation, artifact persistence, and user-facing prediction.

## Features

- Cleans and normalizes raw SMS messages
- Converts text into TF-IDF features using unigram and bigram terms
- Trains Naive Bayes, Random Forest, and soft-voting ensemble classifiers
- Saves trained model artifacts for repeatable inference
- Reports accuracy, precision, recall, and F1 score
- Provides both CLI and Flask web interfaces for prediction
- Uses a modular source structure so preprocessing, training, and prediction logic are separated

## Project Structure

```text
DataMiningProject/
├── data/
│   ├── spam_dataset.csv
│   └── cleaned_spam_dataset.csv
├── models/
│   ├── ensemble_model.pkl
│   └── vectorizer.pkl
├── reports/
│   ├── metrics.csv
│   └── metrics.json
├── src/
│   ├── app.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── train.py
│   └── spam_detector/
│       ├── config.py
│       ├── modeling.py
│       ├── predictor.py
│       └── preprocessing.py
├── static/
│   └── style.css
├── templates/
│   ├── home.html
│   └── result.html
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Flask
- Joblib
- HTML/CSS

## Machine Learning Approach

### Preprocessing

Raw SMS messages are cleaned by:

- Lowercasing text
- Replacing URLs with a normalized token
- Replacing numbers with a normalized token
- Removing punctuation and non-alphabetic characters
- Removing common English stop words

### Feature Engineering

The project uses TF-IDF vectorization with unigram and bigram terms. This captures both individual words and short phrases that are useful for spam detection, such as promotional wording, urgent calls to action, and prize-related language.

### Models

The training pipeline compares:

- Multinomial Naive Bayes
- Random Forest
- Soft Voting Ensemble

The ensemble combines probabilistic outputs from the base models to improve prediction stability.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python src/train.py
```

This trains the models and writes saved artifacts to `models/` and evaluation reports to `reports/`.

## View Evaluation Results

```bash
python src/evaluate.py
```

Example output format:

```text
              model  accuracy  precision  recall      f1
Soft Voting Ensemble    0.9800     0.9500  0.9000  0.9243
        Naive Bayes    0.9700     0.9800  0.8200  0.8929
      Random Forest    0.9650     0.9300  0.8100  0.8659
```

Exact results may vary depending on dataset version and environment.

## Run a Prediction from the Command Line

```bash
python src/predict.py "Congratulations! You won a free prize. Text WIN now."
```

Example output:

```text
Prediction: spam
Spam probability: 0.9132
```

## Run the Web App

```bash
python src/app.py
```

Then open the local Flask URL shown in the terminal and submit an SMS message through the form.

## API Usage

After starting the Flask app, send a JSON request to `/predict`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Free entry! Claim your prize now."}'
```

Example response:

```json
{
  "label": "spam",
  "is_spam": true,
  "spam_probability": 0.9132,
  "clean_text": "free entry claim prize"
}
```

## Design Decisions

- TF-IDF was selected because it provides a strong baseline for sparse text classification without requiring a large dataset.
- Naive Bayes was included because it is efficient and commonly performs well on text classification problems.
- Random Forest was added as a non-linear model to compare against the probabilistic baseline.
- A soft-voting ensemble was used to combine model probabilities instead of relying on a single classifier.
- Training and prediction logic were separated so the saved model can be reused by both the CLI and Flask app.

## Future Improvements

- Add cross-validation and hyperparameter tuning
- Add a confusion matrix visualization
- Add model explainability for top spam-indicating terms
- Add automated tests for preprocessing and prediction
- Package the application with Docker for consistent deployment
