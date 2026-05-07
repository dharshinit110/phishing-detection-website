# Phishing Website Detection Using Machine Learning

## Objective

This project detects whether a given URL is **phishing** or **legitimate** using machine learning, then serves predictions through a Streamlit web app.

## Architecture Overview

- **Feature extraction** from URL strings (`src/feature_extraction.py`)
- **Model training** with selectable models (`src/train_model.py`)
- **Prediction pipeline** (`src/predict.py`)
- **Domain reputation checks** (DNS + HTTP reachability) (`src/domain_reputation.py`)
- **Web UI** built with Streamlit (`app.py`)

## Dataset Sources

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [PhishTank](https://phishtank.org/)

This project currently trains by default on the extracted UCI ARFF file:
- `dataset/uci_phishing_websites/Training Dataset.arff`

A small synthetic CSV also exists for simple demos:
- `dataset/sample_dataset.csv`

## Project Structure

```text
phishing-detection/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
├── dataset/
│   ├── sample_dataset.csv
│   └── uci_phishing_websites/
│       └── Training Dataset.arff
├── models/
│   └── model.pkl
└── src/
    ├── __init__.py
    ├── feature_extraction.py
    ├── domain_reputation.py
    ├── train_model.py
    └── predict.py
```

## Installation

```bash
git clone https://github.com/your-username/phishing-detection.git
cd phishing-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

Default model is **XGBoost**.

```bash
python -m src.train_model --dataset "dataset\uci_phishing_websites\Training Dataset.arff" --model-type xgboost --test-size 0.2
```

Other model options:

```bash
python -m src.train_model --model-type random_forest
python -m src.train_model --model-type logistic_regression
```

## Run Locally

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Deployment (Streamlit Community Cloud)

1. Push this project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your GitHub repo and branch.
5. Set **Main file path** to `app.py`.
6. Click **Deploy**.

### Deployment Notes

- `runtime.txt` pins Python version for cloud builds.
- Ensure `models/model.pkl` exists in the repository before deployment, or train once after deploy.
- DNS/HTTP reputation checks may behave differently depending on cloud/network policies.

## App Behavior

- Takes a URL input.
- Predicts phishing vs legitimate.
- Shows confidence score when available.
- Shows domain checks (DNS + HTTP reachability) as supporting signals.

## Example Screenshots

- `screenshots/home_page.png`
- `screenshots/result_legitimate.png`
- `screenshots/result_phishing.png`

## Limitations

- No phishing detector can guarantee 100% accuracy.
- URL-only models can miss attacks that require page-content or host-history analysis.
- For production, combine URL model + blacklist feeds + WHOIS/content signals + continuous retraining.

