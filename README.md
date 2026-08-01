# Credit Risk Predictor

A simple PyTorch neural network that predicts whether a credit card holder is likely to default on their next payment, trained on the UCI Taiwan credit card default dataset (30,000 customers).

## What it does

Takes in a customer's credit limit, basic demographics, and their last 6 months of payment history and bill amounts, and outputs a default probability, bucketed as:

- **< 50%** → Low Risk — Approve
- **50–70%** → Medium Risk — Review Manually
- **> 70%** → High Risk — Deny

## Model

A small feedforward network (`model.py`):

```
Input (23 features) → Linear(64) → ReLU → BatchNorm → Dropout
                     → Linear(32) → ReLU → BatchNorm → Dropout
                     → Linear(1)  → Sigmoid
```

Trained in `defaultcreditcard.ipynb`, reaching ~82% test accuracy.

## Running it

**Streamlit app:**
```bash
streamlit run streamlit_app.py
```

**Flask API:**
```bash
pip install flask   # not currently in requirements.txt
python app.py
```
`POST /predict` with the 23 features as JSON, returns a probability + risk level.

## Dataset

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — 30,000 Taiwanese credit card holders, April–September 2005.

---
First ML project — built to learn the full pipeline from raw data to a deployed model.
