# Fraud Detection System

End-to-end fraud detection on 6M+ synthetic mobile money transactions (PaySim dataset) — built to handle severe class imbalance and evaluated on metrics that actually reflect real-world fraud detection performance, not misleading accuracy.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic mobile money transaction logs generated from real transaction patterns from a mobile money service in Africa, created specifically for fraud detection research.

- 6,362,620 transactions
- 11 columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`

## Key Findings

### 1. Severe Class Imbalance
```
isFraud
0    0.998709
1    0.001291
```
Fraud accounts for only **0.129%** of transactions. This means **accuracy is not a valid metric** — a model predicting "not fraud" for every transaction would score 99.87% accuracy while catching zero fraud. This project evaluates on **Precision, Recall, F1, and PR-AUC** instead.

### 2. Fraud Occurs in Only Two Transaction Types
```
type
CASH_OUT    4116
TRANSFER    4097
```
Fraud occurs **exclusively** in `TRANSFER` and `CASH_OUT` transactions — zero fraud in `PAYMENT`, `CASH_IN`, or `DEBIT`. This matches the real-world mobile money fraud pattern: funds are moved to another account (`TRANSFER`), then withdrawn as cash (`CASH_OUT`) before detection. `type` is a very strong feature.

### 3. Fraudulent Transactions Are Significantly Larger

| | count | mean | median | max |
|---|---|---|---|---|
| Not Fraud (0) | 6,354,407 | $178,197 | $74,685 | $92,445,517 |
| Fraud (1) | 8,213 | $1,467,967 | $441,423 | $10,000,000 |

Fraud transactions are **~6-8x larger** than legitimate ones, on average and at the median. *Note: fraud's max is suspiciously capped at exactly $10,000,000 — likely a simulator artifact, not expected in real-world data.*

### 4. Disproportionate Financial Impact

| Metric | Value |
|---|---|
| Total transaction volume | $1,144,392,944,759.77 |
| Total fraud volume | $12,056,415,427.84 |
| Fraud as % of total volume | **1.05%** |
| Fraud as % of transaction count | 0.13% |

Fraud is only 0.13% of transactions by count but **1.05% of total transaction value** — roughly 8x disproportionate financial impact, confirming that fraudulent transactions skew large.

### 5. Data Leakage Warning
`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` are **excluded from modeling**. Per the dataset documentation, transactions flagged as fraud are cancelled, so these post-transaction balance columns would directly leak the label into the features.

## Evaluation Approach

Given the extreme class imbalance, this project prioritizes:
- **Recall** — catching actual fraud cases (missed fraud is costly)
- **Precision** — minimizing false alarms
- **PR-AUC** — more informative than ROC-AUC under severe imbalance
- **Accuracy is explicitly not used as a success metric**

## Project Structure

```
fraud-detection-system/
├── config/
│   └── config.yml
├── fraud_detection/
│   ├── processing/
│   │   ├── data_manager.py
│   │   └── features.py
│   ├── pipeline.py
│   ├── train_pipeline.py
│   └── predict.py
├── app/
│   └── main.py
├── notebooks/
│   └── fraud_detection_eda.ipynb
├── tests/
└── requirements.txt
```

## Status

🚧 In progress — EDA complete, modeling in progress.
