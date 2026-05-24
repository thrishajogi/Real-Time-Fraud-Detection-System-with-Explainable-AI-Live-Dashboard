# Real-Time Fraud Detection System with Explainable AI

Machine learning and Explainable AI project built using the IEEE-CIS Fraud Detection dataset.

---

## Project Overview

An end-to-end fraud detection pipeline combining:

- LightGBM / XGBoost / Isolation Forest for classification
- SMOTE for class imbalance handling
- SHAP for Explainable AI (global + per-transaction)
- Streamlit multi-page live dashboard

---

## Dataset

### Credit Card Fraud Detection — IEEE-CIS

Download from Kaggle:  
https://www.kaggle.com/c/ieee-fraud-detection/data

### Steps to Download

Create or log in to your free Kaggle account at:  
https://www.kaggle.com

Search for:  
`IEEE-CIS Fraud Detection`

Download:

- `train_transaction.csv`
- `train_identity.csv`

Place both files inside the `data/` folder.

The project automatically merges both datasets using:
`TransactionID`

---

## Folder Structure

```text
FraudDetection/
├── analysis.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
│
├── dashboard/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── charts/
    ├── confusion_matrices.png
    ├── roc_pr_curves.png
    ├── threshold_optimization.png
    ├── shap_bar.png
    ├── shap_beeswarm.png
    ├── shap_dependence.png
    ├── fraud_by_hour.png
    └── ...
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

```bash
jupyter notebook analysis.ipynb
```

Run all notebook cells in order.

The notebook performs:

- Dataset merging
- Exploratory Data Analysis
- Missing value handling
- Feature engineering
- SMOTE balancing
- Model training and evaluation
- SHAP explainability
- Visualization generation
- Risk segmentation

### 3. Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Dashboard Features

| Page | Description |
|------|-------------|
| Overview | KPI cards, fraud trends, and risk distributions |
| Transaction Explorer | Search and filter transactions with fraud probabilities |
| SHAP Explainer | Transaction-level explainability using SHAP |

---

## Model Performance

| Model | Performance |
|-------|-------------|
| LightGBM | Best overall performance |
| XGBoost | Competitive performance |
| Isolation Forest | Lower performance |

Evaluation metrics used:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-Score

---

## Key Findings

- AmtToMeanRatio was the strongest fraud indicator
- Late-night transactions showed significantly higher fraud rates
- Mobile-device transactions exhibited elevated fraud probability
- PR-AUC was more reliable than accuracy for this imbalanced dataset
- Threshold optimization improved fraud detection efficiency

---

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | LightGBM, XGBoost, Isolation Forest |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Imbalance Handling | SMOTE |

---

## Future Improvements

- Real-time API deployment
- Kafka streaming integration
- Cloud deployment
- Deep learning-based fraud detection
- Live transaction ingestion

---

## Acknowledgements

- IEEE-CIS Fraud Detection Dataset
- Kaggle
- LightGBM
- SHAP
- Streamlit
- XYlofy Internship Program
