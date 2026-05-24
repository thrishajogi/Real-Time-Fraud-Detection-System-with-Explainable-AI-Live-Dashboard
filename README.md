# Real-Time Credit Card Fraud Detection System with Explainable AI

Advanced machine learning and Explainable AI project built using the IEEE-CIS Fraud Detection dataset.

This project implements an end-to-end fraud detection pipeline capable of identifying suspicious credit card transactions using machine learning models and Explainable AI techniques.

The system combines:
- LightGBM
- XGBoost
- Isolation Forest
- SHAP Explainability
- Streamlit Dashboard

to build an interactive fraud analysis and monitoring platform.

---

# Project Highlights

- End-to-end fraud detection workflow
- Handles highly imbalanced fraud data using SMOTE
- Feature engineering for transaction risk analysis
- Multi-model comparison and evaluation
- Explainable AI using SHAP
- Interactive Streamlit dashboard
- Real-time fraud probability scoring
- Transaction-level explainability
- Risk segmentation system

---

# Dataset

## Credit Card Fraud Detection — IEEE-CIS

Download the dataset from Kaggle:

https://www.kaggle.com/c/ieee-fraud-detection/data

### Steps to Download

1. Create or log in to your free Kaggle account:
https://www.kaggle.com

2. Search for:
IEEE-CIS Fraud Detection

3. Download the following files:
- `train_transaction.csv`
- `train_identity.csv`

4. Place both files inside:

```text
FraudDetection/data/
```

5. The project automatically merges both datasets using:
`TransactionID`

---

# Tech Stack

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

# Project Structure

```text
FraudDetection/
│
├── analysis.ipynb
├── README.md
├── requirements.txt
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
    ├── shap_waterfall_fraud.png
    └── ...
```

---

# Installation

Clone the repository:

```bash
git clone <your-repository-link>
cd FraudDetection
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run the Jupyter Notebook

```bash
jupyter notebook analysis.ipynb
```

Run all notebook cells sequentially.

The notebook performs:
- Data cleaning
- Missing value handling
- Feature engineering
- Class balancing using SMOTE
- Model training and evaluation
- Threshold optimization
- SHAP explainability
- Visualization generation

---

## Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Dashboard Features

## Overview Page
- KPI metrics
- Fraud trends
- Risk tier distribution
- Fraud-by-hour analysis

## Transaction Explorer
- Search transactions by TransactionID
- Filter by fraud probability
- Interactive fraud analysis

## SHAP Explainer
- Transaction-level explainability
- SHAP waterfall visualizations
- Plain-English fraud reasoning

---

# Model Performance

| Model | Performance |
|---|---|
| LightGBM | Best overall performance |
| XGBoost | Strong competitive results |
| Isolation Forest | Lower performance compared to supervised models |

Evaluation metrics used:
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-Score

---

# Key Insights

- Large abnormal transactions are strong fraud indicators
- Late-night transactions show elevated fraud probability
- Mobile-device transactions exhibit higher fraud risk
- PR-AUC is more meaningful than accuracy for imbalanced fraud datasets
- Threshold optimization significantly improves business impact

---

# Explainable AI

SHAP was used to provide:
- Global feature importance
- Beeswarm plots
- Dependence plots
- Waterfall explanations
- Transaction-level reasoning

This improves model transparency and interpretability for fraud analysts.

---

# Future Improvements

- Real-time API deployment
- Kafka streaming integration
- User authentication
- Cloud deployment
- Deep learning-based fraud detection
- Real-time transaction ingestion

---

# Author

**Thrisha R.S**  

---

# Acknowledgements

- IEEE-CIS Fraud Detection Dataset
- Kaggle
- LightGBM
- SHAP
- Streamlit
- XYlofy Internship Program
