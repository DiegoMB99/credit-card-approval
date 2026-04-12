# 💳 Credit Card Approval Prediction

A end-to-end Data Science project for credit card approval prediction using machine learning, deployed as a REST API on Google Cloud Run.

---

## 🚀 Live API

The model is deployed and accessible at:

**https://credit-card-approval-842429976001.europe-west1.run.app**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/features` | GET | List of model input features |
| `/predict` | POST | Predict credit risk + SHAP explanation |
| `/docs` | GET | Interactive Swagger UI |

---

## 📁 Project Structure

```
credit-card-approval/
├── application_record.csv      # Applicant demographic & financial data
├── credit_record.csv           # Monthly credit payment history
├── data_processed.csv          # Cleaned & engineered dataset
├── model_final.pkl             # Trained Random Forest model
├── selected_features.pkl       # 16 selected features (RFECV)
├── 01_eda.ipynb                # Exploratory Data Analysis
├── 02_preprocessing.ipynb      # Feature engineering & cleaning
├── 03_modeling.ipynb           # Model training & evaluation
├── 04_shap.ipynb               # SHAP explainability
├── app/
│   └── main.py                 # FastAPI application
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📊 Dataset

Source: [Kaggle - Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

| File | Records | Features | Description |
|---|---|---|---|
| `application_record.csv` | 438,557 | 18 | Applicant profile |
| `credit_record.csv` | 1,048,575 | 3 | Monthly payment history |
| After merge | 36,457 | 47 → 16 | Final dataset |

**Target definition:**
- `BAD client` → any month with STATUS 2, 3, 4 or 5 (30+ days overdue)
- `GOOD client` → STATUS 0, C, X only
- Bad client rate: **1.69%** (extreme class imbalance)

---

## 🔍 EDA Key Findings

- **Gender imbalance**: 2x more female applicants (294k vs 144k); men earn ~23% more
- **Income outliers**: 430 applicants earning >1M (handled via log transformation)
- **DAYS_EMPLOYED anomaly**: 365,243 = placeholder for unemployed (17.18% of records)
- **OCCUPATION_TYPE nulls**: 134k nulls — 75k explained by unemployment, 59k imputed as 'Unknown'
- **FLAG_MOBIL**: zero variance (all = 1) — dropped
- **Age distribution**: stable across 30–60 range; lower income at extremes (20–30 and 60–70)

---

## ⚙️ Preprocessing

1. **Target construction** — join `application_record` + `credit_record` on ID
2. **Anomaly handling** — `DAYS_EMPLOYED = 365243` → `IS_UNEMPLOYED` flag
3. **Feature engineering** — `DAYS_BIRTH → AGE`, `DAYS_EMPLOYED → YEARS_EMPLOYED`
4. **Log transformation** — `AMT_INCOME_TOTAL` (heavy right skew)
5. **Null imputation** — `OCCUPATION_TYPE` nulls → 'Unknown'
6. **Binary encoding** — `Y/N → 1/0` for gender, car, realty
7. **One-hot encoding** — 5 categorical variables → 35 dummy columns
8. **Dropped** — `FLAG_MOBIL` (zero variance), `NAME_INCOME_TYPE_Student` (17 cases)

> ⚠️ **Data Leakage Note**: SMOTE applied *inside* cross-validation folds using `ImbPipeline` to prevent synthetic samples from leaking into validation sets. Naive pre-CV SMOTE inflated ROC-AUC from 0.79 → 0.99 (unrealistic).

---

## 🤖 Modeling

### Models Compared (StratifiedKFold, 5 folds)

| Model | ROC-AUC | Recall (Bad) | Precision (Bad) |
|---|---|---|---|
| Logistic Regression | 0.5164 | 0.15 | 0.03 |
| Neural Network (MLP) | 0.5162 | 0.03 | 0.04 |
| LightGBM | 0.6468 | 0.20 | 0.12 |
| XGBoost | 0.6949 | 0.21 | 0.19 |
| Random Forest | 0.8133 | 0.23 | 0.29 |
| **Random Forest (Tuned)** | **0.8242** | **0.22** | **0.29** |
| **Random Forest (16 features)** | **0.8216** | **0.23** | **0.26** |

### Why Random Forest wins

Tree-based models consistently outperform neural networks on tabular data with limited samples. With only 36k rows and 616 bad clients, the MLP lacked sufficient signal — achieving near-random ROC-AUC (0.51).

### Hyperparameter Tuning

`RandomizedSearchCV` with 20 combinations × 5 folds:

```python
Best params:
  n_estimators: 500
  max_depth: None
  max_features: log2
  min_samples_split: 2
  min_samples_leaf: 1
```

### Feature Selection (RFECV)

Reduced from 46 → **16 features** with negligible performance loss (0.8242 → 0.8216).

**Top features by importance:**
1. AGE (13.4%)
2. AMT_INCOME_TOTAL (11.5%)
3. YEARS_EMPLOYED (10.3%)
4. CNT_FAM_MEMBERS (7.5%)

---

## 🔬 SHAP Explainability

SHAP (SHapley Additive exPlanations) provides model transparency — critical in fintech for regulatory compliance (GDPR, EU AI Act).

**Global insights (beeswarm plot):**
- 🔴 **High CNT_FAM_MEMBERS** → increases risk
- 🔴 **Young AGE** → increases risk
- 🔵 **FLAG_OWN_CAR = 1** → reduces risk (economic stability)
- 🔵 **High YEARS_EMPLOYED** → reduces risk

**Individual client profiles:**

| | High Risk Client | Low Risk Client |
|---|---|---|
| Prediction | Bad (70.4%) | Good (3.1%) |
| Age | 29 | 67 |
| Family members | 4 | 1 |
| Years employed | 0.3 | 0.0 (pensioner) |
| Own car | Yes | No |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine learning | scikit-learn, xgboost, lightgbm |
| Imbalanced data | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Deep learning | TensorFlow / Keras |
| API | FastAPI, uvicorn |
| Containerization | Docker |
| Cloud deployment | Google Cloud Run |

---

## 🐳 Run Locally with Docker

```bash
# Clone the repository
git clone https://github.com/DiegoMB99/credit-card-approval.git
cd credit-card-approval

# Build the Docker image
docker build -t credit-card-approval .

# Run the container
docker run -p 8080:8080 credit-card-approval

# Test the API
curl http://localhost:8080/
```

---

## 📡 API Usage Example

```bash
curl -X POST "https://credit-card-approval-842429976001.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CODE_GENDER": 1,
    "FLAG_OWN_CAR": 0,
    "FLAG_OWN_REALTY": 1,
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 11.97,
    "FLAG_WORK_PHONE": 0,
    "FLAG_PHONE": 1,
    "CNT_FAM_MEMBERS": 2.0,
    "AGE": 45,
    "YEARS_EMPLOYED": 8.5,
    "NAME_INCOME_TYPE_Pensioner": false,
    "NAME_INCOME_TYPE_Working": true,
    "NAME_EDUCATION_TYPE_Secondary_secondary_special": false,
    "NAME_FAMILY_STATUS_Single_not_married": false,
    "OCCUPATION_TYPE_Laborers": false,
    "OCCUPATION_TYPE_Unknown": false
}'
```

**Response:**
```json
{
  "probability": 0.0312,
  "prediction": "Good client",
  "risk_factors": [
    {"feature": "AGE", "impact": -0.094},
    {"feature": "YEARS_EMPLOYED", "impact": -0.087},
    {"feature": "CNT_FAM_MEMBERS", "impact": -0.062}
  ]
}
```

---

## 👤 Author

**Diego MB** — Data Scientist  
[GitHub](https://github.com/DiegoMB99) · [LinkedIn](https://linkedin.com/in/diegomb)
