import pickle
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and features
with open("model_final.pkl", "rb") as f:
    model = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# Extract RF from pipeline
rf_model = model.named_steps["model"]
explainer = shap.TreeExplainer(rf_model)

# FastAPI app
app = FastAPI(title="Credit Card Approval API", version="1.0")

# Input schema
class ClientData(BaseModel):
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    CNT_FAM_MEMBERS: float
    AGE: int
    YEARS_EMPLOYED: float
    NAME_INCOME_TYPE_Pensioner: bool
    NAME_INCOME_TYPE_Working: bool
    NAME_EDUCATION_TYPE_Secondary_secondary_special: bool
    NAME_FAMILY_STATUS_Single_not_married: bool
    OCCUPATION_TYPE_Laborers: bool
    OCCUPATION_TYPE_Unknown: bool

    def to_model_input(self):
        d = self.dict()
        d["NAME_EDUCATION_TYPE_Secondary / secondary special"] = d.pop("NAME_EDUCATION_TYPE_Secondary_secondary_special")
        d["NAME_FAMILY_STATUS_Single / not married"] = d.pop("NAME_FAMILY_STATUS_Single_not_married")
        return pd.DataFrame([d], columns=selected_features)

@app.get("/")
def root():
    return {"message": "Credit Card Approval API is running"}

@app.post("/predict")
def predict(client: ClientData):
    # Build input dataframe
    input_data = client.to_model_input()
    # Prediction
    probability = model.predict_proba(input_data)[0][1]
    prediction = "Bad client" if probability >= 0.5 else "Good client"

    # SHAP values for explanation
    shap_values = explainer.shap_values(input_data)
    shap_bad = shap_values[:, :, 1][0]

    # Top 3 risk factors
    shap_df = pd.DataFrame({
        "feature": selected_features,
        "impact": shap_bad
    }).reindex(pd.Series(shap_bad).abs().sort_values(ascending=False).index)

    top_3 = shap_df.head(3).to_dict(orient="records")

    return {
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_factors": top_3
    }

@app.get("/features")
def features():
    return {"selected_features": selected_features}