"""
Dashboard: live prediction, data validation status, and drift
monitoring in one view. Lightweight stand-in for the Grafana +
Postgres monitoring layer in the full-scale version.
"""
import json
import subprocess
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Wine Quality MLOps Lite", layout="wide")
st.title("Wine quality — lightweight MLOps demo")

API_URL = "http://127.0.0.1:8000/predict"

FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]

tab_predict, tab_validate, tab_drift = st.tabs(["Predict", "Data validation", "Drift monitoring"])

with tab_predict:
    st.subheader("Try a prediction")
    reference = pd.read_csv("data/reference.csv")
    cols = st.columns(3)
    values = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        default = float(reference[col].mean())
        with cols[i % 3]:
            values[col] = st.number_input(col, value=round(default, 3))

    if st.button("Predict"):
        payload = {
            "fixed_acidity": values["fixed acidity"],
            "volatile_acidity": values["volatile acidity"],
            "citric_acid": values["citric acid"],
            "residual_sugar": values["residual sugar"],
            "chlorides": values["chlorides"],
            "free_sulfur_dioxide": values["free sulfur dioxide"],
            "total_sulfur_dioxide": values["total sulfur dioxide"],
            "density": values["density"],
            "pH": values["pH"],
            "sulphates": values["sulphates"],
            "alcohol": values["alcohol"],
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            st.success(f"Prediction: **{result['label']}** (probability good: {result['probability_good']})")
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the FastAPI server — make sure uvicorn is running in another terminal.")

with tab_validate:
    st.subheader("Great Expectations validation results")
    if st.button("Re-run validation"):
        subprocess.run(["python", "data/validate_data.py"])
        st.rerun()
    try:
        with open("reports/validation_report.json") as f:
            report = json.load(f)
        c1, c2 = st.columns(2)
        c1.metric("reference.csv", "PASS" if report["reference_passed"] else "FAIL")
        c2.metric("bad_batch.csv", "PASS" if report["bad_batch_passed"] else "FAIL")
    except FileNotFoundError:
        st.info("No validation report yet — run `python data/validate_data.py` first.")

with tab_drift:
    st.subheader("Drift check (KS-test vs reference distribution)")
    if st.button("Re-run drift check"):
        subprocess.run(["python", "monitoring/check_drift.py"])
        st.rerun()
    try:
        with open("reports/drift_report.json") as f:
            report = json.load(f)
        for label, key in [("New batch (expect no drift)", "new_batch"),
                            ("Drifted batch (synthetic shift)", "drifted_batch")]:
            st.markdown(f"**{label}**")
            df = pd.DataFrame(report[key]["details"]).T
            df["drifted"] = df["drifted"].map({True: "DRIFT", False: "ok"})
            st.dataframe(df, use_container_width=True)
    except FileNotFoundError:
        st.info("No drift report yet — run `python monitoring/check_drift.py` first.")