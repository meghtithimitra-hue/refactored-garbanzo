import streamlit as st
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gb_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("Tobacco-Related Mortality Prediction")

avg_smoking_prevalence = st.slider("Average Smoking Prevalence (%)", 10.0, 30.0, 20.0)
cessation_prescriptions_per_smoker = st.slider("Cessation Prescriptions per Smoker", 0.0, 200.0, 50.0)
affordability_index = st.slider("Affordability of Tobacco Index", 50.0, 100.0, 75.0)
smoking_change = st.slider("Smoking Prevalence Change", -2.0, 2.0, 0.0)

if st.button("Predict Mortality"):
    data = pd.DataFrame([{
        "avg_smoking_prevalence": avg_smoking_prevalence,
        "cessation_prescriptions_per_smoker": cessation_prescriptions_per_smoker,
        "affordability_of_tobacco_index": affordability_index,
        "smoking_prevalence_change": smoking_change
    }])

    prediction = model.predict(data)[0]
    st.success(f"Estimated Tobacco-Related Deaths: {int(prediction):,}")
