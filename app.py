#!/usr/bin/env python
# coding: utf-8

# app.py — Melbourne House Price Predictor (robust, fixed)
import pandas as pd
import numpy as np
import streamlit as st
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Melbourne House Price Predictor", page_icon="🏠", layout="centered")

# ---------------------------
# Load trained PyCaret pipeline
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("melbourne_price_pipeline")

model = get_model()

st.title("Melbourne House Price Predictor")
st.caption("Alagarsamy Dhayanand | 230415b")

with st.expander("Instructions", expanded=False):
    st.markdown(
        "- Enter property details on the left.\n"
        "- Click **Predict Price** to get an estimate.\n"
        "- Some nonessential columns are filled with safe placeholders to match the training schema."
    )

# ---------------------------
# Sidebar inputs (core features)
# ---------------------------
st.sidebar.header("Property Inputs")

Rooms = st.sidebar.number_input("Rooms", min_value=0, value=3, step=1)
Bedroom2 = st.sidebar.number_input("Bedroom2 (scraped)", min_value=0, value=3, step=1)
Bathroom = st.sidebar.number_input("Bathroom", min_value=0, value=2, step=1)
Car = st.sidebar.number_input("Car Spaces", min_value=0, value=1, step=1)

Distance = st.sidebar.number_input("Distance to CBD (km)", min_value=0.0, value=10.0, step=0.1)
Landsize = st.sidebar.number_input("Landsize (sqm)", min_value=0.0, value=450.0, step=10.0)
BuildingArea = st.sidebar.number_input("Building Area (sqm)", min_value=0.0, value=120.0, step=5.0)

YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=1998, step=1)
Propertycount = st.sidebar.number_input("Propertycount (in suburb)", min_value=0, value=6000, step=50)

SaleYear = st.sidebar.number_input("Sale Year", min_value=2007, max_value=2025, value=2017, step=1)
SaleMonth = st.sidebar.slider("Sale Month", min_value=1, max_value=12, value=6)

# Categorical
Method = st.sidebar.selectbox("Sale Method", options=["S", "Other"], index=0)
Type = st.sidebar.selectbox("Property Type", options=['h', 'u', 't', 'dev site', 'o res'], index=0)

Region = st.sidebar.text_input("Region (or 'Other')", value="Other")
CouncilArea = st.sidebar.text_input("Council Area (or 'Other')", value="Other")
Suburb = st.sidebar.text_input("Suburb (or 'Other')", value="Other")

# Optional geo inputs (trained pipeline likely included Latitude/Longitude)
use_location = st.sidebar.checkbox("Provide exact Latitude/Longitude?", value=False)
if use_location:
    Latitude = st.sidebar.number_input("Latitude", value=-37.80, step=0.01, format="%.5f")
    Longitude = st.sidebar.number_input("Longitude", value=145.00, step=0.01, format="%.5f")
else:
    # Sensible Melbourne defaults (median-ish), matches cleaning imputation
    Latitude = -37.80
    Longitude = 145.00

# ---------------------------
# Build prediction row (incl. placeholders for expected columns)
# ---------------------------
def build_input_df():
    property_age = max(0, int(SaleYear) - int(YearBuilt)) if YearBuilt > 0 else 0

    row = {
        # Core numeric/categorical inputs
        'Rooms': int(Rooms),
        'Bedroom2': int(Bedroom2),
        'Bathroom': int(Bathroom),
        'Car': int(Car),
        'Distance': float(Distance),
        'Landsize': float(Landsize),
        'BuildingArea': float(BuildingArea),
        'YearBuilt': float(YearBuilt),
        'CouncilArea': CouncilArea.strip() or 'Other',
        'Region': Region.strip() or 'Other',
        'Suburb': Suburb.strip() or 'Other',
        'Method': Method,
        'Type': Type,                          # important categorical
        'Propertycount': int(Propertycount),
        'SaleYear': int(SaleYear),
        'SaleMonth': int(SaleMonth),
        'PropertyAge': int(property_age),
        'Latitude': float(Latitude),
        'Longitude': float(Longitude),

        # If present during training; safe to send NaN at inference
        'Price_per_sqm': np.nan,

        # Placeholders for columns your pipeline expects (from your error)
        'Address': 'Unknown',
        'Seller': 'Other',
        'Postcode': '3000',                     # string safe if casted in cleaning
        'Date': '2017-06-15',                   # ISO date string
        'LogPrice': 0.0                         # neutral placeholder (proper fix: retrain ignoring this)
    }
    return pd.DataFrame([row])

# ---------------------------
# Predict button + robust prediction column handling
# ---------------------------
st.subheader("Enter details in the sidebar, then click Predict")

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict Price", type="primary", use_container_width=True):
        input_df = build_input_df()
        try:
            preds = predict_model(model, data=input_df)

            # Robustly find the prediction column (PyCaret 2.x vs 3.x)
            added_cols = [c for c in preds.columns if c not in input_df.columns]
            preferred = ['Label', 'prediction_label', 'Prediction', 'Predicted', 'Score']
            pred_col = next((c for c in preferred if c in preds.columns), None)
            if pred_col is None:
                pred_col = added_cols[0] if added_cols else None

            if pred_col is None:
                raise ValueError(f"Could not find prediction column. Columns: {list(preds.columns)}")

            price = float(preds[pred_col].iloc[0])
            st.success(f" **Predicted Price:** ${price:,.0f}")

            with st.expander("See input & output (debug)"):
                st.write("Prediction column used:", pred_col)
                st.dataframe(preds, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
