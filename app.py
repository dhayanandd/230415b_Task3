#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install streamlit


# In[5]:


import pandas as pd
import numpy as np
import streamlit as st

# PyCaret (regression) loader & predictor
from pycaret.regression import load_model, predict_model

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Melbourne House Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource(show_spinner=False)
def get_model():
    # NOTE: load without ".pkl" if you used pycaret's save_model
    # e.g. save_model(final_model, "models/melbourne_price_pipeline")
    # Both forms often work, but this is the most reliable:
    return load_model("models/melbourne_price_pipeline")

model = get_model()

st.title("🏠 Melbourne House Price Predictor")
st.caption("Real-time predictions from your registered PyCaret pipeline")

with st.expander("ℹ️ Instructions", expanded=False):
    st.markdown(
        "- Fill in the property details on the left\n"
        "- Click **Predict Price** to get a real-time estimate\n"
        "- Inputs reflect the features used in training (cleaned/engineered)\n"
    )

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Property Inputs")

# Numeric core features
Rooms = st.sidebar.number_input("Rooms", min_value=0, value=3, step=1)
Bedroom2 = st.sidebar.number_input("Bedroom2 (scraped)", min_value=0, value=3, step=1)
Bathroom = st.sidebar.number_input("Bathroom", min_value=0, value=2, step=1)
Car = st.sidebar.number_input("Car Spaces", min_value=0, value=1, step=1)
Distance = st.sidebar.number_input("Distance to CBD (km)", min_value=0.0, value=10.0, step=0.1)
Landsize = st.sidebar.number_input("Landsize (sqm)", min_value=0.0, value=450.0, step=10.0)
BuildingArea = st.sidebar.number_input("Building Area (sqm)", min_value=0.0, value=120.0, step=5.0)
YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=1998, step=1)
Propertycount = st.sidebar.number_input("Propertycount (in suburb)", min_value=0, value=6000, step=50)

# Date/temporal features used in cleaning/training
SaleYear = st.sidebar.number_input("Sale Year", min_value=2007, max_value=2025, value=2017, step=1)
SaleMonth = st.sidebar.slider("Sale Month", min_value=1, max_value=12, value=6)

# Categorical features (collapsed in cleaning; safe default 'Other')
Method = st.sidebar.selectbox("Sale Method", options=["S","Other"])
Region = st.sidebar.text_input("Region (or 'Other')", value="Other")
CouncilArea = st.sidebar.text_input("Council Area (or 'Other')", value="Other")
Suburb = st.sidebar.text_input("Suburb (or 'Other')", value="Other")

# Optional geo (your training likely included Latitude/Longitude; we imputed medians)
use_location = st.sidebar.checkbox("Provide location (Latitude/Longitude)?", value=False)
if use_location:
    Latitude = st.sidebar.number_input("Latitude", value=-37.80, step=0.01, format="%.5f")
    Longitude = st.sidebar.number_input("Longitude", value=145.00, step=0.01, format="%.5f")
else:
    Latitude = -37.80
    Longitude = 145.00

# ---------------------------
# Build the input row with engineered features
# ---------------------------
def build_input_df():
    # PropertyAge (at sale) as in cleaning: max with 0
    property_age = max(0, int(SaleYear) - int(YearBuilt)) if YearBuilt > 0 else 0
    # Price_per_sqm used in training; compute with safe denom
    price_per_sqm = np.nan  # not known at prediction, leave NaN; pipeline can ignore if present
    # If your training strictly used Price_per_sqm, you can set a proxy: 0 or landsize/building heuristics
    # but leaving NaN is typically fine since model shouldn’t expect target-derived features at inference.

    row = {
        'Rooms': int(Rooms),
        'Bedroom2': int(Bedroom2),
        'Bathroom': int(Bathroom),
        'Car': int(Car),
        'Distance': float(Distance),
        'Landsize': float(Landsize),
        'BuildingArea': float(BuildingArea),
        'YearBuilt': float(YearBuilt),
        'CouncilArea': CouncilArea if CouncilArea.strip() else 'Other',
        'Region': Region if Region.strip() else 'Other',
        'Suburb': Suburb if Suburb.strip() else 'Other',
        'Method': Method,
        'Propertycount': int(Propertycount),
        'SaleYear': int(SaleYear),
        'SaleMonth': int(SaleMonth),
        'PropertyAge': int(property_age),
        'Latitude': float(Latitude),
        'Longitude': float(Longitude),
        # Keep this if it existed in training. If not, the model will ignore extra columns.
        'Price_per_sqm': price_per_sqm,
    }

    df = pd.DataFrame([row])

    # Defensive: drop any columns not seen during training? Usually not necessary;
    # PyCaret pipeline handles unseen columns gracefully by selecting the trained features internally.
    return df

# ---------------------------
# Main panel
# ---------------------------
st.subheader("Enter details in the sidebar, then click Predict")

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict Price", type="primary", use_container_width=True):
        input_df = build_input_df()
        try:
            preds = predict_model(model, data=input_df)
            price = float(preds['Label'].iloc[0])
            st.success(f"💰 **Predicted Price:** ${price:,.0f}")
            st.caption("Tip: save a screenshot for your report/demo.")
            with st.expander("See input row"):
                st.dataframe(input_df, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col2:
    st.info("Your model includes preprocessing + encoding inside the pipeline.\n"
            "Inputs here match columns from your cleaned/training dataset.")

st.markdown("---")
st.caption("Model: `models/melbourne_price_pipeline.pkl` • Built with PyCaret • Logged/Registered in MLflow")


# In[6]:


get_ipython().system('jupyter nbconvert --to script Task3.ipynb --output app.py')

