# app.py — Melbourne House Price Predictor (Streamlit Cloud-ready, robust loader)
import os
import pandas as pd
import numpy as np
import streamlit as st

# Try to import PyCaret; fall back to joblib if needed
try:
    from pycaret.regression import load_model, predict_model
    _HAVE_PYCARET = True
except Exception:
    _HAVE_PYCARET = False
    from joblib import load as joblib_load

st.set_page_config(page_title="Melbourne House Price Predictor", page_icon="🏠", layout="centered")

# ---------------------------
# Model loader with fallbacks
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    """
    Try the common save patterns:
      1) PyCaret base path (no .pkl)
      2) Explicit .pkl path
      3) joblib.load as a fallback
    """
    candidates = [
        "models/melbourne_price_pipeline",        # PyCaret save_model base path
        "models/melbourne_price_pipeline.pkl",    # explicit pickle path
    ]

    last_err = None
    for path in candidates:
        try:
            if _HAVE_PYCARET:
                return load_model(path)
        except Exception as e:
            last_err = e

        # Fallback to joblib if PyCaret load fails or not available
        try:
            return joblib_load(path if path.endswith(".pkl") else f"{path}.pkl")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not load model from {candidates}. Last error: {last_err}")

model = load_pipeline()

st.title("🏠 Melbourne House Price Predictor")
st.caption("Real-time predictions with a saved PyCaret pipeline (tracked in MLflow)")

with st.expander("ℹ️ Instructions", expanded=False):
    st.markdown(
        "- Enter property details on the left and click **Predict Price**.\n"
        "- If you **retrained** the model to remove non-runtime/leaky columns, turn on the toggle below.\n"
        "- If you did **not** retrain yet, keep the toggle OFF (the app will include placeholder columns expected by the original pipeline)."
    )

# Toggle depending on whether you retrained the model to ignore placeholders
retrained_no_placeholders = st.toggle("I retrained without placeholders (ignore Address/Seller/Date/Postcode/LogPrice/Price_per_sqm)", value=False)

# ---------------------------
# Sidebar inputs (runtime features)
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

Method = st.sidebar.selectbox("Sale Method", options=["S", "Other"], index=0)
Type = st.sidebar.selectbox("Property Type", options=['h', 'u', 't', 'dev site', 'o res'], index=0)

Region = st.sidebar.text_input("Region (or 'Other')", value="Other")
CouncilArea = st.sidebar.text_input("Council Area (or 'Other')", value="Other")
Suburb = st.sidebar.text_input("Suburb (or 'Other')", value="Other")

use_location = st.sidebar.checkbox("Provide Latitude/Longitude?", value=False)
if use_location:
    Latitude = st.sidebar.number_input("Latitude", value=-37.80, step=0.01, format="%.5f")
    Longitude = st.sidebar.number_input("Longitude", value=145.00, step=0.01, format="%.5f")
else:
    Latitude = -37.80
    Longitude = 145.00

# ---------------------------
# Build input rows (two modes)
# ---------------------------
def build_runtime_only_row():
    """Row for retrained model that ignores placeholders."""
    property_age = max(0, int(SaleYear) - int(YearBuilt)) if YearBuilt > 0 else 0
    return {
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
        'Type': Type,
        'Propertycount': int(Propertycount),
        'SaleYear': int(SaleYear),
        'SaleMonth': int(SaleMonth),
        'PropertyAge': int(property_age),
        'Latitude': float(Latitude),
        'Longitude': float(Longitude),
    }

def build_with_placeholders_row():
    """Row for original pipeline that still expects the extra columns."""
    row = build_runtime_only_row()
    # Safe placeholders (match earlier training schema)
    row.update({
        'Price_per_sqm': np.nan,
        'Address': 'Unknown',
        'Seller': 'Other',
        'Postcode': '3000',          # keep as string to match cleaning
        'Date': '2017-06-15',        # ISO-format string
        'LogPrice': 0.0
    })
    return row

def build_input_df():
    row = build_runtime_only_row() if retrained_no_placeholders else build_with_placeholders_row()
    return pd.DataFrame([row])

# ---------------------------
# Prediction button
# ---------------------------
st.subheader("Enter details on the left, then click Predict")

left, right = st.columns(2)
with left:
    if st.button("Predict Price", type="primary", use_container_width=True):
        input_df = build_input_df()
        try:
            # Use PyCaret's predict_model if available; else, try scikit-learn-style .predict
            if _HAVE_PYCARET:
                preds = predict_model(model, data=input_df)
                # Find prediction column robustly
                added_cols = [c for c in preds.columns if c not in input_df.columns]
                preferred = ['Label', 'prediction_label', 'Prediction', 'Predicted', 'Score']
                pred_col = next((c for c in preferred if c in preds.columns), None) or (added_cols[0] if added_cols else None)
                if pred_col is None:
                    raise ValueError(f"Could not find prediction column. Columns: {list(preds.columns)}")
                price = float(preds[pred_col].iloc[0])
            else:
                # Fallback for joblib-loaded scikit-learn pipeline
                price = float(model.predict(input_df)[0])

            st.success(f"💰 **Predicted Price:** ${price:,.0f}")

            with st.expander("See input & output (debug)"):
                if _HAVE_PYCARET:
                    st.write("Prediction column detected:", pred_col)
                    st.dataframe(preds, use_container_width=True)
                else:
                    st.dataframe(input_df.assign(Prediction=price), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with right:
    st.info(
        "This app loads a saved pipeline (preprocessing + model). "
        "If you retrained without non-runtime columns, turn on the toggle above. "
        "Otherwise, the app includes placeholder fields so the original pipeline schema matches."
    )

st.markdown("---")
st.caption("Model path tried: `models/melbourne_price_pipeline` / `.pkl` • PyCaret 2/3 compatible")
