import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="UrbanValuate",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# GLOBAL LOAD MODELS (SAFE)
# ---------------------------------------------------------
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

classifier = load_model("models/rf_classifier_pipeline.joblib")
regressor = load_model("models/rf_regressor_pipeline.joblib")


# ---------------------------------------------------------
# CUSTOM CSS + ANIMATION + GLASS HEADER
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    font-family: 'Inter', sans-serif;
    background-color: #0B1221;
}

.header-box {
    width: 100%;
    padding: 28px;
    margin-bottom: 20px;
    border-radius: 18px;
    backdrop-filter: blur(14px);
    background: rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 20px;
    animation: fadeIn 0.8s ease-out;
    border: 1px solid rgba(255,255,255,0.08);
}

.logo-badge {
    width: 60px;
    height: 60px;
    border-radius: 14px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: 900;
    font-size: 22px;
    background: linear-gradient(135deg,#7c3aed,#2563eb);
    color: white;
    animation: glow 3s infinite alternate ease-in-out;
}

@keyframes glow {
  from { box-shadow: 0 0 4px #7c3aed; }
  to { box-shadow: 0 0 22px #2563eb; }
}

.header-title {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: -2px;
}

.header-subtitle {
    font-size: 15px;
    color: #AAB2C9;
}

.card {
    padding: 24px;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}

.result-box {
    padding: 18px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
}

.footer {
    margin-top: 80px;
    padding: 20px 0;
    text-align: center;
    color: #7782A4;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="logo-badge">UV</div>
    <div>
        <div class="header-title">UrbanValuate</div>
        <div class="header-subtitle">Smart property insights for confident decisions.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# NAVIGATION (Streamlit Tabs)
# ---------------------------------------------------------
tabs = st.tabs(["🏠 Home", "🔍 Explore Property", "📈 Price Forecast", "ℹ️ About"])


# ---------------------------------------------------------
# HOME TAB
# ---------------------------------------------------------
with tabs[0]:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("Welcome to UrbanValuate")

    st.markdown("""
UrbanValuate transforms real-estate data into clear, human-friendly insights — engineered for buyers, sellers, and small investors.

### What you can do here:
- Get instant property investment verdicts  
- View a confidence score backed by machine learning  
- Generate rule-based and model-driven 5-year price forecasts  
- Upload property CSVs for quick valuation  
""")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# EXPLORE PROPERTY TAB
# ---------------------------------------------------------
with tabs[1]:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("Explore Property Value")

    uploaded = st.file_uploader("Upload a single-property CSV (Optional)", type=["csv"])
    uploaded_row = None

    if uploaded:
        df = pd.read_csv(uploaded)
        if df.shape[0] == 1:
            uploaded_row = df.iloc[0]
        else:
            st.warning("Please upload a CSV containing exactly ONE property.")

    with st.form("property_form"):

        col1, col2 = st.columns(2)

        with col1:
            city = st.text_input("City", uploaded_row.get("City") if uploaded_row is not None else "Mumbai")
            state = st.text_input("State", uploaded_row.get("State") if uploaded_row is not None else "Maharashtra")

            property_type = st.selectbox(
                "Property Type",
                ["Apartment", "Villa", "Independent House", "Studio"],
                index=0
            )

            bhk = st.number_input(
                "BHK",
                min_value=1.0,
                value=float(uploaded_row.get("BHK", 2.0)) if uploaded_row is not None else 2.0
            )

            amenities = st.number_input(
                "Amenities Count",
                min_value=0,
                max_value=20,
                value=int(uploaded_row.get("Amenities_Count", 3)) if uploaded_row is not None else 3
            )

        with col2:
            size = st.number_input(
                "Size (sqft)",
                min_value=50.0,
                value=float(uploaded_row.get("Size_in_SqFt", 900.0)) if uploaded_row else 900.0
            )

            psf = st.number_input(
                "Price per SqFt",
                min_value=50.0,
                value=float(uploaded_row.get("Price_per_SqFt", 1800.0)) if uploaded_row else 1800.0
            )

            age = st.number_input(
                "Age of Property (Years)",
                min_value=0.0,
                value=float(uploaded_row.get("Age_of_Property", 5.0)) if uploaded_row else 5.0
            )

            parking = st.selectbox("Parking Available", ["No", "Yes"])
            transport = st.selectbox("Public Transport Score", ["Low", "Medium", "High"])

        furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
        owner = st.selectbox("Owner Type", ["Owner", "Builder"])

        submit = st.form_submit_button("Evaluate Property")

    if submit:

        if classifier is None:
            st.error("Model not found. Please ensure models are in /models/")
        else:
            transport_map = {"Low": 0, "Medium": 1, "High": 2}
            parking_flag = 1 if parking == "Yes" else 0

            input_df = pd.DataFrame([{
                "City": city,
                "State": state,
                "Property_Type": property_type,
                "BHK": bhk,
                "Size_in_SqFt": size,
                "Price_per_SqFt": psf,
                "Age_of_Property": age,
                "Amenities_Count": amenities,
                "Parking_Available": parking_flag,
                "Public_Transport_Score": transport_map[transport],
                "Furnished_Status": furnished,
                "Owner_Type": owner
            }])

            prob = float(classifier.predict_proba(input_df)[0][1])

            # RESULT BOX
            if prob >= 0.75:
                st.success(f"High Investment Potential (Confidence: {prob:.2f})")
            elif prob >= 0.50:
                st.info(f"Moderate Investment Potential (Confidence: {prob:.2f})")
            else:
                st.warning(f"Low Investment Potential (Confidence: {prob:.2f})")

            # USER-FRIENDLY EXPLANATION
            st.markdown("""
### 📘 How to Interpret This
The investment score represents how similar this property is to historically strong investments 
based on price trends, size ratios, amenities, and local market behavior.

### Key Influencing Factors:
- Locality appreciation patterns  
- Price per SqFt vs. city benchmarks  
- Age of property  
- Amenities + transport access  
- Ownership type & furnishing  
""")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# FORECAST TAB
# ---------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("5-Year Price Forecast")

    with st.form("forecast_form"):

        colA, colB = st.columns(2)

        with colA:
            f_city = st.text_input("City", "Mumbai")
            f_size = st.number_input("Size (SqFt)", min_value=50.0, value=900.0)

        with colB:
            f_price = st.number_input("Current Price (Lakhs)", min_value=1.0, value=150.0)

        f_submit = st.form_submit_button("Generate Forecast")

    if f_submit:

        rule_future = f_price * ((1 + 0.08) ** 5)

        if regressor:
            pred_df = pd.DataFrame([{
                "City": f_city,
                "State": "Unknown",
                "Property_Type": "Apartment",
                "BHK": 2,
                "Size_in_SqFt": f_size,
                "Price_per_SqFt": 2000,
                "Age_of_Property": 5,
                "Amenities_Count": 3,
                "Parking_Available": 1,
                "Public_Transport_Score": 1,
                "Furnished_Status": "Unfurnished",
                "Owner_Type": "Owner"
            }])

            model_future = float(regressor.predict(pred_df)[0])
        else:
            model_future = None

        st.subheader("Rule-Based 5-Year Projection")
        st.write(f"**{rule_future:.2f} Lakhs**")

        st.subheader("Model-Driven 5-Year Projection")
        st.write(f"**{model_future:.2f} Lakhs**" if model_future else "Model unavailable.")

        st.markdown("""
---

### 📘 What This Means
- **Rule-based projection** assumes a safe 8% annual growth.  
- **Model-driven forecast** uses real-estate patterns learned from thousands of listings.

Use both as guidance, not absolute predictions.

""")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# ABOUT TAB
# ---------------------------------------------------------
with tabs[3]:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.header("About UrbanValuate")

    st.markdown("""
UrbanValuate is designed to make real-estate intelligence accessible to everyone.  
No jargon. No confusion. Just clear insights powered by modern machine learning.

### Core Features
- Property investment analysis  
- 5-year price projection  
- CSV-based valuation  
- Clean, human-first explanations  

Built with ❤️ by Mohammed Akdas Ansari.
""")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
© 2025 Mohammed Akdas Ansari — All Rights Reserved
</div>
""", unsafe_allow_html=True)
