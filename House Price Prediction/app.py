# ['number of bedrooms', 'number of bathrooms', 'number of floors',
#  'condition of the house', 'Built Year']

import streamlit as st
import joblib
import numpy as np
from datetime import datetime

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

# --------- Utilities ---------
def format_inr(value: float) -> str:
    """Format number like ₹ 12,34,567"""
    try:
        x = int(round(value))
        s = str(x)
        # Indian numbering format
        if len(s) <= 3:
            return f"₹ {s}"
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        formatted = ",".join(reversed(parts)) + "," + last3
        return f"₹ {formatted}"
    except Exception:
        return f"₹ {value:,.0f}"

@st.cache_resource
def load_model(path: "model.pkl"):
    return joblib.load(path)

# Load model once
try:
    model = load_model("model.pkl")
except Exception as e:
    st.error("Could not load model.pkl. Please ensure the file exists and is a valid scikit-learn estimator.")
    st.exception(e)
    st.stop()

# --------- Sidebar ---------
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    """
**House Price Prediction**  
Provide the inputs and click **Predict**.  
The model was trained on five features:

1. Bedrooms  
2. Bathrooms  
3. Floors  
4. House Condition (1–5)  
5. Built Year
"""
)
st.sidebar.caption("Tip: Use realistic values for best results.")

# --------- Title & Intro ---------
st.title("🏠 House Price Prediction")
st.write("Predict the price of a house from key features. Enter details below and click **Predict**.")

# --------- Input Form ---------
with st.form("house_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        bedrooms  = st.number_input("No. of Bedrooms", min_value=1, max_value=10, step=1, value=2, help="Typical: 1–5")
        bathrooms = st.number_input("No. of Bathrooms", min_value=1, max_value=5,  step=1, value=1, help="Total bathrooms (integer)")

    with col2:
        
        floors    = st.number_input("No. of Floors", min_value=1, max_value=4, step=1, value=1, help="Stories in the house")
        condition = st.slider("Condition of House (1=Poor, 5=Excellent)", min_value=1, max_value=5, value=3)
        current_year = datetime.now().year
        year = st.number_input("Built Year", min_value=1900, max_value=current_year, step=1, value=2000)
        st.write("YEAR DOESN'Tc MAKE ANY IMPACT")
    # Dynamic hints
    if year < 1950:
        st.info("Older property detected — prices can have higher variance.")
    if condition <= 2:
        st.warning("House condition is quite low; model may predict lower prices.")

    # Form buttons
    colA, colB = st.columns([1, 1])
    predict_btn = colA.form_submit_button("🔮 Predict")
    reset_btn   = colB.form_submit_button("♻️ Reset")

# Reset behavior (optional)
if reset_btn:
    # A simple way to reset is to rerun
    st.rerun()

# --------- Validation & Prediction ---------
if predict_btn:
    # Basic validation rules (extend as per your data reality)
    errors = []
    if not (1 <= bedrooms <= 10):
        errors.append("Bedrooms must be between 1 and 10.")
    if not (1 <= bathrooms <= 5):
        errors.append("Bathrooms must be between 1 and 5.")
    if not (1 <= floors <= 4):
        errors.append("Floors must be between 1 and 4.")
    if not (1 <= condition <= 5):
        errors.append("Condition must be between 1 and 5.")
    if not (1900 <= year <= current_year):
        errors.append(f"Built Year must be between 1900 and {current_year}.")

    if errors:
        st.error("Please correct the following issues:")
        for e in errors:
            st.markdown(f"- {e}")
    else:
        # Keep feature order EXACTLY as during training:
        x = [bedrooms, bathrooms, floors, condition, year]
        X = np.array([x], dtype=float)  # shape (1, 5)

        try:
            with st.spinner("Predicting price..."):
                result = model.predict(X)
            price_text = format_inr(result[0])
            st.success(f"The Estimated Price of the House is **{price_text}**")
            st.caption(f"Inputs → Bedrooms: {bedrooms}, Bathrooms: {bathrooms}, Floors: {floors}, "
                       f"Condition: {condition}, Year: {year}")

        except Exception as e:
            # Helpful troubleshooting if model expects preprocessing/pipeline
            st.error("Prediction failed. Possible causes:\n"
                     "- Feature order or data type mismatch with training\n"
                     "- Model expects preprocessing pipeline (e.g., scaler/encoder)\n"
                     "- Incompatible scikit-learn/joblib versions")
            st.exception(e)

# --------- Optional: Advanced Dynamic Features Ideas ---------
with st.expander("✨ Optional Enhancements"):
    st.markdown(
        """
- **Input presets**: Provide quick preset buttons (e.g., 2BHK Standard).
- **Range guardrails**: Warn if inputs are outside the **training data ranges** (min/max from your training set).
- **Currency toggle**: INR/USD/EUR switch.
- **Confidence intervals**: Show prediction interval via bootstrapping or quantile models.
- **Feature importance**: If the model supports it, show top features impacting price.
- **Persist history**: Keep last 5 predictions in session state for comparison.
- **Download**: Allow user to download prediction summary as a PDF/CSV.
"""
    )