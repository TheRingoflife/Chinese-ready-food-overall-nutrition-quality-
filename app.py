import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("This application uses a trained XGBoost model to classify whether a ready-to-eat food is **healthy** based on simplified input features.")

# Load model and background data
@st.cache_resource
def load_model():
    return joblib.load("XGBoost_retrained_model.pkl")

@st.cache_resource
def load_background_data():
    return np.load("background_data.npy")

model = load_model()
background_data = load_background_data()
explainer = shap.Explainer(model, background_data)

# Helper function for float input
def parse_float_input(label, unit=""):
    value = st.sidebar.text_input(f"{label} {unit}")
    try:
        return float(value)
    except:
        return None

# Sidebar input
st.sidebar.header("🔢 Input Variables")
protein = parse_float_input("Protein", "(g/100g)")
sodium = parse_float_input("Sodium", "(mg/100g)")
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])
total_fat = parse_float_input("Total Fat", "(g/100g)")
energy = parse_float_input("Energy", "(kJ/100g)")

# Prediction trigger
if st.sidebar.button("🧮 Predict"):
    if None in [protein, sodium, total_fat, energy]:
        st.error("Please enter valid numeric values for all fields.")
    else:
        user_input = np.array([[protein, sodium, procef_4, total_fat, energy]])
        prediction = model.predict(user_input)[0]
        prob = model.predict_proba(user_input)[0][1]

        st.subheader("🔍 Prediction Result")
        label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")

        # SHAP force plot
        st.subheader("📊 SHAP Force Plot (Model Explanation)")
        shap_values = explainer(user_input)

        with st.expander("Click to view SHAP force plot"):
            st.markdown("This plot shows how each feature influences the model's prediction.")
            force_html = shap.plots.force(
                explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value,
                shap_values[1][0] if isinstance(shap_values, list) else shap_values[0],
                user_input,
                matplotlib=False
            )
            components.html(force_html.html(), height=300)

# Footer
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
