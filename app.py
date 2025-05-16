import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ========== Page Setup ==========
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("This app uses a trained XGBoost model to classify whether a ready-to-eat food is **healthy**, based on simplified input features.")

# ========== Load Model and Background Data ==========
@st.cache_resource
def load_model():
    return joblib.load("XGBoost_retrained_model.pkl")

@st.cache_resource
def load_background_data():
    return pd.read_pickle("background_data.pkl")  # DataFrame with correct feature names

model = load_model()
background_data = load_background_data()

# Initialize SHAP explainer (Tree-based)
explainer = shap.Explainer(model, background_data)

# ========== Sidebar Input ==========
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])
total_fat = st.sidebar.number_input("Total Fat (g/100g)", min_value=0.0, step=0.1)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)

# ========== Predict & Explain ==========
if st.sidebar.button("🧮 Predict"):
    # 1. Make DataFrame for user input
    user_input_df = pd.DataFrame([{
        'Protein': protein,
        'Sodium': sodium,
        'procef_4': procef_4,
        'Total fat': total_fat,
        'Energy': energy
    }])

    # 2. Make prediction
    prediction = model.predict(user_input_df)[0]
    prob = model.predict_proba(user_input_df)[0][1]

    # 3. Show prediction
    st.subheader("🔍 Prediction Result")
    label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")

    # 4. SHAP force plot
    st.subheader("📊 SHAP Force Plot (Model Explanation)")
    with st.expander("Click to view SHAP force plot"):
        st.markdown("This plot shows how each feature contributes to the prediction.")

        shap_values = explainer(user_input_df)

        # Ensure Explanation format for SHAP compatibility
        if not isinstance(shap_values, shap.Explanation):
            shap_values = shap.Explanation(
                values=shap_values[1] if isinstance(shap_values, list) else shap_values,
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=user_input_df.values,
                feature_names=user_input_df.columns.tolist()
            )

        force_plot_html = shap.force_plot(
            base_value=shap_values.base_values,
            shap_values=shap_values.values,
            features=shap_values.data,
            feature_names=shap_values.feature_names,
            matplotlib=False
        )

        # Display in Streamlit
        components.html(shap.getjs() + force_plot_html.html(), height=300)

# ========== Footer ==========
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
