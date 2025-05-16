import streamlit as st
import pandas as pd
import joblib
import shap
import streamlit.components.v1 as components

# 页面设置
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("This app uses a trained XGBoost model to classify whether a ready-to-eat food is **healthy**, based on simplified input features.")

# ========== 加载模型和背景数据 ==========
@st.cache_resource
def load_model():
    return joblib.load("XGBoost_retrained_model.pkl")

@st.cache_resource
def load_background_data():
    return pd.read_pickle("background_data.pkl")

model = load_model()
background_data = load_background_data()
explainer = shap.Explainer(model, background_data)

# ========== 输入区域 ==========
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])
total_fat = st.sidebar.number_input("Total Fat (g/100g)", min_value=0.0, step=0.1)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)

# ========== 模型预测与解释 ==========
if st.sidebar.button("🧮 Predict"):
    user_input = pd.DataFrame([[protein, sodium, procef_4, total_fat, energy]],
                              columns=['Protein', 'Sodium', 'procef_4', 'Total fat', 'Energy'])
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Prediction:** {'✅ Healthy' if prediction == 1 else '⚠️ Unhealthy'}")
    st.markdown(f"**Confidence:** `{prob:.2f}`")

    st.subheader("📊 SHAP Force Plot (Explanation)")
    shap_values = explainer(user_input)
    with st.expander("Click to view SHAP force plot"):
        st.markdown("This plot shows how each feature contributes to the prediction.")
        force_plot_html = shap.plots.force(shap_values)
        components.html(force_plot_html, height=300)

# 页脚
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
