import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ===== 页面设置 =====
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("This app uses a trained XGBoost model to classify whether a ready-to-eat food is **healthy**, based on simplified input features.")

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    return joblib.load("XGBoost_retrained_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_background_data():
    return pd.read_pickle("background_data.pkl")

model = load_model()
scaler = load_scaler()
background_data = load_background_data()
explainer = shap.Explainer(model, background_data)

# ===== 侧边栏输入 =====
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])
total_fat = st.sidebar.number_input("Total Fat (g/100g)", min_value=0.0, step=0.1)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)

# ===== 模型预测 + SHAP 可解释性 =====
if st.sidebar.button("🧮 Predict"):
    # 用户输入转为 DataFrame
    user_input_df = pd.DataFrame([{
        'Protein': protein,
        'Sodium': sodium,
        'procef_4': procef_4,
        'Total fat': total_fat,
        'Energy': energy
    }])

    # 标准化用户输入
    user_input_scaled = scaler.transform(user_input_df)

    # 模型预测
    prediction = model.predict(user_input_scaled)[0]
    prob = model.predict_proba(user_input_scaled)[0][1]

    # 展示预测结果
    st.subheader("🔍 Prediction Result")
    label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")

    # SHAP 力图解释
    st.subheader("📊 SHAP Force Plot (Explanation)")
    with st.expander("Click to view SHAP force plot"):
        shap_values = explainer(user_input_scaled)

        # 保证为 Explanation 对象
        if not isinstance(shap_values, shap.Explanation):
            shap_values = shap.Explanation(
                values=shap_values[1] if isinstance(shap_values, list) else shap_values,
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=user_input_scaled,
                feature_names=user_input_df.columns.tolist()
            )

        force_plot_html = shap.force_plot(
            base_value=shap_values.base_values,
            shap_values=shap_values.values,
            features=shap_values.data,
            feature_names=shap_values.feature_names,
            matplotlib=False
        )
        components.html(shap.getjs() + force_plot_html.html(), height=300)

# 页脚
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
