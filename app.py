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
    return joblib.load("scaler2.pkl")

@st.cache_resource
def load_background_data():
    return np.load("background_data.npy")

model = load_model()
scaler = load_scaler()
background_data = load_background_data()
explainer = shap.Explainer(model, background_data)

# ===== 侧边栏输入 =====
st.sidebar.header("🔢 Input Variables")
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)

# ===== 模型预测 + SHAP 可解释性 =====
if st.sidebar.button("🧮 Predict"):
    # 1. 准备输入数据（4个特征）
    input_data = np.array([[sodium, protein, procef_4, energy]])
    
    # 2. 标准化
    input_scaled = scaler.transform(input_data)
    
    # 3. 创建DataFrame
    user_scaled_df = pd.DataFrame(input_scaled, columns=['Sodium', 'Protein', 'procef_4', 'Energy'])
    
    # 4. 预测
    prediction = model.predict(user_scaled_df)[0]
    prob = model.predict_proba(user_scaled_df)[0][1]
    
    # 5. 展示结果
    st.subheader("🔍 Prediction Result")
    label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")
    
    # 6. SHAP力图
    st.subheader("📊 SHAP Force Plot (Model Explanation)")
    with st.expander("Click to view SHAP force plot"):
        shap_values = explainer(user_scaled_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if not isinstance(shap_values, shap.Explanation):
            shap_values = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=user_scaled_df.values,
                feature_names=user_scaled_df.columns.tolist()
            )
        force_html = shap.force_plot(
            base_value=shap_values.base_values,
            shap_values=shap_values.values,
            features=shap_values.data,
            feature_names=shap_values.feature_names,
            matplotlib=False
        )
        components.html(shap.getjs() + force_html.html(), height=400)

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
