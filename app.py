import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ===== 页面设置 =====
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")

# ===== 加载模型 =====
@st.cache_resource
def load_model():
    try:
        return joblib.load("XGBoost_retrained_model.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler2.pkl")
    except Exception as e:
        st.error(f"Scaler loading failed: {e}")
        return None

model = load_model()
scaler = load_scaler()

if model is None or scaler is None:
    st.error("❌ Cannot proceed without model and scaler files")
    st.stop()

# 显示模型信息
st.write(f"Model type: {type(model).__name__}")
if hasattr(model, 'steps'):
    final_model = model.steps[-1][1]
    st.write(f"Final model type: {type(final_model).__name__}")

# ===== 输入 =====
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1, value=10.0)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0, value=400.0)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0, value=1000.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])

# ===== 预测 =====
if st.sidebar.button("🧮 Predict"):
    try:
        # 准备数据
        input_data = np.array([[protein, sodium, energy, procef_4]], dtype=float)
        input_scaled = scaler.transform(input_data)
        user_scaled_df = pd.DataFrame(input_scaled, columns=['Protein', 'Sodium', 'Energy', 'procef_4'])
        
        # 预测
        prediction = model.predict(user_scaled_df)[0]
        prob = model.predict_proba(user_scaled_df)[0][1]
        
        # 显示结果
        st.subheader("🔍 Prediction Result")
        label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** `{prob:.2f}`")
        
        # 显示数据
        st.subheader("📊 Input Data")
        st.dataframe(user_scaled_df)
        
        # 测试 SHAP - 分步骤调试
        st.subheader("🔬 SHAP Debug")
        
        try:
            # 步骤1：检查数据
            st.write("Step 1: Data check")
            st.write(f"Input data type: {user_scaled_df.dtypes}")
            st.write(f"Input data values: {user_scaled_df.values}")
            st.write(f"Input data shape: {user_scaled_df.shape}")
            
            # 步骤2：创建解释器
            st.write("Step 2: Creating explainer")
            if hasattr(model, 'steps'):
                final_model = model.steps[-1][1]
                st.write(f"Using final model: {type(final_model).__name__}")
                explainer = shap.TreeExplainer(final_model)
            else:
                st.write(f"Using full model: {type(model).__name__}")
                explainer = shap.TreeExplainer(model)
            
            st.write("Explainer created successfully")
            
            # 步骤3：计算 SHAP 值
            st.write("Step 3: Calculating SHAP values")
            shap_values = explainer.shap_values(user_scaled_df.values)
            st.write(f"SHAP values type: {type(shap_values)}")
            st.write(f"SHAP values shape: {shap_values.shape}")
            st.write(f"SHAP values: {shap_values}")
            
            # 步骤4：创建力图
            st.write("Step 4: Creating force plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                user_scaled_df.iloc[0],
                feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                matplotlib=True,
                show=False
            )
            
            st.pyplot(fig)
            plt.close()
            st.success("✅ SHAP force plot created successfully!")
            
        except Exception as e:
            st.error(f"SHAP failed at step: {e}")
            st.write("Full error details:")
            import traceback
            st.code(traceback.format_exc())
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
