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

# 显示 SHAP 版本
st.info(f"🔍 SHAP version: {shap.__version__}")

# ===== 加载模型、标准化器和背景数据 =====
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

@st.cache_resource
def load_background_data():
    try:
        data = np.load("background_data.npy")
        # 确保背景数据是数值格式
        if data.dtype == object:
            data = data.astype(float)
        return data
    except Exception as e:
        st.warning(f"Background data loading failed: {e}")
        np.random.seed(42)
        return np.random.normal(0, 1, (200, 4))

model = load_model()
scaler = load_scaler()
background_data = load_background_data()

if model is None or scaler is None:
    st.error("❌ Cannot proceed without model and scaler files")
    st.stop()

# 显示调试信息
st.info(f"Model type: {type(model).__name__}")
st.info(f"Scaler features: {len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 'Unknown'}")
st.info(f"Background data shape: {background_data.shape}")
st.info(f"Background data type: {background_data.dtype}")

# ===== 侧边栏输入 =====
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])

# ===== 模型预测 =====
if st.sidebar.button("🧮 Predict"):
    try:
        # 1. 准备输入数据
        input_data = np.array([[protein, sodium, energy, procef_4]], dtype=float)
        
        # 2. 标准化
        input_scaled = scaler.transform(input_data)
        
        # 3. 创建DataFrame并确保数据格式正确
        user_scaled_df = pd.DataFrame(input_scaled, columns=['Protein', 'Sodium', 'Energy', 'procef_4'])
        
        # 4. 强制转换为数值格式
        for col in user_scaled_df.columns:
            user_scaled_df[col] = pd.to_numeric(user_scaled_df[col], errors='coerce')
        
        # 检查是否有 NaN 值
        if user_scaled_df.isnull().any().any():
            st.warning("⚠️ Found NaN values, filling with 0")
            user_scaled_df = user_scaled_df.fillna(0)
        
        # 5. 预测
        prediction = model.predict(user_scaled_df)[0]
        prob = model.predict_proba(user_scaled_df)[0][1]
        
        # 6. 展示结果
        st.subheader("🔍 Prediction Result")
        label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")
        
        # 7. 显示输入数据
        st.subheader("📊 Input Data")
        st.dataframe(user_scaled_df, use_container_width=True)
        
        # 8. SHAP力图 - 完全重写的版本
        st.subheader("📊 SHAP Force Plot")
        
        try:
            # 方法1：使用 TreeExplainer（处理 Pipeline）
            if hasattr(model, 'steps'):
                st.write("🔍 Detected Pipeline model, extracting final model...")
                final_model = model.steps[-1][1]
                st.write(f"Final model type: {type(final_model).__name__}")
                
                # 确保数据是 numpy 数组且为 float 类型
                data_for_shap = user_scaled_df.values.astype(float)
                
                explainer = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(data_for_shap)
                
                with st.expander("Click to view SHAP force plot"):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        data_for_shap[0],
                        feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
            else:
                # 如果不是 Pipeline
                data_for_shap = user_scaled_df.values.astype(float)
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(data_for_shap)
                
                with st.expander("Click to view SHAP force plot"):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        data_for_shap[0],
                        feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
            st.success("✅ SHAP force plot created successfully!")
                    
        except Exception as e:
            st.warning(f"SHAP TreeExplainer failed: {e}")
            
            # 方法2：使用简化的 SHAP 方法
            try:
                st.info("Trying simplified SHAP method...")
                
                # 创建完全数值化的背景数据
                np.random.seed(42)
                clean_background = np.random.normal(0, 1, (100, 4)).astype(float)
                
                # 确保输入数据是 float 类型
                clean_input = user_scaled_df.values.astype(float)
                
                # 使用简化的 SHAP 方法
                explainer = shap.Explainer(model, clean_background)
                shap_values = explainer(clean_input)
                
                # 检查 shap_values 的结构
                st.write(f"SHAP values type: {type(shap_values)}")
                st.write(f"SHAP values shape: {shap_values.values.shape if hasattr(shap_values, 'values') else 'No values attribute'}")
                
                with st.expander("Click to view SHAP force plot"):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # 处理不同的 SHAP 值格式
                    if hasattr(shap_values, 'values'):
                        if len(shap_values.values.shape) == 3:  # 多分类
                            shap_vals = shap_values.values[0, :, 1]  # 健康类别
                            base_val = explainer.expected_value[1]
                        else:  # 二分类
                            shap_vals = shap_values.values[0, :]
                            base_val = explainer.expected_value
                    else:
                        shap_vals = shap_values[0, :]
                        base_val = explainer.expected_value
                    
                    shap.force_plot(
                        base_val,
                        shap_vals,
                        clean_input[0],
                        feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                st.success("✅ SHAP force plot created with simplified method!")
                
            except Exception as e2:
                st.warning(f"Simplified SHAP method failed: {e2}")
                
                # 方法3：只显示特征重要性
                st.info("Falling back to feature importance...")
                
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    feature_importance = model.feature_importances_
                    features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(features, feature_importance)
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importance')
                    
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                                f'{width:.3f}', ha='left', va='center')
                    
                    st.pyplot(fig)
                else:
                    st.info("💡 Neither SHAP nor feature importance is available for this model type.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
