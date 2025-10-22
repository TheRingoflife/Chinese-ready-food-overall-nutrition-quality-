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
        return np.load("background_data.npy")
    except Exception as e:
        st.warning(f"Background data loading failed: {e}")
        # 创建模拟背景数据
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
        input_data = np.array([[protein, sodium, energy, procef_4]])
        
        # 2. 标准化
        input_scaled = scaler.transform(input_data)
        
        # 3. 创建DataFrame
        user_scaled_df = pd.DataFrame(input_scaled, columns=['Protein', 'Sodium', 'Energy', 'procef_4'])
        
        # 4. 预测
        prediction = model.predict(user_scaled_df)[0]
        prob = model.predict_proba(user_scaled_df)[0][1]
        
        # 5. 展示结果
        st.subheader("🔍 Prediction Result")
        label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")
        
        # 6. 显示输入数据
        st.subheader("📊 Input Data")
        st.dataframe(user_scaled_df, use_container_width=True)
        
        # 7. SHAP力图 - 简化版本
        st.subheader("📊 SHAP Force Plot")
        
        try:
            # 创建SHAP解释器
            explainer = shap.Explainer(model, background_data)
            shap_values = explainer(user_scaled_df)
            
            # 显示SHAP值
            st.write("SHAP Values:")
            st.write(shap_values.values)
            
            # 尝试创建力图
            with st.expander("Click to view SHAP force plot"):
                try:
                    force_plot = shap.force_plot(
                        base_value=explainer.expected_value,
                        shap_values=shap_values.values[0],import streamlit as st
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
        return np.load("background_data.npy")
    except Exception as e:
        st.warning(f"Background data loading failed: {e}")
        # 创建模拟背景数据
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
        input_data = np.array([[protein, sodium, energy, procef_4]])
        
        # 2. 标准化
        input_scaled = scaler.transform(input_data)
        
        # 3. 创建DataFrame
        user_scaled_df = pd.DataFrame(input_scaled, columns=['Protein', 'Sodium', 'Energy', 'procef_4'])
        
        # 4. 预测
        prediction = model.predict(user_scaled_df)[0]
        prob = model.predict_proba(user_scaled_df)[0][1]
        
        # 5. 展示结果
        st.subheader("🔍 Prediction Result")
        label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")
        
        # 6. 显示输入数据
        st.subheader("📊 Input Data")
        st.dataframe(user_scaled_df, use_container_width=True)
        
        # 7. SHAP力图 - 针对 SHAP 0.41.0 的解决方案
        st.subheader("📊 SHAP Force Plot")
        
        try:
            # 方法1：使用 TreeExplainer（推荐用于 Pipeline）
            if hasattr(model, 'steps'):
                st.write("🔍 Detected Pipeline model, extracting final model...")
                final_model = model.steps[-1][1]
                st.write(f"Final model type: {type(final_model).__name__}")
                
                # 使用 TreeExplainer
                explainer = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(user_scaled_df)
                
                # 创建力图
                with st.expander("Click to view SHAP force plot"):
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
                    
            else:
                # 如果不是 Pipeline，直接使用模型
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(user_scaled_df)
                
                with st.expander("Click to view SHAP force plot"):
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
            st.warning(f"SHAP TreeExplainer failed: {e}")
            
            # 方法2：使用 predict_proba 函数（SHAP 0.41.0 兼容）
            try:
                st.info("Trying alternative SHAP method with predict_proba...")
                
                # 创建匹配的背景数据
                np.random.seed(42)
                adjusted_background = np.random.normal(0, 1, (200, 4))
                
                # 使用 predict_proba 函数
                explainer = shap.Explainer(model.predict_proba, adjusted_background)
                shap_values = explainer(user_scaled_df)
                
                with st.expander("Click to view SHAP force plot"):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    shap.force_plot(
                        explainer.expected_value[1],  # 健康类别的期望值
                        shap_values.values[0, :, 1],  # 健康类别的SHAP值
                        user_scaled_df.iloc[0],
                        feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                st.success("✅ SHAP force plot created with predict_proba method!")
                
            except Exception as e2:
                st.error(f"All SHAP methods failed: {e2}")
                st.info("💡 SHAP explanation is not available for this model type.")
                
                # 显示特征重要性作为替代
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance (Alternative)")
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
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
                        features=user_scaled_df.iloc[0],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(force_plot)
                except Exception as e:
                    st.warning(f"Force plot creation failed: {e}")
                    st.write("SHAP values are available above.")
                    
        except Exception as e:
            st.warning(f"SHAP analysis failed: {e}")
            st.write("Prediction completed successfully, but SHAP explanation is not available.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
