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
        model = joblib.load("XGBoost_retrained_model.pkl")
        
        # 更彻底的 base_score 修复
        if hasattr(model, 'steps'):
            final_model = model.steps[-1][1]
            if hasattr(final_model, 'get_booster'):
                booster = final_model.get_booster()
                # 获取当前参数
                current_params = booster.get_dump(dump_format='json')
                
                # 强制设置 base_score
                booster.set_param({'base_score': 0.5})
                
                # 验证修复
                new_params = booster.get_dump(dump_format='json')
                st.info("✅ Fixed base_score in Pipeline model")
                
        else:
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
                booster.set_param({'base_score': 0.5})
                st.info("✅ Fixed base_score in direct model")
        
        return model
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
if hasattr(model, 'steps'):
    final_model = model.steps[-1][1]
    st.info(f"Final model type: {type(final_model).__name__}")

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
        
        # 7. 特征重要性
        st.subheader("📊 Feature Importance")
        
        if hasattr(model, 'steps'):
            final_model = model.steps[-1][1]
            if hasattr(final_model, 'feature_importances_'):
                feature_importance = final_model.feature_importances_
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
        
        # 8. SHAP力图 - 跳过 TreeExplainer，直接使用其他方法
        st.subheader("📊 SHAP Force Plot")
        
        # 直接使用方法2：Explainer 与 predict_proba
        try:
            st.write("🔍 Using Explainer with predict_proba...")
            
            # 创建干净的背景数据
            np.random.seed(42)
            clean_background = np.random.normal(0, 1, (100, 4)).astype(float)
            
            explainer = shap.Explainer(model.predict_proba, clean_background)
            shap_values = explainer(user_scaled_df)
            
            # 计算 expected_value
            background_predictions = model.predict_proba(clean_background)
            expected_value = background_predictions.mean(axis=0)
            
            with st.expander("Click to view SHAP force plot"):
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 处理不同的 SHAP 值格式
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:  # 多分类
                        shap_vals = shap_values.values[0, :, 1]  # 健康类别
                        base_val = expected_value[1]
                    else:  # 二分类
                        shap_vals = shap_values.values[0, :]
                        base_val = expected_value[0]
                else:
                    shap_vals = shap_values[0, :]
                    base_val = expected_value[0]
                
                shap.force_plot(
                    base_val,
                    shap_vals,
                    user_scaled_df.iloc[0],
                    feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
            st.success("✅ SHAP force plot created successfully!")
            
        except Exception as e:
            st.warning(f"SHAP method failed: {e}")
            
            # 备用方案：显示 SHAP 值表格
            try:
                st.write("🔍 Trying to show SHAP values as table...")
                
                # 创建干净的背景数据
                np.random.seed(42)
                clean_background = np.random.normal(0, 1, (50, 4)).astype(float)
                
                explainer = shap.Explainer(model.predict_proba, clean_background)
                shap_values = explainer(user_scaled_df)
                
                # 显示 SHAP 值
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:
                        shap_vals = shap_values.values[0, :, 1]
                    else:
                        shap_vals = shap_values.values[0, :]
                else:
                    shap_vals = shap_values[0, :]
                
                # 创建 SHAP 值表格
                shap_df = pd.DataFrame({
                    'Feature': ['Protein', 'Sodium', 'Energy', 'procef_4'],
                    'SHAP Value': shap_vals,
                    'Feature Value': user_scaled_df.iloc[0].values
                })
                
                st.subheader("📊 SHAP Values Table")
                st.dataframe(shap_df, use_container_width=True)
                
                # 创建简单的条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'])
                ax.set_xlabel('SHAP Value')
                ax.set_title('SHAP Values (Feature Impact)')
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                st.pyplot(fig)
                st.success("✅ SHAP values displayed as table and chart!")
                
            except Exception as e2:
                st.error(f"All SHAP methods failed: {e2}")
                st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
