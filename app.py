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
        
        # 修复 base_score 问题
        if hasattr(model, 'steps'):
            final_model = model.steps[-1][1]
            if hasattr(final_model, 'get_booster'):
                booster = final_model.get_booster()
                # 设置正确的 base_score
                booster.set_param({'base_score': 0.5})
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
            else:
                st.warning("Final model does not have feature_importances_ attribute")
        else:
            if hasattr(model, 'feature_importances_'):
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
                st.warning("Model does not have feature_importances_ attribute")
        
        # 8. SHAP力图 - 多种方法尝试
        st.subheader("📊 SHAP Force Plot")
        
        # 方法1：TreeExplainer
        try:
            st.write("🔍 Trying TreeExplainer...")
            
            if hasattr(model, 'steps'):
                final_model = model.steps[-1][1]
                explainer = shap.TreeExplainer(final_model)
            else:
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(user_scaled_df.values)
            
            with st.expander("Click to view SHAP force plot (TreeExplainer)"):
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
                
            st.success("✅ SHAP force plot created with TreeExplainer!")
                    
        except Exception as e:
            st.warning(f"TreeExplainer failed: {e}")
            
            # 方法2：使用 Explainer 与 predict_proba
            try:
                st.write("🔍 Trying Explainer with predict_proba...")
                
                # 创建干净的背景数据
                np.random.seed(42)
                clean_background = np.random.normal(0, 1, (100, 4)).astype(float)
                
                explainer = shap.Explainer(model.predict_proba, clean_background)
                shap_values = explainer(user_scaled_df)
                
                with st.expander("Click to view SHAP force plot (Explainer)"):
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
                        user_scaled_df.iloc[0],
                        feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                st.success("✅ SHAP force plot created with Explainer!")
                
            except Exception as e2:
                st.warning(f"Explainer method failed: {e2}")
                
                # 方法3：使用原始背景数据
                try:
                    st.write("🔍 Trying with original background data...")
                    
                    explainer = shap.Explainer(model.predict_proba, background_data)
                    shap_values = explainer(user_scaled_df)
                    
                    with st.expander("Click to view SHAP force plot (Original background)"):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        if hasattr(shap_values, 'values'):
                            if len(shap_values.values.shape) == 3:
                                shap_vals = shap_values.values[0, :, 1]
                                base_val = explainer.expected_value[1]
                            else:
                                shap_vals = shap_values.values[0, :]
                                base_val = explainer.expected_value
                        else:
                            shap_vals = shap_values[0, :]
                            base_val = explainer.expected_value
                        
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
                        
                    st.success("✅ SHAP force plot created with original background!")
                    
                except Exception as e3:
                    st.error(f"All SHAP methods failed: {e3}")
                    st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
