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

# ===== 侧边栏输入 =====
st.sidebar.header("🔢 Input Variables")
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])

# ===== 模型预测 =====
if st.sidebar.button("🧮 Predict"):
    # 检查输入是否为零
    if protein == 0 and sodium == 0 and energy == 0:
        st.warning("⚠️ Please enter values for at least one feature before predicting.")
        st.stop()
    
    try:
        # 1. 准备输入数据
        input_data = np.array([[protein, sodium, energy, procef_4]], dtype=float)
        input_scaled = scaler.transform(input_data)
        user_scaled_df = pd.DataFrame(input_scaled, columns=['Protein', 'Sodium', 'Energy', 'procef_4'])
        
        # 2. 预测
        prediction = model.predict(user_scaled_df)[0]
        prob = model.predict_proba(user_scaled_df)[0][1]
        
        # 3. 展示结果
        st.subheader("🔍 Prediction Result")
        label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** `{prob:.2f}`")
        
        # 4. 特征重要性 - 调整大小
        st.subheader("📊 Feature Importance")
        
        if hasattr(model, 'steps'):
            final_model = model.steps[-1][1]
            if hasattr(final_model, 'feature_importances_'):
                feature_importance = final_model.feature_importances_
                features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(features, feature_importance)
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # 5. SHAP力图 - 使用您提供的设置
        st.subheader("📊 SHAP Force Plot")
        
        try:
            # 创建背景数据
            np.random.seed(42)
            background_data = np.random.normal(0, 1, (100, 4)).astype(float)
            
            # 使用 Explainer
            explainer = shap.Explainer(model.predict_proba, background_data)
            shap_values = explainer(user_scaled_df)
            
            # 计算期望值
            background_predictions = model.predict_proba(background_data)
            expected_value = background_predictions.mean(axis=0)
            
            # 获取 SHAP 值
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    shap_vals = shap_values.values[0, :, 1]  # 健康类别
                    base_val = expected_value[1]
                else:
                    shap_vals = shap_values.values[0, :]
                    base_val = expected_value[0]
            else:
                shap_vals = shap_values[0, :]
                base_val = expected_value[0]
            
            # 显示 SHAP 值信息
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Base value:** {base_val:.4f}")
            with col2:
                st.write(f"**Final prediction:** {base_val + shap_vals.sum():.4f}")
            
            # 创建 SHAP 力图 - 使用您提供的设置
            with st.expander("Click to view SHAP force plot", expanded=True):
                # 方法1：使用 HTML 版本
                try:
                    force_plot = shap.force_plot(
                        base_val,
                        shap_vals,
                        user_scaled_df.iloc[0],
                        feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                        matplotlib=False
                    )
                    
                    # 转换为 HTML
                    force_html = force_plot.html()
                    components.html(shap.getjs() + force_html, height=400)
                    st.success("✅ SHAP force plot created (HTML version)!")
                    
                except Exception as e:
                    st.warning(f"HTML version failed: {e}")
                    
                    # 方法2：使用您提供的matplotlib设置
                    try:
                        # 参考您的代码设置
                        plt.figure(figsize=(20, 3))
                        shap.force_plot(base_val, shap_vals,
                                       user_scaled_df.iloc[0], matplotlib=True, show=False)
                        plt.title('SHAP Force Plot - Current Prediction', fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(plt)
                        plt.close()
                        st.success("✅ SHAP force plot created (Matplotlib version)!")
                        
                    except Exception as e2:
                        st.error(f"Matplotlib plot failed: {e2}")
                        
                        # 方法3：显示详细的 SHAP 值表格
                        st.subheader("📊 Detailed SHAP Values Table")
                        features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                        feature_values = user_scaled_df.iloc[0].values
                        
                        shap_df = pd.DataFrame({
                            'Feature': features,
                            'Feature Value': feature_values,
                            'SHAP Value': shap_vals,
                            'Impact': ['Negative' if x < 0 else 'Positive' for x in shap_vals],
                            'Contribution': [f"{x:.3f}" for x in shap_vals]
                        })
                        
                        # 按 SHAP 值绝对值排序
                        shap_df['abs_shap'] = np.abs(shap_df['SHAP Value'])
                        shap_df = shap_df.sort_values('abs_shap', ascending=False)
                        shap_df = shap_df.drop('abs_shap', axis=1)
                        
                        st.dataframe(shap_df, use_container_width=True)
                        
                        # 创建简单的条形图
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], 
                                     color=['red' if x < 0 else 'blue' for x in shap_df['SHAP Value']], alpha=0.7)
                        
                        # 添加数值标签
                        for i, (bar, val) in enumerate(zip(bars, shap_df['SHAP Value'])):
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2, 
                                    f'{val:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)
                        
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        ax.set_xlabel('SHAP Value', fontsize=11)
                        ax.set_title('SHAP Values by Feature', fontsize=13)
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        st.info("💡 SHAP values displayed as detailed table and chart")
            
        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
