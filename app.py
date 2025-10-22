import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 设置matplotlib参数，避免重叠
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

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
        
        # 4. 特征重要性
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
        
        # 5. SHAP力图 - 优先使用matplotlib版本
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
            
            # 创建 SHAP 力图
            with st.expander("Click to view SHAP force plot", expanded=True):
                # 方法1：优先使用matplotlib版本
                try:
                    # 设置更大的图形尺寸，避免重叠
                    plt.figure(figsize=(20, 8))  # 增加高度
                    
                    # 创建SHAP力图，确保包含特征名称
                    shap.force_plot(base_val, shap_vals,
                                   user_scaled_df.iloc[0], 
                                   feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],  # 添加特征名称
                                   matplotlib=True, show=False)
                    
                    plt.title('SHAP Force Plot - Current Prediction', fontsize=16, fontweight='bold', pad=30)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                    st.success("✅ SHAP force plot created (Matplotlib version)!")
                    
                except Exception as e:
                    st.warning(f"Matplotlib version failed: {e}")
                    
                    # 方法2：使用 HTML 版本作为备用
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
                        st.success("✅ SHAP force plot created (HTML version - Backup)!")
                        
                    except Exception as e2:
                        st.warning(f"HTML version also failed: {e2}")
                        
                        # 方法3：自定义清晰的条形图（带特征名称）
                        try:
                            fig, ax = plt.subplots(figsize=(15, 8))
                            
                            features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                            feature_values = user_scaled_df.iloc[0].values
                            
                            # 创建条形图
                            colors = ['red' if x < 0 else 'blue' for x in shap_vals]
                            bars = ax.barh(features, shap_vals, color=colors, alpha=0.7, height=0.6)
                            
                            # 添加特征名称和数值标签
                            for i, (bar, shap_val, feature_val, feature_name) in enumerate(zip(bars, shap_vals, feature_values, features)):
                                width = bar.get_width()
                                y_pos = bar.get_y() + bar.get_height()/2
                                
                                # 在条形图内部显示SHAP值
                                ax.text(width/2, y_pos, f'{shap_val:.3f}', 
                                       ha='center', va='center', color='white', fontweight='bold', fontsize=12)
                                
                                # 在条形图外部显示特征名称和值
                                if width > 0:
                                    ax.text(width + 0.05, y_pos, f'{feature_name}: {feature_val:.2f}', 
                                           ha='left', va='center', fontsize=11, fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
                                else:
                                    ax.text(width - 0.05, y_pos, f'{feature_name}: {feature_val:.2f}', 
                                           ha='right', va='center', fontsize=11, fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.8))
                            
                            # 添加零线
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
                            ax.set_xlabel('SHAP Value', fontsize=12)
                            ax.set_ylabel('Features', fontsize=12)
                            ax.set_title('SHAP Force Plot - Feature Contributions', fontsize=14, pad=20)
                            ax.grid(True, alpha=0.3)
                            
                            # 添加图例
                            legend_elements = [
                                plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.7, label='Positive Impact (Higher Health)'),
                                plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Negative Impact (Lower Health)')
                            ]
                            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            st.success("✅ SHAP force plot created (Custom version with feature names)!")
                            
                        except Exception as e3:
                            st.error(f"All SHAP plots failed: {e3}")
                            
                            # 方法4：显示详细表格
                            st.subheader("📊 SHAP Values Table")
                            shap_df = pd.DataFrame({
                                'Feature': features,
                                'Feature Value': feature_values,
                                'SHAP Value': shap_vals,
                                'Impact': ['Negative' if x < 0 else 'Positive' for x in shap_vals]
                            })
                            st.dataframe(shap_df, use_container_width=True)
                            st.info("💡 SHAP values displayed as table")
            
        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
