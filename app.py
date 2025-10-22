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
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1, value=12.0)
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0, value=300.0)
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0, value=400.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])

# ===== 模型预测 =====
if st.sidebar.button("🧮 Predict"):
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
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(features, feature_importance)
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                st.pyplot(fig)
                plt.close()
        
        # 5. SHAP力图 - 使用matplotlib=True
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
            
            # 显示基本信息
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Base value:** {base_val:.4f}")
            with col2:
                st.write(f"**Final prediction:** {base_val + shap_vals.sum():.4f}")
            
            # 创建SHAP力图
            with st.expander("Click to view SHAP force plot", expanded=True):
                
                # 方法1：使用matplotlib=True的SHAP力图
                try:
                    # 设置更大的图形尺寸
                    plt.figure(figsize=(20, 3))
                    
                    # 使用matplotlib=True创建力图
                    shap.force_plot(base_val, shap_vals, user_scaled_df.iloc[0], 
                                   feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                                   matplotlib=True, show=False)
                    
                    plt.title('SHAP Force Plot - Feature Contributions', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    # 在Streamlit中显示
                    st.pyplot(plt.gcf())
                    plt.close()
                    
                    st.success("✅ SHAP force plot created (matplotlib version)!")
                    
                except Exception as e:
                    st.warning(f"matplotlib version failed: {e}")
                    
                    # 方法2：尝试HTML版本
                    try:
                        force_plot = shap.force_plot(
                            base_val,
                            shap_vals,
                            user_scaled_df.iloc[0],
                            feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                            matplotlib=False,
                            show=False
                        )
                        
                        force_html = force_plot.html()
                        components.html(shap.getjs() + force_html, height=500)
                        st.success("✅ SHAP force plot created (HTML version)!")
                        
                    except Exception as e2:
                        st.warning(f"HTML version failed: {e2}")
                        
                        # 方法3：自定义力图
                        try:
                            # 创建两列布局
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                # 创建自定义的SHAP力图
                                fig, ax = plt.subplots(figsize=(16, 4))
                                
                                features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                                feature_values = user_scaled_df.iloc[0].values
                                
                                # 按SHAP值排序
                                sorted_indices = np.argsort(shap_vals)[::-1]
                                sorted_features = [features[i] for i in sorted_indices]
                                sorted_shap_vals = shap_vals[sorted_indices]
                                sorted_feature_vals = feature_values[sorted_indices]
                                
                                # 绘制力图
                                current_pos = base_val
                                
                                # 绘制基线
                                ax.axvline(x=base_val, color='black', linestyle='-', linewidth=3, alpha=0.8)
                                ax.text(base_val, 0.5, f'Base: {base_val:.3f}', 
                                       ha='center', va='center', fontsize=12, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))
                                
                                # 绘制每个特征的贡献
                                for i, (feature, shap_val, feature_val) in enumerate(zip(sorted_features, sorted_shap_vals, sorted_feature_vals)):
                                    start_pos = current_pos
                                    end_pos = current_pos + shap_val
                                    
                                    # 选择颜色
                                    color = 'red' if shap_val < 0 else 'blue'
                                    
                                    # 绘制矩形
                                    rect_height = 0.4
                                    rect = plt.Rectangle((min(start_pos, end_pos), 0.1), 
                                                       abs(shap_val), rect_height, 
                                                       facecolor=color, alpha=0.8, edgecolor='black', linewidth=2)
                                    ax.add_patch(rect)
                                    
                                    # 添加特征标签
                                    label_x = (start_pos + end_pos) / 2
                                    label_y = 0.3 + rect_height
                                    
                                    # 特征名称
                                    ax.text(label_x, label_y, feature, 
                                           ha='center', va='bottom', fontsize=11, fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
                                    
                                    # SHAP值
                                    ax.text(label_x, label_y - 0.15, f'{shap_val:.3f}', 
                                           ha='center', va='top', fontsize=10,
                                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
                                    
                                    # 特征值
                                    ax.text(label_x, label_y - 0.3, f'Val: {feature_val:.2f}', 
                                           ha='center', va='top', fontsize=9,
                                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))
                                    
                                    current_pos = end_pos
                                
                                # 绘制最终预测线
                                final_pred = base_val + shap_vals.sum()
                                ax.axvline(x=final_pred, color='green', linestyle='--', linewidth=3, alpha=0.8)
                                ax.text(final_pred, 0.8, f'Final: {final_pred:.3f}', 
                                       ha='center', va='center', fontsize=12, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
                                
                                # 设置图表属性
                                ax.set_xlim(min(base_val, final_pred) - 0.3, max(base_val, final_pred) + 0.3)
                                ax.set_ylim(0, 1)
                                ax.set_xlabel('Prediction Value', fontsize=12)
                                ax.set_title('SHAP Force Plot - Feature Contributions', fontsize=14, pad=20)
                                ax.grid(True, alpha=0.3, axis='x')
                                
                                # 添加图例
                                legend_elements = [
                                    plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.8, label='Positive Impact'),
                                    plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='Negative Impact'),
                                    plt.Line2D([0],[0], color='black', linewidth=3, label='Base Value'),
                                    plt.Line2D([0],[0], color='green', linewidth=3, linestyle='--', label='Final Prediction')
                                ]
                                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                                
                                # 隐藏y轴
                                ax.set_yticks([])
                                ax.set_ylabel('')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            
                            with col2:
                                # 详细信息表格
                                st.subheader("📋 Feature Details")
                                
                                detail_df = pd.DataFrame({
                                    'Feature': sorted_features,
                                    'SHAP Value': [f'{x:.4f}' for x in sorted_shap_vals],
                                    'Feature Value': [f'{x:.3f}' for x in sorted_feature_vals],
                                    'Impact': ['🔴 Negative' if x < 0 else '🔵 Positive' for x in sorted_shap_vals]
                                })
                                
                                st.dataframe(detail_df, use_container_width=True)
                                
                                # 添加解释说明
                                st.markdown("**📖 力图说明：**")
                                st.markdown("- 🔵 **蓝色矩形**：正向影响")
                                st.markdown("- 🔴 **红色矩形**：负向影响")
                                st.markdown("- **黑色线**：基准值")
                                st.markdown("- **绿色虚线**：最终预测")
                                st.markdown("- **矩形宽度**：影响大小")
                            
                            st.success("✅ SHAP force plot created (Custom version)!")
                            
                        except Exception as e3:
                            st.error(f"All SHAP methods failed: {e3}")
                            st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
