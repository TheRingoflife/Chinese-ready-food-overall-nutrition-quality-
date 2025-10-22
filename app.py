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
        
        # 5. SHAP力图 - 优化版本
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
            
            # 创建优化的 SHAP 力图
            with st.expander("Click to view SHAP force plot", expanded=True):
                # 方法1：尝试HTML版本
                try:
                    force_plot = shap.force_plot(
                        base_val,
                        shap_vals,
                        user_scaled_df.iloc[0],
                        feature_names=['Protein', 'Sodium', 'Energy', 'procef_4'],
                        matplotlib=False
                    )
                    
                    force_html = force_plot.html()
                    components.html(shap.getjs() + force_html, height=400)
                    st.success("✅ SHAP force plot created (HTML version)!")
                    
                except Exception as e:
                    st.warning(f"HTML version failed: {e}")
                    
                    # 方法2：优化的matplotlib版本
                    try:
                        # 创建两列布局
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 优化的条形图
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                            feature_values = user_scaled_df.iloc[0].values
                            
                            # 创建条形图
                            colors = ['red' if x < 0 else 'blue' for x in shap_vals]
                            bars = ax.barh(features, shap_vals, color=colors, alpha=0.7)
                            
                            # 获取x轴范围用于动态调整
                            x_min, x_max = ax.get_xlim()
                            x_range = x_max - x_min
                            
                            # 优化文字显示
                            for i, (bar, shap_val, feature_val) in enumerate(zip(bars, shap_vals, feature_values)):
                                width = bar.get_width()
                                y_pos = bar.get_y() + bar.get_height()/2
                                
                                # 动态计算文字位置，避免重叠
                                if abs(width) < x_range * 0.1:  # 如果条形图太窄
                                    # 在条形图外部显示所有信息
                                    if width >= 0:
                                        text_x = width + x_range * 0.05
                                        ha = 'left'
                                    else:
                                        text_x = width - x_range * 0.05
                                        ha = 'right'
                                    
                                    # 显示组合信息
                                    ax.text(text_x, y_pos, 
                                            f'SHAP: {shap_val:.3f}\nValue: {feature_val:.2f}', 
                                            ha=ha, va='center', fontsize=9,
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                                    alpha=0.9, edgecolor="gray", linewidth=0.5))
                                else:
                                    # 条形图足够宽，分别显示
                                    # SHAP值在条形图内部
                                    ax.text(width/2, y_pos, 
                                            f'{shap_val:.3f}', ha='center', va='center', 
                                            color='white', fontweight='bold', fontsize=10)
                                    
                                    # 特征值在条形图外部
                                    if width > 0:
                                        text_x = width + x_range * 0.02
                                        ha = 'left'
                                    else:
                                        text_x = width - x_range * 0.02
                                        ha = 'right'
                                        
                                    ax.text(text_x, y_pos, 
                                            f'Value: {feature_val:.2f}', ha=ha, va='center', 
                                            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", 
                                            facecolor="lightblue", alpha=0.8, edgecolor="gray"))
                            
                            # 设置图表属性
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=2)
                            ax.set_xlabel('SHAP Value', fontsize=12)
                            ax.set_title('SHAP Force Plot - Feature Contributions', fontsize=14, pad=20)
                            ax.grid(True, alpha=0.3)
                            
                            # 添加图例
                            legend_elements = [
                                plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.7, label='Positive Impact'),
                                plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Negative Impact')
                            ]
                            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # 详细信息表格
                            st.subheader("详细信息")
                            info_df = pd.DataFrame({
                                'Feature': features,
                                'SHAP Value': [f'{x:.3f}' for x in shap_vals],
                                'Feature Value': [f'{x:.2f}' for x in feature_values],
                                'Impact': ['Negative' if x < 0 else 'Positive' for x in shap_vals]
                            })
                            
                            # 按SHAP值绝对值排序
                            info_df['abs_shap'] = np.abs(shap_vals)
                            info_df = info_df.sort_values('abs_shap', ascending=False)
                            info_df = info_df.drop('abs_shap', axis=1)
                            
                            st.dataframe(info_df, use_container_width=True)
                            
                            # 添加解释说明
                            st.markdown("**图例说明：**")
                            st.markdown("- 🔵 蓝色：正向影响（增加健康性）")
                            st.markdown("- 🔴 红色：负向影响（降低健康性）")
                            st.markdown("- 数值越大，影响越强")
                        
                        st.success("✅ SHAP force plot created (Optimized version)!")
                        
                    except Exception as e2:
                        st.error(f"Custom plot failed: {e2}")
                        
                        # 方法3：简化版显示
                        st.subheader("📊 SHAP Values Analysis")
                        
                        # 创建简化的条形图
                        fig, ax = plt.subplots(figsize=(12, 6))
                        features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                        feature_values = user_scaled_df.iloc[0].values
                        
                        bars = ax.barh(features, shap_vals, color=['red' if x < 0 else 'blue' for x in shap_vals], alpha=0.7)
                        
                        # 在条形图右侧显示信息
                        for bar, shap_val, feature_val in zip(bars, shap_vals, feature_values):
                            width = bar.get_width()
                            y_pos = bar.get_y() + bar.get_height()/2
                            
                            # 在条形图右侧显示信息
                            ax.text(max(0, width) + 0.01, y_pos, 
                                    f'SHAP: {shap_val:.3f} | Value: {feature_val:.2f}', 
                                    ha='left', va='center', fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))
                        
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        ax.set_xlabel('SHAP Value')
                        ax.set_title('SHAP Values by Feature')
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        st.info("💡 SHAP values displayed in simplified format")
            
        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
