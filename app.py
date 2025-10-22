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
        
        # 5. SHAP力图 - 完全重新设计
        st.subheader("📊 SHAP Feature Analysis")
        
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
            
            # 创建清晰的特征分析图
            with st.expander("Click to view detailed SHAP analysis", expanded=True):
                
                # 创建两列布局
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # 创建清晰的特征贡献图
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    features = ['Protein', 'Sodium', 'Energy', 'procef_4']
                    feature_values = user_scaled_df.iloc[0].values
                    
                    # 按SHAP值绝对值排序
                    sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
                    sorted_features = [features[i] for i in sorted_indices]
                    sorted_shap_vals = shap_vals[sorted_indices]
                    sorted_feature_vals = feature_values[sorted_indices]
                    
                    # 创建水平条形图
                    y_pos = np.arange(len(sorted_features))
                    colors = ['red' if x < 0 else 'blue' for x in sorted_shap_vals]
                    
                    bars = ax.barh(y_pos, sorted_shap_vals, color=colors, alpha=0.7, height=0.6)
                    
                    # 添加特征标签和数值
                    for i, (bar, shap_val, feature_val, feature_name) in enumerate(zip(bars, sorted_shap_vals, sorted_feature_vals, sorted_features)):
                        width = bar.get_width()
                        y_pos_bar = bar.get_y() + bar.get_height()/2
                        
                        # 在条形图内部显示SHAP值（如果空间足够）
                        if abs(width) > 0.05:
                            ax.text(width/2, y_pos_bar, 
                                    f'{shap_val:.3f}', ha='center', va='center', 
                                    color='white', fontweight='bold', fontsize=10)
                        
                        # 在条形图右侧显示详细信息
                        if width > 0:
                            text_x = width + 0.02
                            ha = 'left'
                        else:
                            text_x = width - 0.02
                            ha = 'right'
                        
                        # 显示特征名称和值
                        ax.text(text_x, y_pos_bar, 
                                f'{feature_name}: {feature_val:.2f}', 
                                ha=ha, va='center', fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                        alpha=0.9, edgecolor="gray", linewidth=0.5))
                    
                    # 设置y轴标签
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(sorted_features)
                    
                    # 添加零线
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                    
                    # 设置标题和标签
                    ax.set_xlabel('SHAP Value (Feature Contribution)', fontsize=12)
                    ax.set_title('SHAP Feature Contributions\n(Features sorted by impact)', fontsize=14, pad=20)
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # 添加图例
                    legend_elements = [
                        plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.7, label='Positive Impact (Increases Health)'),
                        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Negative Impact (Decreases Health)')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # 详细信息表格
                    st.subheader("📋 Feature Details")
                    
                    # 创建详细信息表格
                    detail_df = pd.DataFrame({
                        'Feature': sorted_features,
                        'SHAP Value': [f'{x:.4f}' for x in sorted_shap_vals],
                        'Feature Value': [f'{x:.3f}' for x in sorted_feature_vals],
                        'Impact': ['🔴 Negative' if x < 0 else '🔵 Positive' for x in sorted_shap_vals],
                        'Magnitude': [f'{abs(x):.4f}' for x in sorted_shap_vals]
                    })
                    
                    st.dataframe(detail_df, use_container_width=True)
                    
                    # 添加解释说明
                    st.markdown("**📖 图例说明：**")
                    st.markdown("- 🔵 **蓝色**：正向影响（增加健康性）")
                    st.markdown("- 🔴 **红色**：负向影响（降低健康性）")
                    st.markdown("- **数值越大**：影响越强")
                    st.markdown("- **排序**：按影响强度从大到小排列")
                    
                    # 添加特征含义说明
                    st.markdown("**🔍 特征含义：**")
                    st.markdown("- **Protein**：蛋白质含量 (g/100g)")
                    st.markdown("- **Sodium**：钠含量 (mg/100g)")
                    st.markdown("- **Energy**：能量 (kJ/100g)")
                    st.markdown("- **procef_4**：是否超加工 (0=否, 1=是)")
                
                st.success("✅ SHAP analysis completed successfully!")
            
        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.info("💡 SHAP explanation is not available, but feature importance is shown above.")
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
