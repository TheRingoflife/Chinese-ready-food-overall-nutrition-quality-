import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings
import os

# 添加可能缺失的依赖
try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from imblearn.pipeline import Pipeline
    from imblearn.combine import SMOTETomek, SMOTEENN
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# 忽略警告
warnings.filterwarnings('ignore')

# 忽略警告
warnings.filterwarnings('ignore')

# ===== 页面设置 =====
st.set_page_config(
    page_title="Nutritional Quality Classifier", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 自定义CSS样式 =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .prediction-unhealthy {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# ===== 标题和描述 =====
st.markdown('<h1 class="main-header">🍱 Nutritional Quality Classifier</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        This app uses a trained XGBoost model to classify whether a ready-to-eat food is <strong>healthy</strong>, 
        based on 4 key nutritional features.
    </p>
</div>
""", unsafe_allow_html=True)

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    try:
        return joblib.load("XGBoost_retrained_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'XGBoost_retrained_model.pkl' exists.")
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler2.pkl")
    except FileNotFoundError:
        st.error("❌ Scaler file not found. Please ensure 'scaler2.pkl' exists.")
        return None

@st.cache_resource
def load_background_data():
    try:
        return np.load("background_data.npy")
    except FileNotFoundError:
        st.error("❌ Background data file not found. Please ensure 'background_data.npy' exists.")
        return None

@st.cache_resource
def create_explainer(model, background_data):
    if model is None or background_data is None:
        return None
    try:
        return shap.Explainer(model, background_data)
    except Exception as e:
        st.error(f"❌ Failed to create SHAP explainer: {e}")
        return None

# 加载组件
with st.spinner("🔄 Loading model and data..."):
    model = load_model()
    scaler = load_scaler()
    background_data = load_background_data()
    explainer = create_explainer(model, background_data)

if model is None or scaler is None or background_data is None:
    st.stop()

# ===== 侧边栏输入 =====
st.sidebar.header("🔢 Input Variables")
st.sidebar.markdown("Please enter the nutritional information:")

# 4个特征输入
with st.sidebar.container():
    st.markdown("### 🧂 Sodium Content")
    sodium = st.number_input(
        "Sodium (mg/100g)", 
        min_value=0.0, 
        max_value=5000.0,
        step=1.0, 
        value=400.0,
        help="Sodium content per 100g of food"
    )
    
    st.markdown("### 🥩 Protein Content")
    protein = st.number_input(
        "Protein (g/100g)", 
        min_value=0.0, 
        max_value=100.0,
        step=0.1, 
        value=15.0,
        help="Protein content per 100g of food"
    )
    
    st.markdown("### 🏭 Processing Level")
    procef_4 = st.selectbox(
        "Is Ultra-Processed? (procef_4)", 
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        help="Whether the food is ultra-processed"
    )
    
    st.markdown("### ⚡ Energy Content")
    energy = st.number_input(
        "Energy (kJ/100g)", 
        min_value=0.0, 
        max_value=5000.0,
        step=1.0, 
        value=1000.0,
        help="Energy content per 100g of food"
    )

# 添加示例数据按钮
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Example Data")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🍎 Healthy Example"):
        st.session_state.sodium = 200.0
        st.session_state.protein = 20.0
        st.session_state.procef_4 = 0
        st.session_state.energy = 800.0

with col2:
    if st.button("🍟 Unhealthy Example"):
        st.session_state.sodium = 800.0
        st.session_state.protein = 5.0
        st.session_state.procef_4 = 1
        st.session_state.energy = 1500.0

# ===== 模型预测 + SHAP 可解释性 =====
if st.sidebar.button("🧮 Predict", type="primary", use_container_width=True):
    with st.spinner("🔄 Analyzing nutritional data..."):
        try:
            # 1. 准备输入数据（4个特征）
            features = ['Sodium', 'Protein', 'procef_4', 'Energy']
            input_data = np.array([[sodium, protein, procef_4, energy]])
            
            # 2. 标准化
            input_scaled = scaler.transform(input_data)
            
            # 3. 创建DataFrame
            user_scaled_df = pd.DataFrame(input_scaled, columns=features)
            
            # 4. 预测
            prediction = model.predict(user_scaled_df)[0]
            probabilities = model.predict_proba(user_scaled_df)[0]
            
            # 5. 展示结果
            st.subheader("🔍 Prediction Result")
            
            # 创建结果展示
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.markdown('<div class="prediction-healthy">', unsafe_allow_html=True)
                    st.markdown("### ✅ **HEALTHY**")
                    st.markdown("This food item is classified as healthy!")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-unhealthy">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ **UNHEALTHY**")
                    st.markdown("This food item is classified as unhealthy.")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Healthy Probability", 
                    f"{probabilities[1]:.1%}",
                    delta=f"{(probabilities[1]-0.5)*100:+.1f}%"
                )
            
            with col3:
                st.metric(
                    "Unhealthy Probability", 
                    f"{probabilities[0]:.1%}",
                    delta=f"{(probabilities[0]-0.5)*100:+.1f}%"
                )
            
            # 置信度解释
            confidence = max(probabilities)
            if confidence > 0.8:
                st.success(f"🎯 **High confidence prediction** ({confidence:.1%})")
            elif confidence > 0.6:
                st.warning(f"⚠️ **Medium confidence prediction** ({confidence:.1%})")
            else:
                st.info(f"❓ **Low confidence prediction** ({confidence:.1%})")
            
            # 6. SHAP解释
            if explainer is not None:
                st.subheader("📊 Model Explanation (SHAP)")
                
                try:
                    # 计算SHAP值
                    shap_values = explainer(user_scaled_df)
                    
                    # 创建SHAP可视化
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Feature Importance")
                        shap.plots.bar(shap_values, show=False)
                        st.pyplot(plt.gcf())
                        plt.close()
                    
                    with col2:
                        st.markdown("#### Waterfall Plot")
                        shap.waterfall_plot(shap_values[0], show=False)
                        st.pyplot(plt.gcf())
                        plt.close()
                    
                    # 特征影响分析表格
                    st.markdown("#### Feature Impact Analysis")
                    feature_impact = pd.DataFrame({
                        'Feature': features,
                        'Input Value': input_data[0],
                        'SHAP Value': shap_values.values[0],
                        'Impact': ['Positive' if x > 0 else 'Negative' for x in shap_values.values[0]]
                    })
                    
                    # 按SHAP值绝对值排序
                    feature_impact['Abs_SHAP'] = abs(feature_impact['SHAP Value'])
                    feature_impact = feature_impact.sort_values('Abs_SHAP', ascending=False)
                    
                    st.dataframe(feature_impact[['Feature', 'Input Value', 'SHAP Value', 'Impact']], 
                               use_container_width=True)
                    
                    # 添加解释文本
                    st.markdown("**Impact Explanation:**")
                    for _, row in feature_impact.iterrows():
                        impact_text = "increases" if row['SHAP Value'] > 0 else "decreases"
                        st.write(f"• **{row['Feature']}**: {impact_text} the probability of being healthy by {abs(row['SHAP Value']):.3f}")
                    
                except Exception as e:
                    st.error(f"SHAP visualization failed: {e}")
            
            # 7. 添加建议
            st.subheader("💡 Recommendations")
            if prediction == 0:  # Unhealthy
                st.warning("**This food item is classified as unhealthy. Consider:**")
                if sodium > 400:
                    st.write(f"• Reduce sodium content (current: {sodium:.0f}mg/100g)")
                if energy > 1000:
                    st.write(f"• Lower energy density (current: {energy:.0f}kJ/100g)")
                if protein < 10:
                    st.write(f"• Increase protein content (current: {protein:.1f}g/100g)")
                if procef_4 == 1:
                    st.write("• Consider less processed alternatives")
            else:  # Healthy
                st.success("**This food item is classified as healthy!** ✅")
                st.write("Keep up the good nutritional choices!")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.write("Please check your input data and try again.")

# ===== 添加信息面板 =====
st.markdown("---")
st.subheader("ℹ️ About This App")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔬 Model Information:**
    - Algorithm: XGBoost
    - Features: 4 nutritional indicators
    - Training: Cross-validated
    - Accuracy: High performance
    """)

with col2:
    st.markdown("""
    **📊 Features Used:**
    - Sodium content
    - Protein content
    - Processing level
    - Energy content
    """)

with col3:
    st.markdown("""
    **🎯 Classification:**
    - Healthy: HSR ≥ 3.5
    - Unhealthy: HSR < 3.5
    - Based on nutritional scoring
    """)

# ===== 页脚 =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Developed using Streamlit and XGBoost · For research use only</p>
    <p>© 2024 Nutritional Quality Classifier</p>
</div>
""", unsafe_allow_html=True)
