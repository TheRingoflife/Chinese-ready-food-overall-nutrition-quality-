import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ===== 页面设置 =====
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")
st.title("🍱 Predicting Nutritional Healthiness of Ready Food")
st.markdown("This app uses a trained XGBoost model to classify whether a ready-to-eat food is **healthy**, based on simplified input features.")

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    try:
        # 尝试加载你的模型文件
        model = joblib.load("XGBoost_retrained_model.pkl")
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.error("Please ensure 'XGBoost_retrained_model.pkl' exists and is compatible")
        return None

@st.cache_resource
def load_scaler():
    try:
        # 尝试加载你的标准化器文件
        scaler = joblib.load("scaler2.pkl")
        st.success("✅ Scaler loaded successfully")
        return scaler
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {e}")
        st.error("Please ensure 'scaler2.pkl' exists and is compatible")
        return None

@st.cache_resource
def load_background_data():
    try:
        # 尝试加载背景数据
        data = np.load("background_data.npy")
        st.success("✅ Background data loaded successfully")
        return data
    except Exception as e:
        st.warning(f"⚠️ Failed to load background data: {e}")
        st.warning("Creating simulated background data...")
        # 创建模拟背景数据
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 4))

# 加载组件
with st.spinner("🔄 Loading model and data..."):
    model = load_model()
    scaler = load_scaler()
    background_data = load_background_data()

if model is None or scaler is None:
    st.error("❌ Cannot proceed without model and scaler files")
    st.stop()

# 显示加载状态
st.info(f"📊 Model type: {type(model).__name__}")
st.info(f"📊 Scaler features: {len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 'Unknown'}")
st.info(f"📊 Background data shape: {background_data.shape}")

# ===== 侧边栏输入 =====
st.sidebar.header("🔢 Input Variables")
sodium = st.sidebar.number_input("Sodium (mg/100g)", min_value=0.0, step=1.0, value=400.0)
protein = st.sidebar.number_input("Protein (g/100g)", min_value=0.0, step=0.1, value=15.0)
procef_4 = st.sidebar.selectbox("Is Ultra-Processed? (procef_4)", [0, 1])
energy = st.sidebar.number_input("Energy (kJ/100g)", min_value=0.0, step=1.0, value=1000.0)

# ===== 模型预测 =====
if st.sidebar.button("🧮 Predict"):
    try:
        # 1. 准备输入数据（4个特征）
        input_data = np.array([[sodium, protein, procef_4, energy]])
        
        # 2. 标准化
        input_scaled = scaler.transform(input_data)
        
        # 3. 创建DataFrame
        user_scaled_df = pd.DataFrame(input_scaled, columns=['Sodium', 'Protein', 'procef_4', 'Energy'])
        
        # 4. 预测
        prediction = model.predict(user_scaled_df)[0]
        prob = model.predict_proba(user_scaled_df)[0][1]
        
        # 5. 展示结果
        st.subheader("🔍 Prediction Result")
        label = "✅ Healthy" if prediction == 1 else "⚠️ Unhealthy"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence (probability of being healthy):** `{prob:.2f}`")
        
        # 6. 特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            st.subheader("📊 Feature Importance")
            feature_importance = model.feature_importances_
            features = ['Sodium', 'Protein', 'procef_4', 'Energy']
            
            # 创建重要性图表
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(features, feature_importance)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            st.pyplot(fig)
        
        # 7. 特征影响分析
        st.subheader("📋 Feature Impact Analysis")
        feature_impact = pd.DataFrame({
            'Feature': ['Sodium', 'Protein', 'procef_4', 'Energy'],
            'Input Value': input_data[0],
            'Normalized Value': input_scaled[0]
        })
        
        st.dataframe(feature_impact, use_container_width=True)
        
        # 8. 添加建议
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
    - Healthy: Model prediction = 1
    - Unhealthy: Model prediction = 0
    - Based on nutritional features
    """)

# ===== 页脚 =====
st.markdown("---")
st.markdown("Developed using Streamlit and XGBoost · For research use only.")
