import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings
import shap

# 忽略警告
warnings.filterwarnings('ignore')

# ===== 多语言支持 =====
LANGUAGES = {
    "English": "en",
    "中文": "zh"
}

TEXTS = {
    "en": {
        "title": "🍱 Predicting Nutritional Healthiness of Ready Food",
        "subtitle": "AI-Powered Nutritional Health Assessment for Countries with Limited Nutritional Information",
        "description": "This app uses a trained XGBoost model to classify whether a ready-to-eat food is **healthy**, based on simplified input features.",
        "target_audience": "Target Audience: Countries with Missing Nutritional Information",
        "input_title": "🔢 Input Variables",
        "sodium_label": "Sodium (mg/100g)",
        "protein_label": "Protein (g/100g)",
        "procef_4_label": "Is Ultra-Processed? (procef_4)",
        "energy_label": "Energy (kJ/100g)",
        "predict_button": "🧮 Predict",
        "prediction_title": "🔍 Prediction Result",
        "healthy": "✅ Healthy",
        "unhealthy": "⚠️ Unhealthy",
        "confidence": "Confidence (probability of being healthy)",
        "feature_importance": "📊 Feature Importance",
        "feature_impact": "📋 Feature Impact Analysis",
        "recommendations": "💡 Recommendations",
        "unhealthy_recommendations": "This food item is classified as unhealthy. Consider:",
        "healthy_recommendations": "This food item is classified as healthy! ✅",
        "reduce_sodium": "Reduce sodium content",
        "lower_energy": "Lower energy density",
        "increase_protein": "Increase protein content",
        "less_processed": "Consider less processed alternatives",
        "keep_good_choices": "Keep up the good nutritional choices!",
        "about_title": "ℹ️ About This App",
        "model_info": "Model Information:",
        "features_used": "Features Used:",
        "classification": "Classification:",
        "algorithm": "Algorithm: XGBoost",
        "features_count": "Features: 4 nutritional indicators",
        "training": "Training: Cross-validated",
        "accuracy": "Accuracy: High performance",
        "sodium_content": "Sodium content",
        "protein_content": "Protein content",
        "processing_level": "Processing level",
        "energy_content": "Energy content",
        "healthy_class": "Healthy: Model prediction = 1",
        "unhealthy_class": "Unhealthy: Model prediction = 0",
        "based_on": "Based on nutritional features",
        "shap_explanation": "🔬 SHAP Force Plot",
        "footer": "Developed using Streamlit and XGBoost · For research use only."
    },
    "zh": {
        "title": "🍱 即食食品营养健康度预测器",
        "subtitle": "面向营养素信息缺失国家的AI驱动营养健康评估",
        "description": "本应用使用训练好的XGBoost模型，基于简化的输入特征对即食食品是否**健康**进行分类。",
        "target_audience": "目标用户：营养素信息缺失的国家",
        "input_title": "🔢 输入变量",
        "sodium_label": "钠含量 (mg/100g)",
        "protein_label": "蛋白质含量 (g/100g)",
        "procef_4_label": "是否超加工？(procef_4)",
        "energy_label": "能量 (kJ/100g)",
        "predict_button": "🧮 预测",
        "prediction_title": "🔍 预测结果",
        "healthy": "✅ 健康",
        "unhealthy": "⚠️ 不健康",
        "confidence": "置信度（健康概率）",
        "feature_importance": "📊 特征重要性",
        "feature_impact": "📋 特征影响分析",
        "recommendations": "💡 建议",
        "unhealthy_recommendations": "该食品被分类为不健康。建议考虑：",
        "healthy_recommendations": "该食品被分类为健康！✅",
        "reduce_sodium": "减少钠含量",
        "lower_energy": "降低能量密度",
        "increase_protein": "增加蛋白质含量",
        "less_processed": "考虑较少加工的替代品",
        "keep_good_choices": "继续保持良好的营养选择！",
        "about_title": "ℹ️ 关于本应用",
        "model_info": "模型信息：",
        "features_used": "使用的特征：",
        "classification": "分类：",
        "algorithm": "算法：XGBoost",
        "features_count": "特征：4个营养指标",
        "training": "训练：交叉验证",
        "accuracy": "准确率：高性能",
        "sodium_content": "钠含量",
        "protein_content": "蛋白质含量",
        "processing_level": "加工水平",
        "energy_content": "能量含量",
        "healthy_class": "健康：模型预测 = 1",
        "unhealthy_class": "不健康：模型预测 = 0",
        "based_on": "基于营养特征",
        "shap_explanation": "🔬 SHAP力图",
        "footer": "使用Streamlit和XGBoost开发 · 仅供研究使用。"
    }
}

# ===== 页面设置 =====
st.set_page_config(page_title="Nutritional Quality Classifier", layout="wide")

# 侧边栏语言选择
with st.sidebar:
    st.markdown("### 🌐 Language / 语言")
    lang_key = st.selectbox("", list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_key]

# 根据选择的语言显示内容
st.title(TEXTS[lang]['title'])
st.markdown(f"**{TEXTS[lang]['subtitle']}**")
st.markdown(TEXTS[lang]['description'])

# 显示目标用户信息
st.info(f"🎯 {TEXTS[lang]['target_audience']}")

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    try:
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
        data = np.load("background_data.npy")
        st.success("✅ Background data loaded successfully")
        return data
    except Exception as e:
        st.warning(f"⚠️ Failed to load background data: {e}")
        st.warning("Creating simulated background data...")
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
st.sidebar.header(TEXTS[lang]['input_title'])
sodium = st.sidebar.number_input(TEXTS[lang]['sodium_label'], min_value=0.0, step=1.0, value=400.0)
protein = st.sidebar.number_input(TEXTS[lang]['protein_label'], min_value=0.0, step=0.1, value=15.0)
procef_4 = st.sidebar.selectbox(TEXTS[lang]['procef_4_label'], [0, 1])
energy = st.sidebar.number_input(TEXTS[lang]['energy_label'], min_value=0.0, step=1.0, value=1000.0)

# ===== 模型预测 =====
if st.sidebar.button(TEXTS[lang]['predict_button']):
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
        st.subheader(TEXTS[lang]['prediction_title'])
        label = TEXTS[lang]['healthy'] if prediction == 1 else TEXTS[lang]['unhealthy']
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**{TEXTS[lang]['confidence']}:** `{prob:.2f}`")
        
        # 6. 特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            st.subheader(TEXTS[lang]['feature_importance'])
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
        st.subheader(TEXTS[lang]['feature_impact'])
        feature_impact = pd.DataFrame({
            'Feature': ['Sodium', 'Protein', 'procef_4', 'Energy'],
            'Input Value': input_data[0],
            'Normalized Value': input_scaled[0]
        })
        
        st.dataframe(feature_impact, use_container_width=True)
        
        # 8. SHAP力图
        st.subheader(TEXTS[lang]['shap_explanation'])
        
        try:
            # 检查模型类型
            if hasattr(model, 'steps'):  # 如果是 Pipeline
                # 获取 Pipeline 中的最终模型
                final_model = model.named_steps[list(model.named_steps.keys())[-1]]
                input_transformed = model[:-1].transform(input_data)
                
                # 使用 TreeExplainer
                explainer = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(input_transformed)
                
                # 创建力图
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    input_transformed[0],
                    feature_names=['Sodium', 'Protein', 'procef_4', 'Energy'],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
            else:  # 如果是普通模型
                # 使用 TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                
                # 创建力图
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    input_scaled[0],
                    feature_names=['Sodium', 'Protein', 'procef_4', 'Energy'],
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
        except Exception as e:
            st.warning(f"SHAP force plot error: {str(e)}")
            st.info("💡 Tip: This might be due to Pipeline model structure. SHAP force plot may not be available for this model type.")
        
        # 9. 添加建议
        st.subheader(TEXTS[lang]['recommendations'])
        if prediction == 0:  # Unhealthy
            st.warning(f"**{TEXTS[lang]['unhealthy_recommendations']}**")
            if sodium > 400:
                st.write(f"• {TEXTS[lang]['reduce_sodium']} (current: {sodium:.0f}mg/100g)")
            if energy > 1000:
                st.write(f"• {TEXTS[lang]['lower_energy']} (current: {energy:.0f}kJ/100g)")
            if protein < 10:
                st.write(f"• {TEXTS[lang]['increase_protein']} (current: {protein:.1f}g/100g)")
            if procef_4 == 1:
                st.write(f"• {TEXTS[lang]['less_processed']}")
        else:  # Healthy
            st.success(f"**{TEXTS[lang]['healthy_recommendations']}**")
            st.write(TEXTS[lang]['keep_good_choices'])
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.write("Please check your input data and try again.")

# ===== 添加信息面板 =====
st.markdown("---")
st.subheader(TEXTS[lang]['about_title'])

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    **{TEXTS[lang]['model_info']}**
    - {TEXTS[lang]['algorithm']}
    - {TEXTS[lang]['features_count']}
    - {TEXTS[lang]['training']}
    - {TEXTS[lang]['accuracy']}
    """)

with col2:
    st.markdown(f"""
    **{TEXTS[lang]['features_used']}**
    - {TEXTS[lang]['sodium_content']}
    - {TEXTS[lang]['protein_content']}
    - {TEXTS[lang]['processing_level']}
    - {TEXTS[lang]['energy_content']}
    """)

with col3:
    st.markdown(f"""
    **{TEXTS[lang]['classification']}**
    - {TEXTS[lang]['healthy_class']}
    - {TEXTS[lang]['unhealthy_class']}
    - {TEXTS[lang]['based_on']}
    """)

# ===== 页脚 =====
st.markdown("---")
st.markdown(TEXTS[lang]['footer'])
