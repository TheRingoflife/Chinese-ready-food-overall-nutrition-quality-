import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ===== 多语言配置 =====
LANGUAGES = {
    "English": "en",
    "中文": "zh"
}

TEXTS = {
    "en": {
        "title": "🍱 Nutritional Quality Classifier",
        "subtitle": "Smart Food Health Assessment for Information-Poor Regions",
        "description": "This AI-powered app helps consumers in regions with limited nutritional information understand the healthiness of ready-to-eat foods using just 4 simple inputs.",
        "target_countries": "**Target Regions:** Developing countries and regions with limited nutritional labeling (Southeast Asia, Africa, Latin America, parts of Eastern Europe)",
        "problem_statement": "**The Challenge:** Many countries lack comprehensive nutritional labeling, making it difficult for consumers to make informed food choices.",
        "solution": "**Our Solution:** Use AI to predict food healthiness from basic nutritional data that's commonly available.",
        "input_vars": "Input Variables",
        "sodium": "Sodium (mg/100g)",
        "protein": "Protein (g/100g)",
        "processed": "Is Ultra-Processed? (procef_4)",
        "energy": "Energy (kJ/100g)",
        "predict": "🧮 Assess Healthiness",
        "prediction_result": "Health Assessment Result",
        "healthy": "✅ Healthy",
        "unhealthy": "⚠️ Unhealthy",
        "confidence": "Confidence (probability of being healthy)",
        "model_explanation": "AI Explanation (SHAP)",
        "feature_importance": "Feature Importance",
        "waterfall_plot": "Waterfall Plot",
        "force_plot": "AI Decision Explanation",
        "force_plot_expand": "Click to view detailed AI explanation",
        "feature_impact": "Feature Impact Analysis",
        "impact_explanation": "How each factor affects healthiness:",
        "recommendations": "Health Improvement Suggestions",
        "unhealthy_advice": "This food item is classified as unhealthy. Consider:",
        "healthy_advice": "This food item is classified as healthy! ✅",
        "reduce_sodium": "Reduce sodium content",
        "lower_energy": "Lower energy density",
        "increase_protein": "Increase protein content",
        "less_processed": "Consider less processed alternatives",
        "keep_healthy": "Keep up the good nutritional choices!",
        "about_title": "About This App",
        "mission": "Our Mission",
        "mission_text": "Empowering consumers in nutrition-information-poor regions to make informed food choices through AI technology.",
        "model_info": "AI Model Information",
        "algorithm": "Algorithm",
        "features": "Features",
        "training": "Training",
        "accuracy": "Accuracy",
        "features_used": "Required Inputs",
        "sodium_content": "Sodium content",
        "protein_content": "Protein content",
        "processing_level": "Processing level",
        "energy_content": "Energy content",
        "classification": "Health Classification",
        "healthy_label": "Healthy",
        "unhealthy_label": "Unhealthy",
        "based_on": "Based on nutritional features",
        "use_cases": "Use Cases",
        "use_case_1": "• Consumers in developing countries",
        "use_case_2": "• Food manufacturers without detailed labeling",
        "use_case_3": "• Health organizations in resource-limited areas",
        "use_case_4": "• Import/export food quality assessment",
        "footer": "Developed using Streamlit and XGBoost · For global health equity",
        "copyright": "© 2024 Nutritional Quality Classifier"
    },
    "zh": {
        "title": "🍱 营养质量分类器",
        "subtitle": "信息缺失地区的智能食品健康评估",
        "description": "这个AI驱动的应用程序帮助营养素信息缺失地区的消费者仅使用4个简单输入就能了解即食食品的健康程度。",
        "target_countries": "**目标地区：** 营养素信息缺失的发展中国家和地区（东南亚、非洲、拉丁美洲、东欧部分地区）",
        "problem_statement": "**面临的挑战：** 许多国家缺乏全面的营养标签，消费者难以做出明智的食品选择。",
        "solution": "**我们的解决方案：** 使用AI从常见的基本营养数据预测食品健康程度。",
        "input_vars": "输入变量",
        "sodium": "钠含量 (毫克/100克)",
        "protein": "蛋白质含量 (克/100克)",
        "processed": "是否超加工? (procef_4)",
        "energy": "能量 (千焦/100克)",
        "predict": "🧮 评估健康程度",
        "prediction_result": "健康评估结果",
        "healthy": "✅ 健康",
        "unhealthy": "⚠️ 不健康",
        "confidence": "置信度 (健康的可能性)",
        "model_explanation": "AI解释 (SHAP)",
        "feature_importance": "特征重要性",
        "waterfall_plot": "瀑布图",
        "force_plot": "AI决策解释",
        "force_plot_expand": "点击查看详细AI解释",
        "feature_impact": "特征影响分析",
        "impact_explanation": "每个因素如何影响健康程度：",
        "recommendations": "健康改善建议",
        "unhealthy_advice": "该食品被分类为不健康。建议：",
        "healthy_advice": "该食品被分类为健康！✅",
        "reduce_sodium": "减少钠含量",
        "lower_energy": "降低能量密度",
        "increase_protein": "增加蛋白质含量",
        "less_processed": "考虑较少加工的替代品",
        "keep_healthy": "继续保持良好的营养选择！",
        "about_title": "关于此应用",
        "mission": "我们的使命",
        "mission_text": "通过AI技术赋能营养素信息缺失地区的消费者做出明智的食品选择。",
        "model_info": "AI模型信息",
        "algorithm": "算法",
        "features": "特征",
        "training": "训练",
        "accuracy": "准确率",
        "features_used": "所需输入",
        "sodium_content": "钠含量",
        "protein_content": "蛋白质含量",
        "processing_level": "加工水平",
        "energy_content": "能量含量",
        "classification": "健康分类",
        "healthy_label": "健康",
        "unhealthy_label": "不健康",
        "based_on": "基于营养特征",
        "use_cases": "应用场景",
        "use_case_1": "• 发展中国家的消费者",
        "use_case_2": "• 缺乏详细标签的食品制造商",
        "use_case_3": "• 资源有限地区的健康组织",
        "use_case_4": "• 进出口食品质量评估",
        "footer": "使用Streamlit和XGBoost开发 · 促进全球健康公平",
        "copyright": "© 2024 营养质量分类器"
    }
}

# ===== 页面设置 =====
st.set_page_config(
    page_title="Nutritional Quality Classifier", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 语言选择 =====
col1, col2 = st.columns([1, 4])
with col1:
    selected_lang = st.selectbox("Language / 语言", list(LANGUAGES.keys()))
with col2:
    st.markdown("")

lang_code = LANGUAGES[selected_lang]
texts = TEXTS[lang_code]

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
    .target-countries {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .problem-solution {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .mission-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== 标题和描述 =====
st.markdown(f'<h1 class="main-header">{texts["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 1rem;">
    <p style="font-size: 1.2rem; color: #666;">
        {texts["subtitle"]}
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="target-countries">
    <p style="margin: 0; font-size: 1rem;">
        {texts["description"]}
    </p>
    <br>
    <p style="margin: 0; font-size: 0.9rem; color: #666;">
        {texts["target_countries"]}
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="problem-solution">
    <p style="margin: 0; font-size: 1rem; font-weight: bold;">
        {texts["problem_statement"]}
    </p>
    <br>
    <p style="margin: 0; font-size: 1rem;">
        {texts["solution"]}
    </p>
</div>
""", unsafe_allow_html=True)

# ===== 加载模型、标准化器和背景数据 =====
@st.cache_resource
def load_model():
    try:
        model = joblib.load("XGBoost_retrained_model.pkl")
        st.success("✅ AI Model loaded successfully" if lang_code == "en" else "✅ AI模型加载成功")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}" if lang_code == "en" else f"❌ 模型加载失败: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load("scaler2.pkl")
        st.success("✅ Data processor loaded successfully" if lang_code == "en" else "✅ 数据处理器加载成功")
        return scaler
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {e}" if lang_code == "en" else f"❌ 数据处理器加载失败: {e}")
        return None

@st.cache_resource
def load_background_data():
    try:
        data = np.load("background_data.npy")
        st.success("✅ Reference data loaded successfully" if lang_code == "en" else "✅ 参考数据加载成功")
        return data
    except Exception as e:
        st.warning(f"⚠️ Failed to load background data: {e}" if lang_code == "en" else f"⚠️ 参考数据加载失败: {e}")
        st.warning("Creating simulated reference data..." if lang_code == "en" else "创建模拟参考数据...")
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 4))

@st.cache_resource
def create_explainer(model, background_data):
    try:
        import shap
        explainer = shap.Explainer(model, background_data)
        st.success("✅ AI explanation engine ready" if lang_code == "en" else "✅ AI解释引擎就绪")
        return explainer
    except Exception as e:
        st.warning(f"⚠️ Failed to create explanation engine: {e}" if lang_code == "en" else f"⚠️ 解释引擎创建失败: {e}")
        return None

# 加载组件
with st.spinner("🔄 Loading AI system..." if lang_code == "en" else "🔄 加载AI系统..."):
    model = load_model()
    scaler = load_scaler()
    background_data = load_background_data()
    explainer = create_explainer(model, background_data)

if model is None or scaler is None:
    st.error("❌ Cannot proceed without AI model" if lang_code == "en" else "❌ 无法在没有AI模型的情况下继续")
    st.stop()

# 显示加载状态
st.info(f"🤖 AI Model: {type(model).__name__}" if lang_code == "en" else f"🤖 AI模型: {type(model).__name__}")
st.info(f"📊 Required inputs: {len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 'Unknown'}" if lang_code == "en" else f"📊 所需输入: {len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else '未知'}")
st.info(f"📊 Reference data: {background_data.shape}" if lang_code == "en" else f"📊 参考数据: {background_data.shape}")

# ===== 侧边栏输入 =====
st.sidebar.header(f"🔢 {texts['input_vars']}")
st.sidebar.markdown("*Enter basic nutritional information commonly available on food labels*" if lang_code == "en" else "*输入食品标签上常见的基本营养信息*")

sodium = st.sidebar.number_input(texts["sodium"], min_value=0.0, step=1.0, value=400.0)
protein = st.sidebar.number_input(texts["protein"], min_value=0.0, step=0.1, value=15.0)
procef_4 = st.sidebar.selectbox(texts["processed"], [0, 1])
energy = st.sidebar.number_input(texts["energy"], min_value=0.0, step=1.0, value=1000.0)

# ===== 模型预测 + SHAP 可解释性 =====
if st.sidebar.button(texts["predict"], type="primary", use_container_width=True):
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
        st.subheader(f"🔍 {texts['prediction_result']}")
        label = texts["healthy"] if prediction == 1 else texts["unhealthy"]
        st.markdown(f"**{texts['prediction_result'].split(':')[0] if ':' in texts['prediction_result'] else 'Assessment'}:** {label}")
        st.markdown(f"**{texts['confidence']}:** `{prob:.2f}`")
        
        # 6. SHAP解释
        if explainer is not None:
            st.subheader(f"📊 {texts['model_explanation']}")
            
            try:
                # 计算SHAP值
                shap_values = explainer(user_scaled_df)
                
                # 创建SHAP可视化
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### {texts['feature_importance']}")
                    import shap
                    shap.plots.bar(shap_values, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
                
                with col2:
                    st.markdown(f"#### {texts['waterfall_plot']}")
                    shap.waterfall_plot(shap_values[0], show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
                
                # SHAP Force Plot (力图)
                st.subheader(f"📊 {texts['force_plot']}")
                with st.expander(texts["force_plot_expand"]):
                    if isinstance(shap_values, list):
                        shap_values_force = shap_values[1]
                    else:
                        shap_values_force = shap_values
                    
                    if not isinstance(shap_values_force, shap.Explanation):
                        shap_values_force = shap.Explanation(
                            values=shap_values_force,
                            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                            data=user_scaled_df.values,
                            feature_names=user_scaled_df.columns.tolist()
                        )
                    
                    force_html = shap.force_plot(
                        base_value=shap_values_force.base_values,
                        shap_values=shap_values_force.values,
                        features=shap_values_force.data,
                        feature_names=shap_values_force.feature_names,
                        matplotlib=False
                    )
                    components.html(shap.getjs() + force_html.html(), height=400)
                
                # 特征影响分析表格
                st.markdown(f"#### {texts['feature_impact']}")
                feature_impact = pd.DataFrame({
                    'Feature': ['Sodium', 'Protein', 'procef_4', 'Energy'],
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
                st.markdown(f"**{texts['impact_explanation']}**")
                for _, row in feature_impact.iterrows():
                    impact_text = "increases" if row['SHAP Value'] > 0 else "decreases"
                    st.write(f"• **{row['Feature']}**: {impact_text} the probability of being healthy by {abs(row['SHAP Value']):.3f}")
                
            except Exception as e:
                st.error(f"AI explanation failed: {e}")
        else:
            # 如果没有SHAP，显示简单的特征重要性
            st.subheader(f"📊 {texts['feature_importance']}")
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                features = ['Sodium', 'Protein', 'procef_4', 'Energy']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(features, feature_importance)
                ax.set_xlabel('Importance')
                ax.set_title(texts['feature_importance'])
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center')
                
                st.pyplot(fig)
            else:
                st.warning("⚠️ AI explanation not available" if lang_code == "en" else "⚠️ AI解释不可用")
        
        # 7. 添加建议
        st.subheader(f"💡 {texts['recommendations']}")
        if prediction == 0:  # Unhealthy
            st.warning(f"**{texts['unhealthy_advice']}**")
            if sodium > 400:
                st.write(f"• {texts['reduce_sodium']} (current: {sodium:.0f}mg/100g)")
            if energy > 1000:
                st.write(f"• {texts['lower_energy']} (current: {energy:.0f}kJ/100g)")
            if protein < 10:
                st.write(f"• {texts['increase_protein']} (current: {protein:.1f}g/100g)")
            if procef_4 == 1:
                st.write(f"• {texts['less_processed']}")
        else:  # Healthy
            st.success(f"**{texts['healthy_advice']}**")
            st.write(texts["keep_healthy"])
            
    except Exception as e:
        st.error(f"❌ Assessment failed: {e}" if lang_code == "en" else f"❌ 评估失败: {e}")
        st.write("Please check your input data and try again." if lang_code == "en" else "请检查您的输入数据并重试。")

# ===== 添加信息面板 =====
st.markdown("---")
st.subheader(f"ℹ️ {texts['about_title']}")

# 使命说明
st.markdown(f"""
<div class="mission-box">
    <h4>{texts['mission']}</h4>
    <p style="margin: 0; font-size: 1rem;">
        {texts['mission_text']}
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    **🤖 {texts['model_info']}:**
    - {texts['algorithm']}: XGBoost AI
    - {texts['features']}: 4 nutritional indicators
    - {texts['training']}: Cross-validated
    - {texts['accuracy']}: High performance
    """)

with col2:
    st.markdown(f"""
    **📊 {texts['features_used']}:**
    - {texts['sodium_content']}
    - {texts['protein_content']}
    - {texts['processing_level']}
    - {texts['energy_content']}
    """)

with col3:
    st.markdown(f"""
    **🎯 {texts['classification']}:**
    - {texts['healthy_label']}: AI prediction = 1
    - {texts['unhealthy_label']}: AI prediction = 0
    - {texts['based_on']}
    """)

# 应用场景
st.markdown(f"### 🌍 {texts['use_cases']}")
st.markdown(f"""
- {texts['use_case_1']}
- {texts['use_case_2']}
- {texts['use_case_3']}
- {texts['use_case_4']}
""")

# ===== 页脚 =====
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>{texts['footer']}</p>
    <p>{texts['copyright']}</p>
</div>
""", unsafe_allow_html=True)
