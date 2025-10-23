# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import streamlit.components.v1 as components

# # 设置matplotlib参数，避免重叠
# plt.rcParams.update({
#     'font.size': 10,
#     'axes.titlesize': 14,
#     'axes.labelsize': 12,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 10,
#     'figure.titlesize': 16
# })

# # ===== 多语言支持 =====
# LANGUAGES = {
#     "English": "en",
#     "中文": "zh"
# }

# TEXTS = {
#     "en": {
#         "title": "🍱 Nutritional Quality Classifier",
#         "subtitle": "ML-Powered Ready-to-Eat Food Health Assessment",
#         "description": "This advanced machine learning application uses XGBoost to predict the nutritional healthiness of ready-to-eat foods based on key nutritional features.",
#         "target_audience": "🎯 Target Audience",
#         "audience_desc": "Designed for countries with limited nutritional information and consumers seeking quick, reliable food health assessments.",
#         "problem_statement": "📊 Problem Statement",
#         "problem_desc": "Many countries lack comprehensive nutritional labeling systems, making it difficult to implement generalized positive labeling for food products.",
#         "solution": "💡 Our Solution",
#         "solution_desc": "Advanced ML model analyzes 4 key nutritional features to provide instant, accurate health predictions with detailed explanations.",
#         "mission": "🚀 Mission",
#         "mission_desc": "Providing a practical approach for countries with incomplete nutritional information to implement effective food health assessment systems.",
#         "input_variables": "🔢 Input Variables",
#         "protein_label": "Protein (g/100g)",
#         "sodium_label": "Sodium (mg/100g)",
#         "energy_label": "Energy (kJ/100g)",
#         "processed_label": "Is Ultra-Processed? (procef_4)",
#         "predict_button": "🧮 Predict Healthiness",
#         "prediction_result": "🔍 Prediction Result",
#         "healthy": "✅ Healthy",
#         "unhealthy": "⚠️ Unhealthy",
#         "confidence": "Confidence",
#         "feature_importance": "📊 Feature Importance",
#         "shap_plot": "📊 SHAP Force Plot",
#         "base_value": "Base value",
#         "final_prediction": "Final prediction",
#         "expand_shap": "Click to view SHAP force plot",
#         "shap_success": "✅ SHAP force plot created (Matplotlib version)!",
#         "shap_html_success": "✅ SHAP force plot created (HTML version - Backup)!",
#         "shap_custom_success": "✅ SHAP force plot created (Custom version with feature names)!",
#         "shap_table": "📊 SHAP Values Table",
#         "shap_table_info": "💡 SHAP values displayed as table",
#         "positive_impact": "Positive Impact (Higher Health)",
#         "negative_impact": "Negative Impact (Lower Health)",
#         "warning_input": "⚠️ Please enter values for at least one feature before predicting.",
#         "input_tip": "💡 Tip: Please enter the nutritional information of the food, and the system will predict its healthiness.",
#         "model_error": "❌ Cannot proceed without model and scaler files",
#         "prediction_failed": "Prediction failed",
#         "shap_failed": "SHAP analysis failed",
#         "shap_unavailable": "💡 SHAP explanation is not available, but feature importance is shown above.",
#         "footer": "Developed using Streamlit and XGBoost · For research use only.",
#         "feature_names": ["Protein", "Sodium", "Energy", "procef_4"],
#         "chart_feature_names": ["Protein", "Sodium", "Energy", "procef_4"]  # 图表用英文
#     },
#     "zh": {
#         "title": "🍱 营养质量分类器",
#         "subtitle": "ML驱动的即食食品健康评估",
#         "description": "这个先进的机器学习应用程序使用XGBoost根据关键营养特征预测即食食品的营养健康性。",
#         "target_audience": "🎯 目标用户",
#         "audience_desc": "专为营养信息有限的国家和寻求快速、可靠食品健康评估的消费者设计。",
#         "problem_statement": "📊 问题陈述",
#         "problem_desc": "许多国家缺乏全面的营养标签系统，难以实施食品的概括性正面标签。",
#         "solution": "💡 我们的解决方案",
#         "solution_desc": "先进的ML模型分析4个关键营养特征，提供即时、准确的健康预测和详细解释。",
#         "mission": "🚀 使命",
#         "mission_desc": "为营养信息纰漏不全导致无法使用概括性正面标签的国家提供一个使用思路。",
#         "input_variables": "🔢 输入变量",
#         "protein_label": "蛋白质 (g/100g)",
#         "sodium_label": "钠 (mg/100g)",
#         "energy_label": "能量 (kJ/100g)",
#         "processed_label": "是否超加工？(procef_4)",
#         "predict_button": "🧮 预测健康性",
#         "prediction_result": "🔍 预测结果",
#         "healthy": "✅ 健康",
#         "unhealthy": "⚠️ 不健康",
#         "confidence": "置信度",
#         "feature_importance": "📊 特征重要性",
#         "shap_plot": "📊 SHAP力图",
#         "base_value": "基准值",
#         "final_prediction": "最终预测",
#         "expand_shap": "点击查看SHAP力图",
#         "shap_success": "✅ SHAP力图创建成功 (Matplotlib版本)!",
#         "shap_html_success": "✅ SHAP力图创建成功 (HTML版本 - 备用)!",
#         "shap_custom_success": "✅ SHAP力图创建成功 (自定义版本，包含特征名称)!",
#         "shap_table": "📊 SHAP值表格",
#         "shap_table_info": "💡 SHAP值以表格形式显示",
#         "positive_impact": "积极影响 (更高健康性)",
#         "negative_impact": "消极影响 (更低健康性)",
#         "warning_input": "⚠️ 请在预测前至少输入一个特征的值。",
#         "input_tip": "💡 提示: 请输入食品的营养成分信息，系统将预测其健康性。",
#         "model_error": "❌ 没有模型和标准化器文件无法继续",
#         "prediction_failed": "预测失败",
#         "shap_failed": "SHAP分析失败",
#         "shap_unavailable": "💡 SHAP解释不可用，但上面显示了特征重要性。",
#         "footer": "使用Streamlit和XGBoost开发 · 仅供研究使用。",
#         "feature_names": ["蛋白质", "钠", "能量", "procef_4"],
#         "chart_feature_names": ["Protein", "Sodium", "Energy", "procef_4"]  # 图表用英文
#     }
# }

# # ===== 页面设置 =====
# st.set_page_config(
#     page_title="Nutritional Quality Classifier",
#     page_icon="🍱",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ===== 语言选择器 =====
# def get_language():
#     col1, col2, col3 = st.columns([1, 1, 6])
#     with col1:
#         lang_choice = st.selectbox("🌐 Language", list(LANGUAGES.keys()))
#     return TEXTS[LANGUAGES[lang_choice]]

# # 获取当前语言文本
# texts = get_language()

# # ===== 主标题区域 =====
# st.markdown(f"""
# <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
#     <h1 style="color: white; margin: 0; font-size: 2.5rem;">{texts['title']}</h1>
#     <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['subtitle']}</p>
# </div>
# """, unsafe_allow_html=True)

# # ===== 应用描述 =====
# st.markdown(f"""
# <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin-bottom: 2rem;">
#     <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{texts['description']}</p>
# </div>
# """, unsafe_allow_html=True)

# # ===== 信息卡片 =====
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown(f"""
#     <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
#         <h4 style="color: #1976d2; margin: 0 0 0.5rem 0;">{texts['target_audience']}</h4>
#         <p style="margin: 0; font-size: 0.9rem;">{texts['audience_desc']}</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div style="background: #f3e5f5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
#         <h4 style="color: #7b1fa2; margin: 0 0 0.5rem 0;">{texts['problem_statement']}</h4>
#         <p style="margin: 0; font-size: 0.9rem;">{texts['problem_desc']}</p>
#     </div>
#     """, unsafe_allow_html=True)

# col3, col4 = st.columns(2)

# with col3:
#     st.markdown(f"""
#     <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
#         <h4 style="color: #2e7d32; margin: 0 0 0.5rem 0;">{texts['solution']}</h4>
#         <p style="margin: 0; font-size: 0.9rem;">{texts['solution_desc']}</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col4:
#     st.markdown(f"""
#     <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
#         <h4 style="color: #f57c00; margin: 0 0 0.5rem 0;">{texts['mission']}</h4>
#         <p style="margin: 0; font-size: 0.9rem;">{texts['mission_desc']}</p>
#     </div>
#     """, unsafe_allow_html=True)

# # ===== 加载模型 =====
# @st.cache_resource
# def load_model():
#     try:
#         return joblib.load("XGBoost_retrained_model.pkl")
#     except Exception as e:
#         st.error(f"Model loading failed: {e}")
#         return None

# @st.cache_resource
# def load_scaler():
#     try:
#         return joblib.load("scaler2.pkl")
#     except Exception as e:
#         st.error(f"Scaler loading failed: {e}")
#         return None

# model = load_model()
# scaler = load_scaler()

# if model is None or scaler is None:
#     st.error(texts['model_error'])
#     st.stop()

# # ===== 侧边栏输入 =====
# st.sidebar.markdown(f"## {texts['input_variables']}")

# # 添加输入说明 - 修复语言问题
# st.sidebar.markdown(f"""
# <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
#     <p style="margin: 0; font-size: 0.9rem; color: #1976d2;">
#         <strong>{texts['input_tip']}</strong>
#     </p>
# </div>
# """, unsafe_allow_html=True)

# protein = st.sidebar.number_input(texts['protein_label'], min_value=0.0, step=0.1, help="每100g食品中的蛋白质含量")
# sodium = st.sidebar.number_input(texts['sodium_label'], min_value=0.0, step=1.0, help="每100g食品中的钠含量")
# energy = st.sidebar.number_input(texts['energy_label'], min_value=0.0, step=1.0, help="每100g食品中的能量含量")
# procef_4 = st.sidebar.selectbox(texts['processed_label'], [0, 1], help="0=非超加工, 1=超加工")

# # 添加预测按钮样式
# if st.sidebar.button(texts['predict_button'], type="primary", use_container_width=True):
#     # 检查输入是否为零
#     if protein == 0 and sodium == 0 and energy == 0:
#         st.warning(texts['warning_input'])
#         st.stop()
    
#     try:
#         # 1. 准备输入数据
#         input_data = np.array([[protein, sodium, energy, procef_4]], dtype=float)
#         input_scaled = scaler.transform(input_data)
#         user_scaled_df = pd.DataFrame(input_scaled, columns=texts['chart_feature_names'])  # 使用英文特征名用于数据处理
        
#         # 2. 预测
#         prediction = model.predict(user_scaled_df)[0]
#         prob = model.predict_proba(user_scaled_df)[0][1]
        
#         # 3. 展示结果 - 美化
#         st.markdown(f"## {texts['prediction_result']}")
        
#         # 结果卡片
#         if prediction == 1:
#             result_color = "#28a745"
#             result_icon = "✅"
#             result_text = texts['healthy']
#         else:
#             result_color = "#dc3545"
#             result_icon = "⚠️"
#             result_text = texts['unhealthy']
        
#         st.markdown(f"""
#         <div style="background: {result_color}; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
#             <h2 style="margin: 0; font-size: 2rem;">{result_icon} {result_text}</h2>
#             <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['confidence']}: <strong>{prob:.2f}</strong></p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # 4. 特征重要性
#         st.markdown(f"## {texts['feature_importance']}")
        
#         if hasattr(model, 'steps'):
#             final_model = model.steps[-1][1]
#             if hasattr(final_model, 'feature_importances_'):
#                 feature_importance = final_model.feature_importances_
#                 features = texts['chart_feature_names']  # 使用英文特征名用于图表
                
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 bars = ax.barh(features, feature_importance, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
#                 ax.set_xlabel('Importance', fontsize=12)
#                 ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
                
#                 for i, bar in enumerate(bars):
#                     width = bar.get_width()
#                     ax.text(width, bar.get_y() + bar.get_height()/2, 
#                             f'{width:.3f}', ha='left', va='center', fontweight='bold')
                
#                 plt.tight_layout()
#                 st.pyplot(fig)
#                 plt.close()
        
#         # 5. SHAP力图
#         st.markdown(f"## {texts['shap_plot']}")
        
#         try:
#             # 创建背景数据
#             np.random.seed(42)
#             background_data = np.random.normal(0, 1, (100, 4)).astype(float)
            
#             # 使用 Explainer
#             explainer = shap.Explainer(model.predict_proba, background_data)
#             shap_values = explainer(user_scaled_df)
            
#             # 计算期望值
#             background_predictions = model.predict_proba(background_data)
#             expected_value = background_predictions.mean(axis=0)
            
#             # 获取 SHAP 值
#             if hasattr(shap_values, 'values'):
#                 if len(shap_values.values.shape) == 3:
#                     shap_vals = shap_values.values[0, :, 1]  # 健康类别
#                     base_val = expected_value[1]
#                 else:
#                     shap_vals = shap_values.values[0, :]
#                     base_val = expected_value[0]
#             else:
#                 shap_vals = shap_values[0, :]
#                 base_val = expected_value[0]
            
#             # 显示 SHAP 值信息
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric(texts['base_value'], f"{base_val:.4f}")
#             with col2:
#                 st.metric(texts['final_prediction'], f"{base_val + shap_vals.sum():.4f}")
            
#             # 创建 SHAP 力图
#             with st.expander(texts['expand_shap'], expanded=True):
#                 # 方法1：优先使用matplotlib版本
#                 try:
#                     # 设置更大的图形尺寸，避免重叠
#                     plt.figure(figsize=(20, 8))  # 增加高度
                    
#                     # 创建SHAP力图，确保包含特征名称
#                     shap.force_plot(base_val, shap_vals,
#                                    user_scaled_df.iloc[0], 
#                                    feature_names=texts['chart_feature_names'],  # 使用英文特征名称
#                                    matplotlib=True, show=False)
                    
#                     plt.title('SHAP Force Plot - Current Prediction', fontsize=16, fontweight='bold', pad=30)
#                     plt.tight_layout()
#                     st.pyplot(plt)
#                     plt.close()
#                     st.success(texts['shap_success'])
                    
#                 except Exception as e:
#                     st.warning(f"Matplotlib version failed: {e}")
                    
#                     # 方法2：使用 HTML 版本作为备用
#                     try:
#                         force_plot = shap.force_plot(
#                             base_val,
#                             shap_vals,
#                             user_scaled_df.iloc[0],
#                             feature_names=texts['chart_feature_names'],  # 使用英文特征名称
#                             matplotlib=False
#                         )
                        
#                         # 转换为 HTML
#                         force_html = force_plot.html()
#                         components.html(shap.getjs() + force_html, height=400)
#                         st.success(texts['shap_html_success'])
                        
#                     except Exception as e2:
#                         st.warning(f"HTML version also failed: {e2}")
                        
#                         # 方法3：自定义清晰的条形图（带特征名称）
#                         try:
#                             fig, ax = plt.subplots(figsize=(15, 8))
                            
#                             features = texts['chart_feature_names']  # 使用英文特征名称
#                             feature_values = user_scaled_df.iloc[0].values
                            
#                             # 创建条形图
#                             colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in shap_vals]
#                             bars = ax.barh(features, shap_vals, color=colors, alpha=0.8, height=0.6)
                            
#                             # 添加特征名称和数值标签
#                             for i, (bar, shap_val, feature_val, feature_name) in enumerate(zip(bars, shap_vals, feature_values, features)):
#                                 width = bar.get_width()
#                                 y_pos = bar.get_y() + bar.get_height()/2
                                
#                                 # 在条形图内部显示SHAP值
#                                 ax.text(width/2, y_pos, f'{shap_val:.3f}', 
#                                        ha='center', va='center', color='white', fontweight='bold', fontsize=12)
                                
#                                 # 在条形图外部显示特征名称和值
#                                 if width > 0:
#                                     ax.text(width + 0.05, y_pos, f'{feature_name}: {feature_val:.2f}', 
#                                            ha='left', va='center', fontsize=11, fontweight='bold',
#                                            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
#                                 else:
#                                     ax.text(width - 0.05, y_pos, f'{feature_name}: {feature_val:.2f}', 
#                                            ha='right', va='center', fontsize=11, fontweight='bold',
#                                            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.8))
                            
#                             # 添加零线
#                             ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
#                             ax.set_xlabel('SHAP Value', fontsize=12)
#                             ax.set_ylabel('Features', fontsize=12)
#                             ax.set_title('SHAP Force Plot - Feature Contributions', fontsize=14, pad=20)
#                             ax.grid(True, alpha=0.3)
                            
#                             # 添加图例
#                             legend_elements = [
#                                 plt.Rectangle((0,0),1,1, facecolor='#4ecdc4', alpha=0.8, label=texts['positive_impact']),
#                                 plt.Rectangle((0,0),1,1, facecolor='#ff6b6b', alpha=0.8, label=texts['negative_impact'])
#                             ]
#                             ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                            
#                             plt.tight_layout()
#                             st.pyplot(fig)
#                             plt.close()
#                             st.success(texts['shap_custom_success'])
                            
#                         except Exception as e3:
#                             st.error(f"All SHAP plots failed: {e3}")
                            
#                             # 方法4：显示详细表格
#                             st.markdown(f"### {texts['shap_table']}")
#                             shap_df = pd.DataFrame({
#                                 'Feature': features,
#                                 'Feature Value': feature_values,
#                                 'SHAP Value': shap_vals,
#                                 'Impact': [texts['negative_impact'] if x < 0 else texts['positive_impact'] for x in shap_vals]
#                             })
#                             st.dataframe(shap_df, use_container_width=True)
#                             st.info(texts['shap_table_info'])
            
#         except Exception as e:
#             st.error(f"{texts['shap_failed']}: {e}")
#             st.info(texts['shap_unavailable'])
            
#     except Exception as e:
#         st.error(f"{texts['prediction_failed']}: {e}")

# # ===== 页脚 =====
# st.markdown("---")
# st.markdown(f"""
# <div style="text-align: center; padding: 2rem 0; color: #666;">
#     <p style="margin: 0;">{texts['footer']}</p>
# </div>
# """, unsafe_allow_html=True)


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import sys

# 依赖检查
def check_dependencies():
    missing_deps = []
    try:
        import joblib
    except ImportError:
        missing_deps.append("joblib")
    
    try:
        import shap
    except ImportError:
        missing_deps.append("shap")
    
    try:
        from imblearn.pipeline import Pipeline
    except ImportError:
        try:
            from sklearn.pipeline import Pipeline
        except ImportError:
            missing_deps.append("scikit-learn")
    
    if missing_deps:
        st.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        st.info("Please install with: pip install " + " ".join(missing_deps))
        st.stop()
    
    return True

# 检查依赖
check_dependencies()

# 导入其他库
import joblib
import shap
import streamlit.components.v1 as components

# 忽略警告
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="Nutritional Quality Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 多语言支持
LANGUAGES = {
    "English": "en",
    "中文": "zh"
}

TEXTS = {
    "en": {
        "title": "🍱 Nutritional Quality Classifier",
        "subtitle": "ML-powered food health assessment for countries with limited nutritional information",
        "mission": "Mission",
        "mission_desc": "Providing a practical approach for countries with incomplete nutritional information to implement effective food health assessment systems.",
        "problem": "Problem Statement",
        "problem_desc": "Many countries lack comprehensive nutritional labeling, making it difficult for consumers to make informed food choices. This tool bridges that gap using machine learning.",
        "input_vars": "Input Variables",
        "sodium": "Sodium (mg/100g)",
        "protein": "Protein (g/100g)",
        "procef": "Is Ultra-Processed?",
        "energy": "Energy (kJ/100g)",
        "predict": "Predict",
        "healthy": "HEALTHY",
        "unhealthy": "UNHEALTHY",
        "healthy_prob": "Healthy Probability",
        "unhealthy_prob": "Unhealthy Probability",
        "shap_plot": "SHAP Force Plot",
        "expand_shap": "Click to view SHAP force plot",
        "shap_success": "✅ SHAP force plot generated successfully",
        "shap_failed": "❌ SHAP explanation failed",
        "shap_unavailable": "💡 SHAP explanation is not available for this model type.",
        "prediction_failed": "❌ Prediction failed",
        "chart_feature_names": ["Sodium", "Protein", "Ultra-Processed", "Energy"],
        "positive_impact": "Positive Impact",
        "negative_impact": "Negative Impact",
        "shap_table": "SHAP Values Table",
        "shap_table_info": "This table shows how each feature contributes to the prediction.",
        "base_value": "Base Value",
        "final_prediction": "Final Prediction"
    },
    "zh": {
        "title": "🍱 营养质量分类器",
        "subtitle": "基于机器学习的食品健康评估工具，专为营养信息不完整的国家设计",
        "mission": "使命",
        "mission_desc": "为营养信息不完整的国家提供实用的食品健康评估系统解决方案。",
        "problem": "问题陈述",
        "problem_desc": "许多国家缺乏全面的营养标签，消费者难以做出明智的食品选择。本工具使用机器学习填补这一空白。",
        "input_vars": "输入变量",
        "sodium": "钠含量 (mg/100g)",
        "protein": "蛋白质含量 (g/100g)",
        "procef": "是否超加工？",
        "energy": "能量 (kJ/100g)",
        "predict": "预测",
        "healthy": "健康",
        "unhealthy": "不健康",
        "healthy_prob": "健康概率",
        "unhealthy_prob": "不健康概率",
        "shap_plot": "SHAP力图",
        "expand_shap": "点击查看SHAP力图",
        "shap_success": "✅ SHAP力图生成成功",
        "shap_failed": "❌ SHAP解释失败",
        "shap_unavailable": "💡 此模型类型不支持SHAP解释。",
        "prediction_failed": "❌ 预测失败",
        "chart_feature_names": ["Sodium", "Protein", "Ultra-Processed", "Energy"],
        "positive_impact": "正向影响",
        "negative_impact": "负向影响",
        "shap_table": "SHAP值表格",
        "shap_table_info": "此表格显示每个特征对预测的贡献。",
        "base_value": "基准值",
        "final_prediction": "最终预测"
    }
}

def get_language():
    return st.sidebar.selectbox("Language / 语言", list(LANGUAGES.keys()))

# 获取当前语言
current_lang = get_language()
texts = TEXTS[LANGUAGES[current_lang]]

# 自定义CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    background: linear-gradient(90deg, #2E8B57, #32CD32);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}

.subtitle {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}

.prediction-healthy {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    color: #155724;
    padding: 1.5rem;
    border-radius: 1rem;
    border: 2px solid #c3e6cb;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.prediction-unhealthy {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    color: #721c24;
    padding: 1.5rem;
    border-radius: 1rem;
    border: 2px solid #f5c6cb;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    text-align: center;
}

.sidebar .stSelectbox > div > div {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown(f'<h1 class="main-header">{texts["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{texts["subtitle"]}</p>', unsafe_allow_html=True)

# 使命和问题
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### {texts['mission']}")
    st.info(texts['mission_desc'])
with col2:
    st.markdown(f"### {texts['problem']}")
    st.info(texts['problem_desc'])

# 模型加载函数
@st.cache_resource
def load_model():
    """加载模型，支持多种路径和格式"""
    possible_paths = [
        "results_20251015_112741/models/final_model.pkl",
        "final_model.pkl",
        "XGBoost_retrained_model.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.success(f"✅ Model loaded from {path}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Failed to load {path}: {e}")
                continue
    
    st.error("❌ No valid model file found")
    return None

@st.cache_resource
def load_scaler():
    """加载标准化器"""
    possible_paths = [
        "results_20251015_112741/models/scaler.pkl",
        "scaler.pkl",
        "scaler2.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                scaler = joblib.load(path)
                st.success(f"✅ Scaler loaded from {path}")
                return scaler
            except Exception as e:
                st.warning(f"⚠️ Failed to load {path}: {e}")
                continue
    
    st.error("❌ No valid scaler file found")
    return None

@st.cache_resource
def create_background_data():
    """创建背景数据用于SHAP解释"""
    np.random.seed(42)
    return np.random.normal(0, 1, (100, 4)).astype(float)

# 加载组件
with st.spinner("🔄 Loading model and data..."):
    model = load_model()
    scaler = load_scaler()
    background_data = create_background_data()

if model is None or scaler is None:
    st.error("❌ Cannot proceed without model and scaler files")
    st.stop()

# 侧边栏输入
st.sidebar.header(f"🔢 {texts['input_vars']}")

# 4个特征输入
sodium = st.sidebar.number_input(
    texts['sodium'],
    min_value=0.0,
    max_value=5000.0,
    step=1.0,
    help="Sodium content per 100g of food"
)

protein = st.sidebar.number_input(
    texts['protein'],
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    help="Protein content per 100g of food"
)

procef_4 = st.sidebar.selectbox(
    texts['procef'],
    [0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
    help="Whether the food is ultra-processed"
)

energy = st.sidebar.number_input(
    texts['energy'],
    min_value=0.0,
    max_value=5000.0,
    step=1.0,
    help="Energy content per 100g of food"
)

# 示例数据按钮
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

# 预测按钮
if st.sidebar.button(f"🧮 {texts['predict']}", type="primary", use_container_width=True):
    with st.spinner("🔄 Analyzing nutritional data..."):
        try:
            # 准备输入数据
            features = ['Sodium', 'Protein', 'procef_4', 'Energy']
            input_data = np.array([[sodium, protein, procef_4, energy]])
            
            # 标准化
            input_scaled = scaler.transform(input_data)
            user_scaled_df = pd.DataFrame(input_scaled, columns=features)
            
            # 预测
            prediction = model.predict(user_scaled_df)[0]
            probabilities = model.predict_proba(user_scaled_df)[0]
            
            # 展示结果
            st.subheader("🔍 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if prediction == 1:
                    st.markdown('<div class="prediction-healthy">', unsafe_allow_html=True)
                    st.markdown(f"### ✅ **{texts['healthy']}**")
                    st.markdown("This food item is classified as healthy!")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-unhealthy">', unsafe_allow_html=True)
                    st.markdown(f"### ⚠️ **{texts['unhealthy']}**")
                    st.markdown("This food item is classified as unhealthy.")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    texts['healthy_prob'],
                    f"{probabilities[1]:.1%}",
                    delta=f"{(probabilities[1]-0.5)*100:+.1f}%"
                )
            
            with col3:
                st.metric(
                    texts['unhealthy_prob'],
                    f"{probabilities[0]:.1%}",
                    delta=f"{(probabilities[0]-0.5)*100:+.1f}%"
                )
            
            # SHAP力图
            st.markdown(f"## {texts['shap_plot']}")
            
            try:
                # 创建SHAP解释器
                explainer = shap.Explainer(model.predict_proba, background_data)
                shap_values = explainer(user_scaled_df)
                
                # 计算期望值
                background_predictions = model.predict_proba(background_data)
                expected_value = background_predictions.mean(axis=0)
                
                # 获取SHAP值
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:
                        shap_vals = shap_values.values[0, :, 1]
                        base_val = expected_value[1]
                    else:
                        shap_vals = shap_values.values[0, :]
                        base_val = expected_value[0]
                else:
                    shap_vals = shap_values[0, :]
                    base_val = expected_value[0]
                
                # 显示SHAP值信息
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(texts['base_value'], f"{base_val:.4f}")
                with col2:
                    st.metric(texts['final_prediction'], f"{base_val + shap_vals.sum():.4f}")
                
                # 创建SHAP力图
                with st.expander(texts['expand_shap'], expanded=True):
                    try:
                        # 方法1：HTML版本（推荐）
                        force_plot = shap.force_plot(
                            base_val,
                            shap_vals,
                            user_scaled_df.iloc[0],
                            feature_names=texts['chart_feature_names'],
                            matplotlib=False
                        )
                        
                        force_html = force_plot.html()
                        components.html(shap.getjs() + force_html, height=400)
                        st.success(texts['shap_success'])
                        
                    except Exception as e:
                        st.warning(f"HTML version failed: {e}")
                        
                        # 方法2：自定义条形图
                        try:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            features = texts['chart_feature_names']
                            feature_values = user_scaled_df.iloc[0].values
                            
                            colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in shap_vals]
                            bars = ax.barh(features, shap_vals, color=colors, alpha=0.8, height=0.6)
                            
                            for i, (bar, shap_val, feature_val, feature_name) in enumerate(zip(bars, shap_vals, feature_values, features)):
                                width = bar.get_width()
                                y_pos = bar.get_y() + bar.get_height()/2
                                
                                ax.text(width/2, y_pos, f'{shap_val:.3f}', 
                                       ha='center', va='center', color='white', fontweight='bold', fontsize=10)
                                
                                if width > 0:
                                    ax.text(width + 0.05, y_pos, f'{feature_name}: {feature_val:.2f}', 
                                           ha='left', va='center', fontsize=9, fontweight='bold')
                                else:
                                    ax.text(width - 0.05, y_pos, f'{feature_name}: {feature_val:.2f}', 
                                           ha='right', va='center', fontsize=9, fontweight='bold')
                            
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
                            ax.set_xlabel('SHAP Value', fontsize=12)
                            ax.set_ylabel('Features', fontsize=12)
                            ax.set_title('SHAP Force Plot - Feature Contributions', fontsize=14, pad=20)
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            st.success(texts['shap_success'])
                            
                        except Exception as e2:
                            st.error(f"All SHAP plots failed: {e2}")
                            
                            # 方法3：显示表格
                            st.markdown(f"### {texts['shap_table']}")
                            shap_df = pd.DataFrame({
                                'Feature': features,
                                'Feature Value': feature_values,
                                'SHAP Value': shap_vals,
                                'Impact': [texts['negative_impact'] if x < 0 else texts['positive_impact'] for x in shap_vals]
                            })
                            st.dataframe(shap_df, use_container_width=True)
                            st.info(texts['shap_table_info'])
            
            except Exception as e:
                st.error(f"{texts['shap_failed']}: {e}")
                st.info(texts['shap_unavailable'])
        
        except Exception as e:
            st.error(f"{texts['prediction_failed']}: {e}")
