import streamlit as st
import os
from eda import quick_eda, create_eda_charts
from preprocessing import comprehensive_preprocessing, missing_summary, compare_imputation_methods
from models import get_model, tune_model
from report import create_comprehensive_report
from llm import groq_generate_text
from code_generator import generate_feature_insights, generate_code_notebook, generate_preprocessing_explanation, generate_model_interpretation
import pandas as pd
import json
import uuid
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="Data Science Assistant", layout="wide")
st.title("🤖 Advanced Data Science Assistant")

st.sidebar.header('⚙️ Settings & API')
if 'GROQ_API_KEY' in st.secrets:
    groq_token = st.secrets['GROQ_API_KEY']
    st.sidebar.success("✅ API Key loaded from secrets")
else:
    groq_token = os.environ.get('GROQ_API_KEY', '')
    if groq_token:
        st.sidebar.success("✅ API Key loaded from environment")
    else:
        groq_token = st.sidebar.text_input('Groq API Key', type='password', help="Get your key from https://console.groq.com")

st.sidebar.markdown('---')
nav = st.sidebar.radio('📋 Workflow', ['📤 Upload', '📊 EDA & Profiling', '🔧 Preprocess', '🎨 Feature Engineering', '🤖 Modeling', '📄 Report'])

for key in ['df', 'processed_df', 'preprocessor', 'model_result', 'target_column', 'X_processed', 'y', 'problem_type', 'preprocessing_log', 'imputation_decisions', 'selected_features', 'eda_charts', 'feature_importance', 'engineered_df']:
    if key not in st.session_state:
        st.session_state[key] = None

if nav == '📤 Upload':
    st.header('1️⃣ Upload Dataset')
    uploaded = st.file_uploader('Upload CSV or Excel file', ['csv', 'xlsx', 'xls'])
    if uploaded is not None:
        try:
            with st.spinner('Loading file...'):
                if uploaded.name.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(uploaded, encoding='ISO-8859-1')
                else:
                    df = pd.read_excel(uploaded)
                for key in ['df', 'processed_df', 'preprocessor', 'model_result', 'target_column', 'X_processed', 'y', 'problem_type', 'preprocessing_log', 'selected_features', 'engineered_df']:
                    st.session_state[key] = None
                st.session_state['df'] = df
                st.success(f'✅ Loaded **{uploaded.name}**: {df.shape[0]:,} rows × {df.shape[1]} columns')
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{df.shape[0]:,}")
                col2.metric("Columns", f"{df.shape[1]}")
                col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.subheader('📋 Data Preview')
                st.dataframe(df.head(20), use_container_width=True)
                st.subheader('📊 Column Types')
                type_counts = df.dtypes.value_counts()
                st.write(type_counts)
        except Exception as e:
            st.error(f'❌ Failed to read file: {e}')

elif nav == '📊 EDA & Profiling':
    st.header('2️⃣ Exploratory Data Analysis')
    if st.session_state['df'] is None:
        st.info('👈 Please upload a dataset first (Workflow → Upload).')
    else:
        df = st.session_state['df']
        st.subheader('⚡ Quick EDA')
        with st.spinner('Analyzing dataset...'):
            cards, figs = quick_eda(df)
        cols = st.columns(len(cards))
        for i, (key, value) in enumerate(cards.items()):
            cols[i].metric(key, value)
        for fig in figs:
            st.pyplot(fig)
            plt.close(fig)
        st.markdown('---')
        st.subheader('🔍 Missing Data Analysis')
        missing_df = missing_summary(df)
        st.dataframe(missing_df, use_container_width=True)
        
        st.markdown('---')
        st.subheader('💡 AI Data Insights')
        
        if st.button('🤖 Generate AI Interpretation of Data', type='primary'):
            if not groq_token:
                st.error('❌ Please provide Groq API Key in sidebar')
            else:
                with st.spinner('🤖 AI is analyzing your data...'):
                    try:
                        # Prepare data summary for AI
                        numeric_summary = df.select_dtypes(include='number').describe().to_dict()
                        categorical_info = {}
                        for col in df.select_dtypes(exclude='number').columns[:5]:
                            categorical_info[col] = {
                                'unique': int(df[col].nunique()),
                                'top_values': df[col].value_counts().head(3).to_dict()
                            }
                        
                        missing_info = {col: f"{pct:.1f}%" for col, pct in (df.isna().mean() * 100).items() if pct > 0}
                        
                        correlations = {}
                        numeric_df = df.select_dtypes(include='number')
                        if numeric_df.shape[1] > 1:
                            corr_matrix = numeric_df.corr()
                            # Get top correlations
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                                    corr_val = corr_matrix.iloc[i, j]
                                    if abs(corr_val) > 0.5:
                                        correlations[f"{col1} vs {col2}"] = f"{corr_val:.3f}"
                        
                        prompt = f"""You are a data scientist analyzing a dataset. Provide BUSINESS-FOCUSED insights from this exploratory data analysis.

Dataset Overview:
- Rows: {df.shape[0]:,}
- Columns: {df.shape[1]}
- Numeric Features: {len(df.select_dtypes(include='number').columns)}
- Categorical Features: {len(df.select_dtypes(exclude='number').columns)}

Numeric Features Summary (first 5):
{json.dumps(dict(list(numeric_summary.items())[:5]), indent=2)}

Categorical Features (first 5):
{json.dumps(categorical_info, indent=2)}

Missing Data:
{json.dumps(missing_info, indent=2) if missing_info else "No missing data"}

Strong Correlations (|r| > 0.5):
{json.dumps(correlations, indent=2) if correlations else "No strong correlations found"}

Provide a comprehensive data analysis (4-5 paragraphs) covering:

1. **Data Quality Assessment**: What's the overall quality? Are there red flags?

2. **Key Patterns & Relationships**: What interesting patterns emerge? Which variables are related?

3. **Business Implications**: What do these patterns tell us about the business/domain?

4. **Data Concerns**: What issues need attention? (missing data, outliers, imbalances)

5. **Recommendations**: What should be done before modeling? What features look promising?

Write for a business audience - focus on what the data MEANS, not just statistics."""
                        
                        eda_insights = groq_generate_text(
                            prompt=prompt,
                            api_key=groq_token,
                            model="llama-3.3-70b-versatile",
                            max_tokens=1500
                        )
                        
                        st.session_state['eda_insights'] = eda_insights
                        st.success('✅ AI interpretation generated!')
                        
                        st.markdown('### 📊 Data Analysis & Insights')
                        st.write(eda_insights)
                        
                    except Exception as e:
                        st.error(f'Failed to generate insights: {e}')
        
        if 'eda_insights' in st.session_state and st.session_state['eda_insights']:
            with st.expander('📋 View Saved EDA Insights'):
                st.write(st.session_state['eda_insights'])
        
        st.markdown('---')
        if st.button('🎨 Generate Comprehensive EDA Charts'):
            with st.spinner('Creating detailed visualizations...'):
                chart_paths = create_eda_charts(df)
                st.session_state['eda_charts'] = chart_paths
                st.success(f'✅ Generated {len(chart_paths)} charts for report')
                for path in chart_paths:
                    st.image(path, use_column_width=True)

elif nav == '🔧 Preprocess':
    st.header('3️⃣ Data Preprocessing & Cleaning')
    if st.session_state['df'] is None:
        st.info('👈 Please upload a dataset first.')
        st.stop()
    df = st.session_state['df']
    st.subheader('🎯 Select Target Column')
    target = st.selectbox('Choose target variable for modeling', options=[None] + list(df.columns), help="This is the variable you want to predict")
    if not target:
        st.warning('⚠️ Please select a target column to proceed.')
        st.stop()
    st.session_state['target_column'] = target
    y = df[target]
    if y.dtype == 'object' or y.nunique() < 20:
        problem_type = 'classification'
        st.info(f'🎯 Detected: **Classification** ({y.nunique()} classes)')
    else:
        problem_type = 'regression'
        st.info(f'🎯 Detected: **Regression** (continuous target)')
    st.session_state['problem_type'] = problem_type
    st.markdown('---')
    st.subheader('📊 Missingness Overview')
    missing_df = missing_summary(df)
    missing_df_filtered = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_df_filtered) > 0:
        st.dataframe(missing_df_filtered, use_container_width=True)
    else:
        st.success('✅ No missing values detected!')
    st.markdown('---')
    st.subheader('⚙️ Preprocessing Configuration')
    col1, col2 = st.columns(2)
    with col1:
        drop_threshold = st.slider('Drop threshold (% missing in target)', 0.0, 0.2, 0.05, 0.01, help="If target has less than this % missing, drop those rows")
        knn_neighbors = st.slider('KNN neighbors (k)', 3, 15, 5, help="Number of neighbors for KNN imputation")
    with col2:
        scale_numeric = st.checkbox('Scale numeric features', value=True, help="Apply StandardScaler to numeric features")
        skip_mcar = st.checkbox('Skip MCAR test (faster)', value=False, help="Skip statistical testing for faster processing")
    with st.expander('🧪 Test Imputation Methods (Optional)'):
        st.write("Compare KNN vs Regression imputation for specific columns")
        cols_with_missing = missing_df_filtered['Column'].tolist()
        if cols_with_missing:
            test_col = st.selectbox('Select column to test', cols_with_missing)
            test_k = st.slider('Test with k neighbors', 3, 15, 5, key='test_k')
            if st.button('Run Comparison Test'):
                with st.spinner(f'Testing imputation methods for {test_col}...'):
                    is_numeric = pd.api.types.is_numeric_dtype(df[test_col])
                    if is_numeric:
                        result = compare_imputation_methods(df, test_col, True, test_k)
                        st.write(f"**Best Method:** `{result['best_method'].upper()}`")
                        col_a, col_b = st.columns(2)
                        if result['knn_score']:
                            col_a.metric("KNN RMSE", f"{result['knn_score']:.4f}")
                        if result['regression_score']:
                            col_b.metric("Regression RMSE", f"{result['regression_score']:.4f}")
                        st.info(f"Reason: {result.get('reason', 'N/A')}")
                    else:
                        st.warning('Column is categorical - only mode/constant imputation available')
        else:
            st.info('No columns with missing values')
    st.markdown('---')
    if st.button('🚀 Run Comprehensive Preprocessing', type='primary'):
        try:
            with st.spinner('🔄 Processing... This may take a few minutes for large datasets'):
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text('Running MCAR tests...')
                progress_bar.progress(20)
                result = comprehensive_preprocessing(df=df, target=target, drop_threshold=drop_threshold, knn_neighbors=knn_neighbors, scale_numeric=scale_numeric, skip_mcar=skip_mcar)
                progress_bar.progress(80)
                status_text.text('Finalizing...')
                st.session_state['X_processed'] = result['X_processed']
                st.session_state['y'] = result['y']
                st.session_state['preprocessing_log'] = result['preprocessing_log']
                st.session_state['imputation_decisions'] = result['imputation_decisions']
                st.session_state['preprocessor'] = result['preprocessor']
                progress_bar.progress(100)
                status_text.empty()
                st.success(f'✅ Preprocessing complete! Shape: {result["X_processed"].shape}')
                with st.expander('📝 View Preprocessing Log', expanded=True):
                    for log_entry in result['preprocessing_log']:
                        st.text(log_entry)
                st.subheader('📊 Imputation Summary')
                imputation_df = pd.DataFrame([{'Column': k, 'Method': v} for k, v in result['imputation_decisions'].items()])
                st.dataframe(imputation_df, use_container_width=True)
        except Exception as e:
            st.error(f'❌ Preprocessing failed: {e}')
            import traceback
            with st.expander('🐛 Error Details'):
                st.code(traceback.format_exc())

elif nav == '🎨 Feature Engineering':
    st.header('4️⃣ AI-Powered Feature Engineering')
    if st.session_state.get('df') is None:
        st.info('👈 Please upload a dataset first.')
        st.stop()
    
    if st.session_state.get('engineered_df') is not None:
        df = st.session_state['engineered_df']
        st.success('📝 Working with AI-engineered dataset')
    else:
        df = st.session_state['df'].copy()
        st.info('📝 Working with original dataset')
    
    st.write(f"Current shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    
    if not groq_token:
        st.error('❌ Please provide Groq API Key in sidebar to use AI feature engineering')
        st.stop()
    
    st.markdown('---')
    st.subheader('🤖 AI Feature Engineering Assistant')
    
    st.write("""
    The AI will analyze your dataset and automatically create valuable features based on:
    - Column types and relationships
    - Target variable (if specified)
    - Domain knowledge inferred from column names
    - Statistical patterns in the data
    """)
    
    target_col = st.session_state.get('target_column', None)
    if target_col:
        st.info(f"🎯 Target variable: **{target_col}** ({st.session_state.get('problem_type', 'Unknown')})")
    else:
        st.warning('⚠️ No target variable set. Go to Preprocess tab to set target for better feature suggestions.')
    
    st.markdown('---')
    
    if st.button('🚀 Generate AI Features', type='primary'):
        with st.spinner('🤖 AI is analyzing your data and creating features...'):
            try:
                # Prepare context for AI
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                cat_cols = df.select_dtypes(exclude='number').columns.tolist()
                
                # Sample statistics
                stats_summary = {}
                for col in numeric_cols[:10]:
                    stats_summary[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
                
                prompt = f"""You are a data science expert. Analyze this dataset and CREATE SPECIFIC PYTHON CODE to engineer new features.

Dataset Info:
- Shape: {df.shape}
- Target: {target_col if target_col else 'Not specified'}
- Problem Type: {st.session_state.get('problem_type', 'Not specified')}
- Numeric columns: {numeric_cols[:15]}
- Categorical columns: {cat_cols[:10]}

Sample Statistics:
{json.dumps(stats_summary, indent=2)}

Generate Python code that creates 5-10 valuable new features. Include:
1. Mathematical combinations (ratios, products, differences)
2. Binning of continuous variables
3. Interaction features between important variables
4. Polynomial features for key predictors
5. Domain-specific features based on column names

Return ONLY valid Python code that:
- Assumes df is the DataFrame
- Creates new columns directly: df['new_feature'] = ...
- Includes try-except for safety
- Has comments explaining each feature
- Does NOT include any markdown or explanation outside code

Example format:
# Feature 1: Ratio of X to Y
try:
    df['x_to_y_ratio'] = df['X'] / (df['Y'] + 1e-10)
except:
    pass

Generate complete, runnable code now:"""
                
                feature_code = groq_generate_text(
                    prompt=prompt,
                    api_key=groq_token,
                    model="llama-3.3-70b-versatile",
                    max_tokens=2000,
                    temperature=0.7
                )
                
                # Clean code
                if '```python' in feature_code:
                    feature_code = feature_code.split('```python')[1].split('```')[0].strip()
                elif '```' in feature_code:
                    feature_code = feature_code.split('```')[1].split('```')[0].strip()
                
                st.success('✅ AI generated feature engineering code!')
                
                st.subheader('📝 Generated Code')
                st.code(feature_code, language='python')
                
                st.markdown('---')
                st.markdown('---')
                st.subheader('⚡ Apply Features')
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button('✅ Execute Feature Engineering', type='primary', key='execute_features'):
                        with st.spinner('Creating features...'):
                            try:
                                # Store original columns
                                original_cols = set(df.columns)
                                
                                # Create a safe namespace with all necessary objects
                                namespace = {
                                    'df': df,
                                    'np': np,
                                    'pd': pd,
                                    '__builtins__': __builtins__
                                }
                                
                                # Execute the code
                                exec(feature_code, namespace)
                                
                                # Get the modified dataframe back
                                df = namespace['df']
                                
                                # Find new columns
                                new_cols = set(df.columns) - original_cols
                                
                                if new_cols:
                                    st.session_state['engineered_df'] = df
                                    st.session_state['df'] = df
                                    
                                    st.success(f'✅ Created {len(new_cols)} new features!')
                                    
                                    st.subheader('🎉 New Features Created')
                                    new_features_list = list(new_cols)
                                    for idx, col in enumerate(new_features_list, 1):
                                        st.write(f"{idx}. **{col}**")
                                    
                                    st.subheader('📊 Sample Data with New Features')
                                    display_cols = list(original_cols)[:3] + new_features_list
                                    st.dataframe(df[display_cols].head(10))
                                    
                                    st.info('💾 Features saved! Proceed to Preprocess tab to prepare for modeling.')
                                else:
                                    st.warning('No new features were created. Try regenerating the code.')
                            
                            except Exception as e:
                                st.error(f'❌ Failed to execute code: {e}')
                                st.write('**Debug Info:**')
                                st.code(str(e))
                                st.write('**Problematic Code:**')
                                st.code(feature_code)
                                st.write('Try clicking "Generate AI Features" again to get new code.')
                
                with col2:
                    if st.button('🔄 Regenerate Code', key='regen_code'):
                        st.rerun()
            
            except Exception as e:
                st.error(f'❌ AI feature generation failed: {e}')
    
    st.markdown('---')
    st.subheader('📊 Current Dataset')
    st.write(f"Shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    
    with st.expander('View Dataset Preview'):
        st.dataframe(df.head(20))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🔄 Reset to Original'):
            st.session_state['engineered_df'] = None
            st.info('Reset to original dataset')
            st.rerun()
    
    with col2:
        if st.button('📥 Download Engineered Dataset'):
            csv = df.to_csv(index=False)
            st.download_button(
                label='Download CSV',
                data=csv,
                file_name=f'engineered_dataset_{uuid.uuid4().hex[:6]}.csv',
                mime='text/csv'
            )

elif nav == '🤖 Modeling':
    st.header('5️⃣ Model Training & Evaluation')
    if st.session_state.get('df') is None:
        st.info('👈 Please upload a dataset first.')
        st.stop()
    if st.session_state.get('X_processed') is None:
        st.warning('⚠️ Please run preprocessing first (Preprocess tab).')
        st.stop()
    X_processed = st.session_state['X_processed']
    y = st.session_state['y']
    problem_type = st.session_state['problem_type']
    st.info(f'🎯 Problem Type: **{problem_type.capitalize()}**')
    st.write(f"Features shape: `{X_processed.shape}` | Target shape: `{y.shape}`")
    st.markdown('---')
    st.subheader('🎯 Feature Selection')
    all_features = X_processed.columns.tolist()
    col1, col2 = st.columns([2, 1])
    with col1:
        selection_method = st.radio('Selection method', ['Select All', 'Manual Selection', 'AI-Suggested Features'], horizontal=True)
        if selection_method == 'Select All':
            selected_features = all_features
            st.success(f'✅ Using all {len(selected_features)} features')
        elif selection_method == 'Manual Selection':
            selected_features = st.multiselect('Choose features to include', all_features, default=all_features[:min(20, len(all_features))], help="Select which features to use for modeling")
            st.write(f"Selected: {len(selected_features)} / {len(all_features)} features")
        else:
            if groq_token:
                if st.button('🤖 Get AI Feature Suggestions'):
                    with st.spinner('Analyzing features with AI...'):
                        feature_info = {'feature_names': all_features[:50], 'target': st.session_state['target_column'], 'problem_type': problem_type}
                        insights = generate_feature_insights(feature_info, groq_token, df=st.session_state['df'])
                        st.write(insights)
                        selected_features = all_features
            else:
                st.warning('⚠️ Please provide Groq API key in sidebar')
                selected_features = all_features
    with col2:
        st.metric("Total Features", len(all_features))
        st.metric("Selected", len(selected_features) if 'selected_features' in locals() else 0)
    if 'selected_features' in locals() and selected_features:
        st.session_state['selected_features'] = selected_features
        X_selected = X_processed[selected_features]
    else:
        st.warning('⚠️ No features selected')
        st.stop()
    st.markdown('---')
    st.subheader('🎯 Model Configuration')
    col1, col2 = st.columns(2)
    with col1:
        if problem_type == 'classification':
            model_options = ['RandomForest', 'LogisticRegression', 'XGBoost', 'SVM']
        else:
            model_options = ['RandomForest', 'LinearRegression', 'XGBoost', 'SVR']
        model_name = st.selectbox('Choose Model', model_options)
    st.markdown('### ⚙️ Hyperparameters')
    param_preset = {}
    if model_name == 'RandomForest':
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider('n_estimators', 10, 500, 100)
        with col2:
            max_depth = st.slider('max_depth (0=None)', 0, 50, 0)
        with col3:
            min_samples_split = st.slider('min_samples_split', 2, 20, 2)
        param_preset = {'n_estimators': n_estimators, 'max_depth': None if max_depth == 0 else max_depth, 'min_samples_split': min_samples_split, 'random_state': 42}
    elif model_name == 'XGBoost':
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider('n_estimators', 10, 500, 100)
        with col2:
            max_depth = st.slider('max_depth', 1, 20, 6)
        with col3:
            learning_rate = st.number_input('learning_rate', 0.001, 1.0, 0.1, format="%.3f")
        param_preset = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate, 'random_state': 42}
    elif model_name == 'LogisticRegression':
        C = st.number_input('C (inverse regularization)', 0.0001, 1000.0, 1.0, format="%.4f")
        param_preset = {'C': C, 'random_state': 42, 'max_iter': 1000}
    elif model_name == 'LinearRegression':
        st.info('ℹ️ LinearRegression has no hyperparameters')
    elif model_name in ['SVM', 'SVR']:
        col1, col2 = st.columns(2)
        with col1:
            C = st.number_input('C', 0.0001, 1000.0, 1.0, format="%.4f")
        with col2:
            kernel = st.selectbox('kernel', ['rbf', 'linear', 'poly'])
        param_preset = {'C': C, 'kernel': kernel}
    st.markdown('---')
    st.subheader('🎓 Training Options')
    col1, col2, col3 = st.columns(3)
    with col1:
        tune = st.checkbox('Enable hyperparameter tuning', help="Use RandomizedSearchCV")
    with col2:
        n_iter = st.number_input('Tuning iterations', 5, 200, 20) if tune else 10
    with col3:
        cv = st.number_input('CV folds', 2, 10, 5)
    if st.button('🚀 Train Model', type='primary'):
        try:
            with st.spinner('🔄 Training model... This may take several minutes'):
                progress_bar = st.progress(0)
                progress_bar.progress(20)
                result = get_model(model_name, param_preset, problem_type)
                model_obj = result['model']
                progress_bar.progress(40)
                res = tune_model(model_obj, X_selected, y, tune=tune, n_iter=n_iter, cv=cv, problem_type=problem_type)
                progress_bar.progress(100)
                st.session_state['model_result'] = res
                st.session_state['feature_importance'] = res.get('feature_importance')
                st.success('✅ Training complete!')
                st.markdown('---')
                st.subheader('📊 Model Performance')
                metrics_display = {k: v for k, v in res['metrics'].items() if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'residuals']}
                metrics_df = pd.DataFrame([metrics_display])
                st.dataframe(metrics_df, use_container_width=True)
                if tune and 'best_params' in res:
                    with st.expander('🎯 Best Hyperparameters Found'):
                        st.json(res['best_params'])
                if res['feature_importance'] is not None:
                    st.markdown('---')
                    st.subheader('📈 Feature Importance')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance_df = res['feature_importance'].head(20)
                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 20 Most Important Features')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                if problem_type == 'classification':
                    st.markdown('---')
                    st.subheader('🎯 Classification Metrics')
                    if 'confusion_matrix' in res['metrics']:
                        cm = np.array(res['metrics']['confusion_matrix'])
                        report = res['metrics']['classification_report']
                        class_labels = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
                        plt.close(fig)
                    if 'roc_curve' in res['metrics']:
                        roc_data = res['metrics']['roc_curve']
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(roc_data['fpr'], roc_data['tpr'], label=f"AUC = {res['metrics']['roc_auc']:.3f}")
                        ax.plot([0, 1], [0, 1], 'k--', label='Random')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('ROC Curve')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.markdown('---')
                    st.subheader('📉 Regression Metrics')
                    if 'residuals' in res['metrics']:
                        residuals_data = res['metrics']['residuals']
                        y_test = residuals_data['y_test']
                        y_pred = residuals_data['y_pred']
                        residuals = residuals_data['values']
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        axes[0].scatter(y_test, y_pred, alpha=0.6)
                        axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction')
                        axes[0].set_xlabel('Actual')
                        axes[0].set_ylabel('Predicted')
                        axes[0].set_title('Actual vs Predicted')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        axes[1].scatter(y_pred, residuals, alpha=0.6)
                        axes[1].axhline(y=0, color='r', linestyle='--')
                        axes[1].set_xlabel('Predicted')
                        axes[1].set_ylabel('Residuals')
                        axes[1].set_title('Residual Plot')
                        axes[1].grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                
                # AI Business Insights Section
                st.markdown('---')
                st.subheader('💡 AI Business Insights')
                
                if st.button('🤖 Generate Business Insights from Results', type='primary'):
                    if not groq_token:
                        st.error('❌ Please provide Groq API Key in sidebar')
                    else:
                        with st.spinner('🤖 Analyzing results and generating business insights...'):
                            try:
                                metrics_clean = {k: v for k, v in res['metrics'].items() if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'residuals']}
                                
                                feature_importance_text = ""
                                if res.get('feature_importance') is not None:
                                    top_features = res['feature_importance'].head(10)
                                    feature_importance_text = "\n".join([f"- {row['Feature']}: {row['Importance']:.4f}" for _, row in top_features.iterrows()])
                                
                                prompt = f"""You are a business consultant analyzing machine learning results. Provide ACTIONABLE BUSINESS INSIGHTS.

Problem Type: {problem_type.capitalize()}
Model: {model_name}
Target Variable: {st.session_state.get('target_column')}

Model Performance:
{json.dumps(metrics_clean, indent=2)}

Top 10 Most Important Features:
{feature_importance_text if feature_importance_text else "Not available"}

Training Data: {res.get('X_train_shape', (0,))[0]:,} samples
Test Data: {res.get('X_test_shape', (0,))[0]:,} samples

Provide a comprehensive business analysis (5-6 paragraphs) covering:

1. **Performance Assessment**: What does this performance mean in business terms? Is it good enough for production use?

2. **Key Drivers**: Which factors most influence the outcome? What business actions can we take based on these drivers?

3. **Business Recommendations**: 
   - What immediate actions should the business take?
   - Which areas need improvement or investigation?
   - What opportunities does this model reveal?

4. **Risk Assessment**: What are the limitations? Where might the model fail? What safeguards are needed?

5. **ROI and Implementation**: 
   - What's the expected business impact?
   - How should this be deployed?
   - What resources are needed?

6. **Next Steps**: Concrete action items with priorities

Focus on ACTIONABLE insights that executives and managers can implement. Avoid technical jargon."""
                                
                                business_insights = groq_generate_text(
                                    prompt=prompt,
                                    api_key=groq_token,
                                    model="llama-3.3-70b-versatile",
                                    max_tokens=2000
                                )
                                
                                st.session_state['business_insights'] = business_insights
                                st.success('✅ Business insights generated!')
                                
                                st.markdown('### 📊 Business Analysis')
                                st.write(business_insights)
                                
                            except Exception as e:
                                st.error(f'Failed to generate insights: {e}')
                
                if 'business_insights' in st.session_state and st.session_state['business_insights']:
                    with st.expander('📋 View Saved Business Insights'):
                        st.write(st.session_state['business_insights'])
        except Exception as e:
            st.error(f'❌ Training failed: {e}')
            import traceback
            with st.expander('🐛 Error Details'):
                st.code(traceback.format_exc())

elif nav == '📄 Report':
    st.header('6️⃣ Generate Comprehensive Report')
    if st.session_state['df'] is None:
        st.info('👈 Please upload dataset first')
        st.stop()
    df = st.session_state['df']
    model_res = st.session_state.get('model_result')
    preprocessing_log = st.session_state.get('preprocessing_log', [])
    imputation_decisions = st.session_state.get('imputation_decisions', {})
    st.subheader('📝 Executive Summary')
    if st.button('🤖 Generate AI Executive Summary'):
        if not groq_token:
            st.error('❌ Please provide Groq API Key in sidebar')
        else:
            try:
                with st.spinner('🤖 AI is analyzing your project...'):
                    eda_summary = {'rows': int(df.shape[0]), 'columns': int(df.shape[1]), 'numeric_columns': int(len(df.select_dtypes(include='number').columns)), 'categorical_columns': int(len(df.select_dtypes(exclude='number').columns)), 'total_missing': int(df.isna().sum().sum()), 'missing_percentage': f"{(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%"}
                    prompt = f"""You are a senior data scientist writing an executive summary with BUSINESS-FOCUSED RECOMMENDATIONS.

Dataset Overview:
- Total Records: {eda_summary['rows']:,}
- Features: {eda_summary['columns']}
- Numeric Features: {eda_summary['numeric_columns']}
- Categorical Features: {eda_summary['categorical_columns']}
- Missing Data: {eda_summary['total_missing']} ({eda_summary['missing_percentage']})

Preprocessing Steps:
{chr(10).join(preprocessing_log[:10])}

Model Results:
{json.dumps(model_res, default=str, indent=2) if model_res else "Model not yet trained"}

Write a professional executive summary (4-5 paragraphs) that focuses on:
1. Dataset quality and business implications
2. Key insights from preprocessing (what the data tells us about the business)
3. Model performance in business terms (not just metrics)
4. ACTIONABLE business recommendations based on the model results
5. Concrete next steps for implementation and ROI

Focus on business value, not technical details. What should stakeholders DO with these findings?"""
                    exec_summary = groq_generate_text(prompt=prompt, api_key=groq_token, model="llama-3.3-70b-versatile", max_tokens=1500)
                    st.session_state['exec_summary'] = exec_summary
                    st.success('✅ Executive summary generated!')
                    st.text_area('Executive Summary', value=exec_summary, height=400)
            except Exception as e:
                st.error(f'❌ Failed to generate summary: {e}')
    if 'exec_summary' in st.session_state and st.session_state['exec_summary']:
        st.text_area('Current Executive Summary', value=st.session_state['exec_summary'], height=300, key='existing_summary')
    st.markdown('---')
    st.subheader('🔧 Preprocessing Explanation')
    if imputation_decisions and st.button('🤖 Generate Preprocessing Explanation'):
        if not groq_token:
            st.error('❌ Please provide Groq API Key in sidebar')
        else:
            try:
                with st.spinner('Generating preprocessing explanation...'):
                    prep_explanation = generate_preprocessing_explanation(imputation_decisions, groq_token)
                    st.session_state['prep_explanation'] = prep_explanation
                    st.success('✅ Explanation generated!')
                    st.write(prep_explanation)
            except Exception as e:
                st.error(f'❌ Failed: {e}')
    if 'prep_explanation' in st.session_state:
        with st.expander('View Preprocessing Explanation'):
            st.write(st.session_state['prep_explanation'])
    st.markdown('---')
    st.subheader('📊 Model Results Interpretation')
    if model_res and st.button('🤖 Generate Model Interpretation'):
        if not groq_token:
            st.error('❌ Please provide Groq API Key in sidebar')
        else:
            try:
                with st.spinner('Interpreting model results...'):
                    model_interpretation = generate_model_interpretation(model_res, groq_token)
                    st.session_state['model_interpretation'] = model_interpretation
                    st.success('✅ Interpretation generated!')
                    st.write(model_interpretation)
            except Exception as e:
                st.error(f'❌ Failed: {e}')
    if 'model_interpretation' in st.session_state:
        with st.expander('View Model Interpretation'):
            st.write(st.session_state['model_interpretation'])
    st.markdown('---')
    st.subheader('💻 Generate Reproducible Code')
    if st.button('🤖 Generate Python Notebook Code'):
        if not groq_token:
            st.error('❌ Please provide Groq API Key in sidebar')
        else:
            try:
                with st.spinner('🤖 Generating complete Python code...'):
                    code_context = {'preprocessing_log': preprocessing_log, 'imputation_decisions': imputation_decisions, 'model_result': model_res, 'selected_features': st.session_state.get('selected_features', []), 'target': st.session_state.get('target_column'), 'problem_type': st.session_state.get('problem_type')}
                    notebook_code = generate_code_notebook(code_context, groq_token)
                    st.session_state['generated_code'] = notebook_code
                    st.success('✅ Code generated!')
                    st.code(notebook_code, language='python')
                    st.download_button(label='📥 Download Python Code', data=notebook_code, file_name=f'ml_project_{uuid.uuid4().hex[:6]}.py', mime='text/x-python')
            except Exception as e:
                st.error(f'❌ Code generation failed: {e}')
    st.markdown('---')
    st.subheader('📄 Generate Final Report')
    if st.button('📄 Create Word Report (.docx)', type='primary'):
        try:
            with st.spinner('📝 Creating comprehensive report...'):
                exec_summary = st.session_state.get('exec_summary', 'Executive summary not generated. Please generate it above.')
                prep_explanation = st.session_state.get('prep_explanation', '')
                model_interpretation = st.session_state.get('model_interpretation', '')
                generated_code = st.session_state.get('generated_code', '# Code not generated')
                enhanced_summary = exec_summary
                if prep_explanation:
                    enhanced_summary += f"\n\n## Preprocessing Details\n{prep_explanation}"
                if model_interpretation:
                    enhanced_summary += f"\n\n## Model Analysis\n{model_interpretation}"
                report_path = create_comprehensive_report(df=df, executive_summary=enhanced_summary, preprocessing_log=preprocessing_log, imputation_decisions=imputation_decisions, model_result=model_res, eda_charts=st.session_state.get('eda_charts', []), generated_code=generated_code)
                with open(report_path, 'rb') as f:
                    report_data = f.read()
                st.success('✅ Report generated successfully!')
                st.download_button(label='📥 Download Complete Report', data=report_data, file_name=f'ml_comprehensive_report_{uuid.uuid4().hex[:6]}.docx', mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        except Exception as e:
            st.error(f'❌ Report generation failed: {e}')
            import traceback
            with st.expander('🐛 Error Details'):
                st.code(traceback.format_exc())