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
        cat_fill_value = st.text_input('Categorical fill value', value='Missing', help="Value to use for missing categorical data")
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
                result = comprehensive_preprocessing(df=df, target=target, drop_threshold=drop_threshold, knn_neighbors=knn_neighbors, scale_numeric=scale_numeric, skip_mcar=skip_mcar, cat_fill_value=cat_fill_value)
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
    st.header('4️⃣ Feature Engineering')
    if st.session_state.get('df') is None:
        st.info('👈 Please upload a dataset first.')
        st.stop()
    if st.session_state.get('engineered_df') is not None:
        df = st.session_state['engineered_df']
        st.info('📝 Working with engineered dataset')
    else:
        df = st.session_state['df'].copy()
        st.info('📝 Working with original dataset')
    st.write(f"Current shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.markdown('---')
    st.subheader('➕ Create New Features')
    feature_type = st.selectbox('Feature type to create', ['Mathematical Operation', 'Binning/Categorization', 'Date/Time Features', 'Text Features', 'AI-Suggested Features'])
    if feature_type == 'Mathematical Operation':
        st.write("Create new features using mathematical operations")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                feat1 = st.selectbox('First feature', numeric_cols)
            with col2:
                feat2 = st.selectbox('Second feature', [c for c in numeric_cols if c != feat1])
            operation = st.selectbox('Operation', ['Add (+)', 'Subtract (-)', 'Multiply (*)', 'Divide (/)', 'Power (^)', 'Log Ratio'])
            new_name = st.text_input('New feature name', value=f'{feat1}_{operation}_{feat2}')
            if st.button('Create Feature'):
                try:
                    if operation == 'Add (+)':
                        df[new_name] = df[feat1] + df[feat2]
                    elif operation == 'Subtract (-)':
                        df[new_name] = df[feat1] - df[feat2]
                    elif operation == 'Multiply (*)':
                        df[new_name] = df[feat1] * df[feat2]
                    elif operation == 'Divide (/)':
                        df[new_name] = df[feat1] / (df[feat2] + 1e-10)
                    elif operation == 'Power (^)':
                        df[new_name] = df[feat1] ** df[feat2]
                    elif operation == 'Log Ratio':
                        df[new_name] = np.log1p(df[feat1]) - np.log1p(df[feat2])
                    st.session_state['engineered_df'] = df
                    st.success(f'✅ Created feature: {new_name}')
                    st.dataframe(df[[feat1, feat2, new_name]].head())
                except Exception as e:
                    st.error(f'Failed to create feature: {e}')
        else:
            st.warning('Need at least 2 numeric columns')
    elif feature_type == 'Binning/Categorization':
        st.write("Convert continuous features into categorical bins")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            col_to_bin = st.selectbox('Column to bin', numeric_cols)
            bin_method = st.radio('Binning method', ['Equal Width', 'Equal Frequency', 'Custom Bins'])
            if bin_method in ['Equal Width', 'Equal Frequency']:
                n_bins = st.slider('Number of bins', 2, 10, 5)
                labels = st.text_input('Labels (comma-separated)', value='Low,Medium-Low,Medium,Medium-High,High'[:n_bins*6])
                labels_list = [l.strip() for l in labels.split(',')][:n_bins]
            else:
                bins_str = st.text_input('Custom bin edges (comma-separated)', value='0,25,50,75,100')
                bins_list = [float(b.strip()) for b in bins_str.split(',')]
                labels = st.text_input('Labels (comma-separated)', value='Low,Medium,High')
                labels_list = [l.strip() for l in labels.split(',')]
            new_name = st.text_input('New feature name', value=f'{col_to_bin}_binned')
            if st.button('Create Binned Feature'):
                try:
                    if bin_method == 'Equal Width':
                        df[new_name] = pd.cut(df[col_to_bin], bins=n_bins, labels=labels_list)
                    elif bin_method == 'Equal Frequency':
                        df[new_name] = pd.qcut(df[col_to_bin], q=n_bins, labels=labels_list, duplicates='drop')
                    else:
                        df[new_name] = pd.cut(df[col_to_bin], bins=bins_list, labels=labels_list)
                    st.session_state['engineered_df'] = df
                    st.success(f'✅ Created binned feature: {new_name}')
                    st.write(df[[col_to_bin, new_name]].head(10))
                    st.write(f"Value counts:\n{df[new_name].value_counts()}")
                except Exception as e:
                    st.error(f'Failed to create binned feature: {e}')
        else:
            st.warning('No numeric columns available')
    elif feature_type == 'Date/Time Features':
        st.write("Extract features from date/time columns")
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        if date_cols:
            date_col = st.selectbox('Select date column', date_cols)
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                if st.button('Convert to datetime'):
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        st.success(f'✅ Converted {date_col} to datetime')
                    except:
                        st.error('Failed to convert. Please check date format.')
                        st.stop()
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                features_to_extract = st.multiselect('Features to extract', ['Year', 'Month', 'Day', 'Day of Week', 'Quarter', 'Week of Year', 'Is Weekend'])
                if st.button('Extract Date Features'):
                    try:
                        if 'Year' in features_to_extract:
                            df[f'{date_col}_year'] = df[date_col].dt.year
                        if 'Month' in features_to_extract:
                            df[f'{date_col}_month'] = df[date_col].dt.month
                        if 'Day' in features_to_extract:
                            df[f'{date_col}_day'] = df[date_col].dt.day
                        if 'Day of Week' in features_to_extract:
                            df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
                        if 'Quarter' in features_to_extract:
                            df[f'{date_col}_quarter'] = df[date_col].dt.quarter
                        if 'Week of Year' in features_to_extract:
                            df[f'{date_col}_weekofyear'] = df[date_col].dt.isocalendar().week
                        if 'Is Weekend' in features_to_extract:
                            df[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
                        st.session_state['engineered_df'] = df
                        st.success(f'✅ Extracted {len(features_to_extract)} date features')
                        new_cols = [c for c in df.columns if c.startswith(f'{date_col}_')]
                        st.dataframe(df[[date_col] + new_cols].head())
                    except Exception as e:
                        st.error(f'Failed to extract date features: {e}')
        else:
            st.warning('No date columns detected')
    elif feature_type == 'Text Features':
        st.write("Extract features from text columns")
        text_cols = df.select_dtypes(include='object').columns.tolist()
        if text_cols:
            text_col = st.selectbox('Select text column', text_cols)
            features_to_extract = st.multiselect('Features to extract', ['Length', 'Word Count', 'Contains Digits', 'Contains Special Chars', 'Is Uppercase'])
            if st.button('Extract Text Features'):
                try:
                    if 'Length' in features_to_extract:
                        df[f'{text_col}_length'] = df[text_col].astype(str).str.len()
                    if 'Word Count' in features_to_extract:
                        df[f'{text_col}_word_count'] = df[text_col].astype(str).str.split().str.len()
                    if 'Contains Digits' in features_to_extract:
                        df[f'{text_col}_has_digits'] = df[text_col].astype(str).str.contains(r'\d').astype(int)
                    if 'Contains Special Chars' in features_to_extract:
                        df[f'{text_col}_has_special'] = df[text_col].astype(str).str.contains(r'[^a-zA-Z0-9\s]').astype(int)
                    if 'Is Uppercase' in features_to_extract:
                        df[f'{text_col}_is_upper'] = df[text_col].astype(str).str.isupper().astype(int)
                    st.session_state['engineered_df'] = df
                    st.success(f'✅ Extracted {len(features_to_extract)} text features')
                    new_cols = [c for c in df.columns if c.startswith(f'{text_col}_')]
                    st.dataframe(df[[text_col] + new_cols].head())
                except Exception as e:
                    st.error(f'Failed to extract text features: {e}')
        else:
            st.warning('No text columns available')
    elif feature_type == 'AI-Suggested Features':
        st.write("Get AI suggestions for feature engineering")
        if not groq_token:
            st.error('❌ Please provide Groq API Key in sidebar')
        else:
            if st.button('🤖 Get AI Feature Engineering Suggestions'):
                with st.spinner('Analyzing dataset for feature engineering opportunities...'):
                    try:
                        prompt = f"""You are a data science expert. Analyze this dataset and suggest feature engineering opportunities.

Dataset Info:
- Shape: {df.shape}
- Columns: {list(df.columns[:20])}
- Numeric columns: {list(df.select_dtypes(include='number').columns[:10])}
- Categorical columns: {list(df.select_dtypes(exclude='number').columns[:10])}

Target: {st.session_state.get('target_column', 'Not specified')}
Problem Type: {st.session_state.get('problem_type', 'Not specified')}

Provide 5-7 specific feature engineering suggestions:
1. What mathematical combinations would be valuable?
2. Which features should be binned and how?
3. Are there interaction features to create?
4. What polynomial or log transformations would help?
5. Any domain-specific features based on column names?

Be specific and actionable."""
                        suggestions = groq_generate_text(prompt=prompt, api_key=groq_token, model="llama-3.3-70b-versatile", max_tokens=1000)
                        st.success('✅ AI suggestions generated!')
                        st.write(suggestions)
                    except Exception as e:
                        st.error(f'Failed to get suggestions: {e}')
    st.markdown('---')
    st.subheader('📊 Current Dataset')
    st.write(f"Shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(10))
    col1, col2 = st.columns(2)
    with col1:
        if st.button('💾 Save Engineered Dataset'):
            st.session_state['engineered_df'] = df
            st.session_state['df'] = df
            st.success('✅ Engineered dataset saved! Use it in preprocessing.')
    with col2:
        if st.button('🔄 Reset to Original'):
            st.session_state['engineered_df'] = None
            st.info('Reset to original dataset')
            st.rerun()

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