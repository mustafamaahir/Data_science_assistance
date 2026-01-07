import streamlit as st
import os
from eda import quick_eda, run_full_profile
from preprocessing import build_preprocessor
from models import get_model, tune_model
from report import create_docx_report
from llm import groq_generate_text
import pandas as pd
import json
import uuid


st.set_page_config(page_title="Data Science Assistant", layout="wide")
st.title("Data Science Assistant")

# Sidebar: API & navigation
st.sidebar.header('Settings & API')
use_env = st.sidebar.checkbox('Use GROQ_API_KEY from environment', value=True)
if use_env:
    groq_token = os.environ.get('GROQ_API_KEY', '')
    if not groq_token:
        groq_token = st.sidebar.text_input('Groq API Key', type='password')
else:
    groq_token = st.sidebar.text_input('Groq API Key', type='password')

st.sidebar.markdown('---')
nav = st.sidebar.radio('Workflow', ['Upload', 'EDA & Profiling', 'Preprocess', 'Modeling', 'Report'])

# Global session storage
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'preprocessor' not in st.session_state:
    st.session_state['preprocessor'] = None
if 'model_result' not in st.session_state:
    st.session_state['model_result'] = None
if 'target_column' not in st.session_state:
    st.session_state['target_column'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None

# ---------------- Upload ----------------
if nav == 'Upload':
    st.header('1) Upload dataset')
    uploaded = st.file_uploader('Upload CSV / Excel', ['csv', 'xlsx', 'xls'])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded, encoding='ISO-8859-1')
            else:
                df = pd.read_excel(uploaded)
            
            # Reset all downstream states when new file is uploaded
            st.session_state['df'] = df
            st.session_state['processed_df'] = None
            st.session_state['preprocessor'] = None
            st.session_state['model_result'] = None
            st.session_state['target_column'] = None
            
            st.success(f'Loaded {uploaded.name}: {df.shape[0]} rows × {df.shape[1]} columns')
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f'Failed to read file: {e}')

# ---------------- EDA & Profiling ----------------
elif nav == 'EDA & Profiling':
    st.header('2) EDA & Profiling')
    if st.session_state['df'] is None:
        st.info('Please upload a dataset first (Workflow -> Upload).')
    else:
        df = st.session_state['df']
        st.subheader('Quick EDA (executed live)')
        with st.spinner('Running quick EDA...'):
            cards, figs = quick_eda(df)
        
        # Display summary cards
        cols = st.columns(len(cards))
        for i, (key, value) in enumerate(cards.items()):
            cols[i].metric(key, value)
        
        # Display plots
        for fig in figs:
            st.pyplot(fig)
        
        st.markdown('---')
        if st.button('Generate full profiling report (ydata-profiling)'):
            with st.spinner('Running full profile — this may take some time for large datasets...'):
                profile_html = run_full_profile(df)
            st.success('Profile complete: open the HTML below')
            st.components.v1.html(profile_html, height=700, scrolling=True)

# ---------------- Preprocess ----------------
elif nav == 'Preprocess':
    st.header('3) Preprocessing')
    if st.session_state['df'] is None:
        st.info('Please upload a dataset first.')
    else:
        df = st.session_state['df']
        
        # Select target column first
        st.subheader('Select Target Column')
        target = st.selectbox('Choose target column for modeling', options=[None] + list(df.columns))
        
        if target:
            st.session_state['target_column'] = target
            
            # Show data overview
            st.subheader('Data Overview')
            info_df = pd.DataFrame({
                'dtype': df.dtypes.astype(str), 
                'missing': df.isna().sum(),
                'missing_%': (df.isna().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df)
            
            st.markdown('---')
            st.subheader('Preprocessing Options')
            num_strategy = st.selectbox('Numeric imputation strategy', ['median', 'mean', 'most_frequent'], index=0)
            cat_strategy = st.selectbox('Categorical imputation strategy', ['most_frequent', 'constant'], index=0)
            scale_numeric = st.checkbox('Scale numeric features (StandardScaler)', value=True)
            
            if st.button('Build & Apply Preprocessor'):
                try:
                    with st.spinner('Building preprocessing pipeline...'):
                        # Separate features and target
                        X = df.drop(columns=[target])
                        y = df[target]
                        
                        # Build preprocessor
                        preprocessor = build_preprocessor(
                            X, 
                            num_strategy=num_strategy, 
                            cat_strategy=cat_strategy, 
                            scale_numeric=scale_numeric
                        )
                        
                        # Fit and transform
                        X_processed = preprocessor.fit_transform(X)
                        
                        # Store in session state
                        st.session_state['preprocessor'] = preprocessor
                        st.session_state['processed_df'] = X_processed
                        
                        st.success(f'✅ Preprocessing complete! Shape: {X_processed.shape}')
                        st.info(f'Target column "{target}" has been separated and stored.')
                        
                except Exception as e:
                    st.error(f'Preprocessing failed: {e}')
        else:
            st.warning('Please select a target column to proceed.')

# ---------------- Modeling ----------------
elif nav == 'Modeling':
    st.header('4) Modeling & Hyperparameter Tuning')
    if st.session_state['df'] is None:
        st.info('Please upload a dataset first.')
    elif st.session_state['target_column'] is None:
        st.warning('Please select a target column in the Preprocess tab.')
    elif st.session_state['processed_df'] is None:
        st.warning('Please run preprocessing first (Preprocess tab).')
    else:
        df = st.session_state['df']
        target = st.session_state['target_column']
        X_processed = st.session_state['processed_df']
        
        # Get target variable
        y = df[target]
        
        # Detect problem type
        if y.dtype == 'object' or y.nunique() <= 20:
            problem_type = 'classification'
            st.info(f'🎯 Problem type: **Classification** ({y.nunique()} classes)')
        else:
            problem_type = 'regression'
            st.info(f'🎯 Problem type: **Regression**')
        
        st.markdown('---')
        st.subheader('Model Selection')
        
        if problem_type == 'classification':
            model_options = ['RandomForest', 'LogisticRegression', 'XGBoost', 'SVM']
        else:
            model_options = ['RandomForest', 'LinearRegression', 'XGBoost', 'SVR']
        
        model_name = st.selectbox('Choose Model', model_options)
        
        # Hyperparameters UI
        st.markdown('### Hyperparameters')
        param_preset = {}
        
        if model_name == 'RandomForest':
            n_estimators = st.slider('n_estimators', 10, 500, 100)
            max_depth = st.slider('max_depth (0 = None)', 0, 50, 0)
            param_preset = {
                'n_estimators': n_estimators, 
                'max_depth': None if max_depth == 0 else max_depth,
                'random_state': 42
            }
        
        elif model_name == 'XGBoost':
            n_estimators = st.slider('n_estimators', 10, 500, 100)
            max_depth = st.slider('max_depth', 1, 20, 6)
            learning_rate = st.number_input('learning_rate', 0.001, 1.0, 0.1, format="%.3f")
            param_preset = {
                'n_estimators': n_estimators, 
                'max_depth': max_depth, 
                'learning_rate': learning_rate,
                'random_state': 42
            }
        
        elif model_name == 'LogisticRegression':
            C = st.number_input('C (inverse regularization)', 0.0001, 1000.0, 1.0, format="%.4f")
            param_preset = {'C': C, 'random_state': 42, 'max_iter': 1000}
        
        elif model_name == 'LinearRegression':
            st.info('LinearRegression has no hyperparameters to tune.')
            param_preset = {}
        
        elif model_name in ['SVM', 'SVR']:
            C = st.number_input('C', 0.0001, 1000.0, 1.0, format="%.4f")
            kernel = st.selectbox('kernel', ['rbf', 'linear', 'poly'])
            param_preset = {'C': C, 'kernel': kernel}
        
        st.markdown('---')
        st.subheader('Training Options')
        tune = st.checkbox('Enable hyperparameter tuning (RandomizedSearchCV)', value=False)
        
        if tune:
            n_iter = st.number_input('n_iter for RandomizedSearchCV', min_value=5, max_value=200, value=20)
            cv = st.number_input('CV folds', min_value=2, max_value=10, value=5)
        else:
            n_iter = 10
            cv = 5
        
        if st.button('🚀 Train & Evaluate Model'):
            try:
                with st.spinner('Training model — this may take a while...'):
                    # Get model
                    result = get_model(model_name, param_preset, problem_type)
                    model_obj = result['model']
                    
                    # Train and evaluate
                    res = tune_model(
                        model_obj, 
                        X_processed, 
                        y, 
                        tune=tune, 
                        n_iter=n_iter, 
                        cv=cv,
                        problem_type=problem_type
                    )
                    
                    # Store results
                    st.session_state['model_result'] = res
                    
                    st.success('Training complete!')
                    
                    # Display metrics
                    st.subheader('Model Performance')
                    metrics_df = pd.DataFrame([res['metrics']])
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Show best params if tuning was enabled
                    if tune and 'best_params' in res:
                        st.subheader('Best Hyperparameters Found')
                        st.json(res['best_params'])
                    
            except Exception as e:
                st.error(f'Training failed: {e}')
                import traceback
                st.code(traceback.format_exc())

# ---------------- Report ----------------
elif nav == 'Report':
    st.header('5) Report & Export')
    if st.session_state['df'] is None:
        st.info('Please upload dataset first')
    else:
        df = st.session_state['df']
        model_res = st.session_state.get('model_result')
        
        st.subheader('Generate Executive Summary with AI')
        
        # Prepare summary data
        eda_summary = {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'numeric_columns': int(len(df.select_dtypes(include='number').columns)),
            'categorical_columns': int(len(df.select_dtypes(exclude='number').columns)),
            'total_missing': int(df.isna().sum().sum()),
            'missing_percentage': f"{(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%"
        }
        
        if st.button('Generate Executive Summary (Groq AI)'):
            if not groq_token:
                st.error('Please provide Groq API Key in the sidebar.')
            else:
                try:
                    with st.spinner('Generating AI summary...'):
                        # Create detailed prompt
                        prompt = f"""You are a professional data scientist writing an executive summary for a machine learning project.

Dataset Overview:
- Rows: {eda_summary['rows']}
- Columns: {eda_summary['columns']}
- Numeric columns: {eda_summary['numeric_columns']}
- Categorical columns: {eda_summary['categorical_columns']}
- Missing values: {eda_summary['total_missing']} ({eda_summary['missing_percentage']})

Model Results:
{json.dumps(model_res, default=str, indent=2) if model_res else "No model trained yet"}

Write a concise executive summary (3-4 paragraphs) covering:
1. Dataset characteristics and quality
2. Preprocessing and data preparation steps
3. Model performance and key findings
4. Recommendations for next steps

Keep it professional and actionable."""
                        
                        exec_summary = groq_generate_text(
                            prompt=prompt,
                            api_key=groq_token,
                            model="llama-3.3-70b-versatile",
                            max_tokens=1000
                        )
                        
                        st.success('Summary generated!')
                        st.text_area('Executive Summary', value=exec_summary, height=400)
                        
                        # Store in session state
                        st.session_state['exec_summary'] = exec_summary
                        
                except Exception as e:
                    st.error(f'Failed to generate summary: {e}')
        
        # Export report
        if 'exec_summary' in st.session_state:
            st.markdown('---')
            st.subheader('Export Report')
            
            if st.button('📄 Generate Word Report (.docx)'):
                try:
                    with st.spinner('Creating Word document...'):
                        report_path = create_docx_report(
                            title=f'ML Project Report - {uuid.uuid4().hex[:6]}',
                            executive_summary=st.session_state['exec_summary'],
                            eda_summary=eda_summary,
                            model_summary=str(model_res) if model_res else "No model results available",
                            charts=[]
                        )
                        
                        # Read and encode file
                        with open(report_path, 'rb') as f:
                            data = f.read()
                        
                        st.success('Report generated!')
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Report",
                            data=data,
                            file_name=f"ml_report_{uuid.uuid4().hex[:6]}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
                except Exception as e:
                    st.error(f'Failed to create report: {e}')