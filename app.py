import streamlit as st
import os
from eda import quick_eda, run_full_profile
from preprocessing import build_preprocessor
from models import get_model, tune_model
from report import create_docx_report
from llm import hf_generate_text
import pandas as pd
import base64
import json
import uuid

st.set_page_config(page_title="Data Science Assistant", layout="wide")
st.title("Data Science Assistant")

# Sidebar: API & navigation
st.sidebar.header('Settings & API')
use_env = st.sidebar.checkbox('Use HF_API_TOKEN from environment', value=True)
if use_env:
    hf_token = os.environ.get('HF_API_TOKEN', '')
    if not hf_token:
        hf_token = st.sidebar.text_input('Hugging Face API Token', type='password')
else:
    hf_token = st.sidebar.text_input('Hugging Face API Token', type='password')

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
                    df = pd.read_csv(uploaded, encoding='ISO-8859-1')  # fallback for non-UTF8 files
            else:
                df = pd.read_excel(uploaded)
            st.session_state['df'] = df
            st.success(f'Loaded {uploaded.name}: {df.shape[0]} rows × {df.shape[1]} columns')
            st.dataframe(df.head(5))
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
        st.write(cards)
        for fig in figs:
            st.pyplot(fig)
        st.markdown('---')
        if st.button('Generate full profiling report (ydata-profiling)'):
            st.info('Running full profile — this may take some time for large datasets')
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
        st.write('Columns and types:')
        st.table(pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum()}))
        st.markdown('Select preprocessing options:')
        num_strategy = st.selectbox('Numeric imputation strategy', ['median', 'mean', 'most_frequent'], index=0)
        cat_strategy = st.selectbox('Categorical imputation strategy', ['most_frequent', 'constant'], index=0)
        scale_numeric = st.checkbox('Scale numeric features (StandardScaler)', value=True)
        if st.button('Build & run preprocessor'):
            st.info('Building preprocessing pipeline...')
            preprocessor = build_preprocessor(df, num_strategy=num_strategy, cat_strategy=cat_strategy, scale_numeric=scale_numeric)
            st.session_state['preprocessor'] = preprocessor
            X = df.dropna(axis=1, how='all')
            st.session_state['processed_df'] = preprocessor.fit_transform(X)
            st.success('Preprocessing complete — processed array available in session_state')

# ---------------- Modeling ----------------
elif nav == 'Modeling':
    st.header('4) Modeling & Hyperparameter Tuning')
    if st.session_state['df'] is None:
        st.info('Please upload a dataset first.')
    else:
        df = st.session_state['df']
        target = st.selectbox('Choose target column (or None)', options=[None] + list(df.columns))
        if target:
            X = df.drop(columns=[target])
            y = df[target]
            problem_type = 'classification' if y.dtype == 'object' or (y.dtype in [int, 'int64'] and y.nunique() <= 20) else 'regression'
            st.write(f'Problem type detected: **{problem_type}**')

            model_name = st.selectbox('Model', ['RandomForest', 'LogisticRegression', 'XGBoost', 'SVM'])

            # expose hyperparams UI
            st.markdown('### Hyperparameters')
            if model_name == 'RandomForest':
                n_estimators = st.slider('n_estimators', 10, 500, 100)
                max_depth = st.slider('max_depth (None=0)', 0, 50, 0)
                param_preset = {'n_estimators': n_estimators, 'max_depth': None if max_depth == 0 else max_depth}
            elif model_name == 'XGBoost':
                try:
                    import xgboost as xgb
                    n_estimators = st.slider('n_estimators', 10, 500, 100)
                    max_depth = st.slider('max_depth', 1, 20, 6)
                    learning_rate = st.number_input('learning_rate', 0.001, 1.0, 0.1, format="%.3f")
                    param_preset = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
                except Exception:
                    st.warning('xgboost not installed — add xgboost to requirements if you want XGBoost support')
                    param_preset = {'n_estimators': 100}
            elif model_name == 'LogisticRegression':
                C = st.number_input('C (inverse regularization)', 0.0001, 1000.0, 1.0, format="%.4f")
                param_preset = {'C': C}
            elif model_name == 'SVM':
                C = st.number_input('C', 0.0001, 1000.0, 1.0, format="%.4f")
                kernel = st.selectbox('kernel', ['rbf', 'linear', 'poly'])
                param_preset = {'C': C, 'kernel': kernel}

            tune = st.checkbox('Enable hyperparameter tuning (RandomizedSearchCV)')
            if tune:
                n_iter = st.number_input('n_iter for RandomizedSearchCV', min_value=5, max_value=200, value=20)
                cv = st.number_input('CV folds', min_value=2, max_value=10, value=5)
            else:
                n_iter = None
                cv = 5

            if st.button('Train & Evaluate'):
                st.info('Training model — this may take a while')
                result = get_model(model_name, param_preset, problem_type)
                model_obj = result['model']
                res = tune_model(model_obj, X, y, tune=tune, n_iter=n_iter, cv=cv)
                st.session_state['model_result'] = res
                st.success('Training complete — results in session_state')
                st.write(res['metrics'])

# ---------------- Report ----------------
elif nav == 'Report':
    st.header('5) Report & Export')
    if st.session_state['df'] is None:
        st.info('Please upload dataset first')
    else:
        df = st.session_state['df']
        model_res = st.session_state.get('model_result')
        st.subheader('Draft executive summary (LLM-generated)')
        eda_summary = {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isna().sum().to_dict()
        }
        llm_model = st.text_input('LLM model id (HuggingFace) - or leave default', value='gpt2')
        if st.button('Generate executive summary'):
            prompt = 'You are a professional data scientist. Write an executive summary...' + json.dumps({'eda': eda_summary, 'model': model_res}, default=str)
            if hf_token:
                exec_summary = hf_generate_text(llm_model, prompt, hf_token, max_tokens=512)
            else:
                exec_summary = 'No HF token provided. Please set it in sidebar.'
            st.text_area('Executive Summary', value=exec_summary, height=300)
            if st.button('Build Word report (.docx)'):
                report_path = create_docx_report(f'Executive Report {uuid.uuid4().hex[:6]}', exec_summary, eda_summary, str(model_res), [])
                with open(report_path, 'rb') as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode()
                href = f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}"
                st.markdown(f'[Download report]({href})')
