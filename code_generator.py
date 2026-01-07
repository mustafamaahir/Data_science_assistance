import json
from llm import groq_generate_text


def generate_feature_insights(feature_info: dict, api_key: str, df=None):
    """
    Generate AI insights about feature selection.
    
    Args:
        feature_info: Dict with feature_names, target, problem_type
        api_key: Groq API key
        df: Optional DataFrame for context
    
    Returns:
        String with insights and recommendations
    """
    # Get basic stats if df provided
    feature_stats = ""
    if df is not None:
        feature_names = feature_info.get('feature_names', [])
        stats = []
        for feat in feature_names[:20]:  # Limit for token size
            if feat in df.columns:
                if df[feat].dtype in ['int64', 'float64']:
                    stats.append(f"- {feat}: numeric, mean={df[feat].mean():.2f}, std={df[feat].std():.2f}")
                else:
                    stats.append(f"- {feat}: categorical, unique={df[feat].nunique()}")
        feature_stats = "\n".join(stats)
    
    prompt = f"""You are a data science expert analyzing features for a machine learning model.

Problem Type: {feature_info['problem_type']}
Target Variable: {feature_info['target']}

Available Features (first 50):
{', '.join(feature_info['feature_names'][:50])}

Feature Statistics:
{feature_stats if feature_stats else 'Not available'}

Task: Provide expert insights on:
1. Which features are likely most important for predicting the target
2. Which features might be redundant or collinear
3. Any feature engineering suggestions based on domain knowledge
4. Recommended top 10-15 features to start with

Keep your response concise and actionable (3-4 paragraphs maximum)."""
    
    insights = groq_generate_text(
        prompt=prompt,
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        max_tokens=800
    )
    
    return insights


def generate_code_notebook(context: dict, api_key: str):
    """
    Generate complete Python code for reproducing the ML pipeline.
    
    Args:
        context: Dict with preprocessing_log, model_result, etc.
        api_key: Groq API key
    
    Returns:
        String containing complete Python code
    """
    preprocessing_summary = "\n".join(context.get('preprocessing_log', [])[:15])
    
    model_info = context.get('model_result', {})
    model_name = model_info.get('model_name', 'RandomForestClassifier')
    metrics = model_info.get('metrics', {})
    
    imputation_info = context.get('imputation_decisions', {})
    
    prompt = f"""You are a senior data scientist. Generate a complete, production-ready Python script that reproduces this machine learning pipeline.

Project Details:
- Target Variable: {context.get('target', 'target')}
- Problem Type: {context.get('problem_type', 'classification')}
- Model Used: {model_name}
- Selected Features: {len(context.get('selected_features', []))} features

Preprocessing Summary:
{preprocessing_summary}

Imputation Decisions:
{json.dumps(imputation_info, indent=2)}

Model Performance:
{json.dumps({k: v for k, v in metrics.items() if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'residuals']}, indent=2)}

Generate a complete Python script that includes:

1. All necessary imports
2. Data loading (assume CSV file 'data.csv')
3. Complete preprocessing pipeline with exact imputation methods used
4. Feature selection (exact features used)
5. Model training with exact hyperparameters
6. Model evaluation with all metrics
7. Visualization code for:
   - Feature importance plot
   - Confusion matrix (if classification)
   - ROC curve (if binary classification)
   - Residual plots (if regression)
8. Model saving to disk

Requirements:
- Use only standard libraries (pandas, numpy, sklearn, matplotlib, seaborn)
- Include detailed comments explaining each step
- Make it ready to run (no placeholders)
- Include error handling
- Add a main() function structure

Generate ONLY the Python code, no explanations before or after."""
    
    code = groq_generate_text(
        prompt=prompt,
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        max_tokens=3000
    )
    
    # Clean up the code (remove markdown if present)
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    return code


def generate_preprocessing_explanation(imputation_decisions: dict, api_key: str):
    """
    Generate natural language explanation of preprocessing decisions.
    
    Args:
        imputation_decisions: Dict of column -> method
        api_key: Groq API key
    
    Returns:
        String with detailed explanation
    """
    prompt = f"""You are explaining data preprocessing decisions to a technical audience.

Imputation Methods Used:
{json.dumps(imputation_decisions, indent=2)}

Write a clear, technical explanation (2-3 paragraphs) of:
1. Why each imputation method was chosen
2. The trade-offs of each approach
3. Expected impact on model performance

Be specific about KNN vs regression imputation differences."""
    
    explanation = groq_generate_text(
        prompt=prompt,
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        max_tokens=600
    )
    
    return explanation


def generate_model_interpretation(model_result: dict, api_key: str):
    """
    Generate interpretation of model results.
    
    Args:
        model_result: Model training results
        api_key: Groq API key
    
    Returns:
        String with model interpretation
    """
    metrics = model_result.get('metrics', {})
    model_name = model_result.get('model_name', 'Model')
    feature_importance = model_result.get('feature_importance')
    
    # Extract clean metrics (no complex objects)
    clean_metrics = {
        k: v for k, v in metrics.items() 
        if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'residuals']
    }
    
    top_features = []
    if feature_importance is not None:
        top_features = feature_importance.head(10)['Feature'].tolist()
    
    prompt = f"""You are a data scientist interpreting model results for stakeholders.

Model: {model_name}

Performance Metrics:
{json.dumps(clean_metrics, indent=2)}

Top 10 Important Features:
{', '.join(top_features) if top_features else 'Not available'}

Write a professional interpretation (3-4 paragraphs) covering:
1. Overall model performance assessment
2. What the metrics tell us about model quality
3. Key drivers (important features) and what they mean
4. Limitations and recommendations for improvement

Use clear language suitable for both technical and business audiences."""
    
    interpretation = groq_generate_text(
        prompt=prompt,
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        max_tokens=800
    )
    
    return interpretation