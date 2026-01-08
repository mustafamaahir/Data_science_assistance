import json
from llm import groq_generate_text


def generate_business_insights(df, model_result: dict, api_key: str):
    """
    Generate business-focused insights from model results.
    
    Args:
        df: Original DataFrame
        model_result: Model training results with metrics and feature importance
        api_key: Groq API key
    
    Returns:
        String with business insights and actionable recommendations
    """
    # Extract key information
    metrics = model_result.get('metrics', {})
    feature_importance = model_result.get('feature_importance')
    problem_type = 'classification' if 'accuracy' in metrics else 'regression'
    
    # Get top features
    top_features = []
    if feature_importance is not None:
        top_features = feature_importance.head(10)['Feature'].tolist()
    
    # Clean metrics
    clean_metrics = {k: v for k, v in metrics.items() 
                    if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'residuals']}
    
    # Get data context
    data_context = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'numeric_features': len(df.select_dtypes(include='number').columns),
        'categorical_features': len(df.select_dtypes(exclude='number').columns)
    }
    
    prompt = f"""You are a BUSINESS ANALYST and STRATEGY CONSULTANT translating machine learning results into actionable business recommendations.

CRITICAL INSTRUCTION: Do NOT discuss technical ML details (metrics, algorithms, preprocessing). Focus ONLY on business impact and actions.

Data Context:
- {data_context['rows']:,} records analyzed
- {data_context['columns']} business variables examined

Model Performance:
{json.dumps(clean_metrics, indent=2)}

Top Business Drivers (Most Important Factors):
{', '.join(top_features[:5])}

Your task: Write a business-focused insight report (4-5 paragraphs) covering:

1. **BUSINESS SITUATION**: What does this data tell us about the business reality? What patterns emerged?

2. **KEY DRIVERS**: Which factors matter most for business outcomes? What does this mean for operations?

3. **BUSINESS IMPACT**: Translate model performance into business terms:
   - If accuracy is 85%, what does this mean for decision-making confidence?
   - If R² is 0.75, how much of business outcome variation can we explain?
   - What's the business risk of errors?

4. **ACTIONABLE RECOMMENDATIONS**: Provide 5-7 specific, concrete actions:
   - What should management DO differently?
   - Where should resources be allocated?
   - What processes need improvement?
   - What decisions can now be automated?
   - What ROI can be expected?

5. **IMPLEMENTATION ROADMAP**: 
   - Quick wins (30 days)
   - Medium-term initiatives (3-6 months)
   - Long-term strategy (12+ months)

Use business language - avoid terms like "model," "features," "algorithm," "preprocessing." Instead use:
- "Business factors" instead of "features"
- "Prediction confidence" instead of "accuracy"
- "Key drivers" instead of "important features"
- "Business outcomes" instead of "target variable"

Be specific with numbers, percentages, and dollar impacts where possible."""
    
    insights = groq_generate_text(
        prompt=prompt,
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        temperature=0.7
    )
    
    return insights


def generate_feature_insights(feature_info: dict, api_key: str, df=None):
    """Generate AI insights about feature selection."""
    feature_stats = ""
    if df is not None:
        feature_names = feature_info.get('feature_names', [])
        stats = []
        for feat in feature_names[:20]:
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
    """Generate complete Python code for reproducing the ML pipeline."""
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
    
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    return code


def generate_preprocessing_explanation(imputation_decisions: dict, api_key: str):
    """Generate natural language explanation of preprocessing decisions."""
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
    """Generate interpretation of model results."""
    metrics = model_result.get('metrics', {})
    model_name = model_result.get('model_name', 'Model')
    feature_importance = model_result.get('feature_importance')
    
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


def generate_autonomous_recommendations(context: dict, api_key: str):
    """
    Generate autonomous AI agent recommendations for improving analysis.
    
    Args:
        context: Dict with current analysis state
        api_key: Groq API key
    
    Returns:
        Dict with recommendations and executable actions
    """
    prompt = f"""You are an AUTONOMOUS AI DATA SCIENCE AGENT analyzing this ML project and providing actionable improvement recommendations.

Current State:
- Dataset: {context.get('rows', 0):,} rows, {context.get('columns', 0)} columns
- Target: {context.get('target', 'Unknown')}
- Problem Type: {context.get('problem_type', 'Unknown')}
- Model: {context.get('model_name', 'Not trained')}
- Current Performance: {json.dumps(context.get('metrics', {}), default=str)}
- Features Used: {len(context.get('selected_features', []))} out of {context.get('total_features', 0)}

Preprocessing Issues:
{json.dumps(context.get('preprocessing_issues', []), indent=2)}

Your task: Analyze the entire pipeline and provide specific, prioritized recommendations:

1. **DATA QUALITY IMPROVEMENTS**
   - What data issues need addressing?
   - Should we collect more data? Why?
   - Which missing value handling needs revision?

2. **FEATURE ENGINEERING**
   - What new features should be created? Be specific.
   - Which existing features should be removed?
   - What transformations are needed?

3. **MODEL OPTIMIZATION**
   - Should we try different algorithms? Which ones and why?
   - What hyperparameters need tuning?
   - Is the current train/test split appropriate?

4. **PERFORMANCE BOOST ACTIONS**
   - Rank top 5 actions by expected performance impact
   - Estimate improvement for each action
   - Provide implementation difficulty (Easy/Medium/Hard)

5. **AUTOMATED ACTIONS I CAN TAKE**
   - List specific actions that can be automated
   - Provide exact Python code snippets for each
   - Indicate which need human approval

Format as JSON with this structure:
{{
  "priority_actions": [
    {{
      "action": "description",
      "expected_improvement": "X%",
      "difficulty": "Easy/Medium/Hard",
      "automated": true/false,
      "code": "python code if automated"
    }}
  ],
  "data_recommendations": ["list"],
  "feature_recommendations": ["list"],
  "model_recommendations": ["list"],
  "business_impact": "overall impact summary"
}}

Be specific, quantitative, and actionable."""
    
    recommendations = groq_generate_text(
        prompt=prompt,
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        temperature=0.7
    )
    
    # Try to parse JSON
    try:
        if '```json' in recommendations:
            recommendations = recommendations.split('```json')[1].split('```')[0].strip()
        elif '```' in recommendations:
            recommendations = recommendations.split('```')[1].split('```')[0].strip()
        
        return json.loads(recommendations)
    except:
        # Return as text if JSON parsing fails
        return {'raw_recommendations': recommendations}