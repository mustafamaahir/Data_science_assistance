import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(df: pd.DataFrame,
                       num_strategy="median",
                       cat_strategy="most_frequent",
                       scale_numeric=True):
    """
    Build preprocessing pipeline based on dataframe schema.
    
    Args:
        df: Input DataFrame (features only, no target)
        num_strategy: Imputation strategy for numeric columns
        cat_strategy: Imputation strategy for categorical columns
        scale_numeric: Whether to scale numeric features
    
    Returns:
        ColumnTransformer pipeline
    """
    # Identify column types
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    
    transformers = []
    
    # Numeric pipeline
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy=num_strategy))]
        
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        
        num_pipe = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipe, num_cols))
    
    # Categorical pipeline
    if cat_cols:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=cat_strategy, fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    
    # Build the column transformer
    if not transformers:
        raise ValueError("No valid columns found for preprocessing")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    
    return preprocessor