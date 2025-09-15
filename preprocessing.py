import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(df: pd.DataFrame,
                       num_strategy="median",
                       cat_strategy="most_frequent",
                       scale_numeric=True):
    """Build preprocessing pipeline based on dataframe schema."""
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=num_strategy)),
        ("scaler", StandardScaler() if scale_numeric else "passthrough")
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_strategy, fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor
