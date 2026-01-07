import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(
    df: pd.DataFrame,
    num_strategy="median",
    cat_strategy="most_frequent",
    scale_numeric=True
):
    """
    Build preprocessing pipeline for ALL feature columns.
    """

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []

    # Numeric pipeline
    if num_cols:
        num_steps = [
            ("imputer", SimpleImputer(strategy=num_strategy))
        ]

        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))

        transformers.append((
            "num",
            Pipeline(num_steps),
            num_cols
        ))

    # Categorical pipeline
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy=cat_strategy,
                    fill_value="missing"
                )),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            cat_cols
        ))

    if not transformers:
        raise ValueError("No columns available for preprocessing")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def handle_target_missingness(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    drop_threshold: float = 0.05
):
    """
    Handle missing values in target variable.

    Returns:
        cleaned_df, y
    """

    y = df[target]
    missing_ratio = y.isna().mean()

    # No missing
    if missing_ratio == 0:
        return df.copy(), y.copy()

    # Drop if < threshold
    if missing_ratio < drop_threshold:
        df_clean = df.dropna(subset=[target])
        return df_clean, df_clean[target]

    # Otherwise impute
    if problem_type == "regression":
        fill_value = y.median()
    else:
        fill_value = y.mode()[0]

    df_imputed = df.copy()
    df_imputed[target] = df_imputed[target].fillna(fill_value)

    return df_imputed, df_imputed[target]


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing count and percentage for each column."""
    return pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2)
    })

