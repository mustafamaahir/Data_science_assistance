import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import chi2_contingency
import streamlit as st


def test_mcar(df: pd.DataFrame, col: str) -> dict:
    """
    Test if missing data is MCAR using correlation-based approach.
    
    Returns:
        dict with 'is_mcar' (bool), 'p_value' (float), 'method' (str)
    """
    if df[col].isna().sum() == 0:
        return {'is_mcar': True, 'p_value': 1.0, 'method': 'no_missing'}
    
    df_temp = df.copy()
    df_temp['missing_indicator'] = df_temp[col].isna().astype(int)
    
    numeric_cols = df_temp.select_dtypes(include='number').columns
    numeric_cols = [c for c in numeric_cols if c not in [col, 'missing_indicator']]
    
    if len(numeric_cols) == 0:
        cat_cols = df_temp.select_dtypes(exclude='number').columns
        cat_cols = [c for c in cat_cols if c != col]
        
        if len(cat_cols) == 0:
            return {'is_mcar': True, 'p_value': 1.0, 'method': 'insufficient_data'}
        
        try:
            contingency = pd.crosstab(df_temp['missing_indicator'], df_temp[cat_cols[0]])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            
            return {
                'is_mcar': p_value > 0.05,
                'p_value': float(p_value),
                'method': 'chi_square'
            }
        except:
            return {'is_mcar': True, 'p_value': 1.0, 'method': 'chi_square_failed'}
    
    correlations = []
    for num_col in numeric_cols:
        if df_temp[num_col].isna().sum() < len(df_temp):
            corr = df_temp[['missing_indicator', num_col]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    if len(correlations) == 0:
        return {'is_mcar': True, 'p_value': 1.0, 'method': 'no_valid_correlations'}
    
    max_corr = max(correlations)
    is_mcar = max_corr < 0.1
    
    return {
        'is_mcar': is_mcar,
        'p_value': 1 - max_corr,
        'method': 'correlation',
        'max_correlation': float(max_corr)
    }


def compare_imputation_methods(
    df: pd.DataFrame, 
    col: str, 
    is_numeric: bool,
    n_neighbors: int = 5
) -> dict:
    """Compare KNN and Regression imputation, return best method."""
    df_temp = df.copy()
    missing_mask = df_temp[col].isna()
    
    if missing_mask.sum() > 0.8 * len(df_temp):
        return {
            'best_method': 'simple',
            'reason': 'too_many_missing',
            'knn_score': None,
            'regression_score': None
        }
    
    non_missing_idx = df_temp[~missing_mask].index
    if len(non_missing_idx) < 10:
        return {
            'best_method': 'simple',
            'reason': 'insufficient_samples',
            'knn_score': None,
            'regression_score': None
        }
    
    test_size = min(int(0.2 * len(non_missing_idx)), 100)
    test_idx = np.random.choice(non_missing_idx, size=test_size, replace=False)
    true_values = df_temp.loc[test_idx, col].values
    
    df_test = df_temp.copy()
    df_test.loc[test_idx, col] = np.nan
    
    try:
        if is_numeric:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            numeric_cols = df_test.select_dtypes(include='number').columns.tolist()
            
            df_knn = df_test[numeric_cols].copy()
            df_knn_imputed = pd.DataFrame(
                knn_imputer.fit_transform(df_knn),
                columns=numeric_cols,
                index=df_knn.index
            )
            knn_pred = df_knn_imputed.loc[test_idx, col].values
            knn_score = np.sqrt(np.mean((true_values - knn_pred) ** 2))
        else:
            knn_score = None
        
        if is_numeric:
            reg_imputer = IterativeImputer(
                random_state=42,
                max_iter=10,
                estimator=RandomForestRegressor(n_estimators=10, random_state=42)
            )
            df_reg = df_test[numeric_cols].copy()
            df_reg_imputed = pd.DataFrame(
                reg_imputer.fit_transform(df_reg),
                columns=numeric_cols,
                index=df_reg.index
            )
            reg_pred = df_reg_imputed.loc[test_idx, col].values
            reg_score = np.sqrt(np.mean((true_values - reg_pred) ** 2))
        else:
            reg_score = None
        
        if knn_score is not None and reg_score is not None:
            best_method = 'knn' if knn_score < reg_score else 'regression'
        elif knn_score is not None:
            best_method = 'knn'
        elif reg_score is not None:
            best_method = 'regression'
        else:
            best_method = 'simple'
        
        return {
            'best_method': best_method,
            'knn_score': float(knn_score) if knn_score is not None else None,
            'regression_score': float(reg_score) if reg_score is not None else None,
            'reason': 'comparison_complete'
        }
    
    except Exception as e:
        return {
            'best_method': 'simple',
            'reason': f'error: {str(e)}',
            'knn_score': None,
            'regression_score': None
        }


def smart_impute_column(
    df: pd.DataFrame,
    col: str,
    method: str,
    n_neighbors: int = 5,
    fill_value: str = 'Missing'
) -> pd.Series:
    """Impute a single column using specified method."""
    if method == 'drop':
        return df[col].dropna()
    
    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    
    if method == 'simple':
        if is_numeric:
            return df[col].fillna(df[col].median())
        else:
            # For categorical: use mode or custom fill value
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                return df[col].fillna(mode_val[0])
            else:
                return df[col].fillna(fill_value)
    
    elif method == 'constant' and not is_numeric:
        return df[col].fillna(fill_value)
    
    elif method == 'knn' and is_numeric:
        try:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df_numeric = df[numeric_cols].copy()
            df_imputed = pd.DataFrame(
                knn_imputer.fit_transform(df_numeric),
                columns=numeric_cols,
                index=df_numeric.index
            )
            return df_imputed[col]
        except:
            return df[col].fillna(df[col].median())
    
    elif method == 'regression' and is_numeric:
        try:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            reg_imputer = IterativeImputer(
                random_state=42,
                max_iter=10,
                estimator=RandomForestRegressor(n_estimators=10, random_state=42)
            )
            df_numeric = df[numeric_cols].copy()
            df_imputed = pd.DataFrame(
                reg_imputer.fit_transform(df_numeric),
                columns=numeric_cols,
                index=df_numeric.index
            )
            return df_imputed[col]
        except:
            return df[col].fillna(df[col].median())
    
    else:
        return smart_impute_column(df, col, 'simple', fill_value=fill_value)


def comprehensive_preprocessing(
    df: pd.DataFrame,
    target: str,
    drop_threshold: float = 0.05,
    knn_neighbors: int = 5,
    scale_numeric: bool = True,
    skip_mcar: bool = False
) -> dict:
    """Perform comprehensive preprocessing with MCAR testing and smart imputation."""
    preprocessing_log = []
    imputation_decisions = {}
    
    df_work = df.copy()
    
    # 1. Handle target variable
    y = df_work[target]
    target_missing_ratio = y.isna().mean()
    
    if target_missing_ratio > 0:
        if target_missing_ratio < drop_threshold:
            df_work = df_work.dropna(subset=[target])
            y = df_work[target]
            preprocessing_log.append(
                f"Target '{target}': Dropped {target_missing_ratio:.1%} missing rows"
            )
        else:
            is_numeric = pd.api.types.is_numeric_dtype(y)
            if is_numeric:
                fill_val = y.median()
            else:
                mode_val = y.mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Missing'
            
            df_work[target] = df_work[target].fillna(fill_val)
            y = df_work[target]
            preprocessing_log.append(
                f"Target '{target}': Imputed {target_missing_ratio:.1%} missing with {fill_val}"
            )
    
    # 2. Separate features
    X = df_work.drop(columns=[target])
    
    # 3. Process each feature column
    for col in X.columns:
        missing_ratio = X[col].isna().mean()
        
        if missing_ratio == 0:
            imputation_decisions[col] = 'none'
            continue
        
        is_numeric = pd.api.types.is_numeric_dtype(X[col])
        
        # MCAR Test
        if not skip_mcar:
            mcar_result = test_mcar(X, col)
            is_mcar = mcar_result['is_mcar']
            preprocessing_log.append(
                f"Column '{col}': {missing_ratio:.1%} missing, "
                f"MCAR test: {'PASS' if is_mcar else 'FAIL'} "
                f"(p={mcar_result.get('p_value', 'N/A'):.3f})"
            )
        else:
            is_mcar = missing_ratio < 0.2
            preprocessing_log.append(
                f"Column '{col}': {missing_ratio:.1%} missing (MCAR test skipped)"
            )
        
        # Decision logic
        if is_mcar and missing_ratio < drop_threshold:
            X = X.dropna(subset=[col])
            y = y.loc[X.index]
            imputation_decisions[col] = 'drop'
            preprocessing_log.append(f"  → Action: Dropped rows with missing values")
        
        elif is_numeric:
            comparison = compare_imputation_methods(
                X, col, is_numeric=True, n_neighbors=knn_neighbors
            )
            best_method = comparison['best_method']
            imputation_decisions[col] = best_method
            
            if best_method in ['knn', 'regression']:
                X[col] = smart_impute_column(X, col, best_method, knn_neighbors)
                preprocessing_log.append(
                    f"  → Action: {best_method.upper()} imputation "
                    f"(KNN RMSE: {comparison.get('knn_score', 'N/A')}, "
                    f"Reg RMSE: {comparison.get('regression_score', 'N/A')})"
                )
            else:
                X[col] = smart_impute_column(X, col, 'simple')
                preprocessing_log.append(
                    f"  → Action: Simple median imputation "
                    f"(reason: {comparison.get('reason', 'unknown')})"
                )
        
        else:
            # Categorical: use mode or constant
            X[col] = smart_impute_column(X, col, 'simple', fill_value='Missing')
            imputation_decisions[col] = 'mode/constant (Missing)'
            preprocessing_log.append(f"  → Action: Mode/constant imputation (categorical)")
    
    # 4. Build sklearn pipeline for encoding/scaling
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    
    transformers = []
    
    if num_cols:
        num_steps = []
        if scale_numeric:
            num_steps.append(('scaler', StandardScaler()))
        
        if num_steps:
            transformers.append(('num', Pipeline(num_steps), num_cols))
        else:
            from sklearn.preprocessing import FunctionTransformer
            transformers.append(('num', FunctionTransformer(), num_cols))
    
    if cat_cols:
        transformers.append((
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            cat_cols
        ))
    
    if transformers:
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        X_processed = preprocessor.fit_transform(X)
        
        feature_names = []
        if num_cols and scale_numeric:
            feature_names.extend([f"num__{c}" for c in num_cols])
        elif num_cols:
            feature_names.extend(num_cols)
        
        if cat_cols:
            feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
        
        X_processed = pd.DataFrame(
            X_processed,
            columns=feature_names,
            index=X.index
        )
    else:
        X_processed = X
        preprocessor = None
    
    return {
        'X_processed': X_processed,
        'y': y,
        'preprocessing_log': preprocessing_log,
        'imputation_decisions': imputation_decisions,
        'preprocessor': preprocessor
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing count and percentage for each column."""
    return pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Data Type": df.dtypes.astype(str).values
    }).sort_values("Missing %", ascending=False)