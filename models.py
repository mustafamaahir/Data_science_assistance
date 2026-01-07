import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR


def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract feature importance from model.
    
    Returns:
        DataFrame with feature names and importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    return importance_df


def get_model(name: str, params: dict, problem_type: str):
    """Return initialized model with given hyperparameters."""
    
    if problem_type == "classification":
        if name == "RandomForest":
            model = RandomForestClassifier(**params)
        
        elif name == "LogisticRegression":
            model = LogisticRegression(**params)
        
        elif name == "SVM":
            model = SVC(**params, probability=True)
        
        elif name == "XGBoost":
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(**params)
            except ImportError:
                print("XGBoost not installed, falling back to RandomForest")
                model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100), random_state=42)
        
        else:
            model = RandomForestClassifier(**params)
    
    else:  # regression
        if name == "RandomForest":
            model = RandomForestRegressor(**params)
        
        elif name == "LinearRegression":
            model = LinearRegression()
        
        elif name == "SVR":
            model = SVR(**params)
        
        elif name == "XGBoost":
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(**params)
            except ImportError:
                print("XGBoost not installed, falling back to RandomForest")
                model = RandomForestRegressor(n_estimators=params.get('n_estimators', 100), random_state=42)
        
        else:
            model = RandomForestRegressor(**params)

    return {"model": model}


def tune_model(model, X, y, tune=False, n_iter=10, cv=5, problem_type='classification'):
    """
    Train and evaluate model with comprehensive metrics.
    """
    # Split data
    test_size = min(0.2, max(0.1, 100 / len(X)))
    
    if problem_type == 'classification' and y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    best_params = None
    
    if tune:
        param_distributions = {
            "RandomForestClassifier": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "RandomForestRegressor": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "LogisticRegression": {
                "C": np.logspace(-3, 3, 20),
                "penalty": ['l2'],
                "solver": ['lbfgs', 'saga']
            },
            "SVC": {
                "C": np.logspace(-3, 3, 20),
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ['scale', 'auto']
            },
            "SVR": {
                "C": np.logspace(-3, 3, 20),
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ['scale', 'auto']
            },
            "XGBClassifier": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 0.9, 1.0]
            },
            "XGBRegressor": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 0.9, 1.0]
            }
        }
        
        model_name = model.__class__.__name__
        
        if model_name in param_distributions:
            search = RandomizedSearchCV(
                model,
                param_distributions[model_name],
                n_iter=n_iter,
                cv=min(cv, 5),
                n_jobs=-1,
                random_state=42,
                scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {}
    
    if problem_type == 'classification':
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["f1_score"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        
        # ROC AUC (if binary or has predict_proba)
        if hasattr(model, 'predict_proba') and y.nunique() == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }
    
    else:  # regression
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["r2_score"] = float(r2_score(y_test, y_pred))
        metrics["mse"] = float(mean_squared_error(y_test, y_pred))
        
        # Residuals
        residuals = y_test - y_pred
        metrics["residuals"] = {
            "values": residuals.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist()
        }
    
    # Feature importance
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
    feature_importance = get_feature_importance(model, feature_names)
    
    result = {
        "model": model,
        "metrics": metrics,
        "model_name": model.__class__.__name__,
        "feature_importance": feature_importance,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape
    }
    
    if best_params:
        result["best_params"] = best_params
    
    return result