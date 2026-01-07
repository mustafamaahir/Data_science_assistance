import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

def get_model(name: str, params: dict, problem_type: str):
    """Return initialized model with given hyperparameters."""
    
    if problem_type == "classification":
        if name == "RandomForest":
            model = RandomForestClassifier(**params)
        
        elif name == "LogisticRegression":
            model = LogisticRegression(**params)
        
        elif name == "SVM":
            model = SVC(**params)
        
        elif name == "XGBoost":
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(**params)
            except ImportError:
                print("XGBoost not installed, falling back to RandomForest")
                model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100))
        
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
                model = RandomForestRegressor(n_estimators=params.get('n_estimators', 100))
        
        else:
            model = RandomForestRegressor(**params)

    return {"model": model}


def tune_model(model, X, y, tune=False, n_iter=10, cv=5, problem_type='classification'):
    """
    Train and evaluate model with or without hyperparameter tuning.
    
    Args:
        model: sklearn model instance
        X: Features (can be numpy array or DataFrame)
        y: Target variable
        tune: Whether to perform hyperparameter tuning
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds
        problem_type: 'classification' or 'regression'
    
    Returns:
        Dictionary with model, metrics, and optionally best_params
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' and y.nunique() > 1 else None
    )
    
    best_params = None
    
    if tune:
        # Define parameter distributions for different models
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
                "penalty": ['l1', 'l2'],
                "solver": ['liblinear', 'saga']
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
                cv=cv,
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
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    
    if problem_type == 'classification':
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_score"] = float(f1_score(y_test, y_pred, average="weighted"))
        metrics["precision"] = float(precision_score(y_test, y_pred, average="weighted"))
        metrics["recall"] = float(recall_score(y_test, y_pred, average="weighted"))
    
    else:  # regression
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        metrics["r2_score"] = float(r2_score(y_test, y_pred))
    
    result = {
        "model": model,
        "metrics": metrics,
        "model_name": model.__class__.__name__
    }
    
    if best_params:
        result["best_params"] = best_params
    
    return result