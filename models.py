import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

def get_model(name: str, params: dict, problem_type: str):
    """Return initialized model with given hyperparameters."""
    if problem_type == "classification":
        if name == "RandomForest":
            model = RandomForestClassifier(**params)
        elif name == "LogisticRegression":
            model = LogisticRegression(max_iter=500, **params)
        elif name == "SVM":
            model = SVC(**params)
        elif name == "XGBoost":
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(**params)
            except ImportError:
                model = RandomForestClassifier(**params)
    else:
        if name == "RandomForest":
            model = RandomForestRegressor(**params)
        elif name == "LogisticRegression":
            model = LinearRegression()
        elif name == "SVM":
            model = SVR(**params)
        elif name == "XGBoost":
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(**params)
            except ImportError:
                model = RandomForestRegressor(**params)

    return {"model": model}


def tune_model(model, X, y, tune=False, n_iter=None, cv=5):
    """Train and evaluate model with or without hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if tune:
        param_dist = {
            "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]},
            "RandomForestRegressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]},
            "LogisticRegression": {"C": np.logspace(-3, 3, 10)},
            "SVC": {"C": np.logspace(-3, 3, 10), "kernel": ["linear", "rbf"]},
            "SVR": {"C": np.logspace(-3, 3, 10), "kernel": ["linear", "rbf"]},
        }
        est_name = model.__class__.__name__
        if est_name in param_dist:
            search = RandomizedSearchCV(model, param_dist[est_name], n_iter=n_iter, cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {}
    if len(np.unique(y)) <= 20:  # classification
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["f1"] = f1_score(y_test, y_pred, average="weighted")
    else:  # regression
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["r2"] = r2_score(y_test, y_pred)

    return {"model": model, "metrics": metrics}
