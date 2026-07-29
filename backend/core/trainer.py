import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error, silhouette_score
)
import time

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

NEEDS_SCALING = {"SVM", "KNN", "Logistic Regression"}

PERCENT_METRICS = {"accuracy", "f1_score", "precision", "recall", "silhouette_score", "r2_score"}


def suggest_models(df: pd.DataFrame, target_column: str = None) -> dict:
    if not target_column or target_column not in df.columns:
        return {"problem_type": "clustering", "suggested_models": ["K-Means"]}

    target = df[target_column].dropna()
    n_unique = target.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target)

    if is_numeric and n_unique > 10:
        problem_type = "regression"
        suggested = ["Random Forest", "XGBoost", "Linear Regression", "Decision Tree", "KNN", "SVM"]
    elif n_unique == 2:
        problem_type = "classification"
        suggested = ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Decision Tree", "KNN", "Naive Bayes"]
    else:
        problem_type = "classification"
        suggested = ["Random Forest", "XGBoost", "Decision Tree", "KNN", "SVM", "Naive Bayes", "Logistic Regression"]

    if not HAS_XGB:
        suggested = [m for m in suggested if m != "XGBoost"]

    return {"problem_type": problem_type, "suggested_models": suggested}


def _prepare_data(df: pd.DataFrame, target_column: str):
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    y_raw = df[target_column]

# Preprocess features: encode categoricals, convert numerics, drop datetimes and high-cardinality text
    X_cols = {}
    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if series.dtype == 'object':
            n_uniq = series.nunique()
            if n_uniq > min(50, max(2, len(df) // 3)):
                continue  # drop high-cardinality text
            le = LabelEncoder()
            X_cols[col] = le.fit_transform(series.fillna('__na__').astype(str)).astype(float)
        else:
            X_cols[col] = pd.to_numeric(series, errors='coerce').fillna(0).values.astype(float)

    if not X_cols:
        raise ValueError("No usable feature columns found after preprocessing.")

    X = np.column_stack(list(X_cols.values()))

    # Determine task type and encode target
    n_unique = y_raw.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(y_raw)

    if is_numeric and n_unique > 10:
        task = "regression"
        y_enc = pd.to_numeric(y_raw, errors='coerce').fillna(y_raw.median()).values.astype(float)
    else:
        task = "classification"
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw.astype(str))

    return X, y_enc, task


def _clf_factories(n_neighbors: int, n_classes: int):
    obj = "binary:logistic" if n_classes == 2 else "multi:softmax"
    return {
        "Linear Regression":   None,
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       lambda: DecisionTreeClassifier(random_state=42),
        "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes":         lambda: GaussianNB(),
        "SVM":                 lambda: SVC(random_state=42),
        "KNN":                 lambda: KNeighborsClassifier(n_neighbors=n_neighbors),
        "XGBoost":             (lambda: xgb.XGBClassifier(random_state=42, verbosity=0, objective=obj)) if HAS_XGB else None,
    }


def _reg_factories(n_neighbors: int):
    return {
        "Linear Regression":   lambda: LinearRegression(),
        "Logistic Regression": None,
        "Decision Tree":       lambda: DecisionTreeRegressor(random_state=42),
        "Random Forest":       lambda: RandomForestRegressor(n_estimators=100, random_state=42),
        "Naive Bayes":         None,
        "SVM":                 lambda: SVR(),
        "KNN":                 lambda: KNeighborsRegressor(n_neighbors=n_neighbors),
        "XGBoost":             (lambda: xgb.XGBRegressor(random_state=42, verbosity=0)) if HAS_XGB else None,
    }


def _train_kmeans(df: pd.DataFrame) -> dict:
    try:
        numeric = df.select_dtypes(include=[np.number]).fillna(0)
        if numeric.empty or len(numeric) < 4:
            return {"model": "K-Means", "status": "error",
                    "note": "Not enough numeric data for clustering", "metrics": None}

        X = StandardScaler().fit_transform(numeric)
        n_clusters = max(2, min(5, len(df) // 10))

        t0 = time.time()
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        elapsed = round(time.time() - t0, 2)

        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else 0.0

        return {
            "model": "K-Means",
            "status": "success",
            "task": "clustering",
            "metrics": {
                "silhouette_score": round(sil, 4),
                "n_clusters": int(n_clusters),
                "inertia": round(float(km.inertia_), 2),
            },
            "training_time": f"{elapsed}s",
            "primary_metric": "silhouette_score",
        }
    except Exception as e:
        return {"model": "K-Means", "status": "error", "note": str(e), "metrics": None}


def train_selected_models(df: pd.DataFrame, selected_models: list, target_column: str = None) -> dict:
    results = []
    selected = list(selected_models)

    # Unsupervised
    if "K-Means" in selected:
        results.append(_train_kmeans(df))
        selected = [m for m in selected if m != "K-Means"]

    if not selected:
        best = next((r["model"] for r in results if r.get("status") == "success"), None)
        return {"results": results, "best_model": best, "task": "clustering"}

    # Supervised models need a target column
    if not target_column or target_column not in df.columns:
        for m in selected:
            results.append({
                "model": m, "status": "skipped",
                "note": "Target column required for supervised learning", "metrics": None,
            })
        return {"results": results, "best_model": None, "task": "unknown"}

    # Prepare data
    try:
        X, y, task = _prepare_data(df, target_column)
    except Exception as e:
        return {"results": results, "best_model": None, "task": "unknown", "error": str(e)}

    if len(X) < 10:
        for m in selected:
            results.append({"model": m, "status": "error",
                            "note": "Dataset too small (need ≥ 10 rows)", "metrics": None})
        return {"results": results, "best_model": None, "task": task}

    test_size = max(0.15, min(0.25, 20 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_neighbors = max(1, min(5, len(X_train) - 1))
    n_classes = int(len(np.unique(y)))
    avg = "binary" if n_classes == 2 else "macro"

    factories = _clf_factories(n_neighbors, n_classes) if task == "classification" else _reg_factories(n_neighbors)

    for model_name in selected:
        factory = factories.get(model_name)

        if factory is None:
            results.append({
                "model": model_name, "status": "not_applicable",
                "note": f"Not designed for {task} tasks", "metrics": None,
            })
            continue

        try:
            use_scaled = model_name in NEEDS_SCALING
            X_tr = X_train_s if use_scaled else X_train
            X_te = X_test_s if use_scaled else X_test

            t0 = time.time()
            model = factory()
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            elapsed = round(time.time() - t0, 2)

            if task == "classification":
                metrics = {
                    "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
                    "f1_score":  round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "recall":    round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                }
                primary = "accuracy"
            else:
                metrics = {
                    "r2_score": round(float(r2_score(y_test, y_pred)), 4),
                    "mae":      round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "mse":      round(float(mean_squared_error(y_test, y_pred)), 4),
                }
                primary = "r2_score"

            results.append({
                "model": model_name, "status": "success", "task": task,
                "metrics": metrics, "training_time": f"{elapsed}s", "primary_metric": primary,
            })

        except Exception as e:
            results.append({"model": model_name, "status": "error", "note": str(e)[:300], "metrics": None})

    # Sort supervised results by primary metric; clustering kept separate
    supervised_ok  = [r for r in results if r.get("status") == "success" and r.get("task") == task]
    clustering_ok  = [r for r in results if r.get("status") == "success" and r.get("task") == "clustering"]
    others         = [r for r in results if r.get("status") != "success"]

    if supervised_ok:
        pm = supervised_ok[0]["primary_metric"]
        reverse = pm not in ("mse", "mae")
        supervised_ok.sort(key=lambda r: r["metrics"].get(pm, 0), reverse=reverse)
        best_model = supervised_ok[0]["model"]
    else:
        best_model = None

    return {"results": supervised_ok + clustering_ok + others, "best_model": best_model, "task": task}
