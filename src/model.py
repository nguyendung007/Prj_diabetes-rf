"""
src/model.py
Huấn luyện & đánh giá Random Forest.
So sánh với Logistic Regression và Decision Tree.
Hỗ trợ GridSearchCV để tối ưu hyperparameter.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# ── Hyperparameter grid ──────────────────────────────────────────────────────
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2],
}


# ── Huấn luyện ───────────────────────────────────────────────────────────────

def train_random_forest(
    X_train, y_train, tune: bool = True, random_state: int = 42, cv: int = 5
) -> RandomForestClassifier:
    """Huấn luyện Random Forest, tuỳ chọn GridSearchCV."""
    if tune:
        print("[INFO] GridSearchCV đang chạy … (có thể mất vài phút)")
        base = RandomForestClassifier(random_state=random_state)
        gs = GridSearchCV(base, RF_PARAM_GRID, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"[OK]  Best params : {gs.best_params_}")
        print(f"[OK]  Best CV AUC : {gs.best_score_:.4f}")
        return gs.best_estimator_
    else:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=random_state
        )
        rf.fit(X_train, y_train)
        return rf


def train_baselines(X_train, y_train, random_state: int = 42) -> dict:
    """Huấn luyện Logistic Regression và Decision Tree để so sánh."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
        print(f"[OK]  Huấn luyện xong: {name}")
    return models


# ── Đánh giá ─────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Tính accuracy, F1, AUC và in báo cáo."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Diabetes','Diabetes'])}")

    return {
        "name": model_name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def get_roc_data(y_test, y_prob):
    """Trả về fpr, tpr để vẽ ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return fpr, tpr


# ── Lưu / tải mô hình ────────────────────────────────────────────────────────

def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"[OK]  Mô hình đã lưu tại: {path}")


def load_model(path: str):
    return joblib.load(path)
