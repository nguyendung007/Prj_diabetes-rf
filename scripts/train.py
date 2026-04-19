"""
scripts/train.py
Pipeline huấn luyện hoàn chỉnh:
  1. Tải / kiểm tra dữ liệu
  2. Tiền xử lý
  3. Huấn luyện Random Forest (có GridSearchCV) + baseline
  4. Đánh giá & trực quan hóa
  5. Lưu mô hình
"""

import os
import sys

import joblib
import numpy as np

# ── Thêm thư mục gốc vào path ────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.download_data import download
from src.model import (
    evaluate,
    get_roc_data,
    save_model,
    train_baselines,
    train_random_forest,
)
from src.preprocess import preprocess
from src.visualize import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_histograms,
    plot_boxplots,
)

import pandas as pd

DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")
MODEL_PATH = os.path.join(ROOT, "outputs", "random_forest_model.pkl")
SCALER_PATH = os.path.join(ROOT, "outputs", "scaler.pkl")
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)


def main(tune: bool = False):
    print("\n" + "=" * 60)
    print("  🩺 Dự án AI: Dự đoán Nguy cơ Bệnh Tiểu Đường")
    print("     Mô hình: Random Forest")
    print("=" * 60 + "\n")

    # 1. Tải dữ liệu
    download()
    df_raw = pd.read_csv(DATA_PATH)

    # 2. EDA plots
    print("\n[STEP 1] Vẽ biểu đồ EDA …")
    plot_class_distribution(df_raw["Outcome"], save=True)
    plot_histograms(df_raw, save=True)
    plot_correlation_heatmap(df_raw, save=True)
    plot_boxplots(df_raw, save=True)

    # 3. Tiền xử lý
    print("\n[STEP 2] Tiền xử lý dữ liệu …")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(DATA_PATH)

    # Lưu scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"[OK]  Scaler đã lưu tại: {SCALER_PATH}")

    # 4. Huấn luyện Random Forest
    print("\n[STEP 3] Huấn luyện Random Forest …")
    rf_model = train_random_forest(X_train, y_train, tune=tune)

    # 5. Huấn luyện baseline
    print("\n[STEP 4] Huấn luyện mô hình baseline …")
    baselines = train_baselines(X_train, y_train)

    # 6. Đánh giá
    print("\n[STEP 5] Đánh giá mô hình …")
    rf_result = evaluate(rf_model, X_test, y_test, "Random Forest")
    baseline_results = {
        name: evaluate(m, X_test, y_test, name)
        for name, m in baselines.items()
    }

    all_results = [rf_result] + list(baseline_results.values())

    # 7. Confusion matrix
    for res in all_results:
        plot_confusion_matrix(res["confusion_matrix"], res["name"], save=True)

    # 8. Feature importance
    plot_feature_importance(rf_model, feature_names, save=True)

    # 9. ROC curves
    roc_data = []
    for res in all_results:
        if res["y_prob"] is not None:
            fpr, tpr = get_roc_data(y_test, res["y_prob"])
            roc_data.append({"name": res["name"], "fpr": fpr, "tpr": tpr, "auc": res["auc"]})
    plot_roc_curves(roc_data, save=True)

    # 10. So sánh tổng hợp
    plot_model_comparison(all_results, save=True)

    # 11. Lưu mô hình tốt nhất
    save_model(rf_model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("  ✅ Hoàn thành! Kết quả đã lưu tại thư mục outputs/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Bật GridSearchCV (chậm hơn nhưng tốt hơn)")
    args = parser.parse_args()
    main(tune=args.tune)
