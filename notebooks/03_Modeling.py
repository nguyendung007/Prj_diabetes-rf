# %% [markdown]
# # 03 — Xây dựng & Đánh giá Mô hình Random Forest
# Notebook này thực hiện:  
# 1. Tiền xử lý dữ liệu  
# 2. Huấn luyện Random Forest  
# 3. So sánh với baseline  
# 4. Trực quan hóa kết quả

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

from src.preprocess import preprocess
from src.model import (
    train_random_forest, train_baselines,
    evaluate, get_roc_data, save_model
)
from src.visualize import (
    plot_confusion_matrix, plot_feature_importance,
    plot_roc_curves, plot_model_comparison
)

# %% [markdown]
# ## 1. Tiền xử lý

# %%
X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(
    "../data/diabetes.csv"
)
print("Feature names:", feature_names)

# %% [markdown]
# ## 2. Huấn luyện Random Forest
# Đặt `tune=True` để bật GridSearchCV (chậm hơn).

# %%
rf = train_random_forest(X_train, y_train, tune=False)

# %% [markdown]
# ## 3. Baseline models

# %%
baselines = train_baselines(X_train, y_train)

# %% [markdown]
# ## 4. Đánh giá

# %%
rf_result = evaluate(rf, X_test, y_test, "Random Forest")
bl_results = {n: evaluate(m, X_test, y_test, n) for n, m in baselines.items()}
all_results = [rf_result] + list(bl_results.values())

# %% [markdown]
# ## 5. Confusion Matrix

# %%
for res in all_results:
    plot_confusion_matrix(res["confusion_matrix"], res["name"], save=False)

# %% [markdown]
# ## 6. Feature Importance

# %%
plot_feature_importance(rf, feature_names, save=False)

# %% [markdown]
# ## 7. ROC Curves

# %%
roc_data = []
for res in all_results:
    if res["y_prob"] is not None:
        fpr, tpr = get_roc_data(y_test, res["y_prob"])
        roc_data.append({"name": res["name"], "fpr": fpr, "tpr": tpr, "auc": res["auc"]})
plot_roc_curves(roc_data, save=False)

# %% [markdown]
# ## 8. Tổng hợp kết quả

# %%
plot_model_comparison(all_results, save=False)

# %% [markdown]
# ## 9. Lưu mô hình

# %%
os.makedirs("../outputs", exist_ok=True)
save_model(rf, "../outputs/random_forest_model.pkl")
import joblib
joblib.dump(scaler, "../outputs/scaler.pkl")
print("Đã lưu mô hình và scaler!")
