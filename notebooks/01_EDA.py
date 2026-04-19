# %% [markdown]
# # 01 — Phân tích Dữ liệu Khám phá (EDA)
# **Dataset**: PIMA Indians Diabetes  
# **Mục tiêu**: Hiểu cấu trúc dữ liệu, phân phối và mối tương quan giữa các biến.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.download_data import download

# %% [markdown]
# ## 1. Tải dữ liệu

# %%
download()
df = pd.read_csv("../data/diabetes.csv")
print("Shape:", df.shape)
df.head()

# %% [markdown]
# ## 2. Thống kê mô tả

# %%
df.describe().T.style.background_gradient(cmap="Blues")

# %% [markdown]
# ## 3. Kiểm tra giá trị thiếu và 0 bất hợp lệ

# %%
zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
print("Số giá trị 0 trong mỗi cột:")
for col in zero_cols:
    print(f"  {col}: {(df[col]==0).sum()}")

# %% [markdown]
# ## 4. Phân phối nhãn

# %%
from src.visualize import plot_class_distribution
plot_class_distribution(df["Outcome"], save=False)

# %% [markdown]
# ## 5. Histogram

# %%
from src.visualize import plot_histograms
plot_histograms(df, save=False)

# %% [markdown]
# ## 6. Correlation Heatmap

# %%
from src.visualize import plot_correlation_heatmap
plot_correlation_heatmap(df, save=False)

# %% [markdown]
# ## 7. Boxplot theo Outcome

# %%
from src.visualize import plot_boxplots
plot_boxplots(df, save=False)

# %% [markdown]
# ## Nhận xét
# - **Glucose** và **BMI** có tương quan cao nhất với Outcome.
# - **Insulin** và **SkinThickness** có nhiều giá trị 0 không hợp lệ nhất.
# - Dataset mất cân bằng: ~65% No Diabetes vs ~35% Diabetes.
