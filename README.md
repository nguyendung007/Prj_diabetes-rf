# 🩺 Dự đoán Nguy cơ Bệnh Tiểu Đường — Random Forest AI

Dự án Machine Learning sử dụng **Random Forest** để dự đoán nguy cơ mắc bệnh tiểu đường dựa trên dữ liệu y tế PIMA Indians Diabetes Dataset.

---

## 📁 Cấu trúc dự án (Có thể không đúng hoàn toàn)

```
diabetes-rf-project/
├── data/
│   └── diabetes.csv            ← Tải tự động khi chạy
├── notebooks/
│   ├── 01_EDA.py               ← Phân tích dữ liệu (chạy với Jupyter / VS Code)
│   └── 03_Modeling.py          ← Huấn luyện & đánh giá mô hình
├── outputs/                    ← Biểu đồ (.png) & mô hình (.pkl) được lưu tại đây
├── scripts/
│   ├── download_data.py        ← Tải dataset
│   └── train.py                ← Pipeline huấn luyện hoàn chỉnh
├── src/
│   ├── preprocess.py           ← Tiền xử lý dữ liệu
│   ├── model.py                ← Huấn luyện & đánh giá mô hình
│   └── visualize.py            ← Tất cả hàm vẽ biểu đồ
├── app.py                      ← Streamlit web app
└── requirements.txt
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình

```bash
# Huấn luyện nhanh (không GridSearch)
python scripts/train.py

# Huấn luyện với GridSearchCV (tối ưu hơn, chậm hơn ~5 phút)
python scripts/train.py --tune
```

Kết quả sẽ được lưu vào thư mục `outputs/`:
- `*.png` — Biểu đồ EDA và kết quả mô hình
- `random_forest_model.pkl` — Mô hình đã huấn luyện
- `scaler.pkl` — StandardScaler đã fit

### 3. Chạy Streamlit App

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

### 4. Chạy Notebooks (Jupyter / VS Code)

```bash
jupyter notebook notebooks/01_EDA.py
```

Hoặc mở bằng **VS Code** với extension Jupyter.

---

## 📊 Dataset

| Thông tin | Chi tiết |
|---|---|
| Nguồn | PIMA Indians Diabetes Database |
| Số mẫu | 768 bệnh nhân |
| Số features | 8 |
| Target | Outcome (0 = Không, 1 = Có tiểu đường) |

**Các features:**
- `Pregnancies` — Số lần mang thai
- `Glucose` — Nồng độ glucose huyết tương (mg/dL)
- `BloodPressure` — Huyết áp tâm trương (mmHg)
- `SkinThickness` — Độ dày nếp gấp da (mm)
- `Insulin` — Insulin huyết thanh 2 giờ (µU/mL)
- `BMI` — Chỉ số khối cơ thể (kg/m²)
- `DiabetesPedigreeFunction` — Điểm tiền sử gia đình
- `Age` — Tuổi

---

## 🎯 Kết quả kỳ vọng

| Mô hình | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Random Forest** | ~80% | ~0.73 | ~0.85 |
| Logistic Regression | ~77% | ~0.68 | ~0.83 |
| Decision Tree | ~73% | ~0.67 | ~0.73 |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — Random Forest, GridSearchCV, metrics
- **pandas / numpy** — Xử lý dữ liệu
- **matplotlib / seaborn** — Trực quan hóa
- **Streamlit** — Web app triển khai
- **joblib** — Lưu/tải mô hình

---

## ⚠️ Lưu ý

Đây là dự án học thuật. Kết quả **không thay thế** chẩn đoán của bác sĩ chuyên khoa.

