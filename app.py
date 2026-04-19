"""
app.py
Streamlit web app — Dự đoán nguy cơ bệnh tiểu đường.
Chạy: streamlit run app.py
"""

import os
import sys

import joblib
import numpy as np
import streamlit as st

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "outputs", "random_forest_model.pkl")
SCALER_PATH = os.path.join(ROOT, "outputs", "scaler.pkl")

# ── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🩺 Dự đoán Tiểu Đường",
    page_icon="🩺",
    layout="centered",
)

# ── CSS đơn giản ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .result-box {
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 600;
        text-align: center;
        margin-top: 16px;
    }
    .high-risk   { background: #FFEBEE; color: #C62828; border: 2px solid #EF9A9A; }
    .low-risk    { background: #E8F5E9; color: #1B5E20; border: 2px solid #A5D6A7; }
    .prob-bar    { margin-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler

model, scaler = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Dự đoán Nguy cơ Bệnh Tiểu Đường")
st.markdown(
    "Nhập các chỉ số y tế của bệnh nhân bên dưới để nhận dự đoán sơ bộ "
    "từ mô hình **Random Forest** (độ chính xác ~80%)."
)
st.divider()

if model is None:
    st.error(
        "⚠️ Chưa tìm thấy mô hình. "
        "Hãy chạy `python scripts/train.py` trước để huấn luyện."
    )
    st.stop()

# ── Form nhập liệu ────────────────────────────────────────────────────────────
st.subheader("📋 Thông tin bệnh nhân")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Số lần mang thai", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Huyết áp (mmHg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Độ dày da (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (µU/mL)", min_value=0, max_value=1000, value=79)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01
    )
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=33)

st.divider()

# ── Dự đoán ───────────────────────────────────────────────────────────────────
if st.button("🔍 Dự đoán", use_container_width=True, type="primary"):
    features = np.array(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        dtype=float,
    )

    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.markdown(
            f'<div class="result-box high-risk">'
            f'⚠️ Nguy cơ CAO mắc bệnh tiểu đường<br>'
            f'Xác suất dự đoán: <span style="font-size:1.4rem">{probability*100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-box low-risk">'
            f'✅ Nguy cơ THẤP mắc bệnh tiểu đường<br>'
            f'Xác suất dự đoán: <span style="font-size:1.4rem">{probability*100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.progress(float(probability), text=f"Xác suất: {probability*100:.1f}%")

    # Feature importance nhanh
    st.divider()
    st.subheader("📊 Tầm quan trọng của các chỉ số (mô hình)")
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
    importances = model.feature_importances_
    imp_dict = dict(zip(feature_names, importances))
    imp_sorted = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

    import pandas as pd
    df_imp = pd.DataFrame(
        {"Feature": list(imp_sorted.keys()), "Importance": list(imp_sorted.values())}
    )
    st.bar_chart(df_imp.set_index("Feature"), use_container_width=True)

    st.caption(
        "⚠️ Lưu ý: Đây là công cụ hỗ trợ tham khảo, "
        "không thay thế chẩn đoán của bác sĩ chuyên khoa."
    )

# ── Sidebar thông tin ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Về dự án")
    st.markdown(
        """
        **Dataset**: PIMA Indians Diabetes  
        **Mô hình**: Random Forest Classifier  
        **Thư viện**: scikit-learn, Streamlit  

        ---
        **Các chỉ số đầu vào:**
        - Pregnancies: Số lần mang thai
        - Glucose: Nồng độ glucose huyết tương
        - BloodPressure: Huyết áp tâm trương
        - SkinThickness: Độ dày nếp gấp da tam đầu
        - Insulin: Insulin huyết thanh 2 giờ
        - BMI: Chỉ số khối cơ thể
        - DiabetesPedigreeFunction: Tiền sử gia đình
        - Age: Tuổi

        ---
        *Dự án học thuật — Random Forest AI*
        """
    )
