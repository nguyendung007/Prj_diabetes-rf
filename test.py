"""
test.py — Dự đoán từ file .pkl đã train sẵn
Chạy: python test.py
"""

import joblib
import numpy as np
import sys
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

print("🔄 Loading model...", file=sys.stderr, flush=True)
sys.stdout.flush()

try:
    # ── Load model & scaler ───────────────────────────────────────────────────────
    model  = joblib.load("outputs/random_forest_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    print("✓ Model loaded successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"❌ Error loading model: {e}", file=sys.stderr, flush=True)
    sys.exit(1)


# ── Dữ liệu 1 bệnh nhân mẫu ──────────────────────────────────────────────────
benh_nhan = {
    "Pregnancies"              : 3,
    "Glucose"                  : 158,   # cao → nguy cơ cao
    "BloodPressure"            : 76,
    "SkinThickness"            : 32,
    "Insulin"                  : 100,
    "BMI"                      : 18,
    "DiabetesPedigreeFunction" : 0.587,
    "Age"                      : 45,
}

# ── Predict ───────────────────────────────────────────────────────────────────
X = np.array([list(benh_nhan.values())])
X_scaled = scaler.transform(X)

ket_qua  = model.predict(X_scaled)[0]
xac_suat = model.predict_proba(X_scaled)[0][1]

# ── In kết quả ───────────────────────────────────────────────────────────────
print("\n📋 Thông tin bệnh nhân:")
sys.stdout.flush()
for k, v in benh_nhan.items():
    print(f"   {k:<30}: {v}")
    sys.stdout.flush()

print(f"\n🔍 Kết quả dự đoán : {'⚠️  CÓ nguy cơ tiểu đường' if ket_qua == 1 else '✅ KHÔNG có nguy cơ'}")
print(f"📊 Xác suất        : {xac_suat*100:.1f}%")
sys.stdout.flush()