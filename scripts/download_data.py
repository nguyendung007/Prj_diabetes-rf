"""
scripts/download_data.py
Tải PIMA Indians Diabetes Dataset từ URL công khai.
"""

import os
import urllib.request

DATA_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "diabetes.csv")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"[INFO] Dataset đã tồn tại tại: {OUTPUT_PATH}")
        return OUTPUT_PATH

    print("[INFO] Đang tải dữ liệu...")
#lệnh tải dữ liệu chỉ đơn giản vầy thôi,ML cơ bản thôi mà 
    urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
    print(f"[OK]  Đã lưu tại: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    download()
