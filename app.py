import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="App Xổ Số Lịch Vạn Niên", page_icon="📅", layout="centered")
st.title("📅 Dự Đoán Theo Lịch (Chính Xác 100%)")
st.write("---")

# --- 1. KHU VỰC TẢI FILE ---
st.info("Bước 1: Tải các file Excel (Code sẽ tự đọc Tháng/Năm trong tên file)")
uploaded_files = st.file_uploader("Chọn file (Ví dụ: File T12.2025 và T1.2026)", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
    st.caption("Ví dụ: Chọn 10 ngày thì khi dự đoán ngày 2/1, máy sẽ xem lại từ 23/12 đến 1/1.")

# --- CÁC HÀM XỬ LÝ ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}
RE_CLEAN = re.compile(r'[^A-Z0-9\/]')
RE_FIND_NUMS = re.compile(r'\d{1,2}') 

def clean_text(s):
    if pd.isna(s): return ""
    s_str = str(s).upper().replace('.', '/').replace('-', '/').replace('_', '/')
    return RE_CLEAN.sub('', s_str)

def get_nums(s):
    if pd.isna(s): return []
    raw_nums = RE_FIND_NUMS.findall(str(s))
    return [n.zfill(2) for n in raw_nums]

def get_col_score(col_name):
    clean = col_name 
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def get_header_row_index(df_raw):
    for i, row in df_raw.head(10).iterrows():
        row_str = clean_text("".join(row.values.astype(str)))
        if "THANHVIEN" in row_str and "STT" in row_str: return i
    return 3

# --- HÀM THÔNG MINH: ĐỌC NGÀY THÁNG TỪ TÊN FILE ---
def parse_month_year_from_filename(filename):
    # Tìm năm (4 chữ số, vd 2025, 2026)
    year_match = re.search(r'(20\d{2})', filename)
    year = int(year_match.group(1)) if year_match else None
    
    # Tìm tháng (Chữ THÁNG hoặc T theo sau là số)
    # Ví dụ: THANG 12, THÁNG 1, T12, T01
    name_clean = clean_text(filename)
    month_match = re.search(r'(?:THANG|T)(\d+)', name_clean)
    month = int(month
