import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V15 (Fix Hiển Thị)", page_icon="🚨", layout="centered")
st.title("🚨 Dự Đoán & Backtest (V15)")

# --- 1. TẢI FILE ---
st.info("Bước 1: Tải các file Excel")
uploaded_files = st.file_uploader("Chọn file:", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)

# --- HÀM XỬ LÝ SỐ LIỆU ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}

def get_nums(s):
    if pd.isna(s): return []
    raw_nums = re.findall(r'\d+', str(s))
    # Chỉ lấy số có 1-2 chữ số
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

def get_col_score(col_name):
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

# --- HÀM XỬ LÝ NGÀY THÔNG MINH ---
def parse_date_magic(col_str, file_month, file_year):
    s = str(col_str).strip().upper()
    
    # Case 1: 30/11, 1/12 (Dạng thường)
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        y = file_year
        # Xử lý giao thừa (File T1 có cột 31/12)
        if m == 12 and file_month == 1: y -= 1
        elif m == 1 and file_month == 12: y += 1
        try: return datetime.date(y, m, d)
        except: pass

    # Case 2: 2025-01-12 (Lỗi đảo ngày tháng trong file của bạn)
    # File tháng 12 mà lại hiện 2025-01-12 -> Thực ra là ngày 01/12
    match_iso = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        
        # Logic sửa lỗi:
        # Nếu p1 (vị trí tháng) != file_month, mà p2 (vị trí ngày) == file_month
        # => ĐẢO NGƯỢC
        if p1 != file_month and p2 == file_month:
            try: return datetime.date(y, p2, p1) # p2 là Tháng, p1 là Ngày
            except: pass
            
        # Nếu thuận:
        if p1 == file_month:
            try: return datetime.date(y, p1, p2)
            except: pass
            
        # Nếu cả 2 không khớp, thử ưu tiên p1 là tháng
        try: return datetime.date(y, p1, p2)
        except: pass
        
    return None

def get_file_info(filename):
    y_match = re.search(r'20\d{2}', filename)
    y = int(y_match.group(0)) if y_match else 2025
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m = int(m_match.group(1)) if m_match else 1
    return m, y

@st.cache_data(ttl=600)
def load_data_v15(files):
    data_cache = {}
    kq_db = {}
    logs = []
    
    for file in files:
        f_m, f_y = get_file_info(file.name)
        logs.append(f"📂 Đọc file: {file.name} (Hiểu là T{f_m}/{f_y})")
        
        try:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                try:
                    # Tìm dòng Header (chứa TV TOP)
                    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    header_row = 3
                    for idx, row in preview.iterrows():
                        r_s = str(row.values).upper()
                        if "TV TOP" in r_s or "THÀNH VIÊN" in r_s:
                            header_row = idx; break
                    
                    df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
                    
                    # Map Cột Ngày
                    col_map = {}
                    found_dates = []
                    
                    for col in df.columns:
                        d_obj = parse_date_magic(col, f_m, f_y)
                        if d_obj:
                            col_map[col] = d_obj
                            found_dates.append(d_obj)
                            
                    # Tìm KQ
                    kq_row = None
