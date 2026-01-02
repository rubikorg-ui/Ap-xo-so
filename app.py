import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V12 Final", page_icon="💎", layout="centered")
st.title("💎 Dự Đoán (V12 - Fix Lỗi Cột Ngày)")

# --- 1. TẢI FILE ---
st.info("Bước 1: Tải file Excel (T12.2025, T1.2026...)")
uploaded_files = st.file_uploader("Chọn file:", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)

# --- HÀM XỬ LÝ ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}
RE_CLEAN = re.compile(r'[^A-Z0-9\/]')
RE_FIND_NUMS = re.compile(r'\d{1,2}') 

def clean_text(s):
    if pd.isna(s): return ""
    s_str = str(s).upper().replace('.', '/').replace('-', '/').replace('_', '/')
    # Giữ lại các ký tự số, chữ và dấu / để so sánh ngày
    return s_str

def get_nums(s):
    if pd.isna(s): return []
    raw_nums = re.findall(r'\d+', str(s)) # Lấy mọi con số
    # Lọc số có 1-2 chữ số (để tránh lấy nhầm năm 2026)
    valid_nums = [n.zfill(2) for n in raw_nums if len(n) <= 2]
    return valid_nums

def get_col_score(col_name):
    # Làm sạch tên cột để check M1...M10
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def get_header_row_index(df_raw):
    for i, row in df_raw.head(10).iterrows():
        row_str = str(row.values).upper()
        if "THÀNH VIÊN" in row_str or "THANH VIEN" in row_str: return i
    return 3

# --- [FIX] HÀM ĐỌC NGÀY TỪ SHEET ---
def parse_date_from_sheet(sheet_name, filename):
    # 1. Lấy Năm/Tháng từ Tên File
    year_match = re.search(r'(20\d{2})', filename)
    year = int(year_match.group(1)) if year_match else None
    
    month_match = re.search(r'(?:THANG|THÁNG|TH|T|M)[^0-9]*(\d+)', filename, re.IGNORECASE)
    if not month_match:
        # Fallback: tìm cụm d-yyyy hoặc d.yyyy
        alt_match = re.search(r'(\d+)[\.\-_]+' + str(year), filename) if year else None
        month = int(alt_match.group(1)) if alt_match else None
    else:
        month = int(month_match.group(1))

    # 2. Lấy Ngày từ Tên Sheet
    # Regex lấy số đầu tiên trong tên sheet (VD: "1.12" -> lấy 1, "2" -> lấy 2)
    day_match = re.search(r'^(\d+)', sheet_name.strip())
    day = int(day_match.group(1)) if day_match else None
    
    if day and month and year:
        try: return datetime.date(year, month, day)
        except: return None
    return None

@st.cache_data(ttl=600)
def load_data_v12(files):
    data_cache = {}
    kq_db = {} 
    debug_logs = []
    
    for file in files:
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                try:
                    current_date = parse_date_from_sheet(sheet_name, file.name)
                    if not current_date: continue 

                    # Đọc file
                    temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=15)
                    h = get_header_row_index(temp)
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h)
                    
                    # Quan trọng: Chuyển tên cột về String hết để dễ tìm
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    data_cache[current_date] = df
                    
                    # TÌM KẾT QUẢ (KQ)
                    # Tìm dòng chứa chữ "KQ"
                    kq_row_idx = None
                    for idx, row in df.iterrows():
                        row_s = str(row.values).upper()
                        if "KQ" in row_s and ("9X" not in row_s): # Tránh nhầm tiêu đề
                             kq_row_idx = idx; break
                    
                    if kq_row_idx is not None:
                        kq_row = df.iloc[kq_row_idx]
                        
                        # [FIX MẠNH] Tạo mọi định dạng ngày có thể để tìm cột
                        d, m, y = current_date.day, current_date.month, current_date.year
                        possible_cols = [
                            f"{d}/{m}", f"{d:02d}/{m}", f"{d}/{m:02d}", f"{d:02d}/{m:02d}", # 1/1, 01/01
                            str(d), # 1
                            f"{y}-{m:02d}-{d:02d}", # 2026-01-01 (Định dạng Excel hay dùng)
                            f"{y}-{m}-{d}", # 2026-1-1
                            f"{d}-{m}-{y}", # 01-01-2026
                        ]
                        
                        found_val = None
                        found_col_name = ""
                        
                        # Duyệt qua các cột trong file
                        for col in df.columns:
                            # So sánh: cột trong file có CHỨA một trong các pattern không?
                            col_upper = str(col).upper()
                            for p in possible_cols:
                                if p in col_upper:
                                    # Kiểm tra kỹ hơn: Nếu tìm số 1, tránh nhầm số 10, 11...
                                    # Nhưng với định dạng ngày tháng (có dấu / hoặc -) thì khá an toàn
                                    try:
                                        val = str(kq_row[col])
                                        nums = get_nums
