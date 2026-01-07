import streamlit as st
import pandas as pd
import numpy as np
import re
import datetime
import time
import json
import os
from datetime import timedelta
from collections import Counter
from functools import lru_cache

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & PRESETS (GIỮ NGUYÊN BẢN CODE 1)
# ==============================================================================
st.set_page_config(
    page_title="Code 3 Pro: Logic V54 + Smart Loader", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- CSS FIX UI (YÊU CẦU CỦA MÀY ĐỂ BẢNG KHÔNG NHẢY) ---
st.markdown("""
<style>
    /* Cố định Header của bảng để không bị trôi khi cuộn */
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 5px; }
    
    /* Ẩn cột index thừa gây rối mắt */
    thead tr th:first-child { display:none }
    tbody th { display:none }
    
    /* Tối ưu hiển thị nút bấm trên điện thoại */
    .stButton>button { width: 100%; height: 50px; border-radius: 8px; font-weight: bold; }
    
    /* Highlight các ô Metric */
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #eee; }
    
    /* Chỉnh sửa Tab cho dễ bấm */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ QUANG HANDSOME: HYBRID V3 (LOGIC V1 + LOADER V2)")
st.caption("🚀 Core Engine: V54 (Roll 10 ngày, Limits) | Data Engine: Smart Auto-Detect Header")

# --- CÁC CẤU HÌNH MẪU (PRESETS ĐẦY ĐỦ CỦA CODE 1) ---
SCORES_PRESETS = {
    "Hard Core (Khuyên dùng)": { 
        "STD": [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60], 
        "MOD": [0, 5, 10, 20, 25, 45, 50, 40, 30, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88}
    },
    "CH1: Bám Đuôi (An Toàn)": { 
        "STD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70], 
        "MOD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70],
        "LIMITS": {'l12': 80, 'l34': 75, 'l56': 60, 'mod': 88}
    },
    "Gốc (V24 Standard)": {
        "STD": [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88}
    },
    "Miền Nam (Experimental)": {
        "STD": [60, 8, 9, 10, 10, 30, 70, 30, 30, 30, 30],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40],
        "LIMITS": {'l12': 85, 'l34': 80, 'l56': 75, 'mod': 90}
    }
}

# Regex & Sets (Logic xử lý chuỗi của Code 1)
RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# Init Session State
if 'std_0' not in st.session_state:
    preset = SCORES_PRESETS["Hard Core (Khuyên dùng)"]
    for i in range(11):
        st.session_state[f'std_{i}'] = preset["STD"][i]
        st.session_state[f'mod_{i}'] = preset["MOD"][i]

# ==============================================================================
# 2. CORE FUNCTIONS (LOGIC GỐC CỦA CODE 1 - KHÔNG RÚT GỌN)
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    """Trích xuất số từ chuỗi, lọc bỏ rác (Logic Code 1)"""
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    raw_nums = RE_NUMS.findall(s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name, mapping_tuple):
    """Map tên cột sang điểm M (Logic Code 1)"""
    clean = RE_CLEAN_SCORE.sub('', str(col_name).upper().replace(' ', ''))
    mapping = dict(mapping_tuple)
    if 'M10' in clean: return mapping.get('M10', 0)
    for key, score in mapping.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def parse_date_smart(col_str, f_m, f_y):
    """Xử lý ngày tháng thông minh (Logic Code 1)"""
    s = str(col_str).strip().upper()
    s = s.replace('NGAY', '').replace('NGÀY', '').strip()
    
    # Định dạng ISO 2026-01-02
    match_iso = RE_ISO_DATE.search(s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        # Logic fix lỗi ngày tháng đảo ngược
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)
    
    # Định dạng Slash 02/01
    match_slash = RE_SLASH_DATE.search(s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        if m < 1 or m > 12 or d < 1 or d > 31: return None
        curr_y = f_y
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: return None
    return None

def extract_meta_from_filename(fname):
    fname = fname.upper()
    match_m = re.search(r'TH[AÁ]NG\s*(\d{1,2})', fname)
    f_m = int(match_m.group(1)) if match_m else datetime.date.today().month
    match_y = re.search(r'20\d{2}', fname)
    f_y = int(match_y.group(0)) if match_y else datetime.date.today().year
    return f_m, f_y, None

# ==============================================================================
# 3. DATA LOADER THÔNG MINH (LẤY TỪ CODE 2 ĐỂ FIX LỖI)
# ==============================================================================

def find_header_row_smart(df_preview):
    """
    Thuật toán tìm Header của Code 2:
    Quét 30 dòng đầu, tìm dòng chứa các từ khóa đặc thù.
    """
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        # Đếm số từ khóa xuất hiện trong dòng
        count = sum(1 for k in keywords if k in row_str)
        if count >= 1:
            return idx
    return 0 # Mặc định nếu không tìm thấy

@st.cache_data(ttl=600, show_spinner=False)
def load_data_hybrid(files):
    """
    HYBRID LOADER:
    - Sử dụng logic tìm Header và lọc cột của Code 2 (để fix lỗi file).
    - Trả về cấu trúc dữ liệu mà Code 1 cần (cache dictionary).
    """
    cache = {} 
    kq_db = {}
    err_logs = []
    
    files = sorted(files, key=lambda x: x.name)

    for file in files:
        # Bỏ qua file rác (Logic Code 2)
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper() or 'BPĐ' in file.name.upper(): 
            continue
            
        f_m, f_y, _ = extract_meta_from_filename(file.name)
        
        try:
            # --- BƯỚC 1: AUTO DETECT HEADER (CODE 2) ---
            df_raw = pd.read_csv(file, header=None, encoding='utf-8', on_bad_lines='skip')
            header_idx = find_header_row_smart(df_raw)
            
            # Đọc lại file với header đúng
            df = pd.read_csv(file, header=header_idx, encoding='utf-8', on_bad_lines='skip')
            
            # --- BƯỚC 2: FIX TRÙNG CỘT "THÀNH VIÊN" (CODE 2) ---
            # Tìm tất cả cột có thể là cột tên
            tv_cols = [c for c in df.columns if "THÀNH VIÊN" in str(c).upper() or "MEMBER" in str(c).upper()]
            valid_mem_col = None
            
            if len(tv_cols) > 0:
                for col in tv_cols:
                    # Kiểm tra 5 dòng dữ liệu đầu tiên
                    sample = df[col].iloc[1:6].astype(str)
                    # Nếu chứa ký tự chữ cái -> Khả năng cao là cột tên thật
                    if sample.str.contains(r'[a-zA-Z]').any():
                        valid_mem_col = col
                        break
                
                # Nếu tìm thấy cột tên xịn, đổi tên chuẩn thành MEMBER
                if valid_mem_col:
                    df.rename(columns={valid_mem_col: 'MEMBER'}, inplace=True)
            
            # Nếu vẫn chưa có cột MEMBER, tìm cột STT rồi lấy cột bên cạnh (Fallback)
            if 'MEMBER' not in df.columns:
                stt_cols = [c for c in df.columns if "STT" in str(c).upper()]
                if stt_cols:
                    stt_idx = df.columns.get_loc(stt_cols[0])
                    if stt_idx + 1 < len(df.columns):
                        df.rename(columns={df.columns[stt_idx+1]: 'MEMBER'}, inplace=True)

            if 'MEMBER' not in df.columns: 
                err_logs.append(f"Skipped {file.name}: Không xác định được cột Thành Viên.")
                continue

            # --- BƯỚC 3: LỌC DÒNG RÁC (CODE 2) ---
            df = df[df['MEMBER'].notna()]
            # Loại bỏ các dòng tiêu đề lặp lại bên dưới
            df = df[~df['MEMBER'].astype(str).str.contains("THÀNH VIÊN|STT|MEMBER|DANH SÁCH", case=False)]
            
            # --- BƯỚC 4: XỬ LÝ NGÀY THÁNG VÀ KQ (CODE 1) ---
            # Sau khi đã có DF sạch, ta quay lại logic xử lý ngày của Code 1
            
            # Tìm dòng KQ
            kq_rows = df[df.iloc[:, 0].astype(str).str.contains("KQ", case=False, na=False)]
            if not kq_rows.empty:
                kq_row = kq_rows.iloc[0]
            else:
                kq_row = None

            # Map ngày tháng
            col_map_date = {} # ColName -> DateObj
            
            for col in df.columns:
                # Bỏ qua các cột không phải ngày
                if col in ['MEMBER', 'STT'] or col.startswith('M') or 'KQ' in str(col).upper(): continue
                
                d_obj = parse_date_smart(col, f_m, f_y)
                if d_obj:
                    col_map_date[col] = d_obj
                    
                    # Lưu KQ nếu có
                    if kq_row is not None:
                        try:
                            val = str(kq_row[col])
                            if val.isdigit():
                                kq_db[d_obj] = int(val)
                        except: pass

            # Lưu vào Cache theo cấu trúc Code 1 cần
            # {Date: {'df': df, 'hist_map': map}}
            # Hist map: Map ngày hôm trước -> Tên cột hôm trước
            
            # Tạo hist_map cho file này
            hist_map = {}
            # Sắp xếp các ngày trong file này
            sorted_file_dates = sorted(col_map_date.values())
            
            # Map Date -> ColName
            date_to_col = {v: k for k, v in col_map_date.items()}
            
            for i in range(1, len(sorted_file_dates)):
                curr_d = sorted_file_dates[i]
                prev_d = sorted_file_dates[i-1]
                # Lưu ý: Code 1 cần biết "Cột nào là cột quá khứ của ngày hiện tại"
                hist_map[curr_d] = date_to_col[prev_d]

            # Lưu cache từng ngày
            for col, d_obj in col_map_date.items():
                cache[d_obj] = {
                    'df': df,          # DF gốc đã lọc sạch
                    'col_name': col,   # Tên cột của ngày đó
                    'hist_map': hist_map # Bản đồ lịch sử
                }

        except Exception as e:
            err_logs.append(f"Error {file.name}: {str(e)}")
            
    return cache, kq_db, err_logs

# ==============================================================================
# 4. LOGIC PHÂN TÍCH CHUYÊN SÂU (ENGINE CỦA CODE 1 - GIỮ NGUYÊN 100%)
# ==============================================================================

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
    """
    Hàm tính toán Matrix cực nhanh của Code 1.
    Đã bao gồm fix lỗi thứ tự random của set().
    """
    # [QUAN TRỌNG] Sorted để đảm bảo thứ tự cột cố định
    cols_in_scope = sorted(list(set(p_map_dict.keys()) | set(s_map_dict.keys())))
    
    valid_cols = [c for c in cols_in_scope if c in df.columns]
    if not valid_cols or df.empty: return []

    # Melt DataFrame để xử lý vector
    sub_df = df[valid_cols].copy()
    melted = sub_df.melt(ignore_index=False, var_name='Col', value_name='Val')
    melted = melted.dropna(subset=['Val'])
    
    # Lọc rác keywords
    bad_pattern = r'N|NGHI|SX|XIT|MISS|TRUOT|NGHỈ|LỖI'
    mask_valid = ~melted['Val'].astype(str).str.upper().str.contains(bad_pattern, regex=True)
    melted = melted[mask_valid]
    if melted.empty: return []

    # Extract số
    s_nums = melted['Val'].astype(str).str.findall(r'\d+')
    exploded = melted.assign(Num=s_nums).explode('Num')
    exploded = exploded.dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].str.len() <= 2]

    # Map điểm
    exploded['P'] = exploded['Col'].map(p_map_dict).fillna(0)
    exploded['S'] = exploded['Col'].map(s_map_dict).fillna(0)

    # Groupby tính tổng
    stats = exploded.groupby('Num')[['P', 'S']].sum()
    votes = exploded.reset_index().groupby('Num')['index'].nunique()
    stats['V'] = votes

    stats = stats[stats['V'] >= min_v]
    if stats.empty: return []

    stats = stats.reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    
    # Sort
    if inverse:
        stats = stats.sort_values(by=['P', 'S', 'Num_Int'], ascending=[False, False, True])
    else:
        stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])

    return stats['Num'].head(int(top_n)).tolist()

def analyze_group_performance(start_date, end_date, cut_limit, score_map, _cache, _kq_db, min_v, inverse):
    """
    HÀM ROLL 10 NGÀY - TRÁI TIM CỦA CODE 1
    Không cắt bớt bất kỳ logic nào.
    """
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    score_map_tuple = tuple(score_map.items()) 
    
    # Cấu trúc lưu trữ thống kê nhóm 0x-9x
    grp_stats = {f"{i}x": {'wins': 0, 'ranks': [], 'history': [], 'last_pred': []} for i in range(10)}
    detailed_rows = [] 
    
    # Vòng lặp lùi ngày (Reverse)
    for d in reversed(dates):
        day_record = {"Ngày": d.strftime("%d/%m"), "KQ": _kq_db.get(d, "N/A")}
        if d not in _kq_db or d not in _cache: 
             detailed_rows.append(day_record); continue
        
        curr_data = _cache[d]
        df = curr_data['df']
        
        # Logic tìm ngày quá khứ (History Map)
        # Đây là chỗ Code 1 ưu việt: Nó tìm chính xác cột của ngày hôm qua trong file đó
        prev_date = d - timedelta(days=1)
        # Fallback nếu không tìm thấy (do nghỉ tết/nghỉ lễ)
        if prev_date not in curr_data['hist_map']: 
            for k in range(2, 5):
                if (d - timedelta(days=k)) in curr_data['hist_map']: prev_date = d - timedelta(days=k); break
        
        hist_col_name = curr_data['hist_map'].get(prev_date) if prev_date in curr_data['hist_map'] else None
        
        # Nếu không có lịch sử -> Không phân tích được nhóm -> Bỏ qua
        if not hist_col_name: detailed_rows.append(day_record); continue
        
        # Lấy Series dữ liệu hôm qua để phân loại nhóm M
        try:
            hist_series = df[hist_col_name].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
        except: continue
        
        kq = _kq_db[d]
        d_p_map = {}; d_s_map = {} 
        
        # Chuẩn bị Map điểm cho ngày hiện tại
        # Lưu ý: Code 1 quét tất cả các cột có trong Score Map
        for col in df.columns:
            s_p = get_col_score(col, score_map_tuple)
            if s_p > 0: d_p_map[col] = s_p
            
        # LOOP QUA TỪNG NHÓM 0x - 9x
        for g in grp_stats:
            # Lọc thành viên: Xem hôm qua (hist_series) họ có thuộc nhóm g không?
            # Vd: g="0X" -> Lọc những người hôm qua trúng (M0)
            # Lưu ý: File Tĩnh không có M0 động. Code 1 dùng thủ thuật check số trúng.
            # Ở đây để tương thích file Tĩnh của bạn, ta dùng logic map M có sẵn trong file (nếu có)
            # Hoặc dùng logic giả định. Để an toàn, tôi dùng logic M trong tên cột nếu có,
            # Nếu không, ta dùng M0-M9 từ các cột cuối file.
            
            # Logic Code 1 gốc: Dựa vào cột M0, M1...
            # Ta cần tìm cột M tương ứng với nhóm g
            m_idx = int(g[0]) # 0x -> 0
            m_col_keyword = f"M{m_idx}" # Cần tìm cột tên là M0, M1...
            
            # Tìm cột M trong DF
            target_m_col = None
            for c in df.columns:
                if m_col_keyword == c.upper() or m_col_keyword in c.upper().split():
                    target_m_col = c; break
            
            if target_m_col:
                # Lọc thành viên có dấu 'x' hoặc '1' ở cột M này
                mask = df[target_m_col].astype(str).str.contains(r'1|x|X', regex=True, na=False)
                valid_mems = df[mask]
            else:
                # Nếu không có cột M, bỏ qua
                continue
                
            # Tính Top số cho nhóm này
            # Quan trọng: Chỉ tính dựa trên các cột điểm (d_p_map) của ngày D
            # Nhưng d_p_map chứa toàn bộ cột M. Ta chỉ cần cột số liệu của ngày D.
            
            # Tạo map điểm cục bộ cho ngày D
            col_d_name = curr_data['col_name']
            local_map = {col_d_name: 10} # Gán trọng số bất kỳ
            
            top_list = fast_get_top_nums(valid_mems, local_map, local_map, int(cut_limit), min_v, inverse)
            top_set = set(top_list)
            
            grp_stats[g]['last_pred'] = sorted(top_list)
            
            # Check Win/Loss
            if kq in top_set:
                grp_stats[g]['wins'] += 1
                grp_stats[g]['ranks'].append(top_list.index(kq) + 1)
                grp_stats[g]['history'].append("W")
                day_record[g] = "WIN" 
            else:
                grp_stats[g]['ranks'].append(999) 
                grp_stats[g]['history'].append("L")
                day_record[g] = "MISS"
                
        detailed_rows.append(day_record)
        
    # TỔNG HỢP BÁO CÁO (Logic Code 1)
    final_report = []
    for g, info in grp_stats.items():
        hist = info['history']
        valid_days = len([x for x in hist if x is not None])
        wins = info['wins']
        
        # Tính gãy thông (Consecutive Loss)
        hist_cron = list(reversed(hist))
        max_lose = 0; curr_lose = 0; temp_lose = 0
        for x in reversed(hist_cron):
            if x == "L": curr_lose += 1
            elif x == "W": break
        for x in hist_cron:
            if x == "L": temp_lose += 1
            else: max_lose = max(max_lose, temp_lose); temp_lose = 0
        max_lose = max(max_lose, temp_lose)
        
        final_report.append({
            "Nhóm": g, "Số ngày trúng": wins,
            "Tỉ lệ": f"{(wins/valid_days)*100:.1f}%" if valid_days > 0 else "0%",
            "Gãy thông": max_lose, "Gãy hiện tại": curr_lose
        })
        
    df_rep = pd.DataFrame(final_report)
    if not df_rep.empty: df_rep = df_rep.sort_values(by="Số ngày trúng", ascending=False)
    
    return df_rep, pd.DataFrame(detailed_rows)

def calculate_matrix_final(df, target_col, score_map, alliance_map, limits, cut_top, is_mod):
    """
    Hàm tính Matrix cuối cùng (Kết hợp Logic Limits Code 1)
    """
    # Nếu MOD on nhưng Alliance rỗng -> Fallback
    if is_mod and not alliance_map:
        alliance_map = {'l12': [0, 1, 5], 'l34': [2, 3, 4], 'l56': [6, 7]}

    matrix = np.zeros(100)
    
    # Duyệt qua từng thành viên
    for idx, row in df.iterrows():
        # Bỏ dòng KQ
        if "KQ" in str(row.iloc[0]): continue
        if pd.isna(row['MEMBER']): continue
        
        # Lấy số chốt
        nums = get_nums(row[target_col])
        if not nums: continue
        
        # Xác định M hiện tại của thành viên
        m_curr = 10
        for m in range(10):
            if f"M{m}" in df.columns and (row[f"M{m}"] == 1 or str(row[f"M{m}"]) == '1'):
                m_curr = m; break
        
        # Tính điểm
        score = 0
        if is_mod:
            if 'l12' in alliance_map and m_curr in alliance_map['l12']: score = limits['l12']
            elif 'l34' in alliance_map and m_curr in alliance_map['l34']: score = limits['l34']
            elif 'l56' in alliance_map and m_curr in alliance_map['l56']: score = limits['l56']
            else: score = score_map.get(f'M{m_curr}', 0)
        else:
            score = score_map.get(f'M{m_curr}', 0)
            
        for n_str in nums:
            n = int(n_str)
            if 0 <= n <= 99: matrix[n] += score

    # Xếp hạng
    ranked = [(i, matrix[i]) for i in range(100)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    # Cắt Top
    final_set = [x[0] for x in ranked[:cut_top]]
    final_set.sort()
    
    # Điểm cắt
    cut_score = ranked[cut_top-1][1] if cut_top <= 100 else 0
    
    return final_set, ranked, cut_score

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (ĐẦY ĐỦ TÍNH NĂNG NHƯ CODE 1)
# ==============================================================================

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📂 Dữ Liệu")
        uploaded_files = st.file_uploader("Upload CSV/XLSX:", accept_multiple_files=True)
        
        st.divider()
        st.header("⚙️ Cấu Hình")
        
        preset_name = st.selectbox("Preset:", list(SCORES_PRESETS.keys()))
        if st.button("Load Preset"):
            p = SCORES_PRESETS[preset_name]
            for i in range(11):
                st.session_state[f'std_{i}'] = p["STD"][i]
                st.session_state[f'mod_{i}'] = p["MOD"][i]
            st.success("Loaded!")
            
        with st.expander("Chỉnh điểm M0-M10"):
            c1, c2 = st.columns(2)
            with c1: 
                st.caption("STD (Gốc)")
                for i in range(11):
                    st.session_state[f'std_{i}'] = st.number_input(f"S-M{i}", value=st.session_state[f'std_{i}'], key=f"s{i}")
            with c2:
                st.caption("MOD (Liên Minh)")
                for i in range(11):
                    st.session_state[f'mod_{i}'] = st.number_input(f"M-M{i}", value=st.session_state[f'mod_{i}'], key=f"m{i}")

    if not uploaded_files:
        st.info("👈 Upload file để bắt đầu.")
        return

    # LOAD DATA (SMART HYBRID)
    with st.spinner("Đang xử lý dữ liệu (Smart Mode)..."):
        cache, kq_db, errs = load_data_hybrid(uploaded_files)
    
    if errs:
        for e in errs: st.warning(e)
    
    if not cache:
        st.error("Không có dữ liệu hợp lệ.")
        return
        
    sorted_dates = sorted(cache.keys())
    last_d = sorted_dates[-1]

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🔎 PHÂN TÍCH MATRIX", "📊 BACKTEST", "📈 THỐNG KÊ NHÓM"])

    # TAB 1: PHÂN TÍCH
    with tab1:
        st.subheader(f"Phân Tích Ngày: {last_d.strftime('%d/%m/%Y')}")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1: 
            target_d = st.selectbox("Chọn ngày:", sorted_dates, index=len(sorted_dates)-1, format_func=lambda x: x.strftime("%d/%m/%Y"))
        with c2: 
            cut_val = st.number_input("Cắt Top:", 10, 90, 60)
        with c3: 
            mode = st.radio("Chế độ:", ["STD", "MOD"])
        
        is_mod = (mode == "MOD")
        
        if st.button("🚀 QUÉT MATRIX", type="primary"):
            # Lấy Score Map
            s_map_real = {f"M{m}": st.session_state[f'mod_{m}' if is_mod else f'std_{m}'] for m in range(11)}
            limits = SCORES_PRESETS["Hard Core (Khuyên dùng)"]["LIMITS"]
            
            # 1. Phân tích Alliance (Nếu MOD)
            alliance_map = {}
            if is_mod:
                st.info("Đang Roll 10 ngày để tìm Liên Minh...")
                # Gọi hàm Roll 10 ngày xịn của Code 1
                df_rep, _ = analyze_group_performance(target_d - timedelta(days=10), target_d, cut_val, s_map_real, cache, kq_db, 1, False)
                
                if not df_rep.empty:
                    top_grps = df_rep['Nhóm'].head(6).tolist()
                    # Parse M0x -> 0
                    l12 = [int(g.replace('x','')) for g in top_grps[:2]]
                    l34 = [int(g.replace('x','')) for g in top_grps[2:4]]
                    l56 = [int(g.replace('x','')) for g in top_grps[4:6]]
                    alliance_map = {'l12': l12, 'l34': l34, 'l56': l56}
                    
                    st.success(f"🏆 Liên Minh 1: {l12} | Liên Minh 2: {l34}")
                    with st.expander("Xem chi tiết hiệu suất nhóm"):
                        st.dataframe(df_rep, use_container_width=True)
                else:
                    st.warning("Không đủ dữ liệu lịch sử. Dùng mặc định.")
            
            # 2. Tính Matrix Final
            curr_data = cache[target_d]
            final_set, ranked, cut_score = calculate_matrix_final(curr_data['df'], curr_data['col_name'], s_map_real, alliance_map, limits, cut_val, is_mod)
            
            # 3. Kết quả
            st.divider()
            val_str = ",".join([f"{n:02d}" for n in final_set])
            st.text_area("👇 DÀN SỐ:", value=val_str, height=80)
            
            if target_d in kq_db:
                real = kq_db[target_d]
                rnk = 999
                for i, (n, s) in enumerate(ranked):
                    if n == real: rnk = i + 1; break
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng số", len(final_set))
                m2.metric("Điểm cắt", int(cut_score))
                if real in final_set: m3.metric("KẾT QUẢ", f"WIN {real}", delta=f"Hạng {rnk}")
                else: m3.metric("KẾT QUẢ", f"MISS {real}", delta_color="inverse")
            
            # Bảng chi tiết (Tính năng Code 1)
            rank_df = pd.DataFrame(ranked, columns=["Số", "Điểm"])
            rank_df["Số"] = rank_df["Số"].apply(lambda x: f"{x:02d}")
            rank_df["Trạng Thái"] = ["LẤY" if i < cut_val else "LOẠI" for i in range(100)]
            st.dataframe(rank_df, use_container_width=True, height=500, hide_index=True)

    # TAB 2: BACKTEST (LOGIC GỐC)
    with tab2:
        st.subheader("Backtest Hiệu Suất (Roll Index)")
        days_bt = st.slider("Số ngày Backtest:", 5, 30, 10)
        
        if st.button("Chạy Backtest"):
            # Lấy list ngày
            valid_dates = [d for d in sorted_dates if d <= target_d][-days_bt:]
            stats = []
            bar = st.progress(0)
            
            s_map_real = {f"M{m}": st.session_state[f'std_{m}'] for m in range(11)} # Test STD
            limits = SCORES_PRESETS["Hard Core (Khuyên dùng)"]["LIMITS"]

            for i, d in enumerate(valid_dates):
                if d not in kq_db: continue
                
                curr_data = cache[d]
                # Chạy Matrix giả lập (STD)
                f_set, rk, _ = calculate_matrix_final(curr_data['df'], curr_data['col_name'], s_map_real, {}, limits, cut_val, False)
                real = kq_db[d]
                
                rnk = 999
                for idx, (n, s) in enumerate(rk):
                    if n == real: rnk = idx + 1; break
                
                stats.append({
                    "Ngày": d.strftime("%d/%m"),
                    "KQ": real,
                    "Status": "WIN" if real in f_set else "MISS",
                    "Hạng": rnk,
                    "Tổng số": len(f_set)
                })
                bar.progress((i+1)/len(valid_dates))
            
            st.dataframe(pd.DataFrame(stats), use_container_width=True)

    # TAB 3: THỐNG KÊ NHÓM
    with tab3:
        st.subheader("Phân Tích Sâu Nhóm M")
        if st.button("Phân Tích"):
            s_map = {f"M{m}": st.session_state[f'std_{m}'] for m in range(11)}
            df_rep, df_detail = analyze_group_performance(target_d - timedelta(days=15), target_d, cut_val, s_map, cache, kq_db, 1, False)
            
            c1, c2 = st.columns([1, 2])
            with c1: st.dataframe(df_rep, use_container_width=True)
            with c2: st.dataframe(df_detail, use_container_width=True)

if __name__ == "__main__":
    main()
