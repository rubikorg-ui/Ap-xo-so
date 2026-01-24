import streamlit as st
import pandas as pd
import re
import datetime
import time
import json
import os
from datetime import timedelta
from collections import Counter
from functools import lru_cache
import numpy as np
import pa2_preanalysis_text as pa2

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & PRESETS (GIỮ NGUYÊN 100%)
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V62 - Dynamic Hybrid", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🛡️ Quang Handsome: V62 Dynamic Hybrid")
st.caption("🚀 Tính năng mới: Hybrid thay đổi theo tinh chỉnh màn hình | Backtest Đơn | M Động")

CONFIG_FILE = 'config.json'

SCORES_PRESETS = {
    "Balanced (Khuyên dùng 2026)": { 
        "STD": [5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70], 
        "MOD": [5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70],
        "LIMITS": {'l12': 75, 'l34': 70, 'l56': 65, 'mod': 75},
        "ROLLING": 10
    },
    "CH1 Fix (Siết chặt)": { 
        "STD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70], 
        "MOD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70],
        "LIMITS": {'l12': 70, 'l34': 65, 'l56': 55, 'mod': 80},
        "ROLLING": 10
    },
    "Hard Core (Gốc)": { 
        "STD": [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60], 
        "MOD": [0, 5, 10, 20, 25, 45, 50, 40, 30, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88},
        "ROLLING": 10
    },
    "CH1: Bám Đuôi (Gốc)": { 
        "STD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70], 
        "MOD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70],
        "LIMITS": {'l12': 80, 'l34': 75, 'l56': 60, 'mod': 88},
        "ROLLING": 10
    }
}

RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CORE FUNCTIONS
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    raw_nums = RE_NUMS.findall(s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name, mapping_tuple):
    clean = RE_CLEAN_SCORE.sub('', str(col_name).upper().replace(' ', ''))
    mapping = dict(mapping_tuple)
    if 'M10' in clean: return mapping.get('M10', 0)
    for key, score in mapping.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def get_adaptive_weights(target_date, base_weights, data_cache, kq_db, window=3, factor=1.5):
    m_hits = {i: 0 for i in range(11)}
    m_total = {i: 0 for i in range(11)}
    past_days = []
    check_d = target_date - timedelta(days=1)
    while len(past_days) < window:
        if check_d in data_cache and check_d in kq_db:
            past_days.append(check_d)
        check_d -= timedelta(days=1)
        if (target_date - check_d).days > 20: break 
    if not past_days: return base_weights 
    for d in past_days:
        kq = str(kq_db[d]).zfill(2)
        df = data_cache[d]['df']
        m_cols = [c for c in df.columns if re.match(r'^M\s*\d+', c) or c in ['M10', 'M 1 0']]
        m_map = {}
        for c in m_cols:
            clean = c.replace(' ', '').replace('M', '')
            try: idx = int(clean); m_map[c] = idx
            except: pass
        for _, row in df.iterrows():
            if 'KQ' in str(row.iloc[0]): continue
            for col, w_idx in m_map.items():
                m_total[w_idx] += 1
                nums = get_nums(row[col])
                if kq in nums: m_hits[w_idx] += 1
    new_weights = {}
    for i, base_w in base_weights.items():
        idx = int(i.replace('M', ''))
        eff = m_hits[idx] / m_total[idx] if m_total[idx] > 0 else 0
        adjusted_w = base_w * (1 + factor * eff)
        new_weights[i] = round(adjusted_w, 1)
    return new_weights

def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper().replace('NGAY', '').replace('NGÀY', '').strip()
    match_iso = RE_ISO_DATE.search(s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)
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

def extract_meta_from_filename(filename):
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    clean_name = re.sub(r'\s*-\s*', '-', clean_name) 
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    full_date_match = re.search(r'(\d{1,2})[\.\-](\d{1,2})(?:[\.\-]20\d{2})?', clean_name)
    if full_date_match:
        try:
            d = int(full_date_match.group(1)); m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            if m == 12 and m_global == 1: y -= 1 
            elif m == 1 and m_global == 12: y += 1
            return m, y, datetime.date(y, m, d)
        except: pass
    return m_global, y_global, None

def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords): return idx
    return 3

@st.cache_data(ttl=600, show_spinner=False)
def load_data_v24(files):
    cache = {}; kq_db = {}; err_logs = []; file_status = []
    files = sorted(files, key=lambda x: x.name)
    for file in files:
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper(): continue
        f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
        try:
            dfs_to_process = []
            if file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(file, engine='openpyxl')
                for sheet in xls.sheet_names:
                    s_date = None
                    try:
                        clean_s = re.sub(r'[^0-9]', ' ', sheet).strip()
                        parts = [int(x) for x in clean_s.split()]
                        if parts:
                            d_s, m_s, y_s = parts[0], f_m, f_y
                            if len(parts) >= 3 and parts[2] > 2000: y_s = parts[2]; m_s = parts[1]
                            s_date = datetime.date(y_s, m_s, d_s)
                    except: pass
                    if not s_date: s_date = date_from_name
                    if s_date:
                        preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=30, engine='openpyxl')
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row, engine='openpyxl')
                        dfs_to_process.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")
            elif file.name.endswith('.csv'):
                if not date_from_name: continue
                encodings_to_try = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'utf-16']
                df_raw = None; h_row = 0
                for enc in encodings_to_try:
                    try:
                        file.seek(0)
                        preview = pd.read_csv(file, header=None, nrows=30, encoding=enc)
                        h_row = find_header_row(preview)
                        file.seek(0)
                        df_raw = pd.read_csv(file, header=None, encoding=enc); break
                    except: continue
                if df_raw is None: err_logs.append(f"❌ Lỗi Encoding: {file.name}"); continue
                df = df_raw.iloc[h_row+1:].copy()
                raw_cols = df_raw.iloc[h_row].astype(str).tolist()
                seen = {}; final_cols = []
                for c in raw_cols:
                    c = str(c).strip().upper().replace('M 1 0', 'M10')
                    if c in seen: seen[c] += 1; final_cols.append(f"{c}.{seen[c]}")
                    else: seen[c] = 0; final_cols.append(c)
                df.columns = final_cols
                dfs_to_process.append((date_from_name, df))
                file_status.append(f"✅ CSV: {file.name}")

            for t_date, df in dfs_to_process:
                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]
                score_col = next((c for c in df.columns if 'Đ9' in c or 'DIEM' in c or 'ĐIỂM' in c), None)
                if score_col: df['SCORE_SORT'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
                else: df['SCORE_SORT'] = 0
                rename_map = {}
                for c in df.columns:
                    clean_c = c.replace(" ", "")
                    if re.match(r'^M\d+$', clean_c) or clean_c == 'M10': rename_map[c] = clean_c
                if rename_map: df = df.rename(columns=rename_map)
                hist_map = {}
                for col in df.columns:
                    if "UNNAMED" in col or col.startswith("M") or col in ["STT", "SCORE_SORT"]: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: hist_map[d_obj] = col
                kq_row = None
                if not df.empty:
                    for c_idx in range(min(5, len(df.columns))):
                        col_check = df.columns[c_idx]
                        try:
                            mask_kq = df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')
                            if mask_kq.any(): kq_row = df[mask_kq].iloc[0]; break
                        except: continue
                if kq_row is not None:
                    for d_val, c_name in hist_map.items():
                        try:
                            nums = get_nums(str(kq_row[c_name]))
                            if nums: kq_db[d_val] = nums[0]
                        except: pass
                cache[t_date] = {'df': df, 'hist_map': hist_map}
        except Exception as e: err_logs.append(f"Lỗi '{file.name}': {str(e)}"); continue
    return cache, kq_db, file_status, err_logs

# --- PHẦN 1 KẾT THÚC ---
# ==============================================================================
# 3. CORE LOGIC (V24 & ALLIANCE 8X)
# ==============================================================================

def calculate_v24_final(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None, max_trim=None):
    res = calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups, max_trim)
    if not res: return None, "Lỗi dữ liệu hoặc không tìm thấy lịch sử nhóm."
    return res, None

# --- 🛡️ HÀM LIÊN MINH 8X (GIAO THOA 1-6-4 & 2-5-3) ---
def calculate_8x_alliance_custom(df_target, top_6_names, limits_config, col_name="8X", min_v=2):
    """
    Logic: Tìm Top 6 -> Chia 2 liên minh (1-6-4 và 2-5-3) -> Lọc vote >= 2 -> Lấy GIAO THOA.
    """
    def get_set_from_member(name, limit):
        # Xác định dòng thành viên (Cột Tên thường ở index 15)
        m_row = df_target[df_target.iloc[:, 15].astype(str).str.strip() == name]
        if m_row.empty: return set()
        # Lấy dữ liệu từ cột 8X (thường ở index 17)
        c_idx = 17 if col_name == "8X" else 27
        nums = get_nums(str(m_row.iloc[0, c_idx]))
        return set(nums[:limit])

    # Lấy giới hạn cắt số cho từng vị trí Top
    lim_map = {
        top_6_names[0]: limits_config['l12'], top_6_names[1]: limits_config['l12'],
        top_6_names[2]: limits_config['l34'], top_6_names[3]: limits_config['l34'],
        top_6_names[4]: limits_config['l56'], top_6_names[5]: limits_config['l56']
    }

    # Liên minh 1: Top 1, 6, 4
    set1 = get_set_from_member(top_6_names[0], lim_map[top_6_names[0]])
    set6 = get_set_from_member(top_6_names[5], lim_map[top_6_names[5]])
    set4 = get_set_from_member(top_6_names[3], lim_map[top_6_names[3]])
    c1 = Counter(list(set1) + list(set6) + list(set4))
    lm1 = {n for n, c in c1.items() if c >= min_v}

    # Liên minh 2: Top 2, 5, 3
    set2 = get_set_from_member(top_6_names[1], lim_map[top_6_names[1]])
    set5 = get_set_from_member(top_6_names[4], lim_map[top_6_names[4]])
    set3 = get_set_from_member(top_6_names[2], lim_map[top_6_names[2]])
    c2 = Counter(list(set2) + list(set5) + list(set3))
    lm2 = {n for n, c in c2.items() if c >= min_v}

    return sorted(list(lm1.intersection(lm2)))

# ==============================================================================
# 4. QUẢN LÝ TRẠNG THÁI (SESSION STATE) & SIDEBAR
# ==============================================================================

if 'std_0' not in st.session_state:
    s_std, s_mod, s_lim, s_roll = get_preset_params("Balanced (Khuyên dùng 2026)")
    for i in range(11):
        st.session_state[f'std_{i}'] = s_std[f'M{i}']
        st.session_state[f'mod_{i}'] = s_mod[f'M{i}']
    st.session_state['L12'] = s_lim['l12']
    st.session_state['L34'] = s_lim['l34']
    st.session_state['L56'] = s_lim['l56']
    st.session_state['LMOD'] = s_lim['mod']
    st.session_state['ROLLING_WINDOW'] = s_roll
    st.session_state['MAX_TRIM'] = 80
    st.session_state['MIN_VOTES'] = 1

with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    
    with st.expander("🛡️ Alliance 8X (Giao thoa)", expanded=True):
        USE_ALLIANCE_8X = st.toggle("Kích hoạt Liên minh 8X", value=True)
        COL_TARGET_8X = st.selectbox("🎯 Cột dữ liệu", ["8X", "M0", "M1"], index=0)
        MIN_VOTES_LM = st.slider("🗳️ Vote tối thiểu LM", 1, 3, 2)
    
    st.divider()
    STRATEGY_MODE = st.selectbox("🧩 Chế độ Chiến thuật", ["🛡️ V24 Cổ Điển", "🧪 Gốc 3 (Test)"])
    
    menu_ops = ["Cấu hình hiện tại"] + list(SCORES_PRESETS.keys())
    selected_cfg = st.selectbox("📚 Chọn bộ mẫu:", menu_ops)
    if st.button("Áp dụng Preset"):
        if selected_cfg != "Cấu hình hiện tại":
            vals = SCORES_PRESETS[selected_cfg]
            for i in range(11):
                st.session_state[f'std_{i}'] = vals["STD"][i]
                st.session_state[f'mod_{i}'] = vals["MOD"][i]
            st.session_state['L12'] = vals['LIMITS']['l12']
            st.session_state['L34'] = vals['LIMITS']['l34']
            st.session_state['L56'] = vals['LIMITS']['l56']
            st.session_state['LMOD'] = vals['LIMITS']['mod']
            st.session_state['ROLLING_WINDOW'] = vals.get('ROLLING', 10)
            st.rerun()

    st.subheader("📊 Trọng số Ma trận")
    col_w1, col_w2 = st.columns(2)
    curr_std_w = {}
    curr_mod_w = {}
    for i in range(11):
        with col_w1:
            st.session_state[f'std_{i}'] = st.number_input(f"STD M{i}", 0, 100, st.session_state[f'std_{i}'], key=f"s_{i}")
            curr_std_w[f'M{i}'] = st.session_state[f'std_{i}']
        with col_w2:
            st.session_state[f'mod_{i}'] = st.number_input(f"MOD M{i}", 0, 100, st.session_state[f'mod_{i}'], key=f"m_{i}")
            curr_mod_w[f'M{i}'] = st.session_state[f'mod_{i}']

    st.divider()
    ROLL_W = st.number_input("📅 Rolling (Ngày)", 1, 30, st.session_state['ROLLING_WINDOW'])
    L12 = st.number_input("✂️ Limit L1,2", 1, 100, st.session_state['L12'])
    L34 = st.number_input("✂️ Limit L3,4", 1, 100, st.session_state['L34'])
    L56 = st.number_input("✂️ Limit L5,6", 1, 100, st.session_state['L56'])
    LMOD = st.number_input("✂️ Limit MOD", 1, 100, st.session_state['LMOD'])
    MAX_T = st.slider("📏 Max Trim", 50, 95, st.session_state['MAX_TRIM'])
    MIN_V = st.slider("🗳️ Min Vote", 1, 5, st.session_state['MIN_VOTES'])
    USE_INV = st.checkbox("🔄 Inverse Mode", value=False)
    USE_ADAPT = st.checkbox("🧠 Adaptive Weights", value=False)

# ==============================================================================
# 5. MAIN APP LOGIC
# ==============================================================================

uploaded_files = st.file_uploader("📂 Tải lên file tổng hợp", accept_multiple_files=True)

if uploaded_files:
    data_cache, kq_db, status, logs = load_data_v24(uploaded_files)
    
    if data_cache:
        st.success(f"⚡ Đã nạp {len(data_cache)} ngày dữ liệu.")
        tab_main, tab_backtest, tab_manual = st.tabs(["🎯 Soi cầu", "📊 Backtest", "🛠️ Công cụ phụ"])
        
        with tab_main:
            all_dates = sorted(list(data_cache.keys()), reverse=True)
            target_date = st.selectbox("📅 Chọn ngày:", all_dates)
            
            if target_date:
                u_lims = {'l12': L12, 'l34': L34, 'l56': L56, 'mod': LMOD}
                if USE_ADAPT: curr_std_w = get_adaptive_weights(target_date, curr_std_w, data_cache, kq_db)
                
                if STRATEGY_MODE == "🛡️ V24 Cổ Điển":
                    res, err = calculate_v24_final(target_date, ROLL_W, data_cache, kq_db, u_lims, MIN_V, curr_std_w, curr_mod_w, USE_INV, max_trim=MAX_T)
                else:
                    g3 = calculate_goc_3_logic(target_date, ROLL_W, data_cache, kq_db, L12, MAX_T, curr_std_w, USE_INV, MIN_V)
                    res = {"top6_std": g3['top3'] + ["N/A"]*3, "dan_final": g3['dan_final'], "source_col": g3['source_col'], "dan_goc": [], "dan_mod": []}
                
                if res:
                    st.header(f"🔮 Phân tích ngày: {target_date.strftime('%d/%m/%Y')}")
                    
                    if USE_ALLIANCE_8X:
                        st.subheader("🛡️ Dàn Tinh hoa Liên minh 8X (Giao thoa 1-6-4 & 2-5-3)")
                        dan_8x = calculate_8x_alliance_custom(data_cache[target_date]['df'], res['top6_std'], u_lims, COL_TARGET_8X, MIN_VOTES_LM)
                        st.text_area(f"👇 Dàn {len(dan_8x)} số (Copy):", value=",".join(dan_8x), height=150)
                        if target_date in kq_db:
                            real = str(kq_db[target_date]).zfill(2)
                            if real in dan_8x: st.success(f"✅ ALLIANCE WIN: {real}")
                            else: st.error(f"❌ ALLIANCE MISS: {real}")
                        st.divider()

                    st.subheader("💎 Dàn Tinh hoa V24 (Gốc)")
                    st.text_area(f"👇 Dàn {len(res['dan_final'])} số:", value=",".join(res['dan_final']), height=150)
                    if target_date in kq_db:
                        real = str(kq_db[target_date]).zfill(2)
                        if real in res['dan_final']: st.success(f"✅ V24 WIN: {real}")
                        else: st.error(f"❌ V24 MISS: {real}")

        with tab_backtest:
            st.subheader("📊 Backtest Hệ thống")
            if st.button("🚀 Chạy Backtest"):
                bt_dates = sorted([d for d in data_cache.keys() if d in kq_db])
                bt_res = []
                for d in bt_dates:
                    r, _ = calculate_v24_final(d, ROLL_W, data_cache, kq_db, u_lims, MIN_V, curr_std_w, curr_mod_w, USE_INV, max_trim=MAX_T)
                    if r:
                        real = str(kq_db[d]).zfill(2)
                        v24_win = real in r['dan_final']
                        d_8x = calculate_8x_alliance_custom(data_cache[d]['df'], r['top6_std'], u_lims, COL_TARGET_8X, MIN_VOTES_LM)
                        all_win = real in d_8x
                        bt_res.append({"Ngày": d.strftime("%d/%m"), "KQ": real, "V24": "✅" if v24_win else "❌", "Alliance 8X": "🌟" if all_win else "☁️", "Size 8X": len(d_8x)})
                st.table(pd.DataFrame(bt_res))

        with tab_manual:
            st.subheader("🛠️ Công cụ tạo dàn thủ công")
            target_d = st.selectbox("Chọn ngày dữ liệu:", all_dates, key="manual_d")
            if target_d:
                df_target = data_cache[target_d]['df']
                filter_mode = st.radio("Sắp xếp theo:", ["score", "stt"])
                top_n_select = st.number_input("Top N cao thủ:", 1, 50, 10)
                skip_val = st.number_input("Bỏ qua X số đầu:", 0, 50, 0)
                cut_val = st.number_input("Lấy X số:", 1, 100, 80)
                
                input_df = get_elite_members(df_target, top_n=top_n_select, sort_by=filter_mode)
                with st.expander("📋 Danh sách Cao thủ"):
                    st.dataframe(input_df[['STT', 'THÀNH VIÊN', 'SCORE_SORT'] if 'THÀNH VIÊN' in input_df.columns else input_df.columns], use_container_width=True)
                
                c_weights = [st.session_state[f'std_{i}'] for i in range(11)]
                ranked_n = calculate_matrix_simple(input_df, c_weights)
                
                s_idx, e_idx = skip_val, skip_val + cut_val
                f_set = [f"{n:02d}" for n, score in ranked_n[s_idx:e_idx]]
                f_set.sort()
                
                st.divider()
                st.text_area("👇 Dàn số thủ công:", value=",".join(f_set), height=150)
                if target_d in kq_db:
                    real = str(kq_db[target_d]).zfill(2)
                    rank = next((i+1 for i, (n,s) in enumerate(ranked_n) if n == int(real)), 999)
                    if s_idx < rank <= e_idx: st.success(f"WIN: {real} (Hạng {rank})")
                    else: st.error(f"MISS: {real} (Hạng {rank})")

    if logs:
        with st.expander("⚠️ Nhật ký lỗi file"):
            for l in logs: st.warning(l)

if __name__ == "__main__":
    pass
