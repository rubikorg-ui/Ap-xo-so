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
# 1. CẤU HÌNH HỆ THỐNG & PRESETS
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
                if not any(c.startswith('M') for c in final_cols):
                    if h_row != 3:
                        h_row = 3
                        df = df_raw.iloc[h_row+1:].copy()
                        df.columns = [str(c).strip().upper().replace('M 1 0', 'M10') for c in df_raw.iloc[h_row]]
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
                    for c_idx in range(min(2, len(df.columns))):
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

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
# (Phần xử lý Logic tiếp theo trong Part 2...)
    cols_in_scope = sorted(list(set(p_map_dict.keys()) | set(s_map_dict.keys())))
    valid_cols = [c for c in cols_in_scope if c in df.columns]
    if not valid_cols or df.empty: return []
    sub_df = df[valid_cols].copy()
    melted = sub_df.melt(ignore_index=False, var_name='Col', value_name='Val').dropna(subset=['Val'])
    mask_valid = ~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT|MISS|TRUOT|NGHỈ|LỖI', regex=True)
    melted = melted[mask_valid]
    if melted.empty: return []
    s_nums = melted['Val'].astype(str).str.findall(r'\d+')
    exploded = melted.assign(Num=s_nums).explode('Num').dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].str.len() <= 2]
    exploded['P'] = exploded['Col'].map(p_map_dict).fillna(0)
    exploded['S'] = exploded['Col'].map(s_map_dict).fillna(0)
    stats = exploded.groupby('Num')[['P', 'S']].sum()
    votes = exploded.reset_index().groupby('Num')['index'].nunique()
    stats['V'] = votes
    stats = stats[stats['V'] >= min_v]
    if stats.empty: return []
    stats = stats.reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    if inverse: stats = stats.sort_values(by=['P', 'S', 'Num_Int'], ascending=[False, False, True])
    else: stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])
    return stats['Num'].head(int(top_n)).tolist()

def smart_trim_by_score(number_list, df, p_map, s_map, target_size):
    if len(number_list) <= target_size: return sorted(number_list)
    temp_df = df.copy()
    melted = temp_df.melt(value_name='Val').dropna(subset=['Val'])
    mask_bad = ~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT', regex=True)
    melted = melted[mask_bad]
    s_nums = melted['Val'].astype(str).str.findall(r'\d+')
    exploded = melted.assign(Num=s_nums).explode('Num').dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].isin(number_list)]
    exploded['Score'] = exploded['variable'].map(p_map).fillna(0) + exploded['variable'].map(s_map).fillna(0)
    final_scores = exploded.groupby('Num')['Score'].sum().reset_index()
    final_scores = final_scores.sort_values(by='Score', ascending=False)
    return sorted(final_scores.head(int(target_size))['Num'].tolist())

def calculate_goc_3_logic(target_date, rolling_window, _cache, _kq_db, input_limit, target_limit, score_std, use_inverse, min_votes):
    dummy_lim = {'l12':1, 'l34':1, 'l56':1, 'mod':1}
    res_v24 = calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, dummy_lim, min_votes, score_std, score_std, use_inverse)
    if not res_v24: return None
    top3 = res_v24['top6_std'][:3]
    col_hist = res_v24['source_col']
    curr_data = _cache[target_date]; df = curr_data['df']
    p_map_dict = {}
    score_std_tuple = tuple(score_std.items())
    for col in df.columns:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map_dict[col] = s_p
    hist_series = df[col_hist].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
    pool_sets = []
    for g in top3:
        mask = hist_series == g.upper(); valid_mems = df[mask]
        res = fast_get_top_nums(valid_mems, p_map_dict, p_map_dict, int(input_limit), min_votes, use_inverse)
        pool_sets.append(set(res))
    all_nums = []
    for s in pool_sets: all_nums.extend(list(s))
    counts = Counter(all_nums)
    overlap_nums = [n for n, c in counts.items() if c >= 2]
    final_set = smart_trim_by_score(overlap_nums, df, p_map_dict, {}, target_limit)
    return {"top3": top3, "dan_final": final_set, "source_col": col_hist}

# --- 🛡️ NEW: ALLIANCE 8X (GIAO THOA 1-6-4 & 2-5-3) ---
def calculate_8x_alliance_custom(df, top6, limits, col_name="8X", min_vote=2):
    def get_set(name, lim):
        m_row = df[df.iloc[:, 15].astype(str).str.strip() == name]
        if m_row.empty: return set()
        c_idx = 17 if col_name == "8X" else 27
        return set(get_nums(str(m_row.iloc[0, c_idx]))[:lim])
    lms = {top6[0]: limits['l12'], top6[1]: limits['l12'], top6[2]: limits['l34'], top6[3]: limits['l34'], top6[4]: limits['l56'], top6[5]: limits['l56']}
    s1 = {n for n, c in Counter(list(get_set(top6[0], lms[top6[0]])) + list(get_set(top6[5], lms[top6[5]])) + list(get_set(top6[3], lms[top6[3]]))).items() if c >= min_vote}
    s2 = {n for n, c in Counter(list(get_set(top6[1], lms[top6[1]])) + list(get_set(top6[4], lms[top6[4]])) + list(get_set(top6[2], lms[top6[2]]))).items() if c >= min_vote}
    return sorted(list(s1.intersection(s2)))

@st.cache_data(show_spinner=False)
def calculate_v24_final(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None, max_trim=None):
    res = calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups, max_trim)
    if not res: return None, "Lỗi dữ liệu"
    return res, None

def get_elite_members(df, top_n=10, sort_by='score'):
    if df.empty: return df
    m_cols = [c for c in df.columns if c.startswith('M')]
    df = df.dropna(subset=m_cols, how='all')
    if sort_by == 'score': return df.sort_values(by='SCORE_SORT', ascending=False).head(top_n)
    else:
        if 'STT' in df.columns: return df.sort_values(by='STT', ascending=True).head(top_n)
        return df.head(top_n)

def calculate_matrix_simple(df_members, weights_list):
    scores = np.zeros(100)
    for _, row in df_members.iterrows():
        for i in range(len(weights_list)):
            col_name = f"M{i}"; w = weights_list[i]
            if col_name in df_members.columns and w > 0:
                nums = get_nums(row[col_name])
                for n in nums:
                    try:
                        n_int = int(n)
                        if 0 <= n_int <= 99: scores[n_int] += w
                    except: pass
    result = []
    for i in range(100):
        if scores[i] > 0: result.append((i, scores[i]))
    result.sort(key=lambda x: x[1], reverse=True)
    return result

def get_preset_params(preset_name):
    if preset_name not in SCORES_PRESETS: return None
    p = SCORES_PRESETS[preset_name]
    std = {f'M{i}': p['STD'][i] for i in range(11)}
    mod = {f'M{i}': p['MOD'][i] for i in range(11)}
    lim = p['LIMITS']
    rolling = p.get('ROLLING', 10) 
    return std, mod, lim, rolling

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except: return None
    return None

def save_config(config_data):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=4); return True
    except: return False

# --- KHỞI TẠO SESSION STATE (Dòng 550 gốc) ---
if 'std_0' not in st.session_state:
    s_std, s_mod, s_lim, s_roll = get_preset_params("Balanced (Khuyên dùng 2026)")
    for i in range(11):
        st.session_state[f'std_{i}'] = s_std[f'M{i}']
        st.session_state[f'mod_{i}'] = s_mod[f'M{i}']
    st.session_state['L12'], st.session_state['L34'] = s_lim['l12'], s_lim['l34']
    st.session_state['L56'], st.session_state['LMOD'] = s_lim['l56'], s_lim['mod']
    st.session_state['ROLLING_WINDOW'] = s_roll
    st.session_state['MAX_TRIM'] = 80
    st.session_state['MIN_VOTES'] = 1
    st.session_state['STRATEGY_MODE'] = "🛡️ V24 Cổ Điển"
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    
    # --- CHÈN THÊM: ALLIANCE 8X CONFIG ---
    with st.expander("🛡️ Alliance 8X Custom Settings", expanded=True):
        USE_ALLIANCE_8X = st.toggle("Kích hoạt Giao thoa 8X", value=True)
        COL_TARGET_8X = st.selectbox("🎯 Cột lấy số mục tiêu", ["8X", "M0", "M1", "M2"], index=0)
        MIN_VOTES_LM = st.slider("🗳️ Vote tối thiểu Liên minh", 1, 3, 2)
    st.divider()

    # Quản lý Presets (Dòng 620 gốc)
    menu_ops = ["Cấu hình hiện tại"] + list(SCORES_PRESETS.keys())
    selected_cfg = st.selectbox("📚 Chọn bộ mẫu:", menu_ops)
    if st.button("🚀 Áp dụng mẫu này"):
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
    c_std_final = {}
    c_mod_final = {}
    for i in range(11):
        with col_w1:
            st.session_state[f'std_{i}'] = st.number_input(f"STD M{i}", 0, 100, st.session_state[f'std_{i}'], key=f"sidebar_s_{i}")
            c_std_final[f'M{i}'] = st.session_state[f'std_{i}']
        with col_w2:
            st.session_state[f'mod_{i}'] = st.number_input(f"MOD M{i}", 0, 100, st.session_state[f'mod_{i}'], key=f"sidebar_m_{i}")
            c_mod_final[f'M{i}'] = st.session_state[f'mod_{i}']

    st.divider()
    ROLL_WINDOW = st.number_input("📅 Rolling Window", 1, 30, st.session_state['ROLLING_WINDOW'])
    L_12 = st.number_input("✂️ Limit L1,2", 1, 100, st.session_state['L12'])
    L_34 = st.number_input("✂️ Limit L3,4", 1, 100, st.session_state['L34'])
    L_56 = st.number_input("✂️ Limit L5,6", 1, 100, st.session_state['L56'])
    L_MOD = st.number_input("✂️ Limit MOD", 1, 100, st.session_state['LMOD'])
    
    MAX_TRIM_V24 = st.slider("📏 Cắt dàn (Max Trim)", 50, 95, st.session_state.get('MAX_TRIM', 80))
    MIN_VOTES_V24 = st.slider("🗳️ Vote tối thiểu (V24)", 1, 5, st.session_state.get('MIN_VOTES', 1))
    
    USE_INVERSE_MODE = st.checkbox("🔄 Nghịch đảo", value=False)
    USE_ADAPTIVE_MODE = st.checkbox("🧠 Adaptive Weights", value=False)

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (GIỮ NGUYÊN 100% TỪ FILE GỐC)
# ==============================================================================

uploaded_files = st.file_uploader("📂 Tải lên dữ liệu tổng hợp (CSV/XLSX)", accept_multiple_files=True)

if uploaded_files:
    data_cache, kq_db, status, logs = load_data_v24(uploaded_files)
    
    if data_cache:
        st.success(f"⚡ Đã nạp {len(data_cache)} ngày dữ liệu.")
        tab_soi, tab_backtest, tab_manual = st.tabs(["🎯 Soi cầu hằng ngày", "📊 Backtest Hệ thống", "🛠️ Công cụ phụ"])
        
        with tab_soi:
            all_dates = sorted(list(data_cache.keys()), reverse=True)
            target_date = st.selectbox("📅 Chọn ngày soi cầu:", all_dates, key="main_sb_date")
            
            if target_date:
                u_limits = {'l12': L_12, 'l34': L_34, 'l56': L_56, 'mod': L_MOD}
                if USE_ADAPTIVE_MODE: c_std_final = get_adaptive_weights(target_date, c_std_final, data_cache, kq_db)
                
                with st.spinner("🚀 Đang tính toán ma trận..."):
                    if st.session_state['STRATEGY_MODE'] == "🛡️ V24 Cổ Điển":
                        res, err = calculate_v24_final(target_date, ROLL_WINDOW, data_cache, kq_db, u_limits, MIN_VOTES_V24, c_std_final, c_mod_final, USE_INVERSE_MODE, max_trim=MAX_TRIM_V24)
                    else:
                        g3_res = calculate_goc_3_logic(target_date, ROLL_WINDOW, data_cache, kq_db, L_12, MAX_TRIM_V24, c_std_final, USE_INVERSE_MODE, MIN_VOTES_V24)
                        res = {"top6_std": g3_res['top3'] + ["N/A"]*3, "dan_final": g3_res['dan_final'], "source_col": g3_res['source_col'], "dan_goc": [], "dan_mod": []}
                
                if res:
                    st.header(f"🔮 Kết quả soi cầu: {target_date.strftime('%d/%m/%Y')}")
                    
                    # HIỂN THỊ ALLIANCE 8X (HIỆN FULL SỐ)
                    if USE_ALLIANCE_8X:
                        st.markdown("### 🛡️ Dàn Tinh hoa Alliance 8X (Giao thoa)")
                        dan_8x = calculate_8x_alliance_custom(data_cache[target_date]['df'], res['top6_std'], u_limits, col_name=COL_TARGET_8X, min_v=MIN_VOTES_LM)
                        st.text_area(f"👇 Copy dàn Alliance ({len(dan_8x)} số):", value=",".join(dan_8x), height=150, key="ta_8x")
                        
                        if target_date in kq_db:
                            real_val = str(kq_db[target_date]).zfill(2)
                            if real_val in dan_8x: st.success(f"✅ ALLIANCE WIN: {real_val}")
                            else: st.error(f"❌ ALLIANCE MISS: {real_val}")
                        st.divider()

                    # HIỂN THỊ V24 GỐC
                    st.markdown("### 💎 Dàn Tinh hoa V24 (Gốc)")
                    st.text_area(f"👇 Copy dàn V24 ({len(res['dan_final'])} số):", value=",".join(res['dan_final']), height=150, key="ta_v24")
                    
                    if target_date in kq_db:
                        real_val = str(kq_db[target_date]).zfill(2)
                        if real_val in res['dan_final']: st.success(f"✅ V24 WIN: {real_val}")
                        else: st.error(f"❌ V24 MISS: {real_val}")

                    with st.expander("🔎 Xem chi tiết phân tích cao thủ"):
                        st.write(f"**Top 6 Phong độ:** {', '.join(res['top6_std'])}")
                        st.write(f"**Cột lịch sử quét:** {res['source_col']}")

        with tab_backtest:
            st.subheader("📊 Backtest Hiệu suất Cấu hình")
            if st.button("▶️ Chạy Backtest Toàn bộ"):
                bt_dates = sorted([d for d in data_cache.keys() if d in kq_db])
                if not bt_dates: st.warning("Không tìm thấy dữ liệu KQ.")
                else:
                    results = []
                    pb = st.progress(0)
                    for idx, d in enumerate(bt_dates):
                        r, _ = calculate_v24_final(d, ROLL_WINDOW, data_cache, kq_db, u_limits, MIN_VOTES_V24, c_std_final, c_mod_final, USE_INVERSE_MODE, max_trim=MAX_TRIM_V24)
                        if r:
                            real_kq = str(kq_db[d]).zfill(2)
                            v24_hit = real_kq in r['dan_final']
                            d_8x = calculate_8x_alliance_custom(data_cache[d]['df'], r['top6_std'], u_limits, col_name=COL_TARGET_8X, min_v=MIN_VOTES_LM)
                            all_hit = real_kq in d_8x
                            results.append({
                                "Ngày": d.strftime("%d/%m"),
                                "KQ": real_kq,
                                "V24": "✅" if v24_hit else "❌",
                                "Alliance 8X": "🌟 WIN" if all_hit else "MISS",
                                "Size 8X": len(d_8x)
                            })
                        pb.progress((idx + 1) / len(bt_dates))
                    st.table(pd.DataFrame(results))

        with tab_manual:
            # GIỮ NGUYÊN 100% CÔNG CỤ THỦ CÔNG CỦA ÔNG (TỪ DÒNG 820 GỐC)
            st.subheader("🛠️ Công cụ tạo dàn thủ công (Lấy Top Cao thủ)")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                target_d_man = st.selectbox("Chọn ngày dữ liệu:", all_dates, key="manual_date_select")
                sort_mode_man = st.radio("Sắp xếp theo:", ["score", "stt"])
            with col_m2:
                top_n_man = st.number_input("Lấy Top N người:", 1, 50, 10)
                skip_val_man = st.number_input("Bỏ qua X số đầu (Skip):", 0, 50, 0)
                cut_val_man = st.number_input("Lấy X số (Limit):", 1, 100, 80)

            if target_d_man:
                df_manual = data_cache[target_d_man]['df']
                input_elite = get_elite_members(df_manual, top_n=top_n_man, sort_by=sort_mode_man)
                with st.expander("📋 Danh sách cao thủ đang chọn"):
                    st.dataframe(input_elite, use_container_width=True)
                
                w_list_man = [st.session_state[f'std_{i}'] for i in range(11)]
                ranked_nums_man = calculate_matrix_simple(input_elite, w_list_man)
                
                s_idx, e_idx = skip_val_man, skip_val_man + cut_val_man
                final_dan_man = [f"{n:02d}" for n, score in ranked_nums_man[s_idx:e_idx]]
                final_dan_man.sort()
                
                st.divider()
                st.text_area(f"👇 Dàn thủ công ({len(final_dan_man)} số):", value=",".join(final_dan_man), height=150, key="ta_manual")
                
                if target_d_man in kq_db:
                    real_val = str(kq_db[target_d_man]).zfill(2)
                    rank_pos = next((i+1 for i, (n,s) in enumerate(ranked_nums_man) if n == int(real_val)), 999)
                    if s_idx < rank_pos <= e_idx: st.success(f"✅ WIN: {real_val} (Hạng {rank_pos})")
                    else: st.error(f"❌ MISS: {real_val} (Hạng {rank_pos})")

    if logs:
        with st.expander("⚠️ Nhật ký lỗi hệ thống"):
            for log in logs: st.warning(log)
else:
    st.info("👋 Chào Quang Handsome! Hãy tải các file tổng hợp lên để bắt đầu.")
