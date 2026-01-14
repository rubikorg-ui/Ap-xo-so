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

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Ly Thong V62 - Ultimate", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🛡️ Lý Thị Thông: V62 Ultimate")
st.caption("🚀 Đầy đủ: V24 Cổ Điển | V24 Vote 8x (Fix) | Gốc 3 | Matrix")

CONFIG_FILE = 'config.json'

SCORES_PRESETS = {
    "Balanced (Khuyên dùng 2026)": { 
        "STD": [5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70], 
        "MOD": [5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 70],
        "LIMITS": {'l12': 75, 'l34': 70, 'l56': 65, 'mod': 75},
        "ROLLING": 10
    },
    "Hard Core (Gốc)": { 
        "STD": [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60], 
        "MOD": [0, 5, 10, 20, 25, 45, 50, 40, 30, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88},
        "ROLLING": 10
    }
}

RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CORE UTILS
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    raw_nums = RE_NUMS.findall(s_str)
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
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords): return idx
    return 3

# --- 3. CÁC HÀM XỬ LÝ SỐ (Vote, Score, Matrix) ---

def get_top_nums_by_vote(df_members, col_name, limit):
    if df_members.empty: return []
    all_nums = []
    vals = df_members[col_name].dropna().astype(str).tolist()
    for val in vals:
        if any(kw in val.upper() for kw in BAD_KEYWORDS): continue
        all_nums.extend(get_nums(val))
    counts = Counter(all_nums)
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], int(x[0])))
    return [n for n, c in sorted_items[:int(limit)]]

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse, sort_by_vote=False):
    # Hàm này dùng cho V24 Cổ điển & Gốc 3
    cols_in_scope = sorted(list(set(p_map_dict.keys()) | set(s_map_dict.keys())))
    valid_cols = [c for c in cols_in_scope if c in df.columns]
    if not valid_cols or df.empty: return []
    sub_df = df[valid_cols].copy()
    melted = sub_df.melt(ignore_index=False, var_name='Col', value_name='Val').dropna(subset=['Val'])
    mask_valid = ~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT|MISS|TRUOT', regex=True)
    melted = melted[mask_valid]
    if melted.empty: return []
    s_nums = melted['Val'].astype(str).str.findall(r'\d+')
    exploded = melted.assign(Num=s_nums).explode('Num').dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].str.len() <= 2]
    exploded['P'] = exploded['Col'].map(p_map_dict).fillna(0)
    exploded['S'] = exploded['Col'].map(s_map_dict).fillna(0)
    stats = exploded.groupby('Num')[['P', 'S']].sum()
    stats['V'] = exploded.reset_index().groupby('Num')['index'].nunique()
    stats = stats[stats['V'] >= min_v].reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    if sort_by_vote:
        stats = stats.sort_values(by=['V', 'P', 'Num_Int'], ascending=[True, True, True] if inverse else [False, False, True])
    else:
        stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[True, True, True] if inverse else [False, False, True])
    return stats['Num'].head(int(top_n)).tolist()

def smart_trim_by_score(number_list, df, p_map, s_map, target_size):
    if len(number_list) <= target_size: return sorted(number_list)
    temp_df = df.copy()
    melted = temp_df.melt(value_name='Val').dropna(subset=['Val'])
    melted = melted[~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT', regex=True)]
    s_nums = melted['Val'].astype(str).str.findall(r'\d+')
    exploded = melted.assign(Num=s_nums).explode('Num').dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].isin(number_list)]
    exploded['Score'] = exploded['variable'].map(p_map).fillna(0) 
    final_scores = exploded.groupby('Num')['Score'].sum().reset_index().sort_values(by='Score', ascending=False)
    return sorted(final_scores.head(int(target_size))['Num'].tolist())

# --- 4. CÁC CHIẾN THUẬT (LOGIC) ---

# A. V24 8X VOTE (ĐÃ SỬA CHUẨN 63 SỐ)
def calculate_vote_8x_strict(target_date, rolling_window, _cache, _kq_db, limits_config):
    if target_date not in _cache: return None
    curr_data = _cache[target_date]; df = curr_data['df']
    col_8x = next((c for c in df.columns if re.match(r'^(8X|80|DÀN|DAN)$', c.strip().upper()) or '8X' in c.strip().upper()), None)
    if not col_8x: return None
    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache: prev_date = target_date - timedelta(days=i); break
    col_group = curr_data['hist_map'].get(prev_date)
    if not col_group and prev_date in _cache: col_group = _cache[prev_date]['hist_map'].get(prev_date)
    if not col_group: return None

    # Backtest
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    past_dates = []
    check_d = target_date - timedelta(days=1)
    while len(past_dates) < rolling_window:
        if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
        check_d -= timedelta(days=1)
        if (target_date - check_d).days > 60: break
    
    for d in past_dates:
        d_df = _cache[d]['df']; kq = _kq_db[d]
        d_c8 = next((c for c in d_df.columns if '8X' in c.upper()), None)
        sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
        d_c_grp = _cache[d]['hist_map'].get(sorted_dates[0]) if sorted_dates else None
        if d_c8 and d_c_grp:
            try:
                grp_series = d_df[d_c_grp].astype(str).str.upper().str.replace('S', '6').str.replace(r'[^0-9X]', '', regex=True)
                for g in groups:
                    mems = d_df[grp_series == g.upper()]
                    top80 = get_top_nums_by_vote(mems, d_c8, 80)
                    if kq in top80:
                        stats[g]['wins'] += 1
                        stats[g]['ranks'].append(top80.index(kq))
                    else: stats[g]['ranks'].append(999)
            except: continue

    final_rank = []
    for g, inf in stats.items(): final_rank.append((g, -inf['wins'], sum(inf['ranks'])))
    final_rank.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_rank[:6]]

    hist_series = df[col_group].astype(str).str.upper().str.replace('S', '6').str.replace(r'[^0-9X]', '', regex=True)
    
    pool1 = get_top_nums_by_vote(df[hist_series == top6[0].upper()], col_8x, limits_config['l12']) + \
            get_top_nums_by_vote(df[hist_series == top6[4].upper()], col_8x, limits_config['l56']) + \
            get_top_nums_by_vote(df[hist_series == top6[2].upper()], col_8x, limits_config['l34'])
    s1 = {n for n, c in Counter(pool1).items() if c >= 2}

    pool2 = get_top_nums_by_vote(df[hist_series == top6[1].upper()], col_8x, limits_config['l12']) + \
            get_top_nums_by_vote(df[hist_series == top6[3].upper()], col_8x, limits_config['l34']) + \
            get_top_nums_by_vote(df[hist_series == top6[5].upper()], col_8x, limits_config['l56'])
    s2 = {n for n, c in Counter(pool2).items() if c >= 2}

    final_dan = sorted(list(s1.intersection(s2)))
    return {"top6_std": top6, "dan_goc": final_dan, "dan_final": final_dan, "source_col": col_group}

# B. V24 CỔ ĐIỂN & GỐC 3 (LOGIC CŨ ĐÃ ĐƯỢC PHỤC HỒI)
def calculate_v24_classic(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, max_trim=None):
    if target_date not in _cache: return None
    curr_data = _cache[target_date]; df = curr_data['df']
    p_map_dict = {}; s_map_dict = {}
    score_std_tuple = tuple(score_std.items()); score_mod_tuple = tuple(score_mod.items())
    for col in df.columns:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map_dict[col] = s_p
        s_s = get_col_score(col, score_mod_tuple)
        if s_s > 0: s_map_dict[col] = s_s

    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache: prev_date = target_date - timedelta(days=i); break
    col_hist = curr_data['hist_map'].get(prev_date)
    if not col_hist and prev_date in _cache: col_hist = _cache[prev_date]['hist_map'].get(prev_date)
    if not col_hist: return None

    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}
    past_dates = []
    check_d = target_date - timedelta(days=1)
    while len(past_dates) < rolling_window:
        if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
        check_d -= timedelta(days=1)
        if (target_date - check_d).days > 60: break
        
    for d in past_dates:
        d_df = _cache[d]['df']; kq = _kq_db[d]
        d_p_map = {}; d_s_map = {}
        for col in d_df.columns:
            s_p = get_col_score(col, tuple(score_std.items()))
            if s_p > 0: d_p_map[col] = s_p
            s_s = get_col_score(col, tuple(score_mod.items()))
            if s_s > 0: d_s_map[col] = s_s
        sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
        d_hist_col = _cache[d]['hist_map'][sorted_dates[0]] if sorted_dates else None
        if not d_hist_col: continue
        
        try:
            hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
            for g in groups:
                mems = d_df[hist_series_d == g.upper()]
                if mems.empty: stats_std[g]['ranks'].append(999); continue
                top80 = fast_get_top_nums(mems, d_p_map, d_s_map, 80, min_votes, use_inverse)
                if kq in top80: stats_std[g]['wins']+=1; stats_std[g]['ranks'].append(top80.index(kq)+1)
                else: stats_std[g]['ranks'].append(999)
                top86_mod = fast_get_top_nums(mems, d_s_map, d_p_map, int(limits_config['mod']), min_votes, use_inverse)
                if kq in top86_mod: stats_mod[g]['wins']+=1
        except: continue

    final_std = []
    for g, inf in stats_std.items(): final_std.append((g, -inf['wins'], sum(inf['ranks'])))
    final_std.sort(key=lambda x: (x[1], x[2])) 
    top6 = [x[0] for x in final_std[:6]]
    best_mod = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]

    hist_series = df[col_hist].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
    def get_pool(grp_list, lim_dict):
        p = []
        for g in grp_list:
            res = fast_get_top_nums(df[hist_series==g.upper()], p_map_dict, s_map_dict, lim_dict.get(g, 80), min_votes, use_inverse)
            p.extend(res)
        return p
    
    lim_map = {top6[0]: limits_config['l12'], top6[1]: limits_config['l12'], top6[2]: limits_config['l34'], top6[3]: limits_config['l34'], top6[4]: limits_config['l56'], top6[5]: limits_config['l56']}
    s1 = {n for n, c in Counter(get_pool([top6[0], top6[4], top6[2]], lim_map)).items() if c>=2}
    s2 = {n for n, c in Counter(get_pool([top6[1], top6[3], top6[5]], lim_map)).items() if c>=2}
    dan_goc = sorted(list(s1.intersection(s2)))
    
    dan_mod = sorted(fast_get_top_nums(df[hist_series==best_mod.upper()], s_map_dict, p_map_dict, int(limits_config['mod']), min_votes, use_inverse))
    final = sorted(list(set(dan_goc).intersection(set(dan_mod))))
    
    if max_trim and len(final) > max_trim:
        final = smart_trim_by_score(final, df, p_map_dict, s_map_dict, max_trim)
        
    return {"top6_std": top6, "best_mod": best_mod, "dan_goc": dan_goc, "dan_final": final, "source_col": col_hist}

# C. MATRIX LOGIC
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
                    try: scores[int(n)] += w
                    except: pass
    result = [(i, scores[i]) for i in range(100) if scores[i] > 0]
    result.sort(key=lambda x: x[1], reverse=True)
    return result
# --- 5. LOGIC GỐC 3 (WRAPPER) ---
def calculate_goc_3_logic(target_date, rolling_window, _cache, _kq_db, input_limit, target_limit, score_std, min_votes, use_inverse):
    # Dùng core V24 để tìm Top 6
    dummy_lim = {'l12':1, 'l34':1, 'l56':1, 'mod':1}
    res_v24 = calculate_v24_classic(target_date, rolling_window, _cache, _kq_db, dummy_lim, min_votes, score_std, score_std, use_inverse)
    if not res_v24: return None
    
    top3 = res_v24['top6_std'][:3]
    col_hist = res_v24['source_col']
    curr_data = _cache[target_date]; df = curr_data['df']
    
    # Map Score
    p_map = {}
    score_std_tuple = tuple(score_std.items())
    for col in df.columns:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map[col] = s_p
        
    hist_series = df[col_hist].astype(str).str.upper().replace('S', '6').str.replace(r'[^0-9X]', '', regex=True)
    
    pool_sets = []
    for g in top3:
        mems = df[hist_series == g.upper()]
        res = fast_get_top_nums(mems, p_map, p_map, int(input_limit), min_votes, use_inverse)
        pool_sets.append(res)
        
    # Gộp tất cả số từ 3 nhóm
    all_nums = []
    for s in pool_sets: all_nums.extend(s)
    
    # Lấy số trùng (xuất hiện >= 2 lần trong 3 nhóm)
    counts = Counter(all_nums)
    overlap_nums = [n for n, c in counts.items() if c >= 2]
    
    # Cắt gọn theo điểm số (Target Limit)
    final_set = smart_trim_by_score(overlap_nums, df, p_map, {}, target_limit)
    
    return {"top3": top3, "dan_final": final_set, "source_col": col_hist}

# ==============================================================================
# 6. HÀM LOAD DỮ LIỆU (ĐỌC FILE)
# ==============================================================================

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
                
                hist_map = {}
                kq_row = None
                if not df.empty:
                    for c_idx in range(min(2, len(df.columns))):
                        col_check = df.columns[c_idx]
                        try:
                            mask_kq = df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')
                            if mask_kq.any(): kq_row = df[mask_kq].iloc[0]; break
                        except: continue
                
                for col in df.columns:
                    if "UNNAMED" in col or col.startswith("M") or col in ["STT", "SCORE_SORT"]: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: 
                        hist_map[d_obj] = col
                        if kq_row is not None:
                            try:
                                nums = get_nums(str(kq_row[col]))
                                if nums: kq_db[d_obj] = nums[0]
                            except: pass
                cache[t_date] = {'df': df, 'hist_map': hist_map}
        except Exception as e: err_logs.append(f"Lỗi '{file.name}': {str(e)}"); continue
    return cache, kq_db, file_status, err_logs

# ==============================================================================
# 7. GIAO DIỆN CHÍNH (MAIN APP)
# ==============================================================================

def main():
    uploaded_files = st.file_uploader("📂 Tải file CSV/Excel", type=['xlsx', 'csv'], accept_multiple_files=True)

    saved_cfg = None
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: saved_cfg = json.load(f)

    # Initialize Session State
    if 'std_0' not in st.session_state:
        source = saved_cfg if saved_cfg else SCORES_PRESETS["Balanced (Khuyên dùng 2026)"]
        # Flat structure for scores
        if 'STD' in source:
            for i in range(11):
                st.session_state[f'std_{i}'] = source['STD'][i]
                st.session_state[f'mod_{i}'] = source['MOD'][i]
        # Limits
        if 'LIMITS' in source:
            st.session_state['L12'] = source['LIMITS']['l12']
            st.session_state['L34'] = source['LIMITS']['l34']
            st.session_state['L56'] = source['LIMITS']['l56']
            st.session_state['LMOD'] = source['LIMITS']['mod']
        else:
            st.session_state['L12']=80; st.session_state['L34']=70; st.session_state['L56']=60; st.session_state['LMOD']=80
            
        st.session_state['ROLLING_WINDOW'] = source.get('ROLLING', 10)
        st.session_state['STRATEGY_MODE'] = "🛡️ V24 Vote 8x"

    with st.sidebar:
        st.header("⚙️ Cài đặt")
        st.session_state['STRATEGY_MODE'] = st.radio(
            "🎯 CHIẾN THUẬT:",
            ["🛡️ V24 Vote 8x", "🛡️ V24 Cổ Điển", "⚔️ Gốc 3", "🎯 Matrix"], 
            index=["🛡️ V24 Vote 8x", "🛡️ V24 Cổ Điển", "⚔️ Gốc 3", "🎯 Matrix"].index(st.session_state.get('STRATEGY_MODE', "🛡️ V24 Vote 8x"))
        )
        STRATEGY_MODE = st.session_state['STRATEGY_MODE']
        
        if STRATEGY_MODE == "🛡️ V24 Vote 8x":
            st.info("💡 8X Mode: Lấy 8X -> Sort Vote -> Giao thoa 2 Liên Minh (Chuẩn 63 số).")
        
        st.markdown("---")
        def update_scores():
            choice = st.session_state.preset_choice
            vals = SCORES_PRESETS.get(choice, {})
            if vals:
                for i in range(11): 
                    st.session_state[f'std_{i}'] = vals['STD'][i]
                    st.session_state[f'mod_{i}'] = vals['MOD'][i]
                st.session_state['L12'] = vals['LIMITS']['l12']
                st.session_state['L34'] = vals['LIMITS']['l34']
                st.session_state['L56'] = vals['LIMITS']['l56']
                st.session_state['LMOD'] = vals['LIMITS']['mod']

        st.selectbox("📚 Bộ Mẫu:", list(SCORES_PRESETS.keys()), key="preset_choice", on_change=update_scores)
        ROLLING_WINDOW = st.number_input("Chu kỳ (Ngày)", value=st.session_state['ROLLING_WINDOW'])

        if STRATEGY_MODE == "⚔️ Gốc 3":
            with st.expander("⚔️ Cấu hình Gốc 3", expanded=True):
                G3_INPUT = st.slider("Input Top (Lấy vào):", 60, 100, 75, key="G3_INPUT")
                G3_TARGET = st.slider("Target (Giữ lại):", 50, 80, 70, key="G3_TARGET")
        
        with st.expander("✂️ Cắt Số (Limits)", expanded=True):
            L12 = st.number_input("Top 1 & 2:", value=st.session_state['L12'], step=1, key="L12")
            L34 = st.number_input("Top 3 & 4:", value=st.session_state['L34'], step=1, key="L34")
            L56 = st.number_input("Top 5 & 6:", value=st.session_state['L56'], step=1, key="L56")
            LMOD = st.number_input("Mod (V24 Cổ điển):", value=st.session_state['LMOD'], step=1, key="LMOD")

        with st.expander("🎚️ Điểm số (V24 Cổ điển/Gốc 3)", expanded=False):
            c1, c2 = st.columns(2)
            with c1: 
                st.write("Gốc")
                for i in range(11): st.number_input(f"M{i}", key=f"std_{i}")
            with c2:
                st.write("Mod")
                for i in range(11): st.number_input(f"M{i}", key=f"mod_{i}")

        MIN_VOTES = st.number_input("Vote tối thiểu:", 1)
        USE_INVERSE = st.checkbox("Chấm điểm Đảo")
        MAX_TRIM = st.slider("Max Trim Final:", 50, 90, 75)

        if st.button("💾 LƯU CẤU HÌNH"):
            save_data = {
                'STD': [st.session_state[f'std_{i}'] for i in range(11)],
                'MOD': [st.session_state[f'mod_{i}'] for i in range(11)],
                'LIMITS': {'l12': L12, 'l34': L34, 'l56': L56, 'mod': LMOD},
                'ROLLING': ROLLING_WINDOW
            }
            with open(CONFIG_FILE, 'w') as f: json.dump(save_data, f)
            st.success("Đã lưu!")
        
        if st.button("🗑️ XÓA CACHE"): st.cache_data.clear(); st.rerun()

    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        with st.expander("Debug File"):
            for s in f_status: st.write(s)
            for e in err_logs: st.error(e)
        
        if data_cache:
            last_d = max(data_cache.keys())
            tab1, tab2, tab3 = st.tabs(["📊 DỰ ĐOÁN", "🔙 BACKTEST", "🎯 MATRIX"])

            # ------------------------------------------------------------------
            # TAB 1: PREDICTION
            # ------------------------------------------------------------------
            with tab1:
                st.subheader(f"Soi Cầu: {STRATEGY_MODE}")
                col_d, col_btn = st.columns([1, 2])
                with col_d: target_d = st.date_input("Ngày:", value=last_d)
                
                if st.button("🚀 PHÂN TÍCH", type="primary"):
                    score_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                    score_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    limits = {'l12': L12, 'l34': L34, 'l56': L56, 'mod': LMOD}
                    
                    res = None; err = None
                    
                    if STRATEGY_MODE == "🛡️ V24 Vote 8x":
                        res, err = calculate_vote_8x_strict(target_d, ROLLING_WINDOW, data_cache, kq_db, limits)
                    elif STRATEGY_MODE == "🛡️ V24 Cổ Điển":
                        res = calculate_v24_classic(target_d, ROLLING_WINDOW, data_cache, kq_db, limits, MIN_VOTES, score_std, score_mod, USE_INVERSE, MAX_TRIM)
                    elif STRATEGY_MODE == "⚔️ Gốc 3":
                        res = calculate_goc_3_logic(target_d, ROLLING_WINDOW, data_cache, kq_db, st.session_state['G3_INPUT'], st.session_state['G3_TARGET'], score_std, MIN_VOTES, USE_INVERSE)

                    if err: st.error(err)
                    elif res:
                        st.success("Đã tính toán xong!")
                        if "top6_std" in res: st.info(f"🏆 Top Nhóm: {', '.join(res['top6_std'])}")
                        if "top3" in res: st.info(f"🏆 Top 3: {', '.join(res['top3'])}")
                        
                        st.divider()
                        c1, c2 = st.columns(2)
                        with c1:
                            if "dan_goc" in res:
                                st.text_area(f"Dàn Gốc/LM ({len(res['dan_goc'])})", ",".join(res['dan_goc']), height=120)
                        with c2:
                            st.text_area(f"🔥 FINAL ({len(res['dan_final'])})", ",".join(res['dan_final']), height=120)
                        
                        if target_d in kq_db:
                            kq = kq_db[target_d]
                            st.markdown(f"### KQ: `{kq}`")
                            if kq in res['dan_final']: st.success("WIN Final")
                            else: st.error("MISS Final")

            # ------------------------------------------------------------------
            # TAB 2: BACKTEST
            # ------------------------------------------------------------------
            with tab2:
                c1, c2 = st.columns(2)
                with c1: d_start = st.date_input("Từ:", value=last_d - timedelta(days=5))
                with c2: d_end = st.date_input("Đến:", value=last_d)
                
                if st.button("▶️ CHẠY BACKTEST"):
                    logs = []
                    score_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                    score_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    limits = {'l12': L12, 'l34': L34, 'l56': L56, 'mod': LMOD}
                    
                    bar = st.progress(0)
                    days = [d_start + timedelta(days=x) for x in range((d_end - d_start).days + 1)]
                    
                    for i, d in enumerate(days):
                        bar.progress((i+1)/len(days))
                        if d not in kq_db: continue
                        
                        r = None
                        if STRATEGY_MODE == "🛡️ V24 Vote 8x":
                            r, _ = calculate_vote_8x_strict(d, ROLLING_WINDOW, data_cache, kq_db, limits)
                        elif STRATEGY_MODE == "🛡️ V24 Cổ Điển":
                            r = calculate_v24_classic(d, ROLLING_WINDOW, data_cache, kq_db, limits, MIN_VOTES, score_std, score_mod, USE_INVERSE, MAX_TRIM)
                        elif STRATEGY_MODE == "⚔️ Gốc 3":
                            r = calculate_goc_3_logic(d, ROLLING_WINDOW, data_cache, kq_db, st.session_state['G3_INPUT'], st.session_state['G3_TARGET'], score_std, MIN_VOTES, USE_INVERSE)
                        
                        if r:
                            kq = kq_db[d]
                            win = "✅" if kq in r['dan_final'] else "❌"
                            logs.append({"Ngày": d.strftime("%d/%m"), "KQ": kq, "Res": win, "Size": len(r['dan_final'])})
                            
                    if logs:
                        st.dataframe(pd.DataFrame(logs), use_container_width=True)

            # ------------------------------------------------------------------
            # TAB 3: MATRIX (ĐÃ KHÔI PHỤC)
            # ------------------------------------------------------------------
            with tab3:
                st.subheader("🎯 MATRIX QUANT HUNTER")
                c_m1, c_m2, c_m3 = st.columns([2, 1, 1])
                with c_m1:
                    mtx_strat = st.selectbox("Chiến thuật:", ["🔥 Săn M6-M9", "🛡️ Thủ M10", "💎 Elite 5", "👥 Top 10 File"])
                with c_m2:
                    mtx_cut = st.number_input("Lấy:", 40, step=5)
                with c_m3:
                    mtx_skip = st.number_input("Bỏ:", 0, step=5)
                
                target_mtx = st.date_input("Ngày soi Matrix:", value=last_d)
                
                if st.button("🚀 QUÉT MATRIX"):
                    if target_mtx in data_cache:
                        df_t = data_cache[target_mtx]['df']
                        
                        # Config Weights
                        if "Săn M6-M9" in mtx_strat:
                            w = [0,0,0,0,0,0,30,40,50,60,0]; top_n=10; sort_by='score'
                        elif "Thủ M10" in mtx_strat:
                            w = [0,0,0,0,0,0,0,0,0,0,60]; top_n=20; sort_by='score'
                        elif "Elite 5" in mtx_strat:
                            w = [0,0,5,10,15,25,30,35,40,50,60]; top_n=5; sort_by='score'
                        else:
                            w = [0,0,5,10,15,25,30,35,40,50,60]; top_n=10; sort_by='stt'
                            
                        elite_df = get_elite_members(df_t, top_n, sort_by)
                        st.dataframe(elite_df[['STT', 'MEMBER', 'SCORE_SORT'] if 'MEMBER' in elite_df.columns else elite_df.columns])
                        
                        ranked = calculate_matrix_simple(elite_df, w)
                        final_mtx = [f"{n:02d}" for n, s in ranked[mtx_skip:mtx_skip+mtx_cut]]
                        final_mtx.sort()
                        
                        st.text_area("Kết quả Matrix:", ",".join(final_mtx))
                        
                        if target_mtx in kq_db:
                            kq = kq_db[target_mtx]
                            rank = next((i+1 for i, (n,s) in enumerate(ranked) if f"{n:02d}" == kq), 999)
                            if kq in final_mtx: st.success(f"WIN (Hạng {rank})")
                            else: st.error(f"MISS (Hạng {rank})")

if __name__ == "__main__":
    main()
