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

# --- V24 LOGIC ---
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
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache: prev_date = target_date - timedelta(days=i); break
    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in _cache: col_hist_used = _cache[prev_date]['hist_map'].get(prev_date)
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
                s_p = get_col_score(col, score_std_tuple)
                if s_p > 0: d_p_map[col] = s_p
                s_s = get_col_score(col, score_mod_tuple)
                if s_s > 0: d_s_map[col] = s_s
            d_hist_col = None
            sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
            if sorted_dates: d_hist_col = _cache[d]['hist_map'][sorted_dates[0]]
            if not d_hist_col: continue
            try:
                hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False)
                hist_series_d = hist_series_d.str.replace(r'[^0-9X]', '', regex=True)
            except: continue
            for g in groups:
                mask = hist_series_d == g.upper()
                mems = d_df[mask]
                if mems.empty: stats_std[g]['ranks'].append(999); continue
                top80_std = fast_get_top_nums(mems, d_p_map, d_s_map, 80, min_votes, use_inverse)
                if kq in top80_std:
                    stats_std[g]['wins'] += 1; stats_std[g]['ranks'].append(top80_std.index(kq) + 1)
                else: stats_std[g]['ranks'].append(999)
                top86_mod = fast_get_top_nums(mems, d_s_map, d_p_map, int(limits_config['mod']), min_votes, use_inverse)
                if kq in top86_mod: stats_mod[g]['wins'] += 1
    top6_std = []; best_mod_grp = ""
    if not manual_groups:
        final_std = []
        for g, inf in stats_std.items(): final_std.append((g, -inf['wins'], sum(inf['ranks']), sorted(inf['ranks'])))
        final_std.sort(key=lambda x: (x[1], x[2], x[3], x[0])) 
        top6_std = [x[0] for x in final_std[:6]]
        best_mod_grp = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]
    hist_series = df[col_hist_used].astype(str).str.upper().replace('S', '6', regex=False)
    hist_series = hist_series.str.replace(r'[^0-9X]', '', regex=True)
    def get_final_pool(group_list, limit_dict, p_map, s_map):
        pool = []
        for g in group_list:
            mask = hist_series == g.upper(); valid_mems = df[mask]
            lim = limit_dict.get(g, limit_dict.get('default', 80))
            res = fast_get_top_nums(valid_mems, p_map, s_map, int(lim), min_votes, use_inverse)
            pool.extend(res)
        return pool
    final_original = []; final_modified = []
    if manual_groups:
        limit_map = {'default': limits_config['l12']}
        final_original = sorted(list(set(get_final_pool(manual_groups, limit_map, p_map_dict, s_map_dict))))
        final_modified = sorted(list(set(get_final_pool(manual_groups, {'default': limits_config['mod']}, s_map_dict, p_map_dict))))
    else:
        limits_std = {
            top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], 
            top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], 
            top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']
        }
        g_set1 = [top6_std[0], top6_std[5], top6_std[3]]
        pool1 = get_final_pool(g_set1, limits_std, p_map_dict, s_map_dict)
        s1 = {n for n, c in Counter(pool1).items() if c >= 2} 
        g_set2 = [top6_std[1], top6_std[4], top6_std[2]]
        pool2 = get_final_pool(g_set2, limits_std, p_map_dict, s_map_dict)
        s2 = {n for n, c in Counter(pool2).items() if c >= 2}
        final_original = sorted(list(s1.intersection(s2)))
        mask_mod = hist_series == best_mod_grp.upper()
        final_modified = sorted(fast_get_top_nums(df[mask_mod], s_map_dict, p_map_dict, int(limits_config['mod']), min_votes, use_inverse))
    intersect_list = list(set(final_original).intersection(set(final_modified)))
    if max_trim and len(intersect_list) > max_trim:
        temp_df = df.copy()
        melted = temp_df.melt(value_name='Val').dropna(subset=['Val'])
        mask_bad = ~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT', regex=True)
        melted = melted[mask_bad]
        s_nums = melted['Val'].astype(str).str.findall(r'\d+')
        exploded = melted.assign(Num=s_nums).explode('Num').dropna(subset=['Num'])
        exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
        exploded = exploded[exploded['Num'].isin(intersect_list)]
        exploded['Score'] = exploded['variable'].map(p_map_dict).fillna(0) + exploded['variable'].map(s_map_dict).fillna(0)
        final_scores = exploded.groupby('Num')['Score'].sum().reset_index()
        final_scores = final_scores.sort_values(by='Score', ascending=False)
        final_intersect = sorted(final_scores.head(int(max_trim))['Num'].tolist()) 
    else: final_intersect = sorted(intersect_list)
    return {
        "top6_std": top6_std, "best_mod": best_mod_grp, "dan_goc": final_original, 
        "dan_mod": final_modified, "dan_final": final_intersect, "source_col": col_hist_used
    }

# --- GỐC 3 LOGIC (CÓ SMART CUT) ---
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
    exploded['Score'] = exploded['variable'].map(p_map).fillna(0) 
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
    real_cols = df.columns
    p_map_dict = {}
    score_std_tuple = tuple(score_std.items())
    for col in real_cols:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map_dict[col] = s_p
    hist_series = df[col_hist].astype(str).str.upper().replace('S', '6', regex=False)
    hist_series = hist_series.str.replace(r'[^0-9X]', '', regex=True)
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
# ==============================================================================
# 3. GIAO DIỆN CHÍNH (MAIN APP) - PHẦN 2
# ==============================================================================

def main():
    uploaded_files = st.file_uploader("📂 Tải file CSV/Excel", type=['xlsx', 'csv'], accept_multiple_files=True)

    # LOAD CONFIG
    saved_cfg = load_config()
    if 'std_0' not in st.session_state:
        if saved_cfg:
            source = saved_cfg
            st.session_state['preset_choice'] = "Cấu hình đã lưu (Saved)"
        else:
            # Mặc định dùng Balanced
            source = SCORES_PRESETS["Balanced (Khuyên dùng 2026)"]
            source_flat = {}
            for i in range(11):
                source_flat[f'std_{i}'] = source['STD'][i]
                source_flat[f'mod_{i}'] = source['MOD'][i]
            source_flat['L12'] = source['LIMITS']['l12']
            source_flat['L34'] = source['LIMITS']['l34']
            source_flat['L56'] = source['LIMITS']['l56']
            source_flat['LMOD'] = source['LIMITS']['mod']
            source_flat['MAX_TRIM'] = 75 
            source_flat['ROLLING_WINDOW'] = source.get('ROLLING', 10)
            source_flat['MIN_VOTES'] = 1
            source_flat['USE_INVERSE'] = False
            source_flat['USE_ADAPTIVE'] = False
            source_flat['STRATEGY_MODE'] = "🛡️ V24 Cổ Điển"
            source_flat['G3_INPUT'] = 75
            source_flat['G3_TARGET'] = 70
            source = source_flat
        for k, v in source.items():
            if k in ['STD', 'MOD', 'LIMITS']: continue 
            st.session_state[k] = v

    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # --- MASTER SWITCH ---
        st.session_state['STRATEGY_MODE'] = st.radio(
            "🎯 CHỌN CHIẾN THUẬT:",
            ["🛡️ V24 Cổ Điển", "⚔️ Gốc 3 Bá Đạo"],
            index=0 if st.session_state.get('STRATEGY_MODE') == "🛡️ V24 Cổ Điển" else 1
        )
        STRATEGY_MODE = st.session_state['STRATEGY_MODE']
        st.markdown("---")

        def update_scores():
            choice = st.session_state.preset_choice
            if choice == "Cấu hình đã lưu (Saved)":
                cfg = load_config()
                if cfg:
                    for k, v in cfg.items(): st.session_state[k] = v
            elif choice in SCORES_PRESETS:
                vals = SCORES_PRESETS[choice]
                for i in range(11):
                    st.session_state[f'std_{i}'] = vals["STD"][i]
                    st.session_state[f'mod_{i}'] = vals["MOD"][i]
                if 'LIMITS' in vals:
                    st.session_state['L12'] = vals['LIMITS']['l12']
                    st.session_state['L34'] = vals['LIMITS']['l34']
                    st.session_state['L56'] = vals['LIMITS']['l56']
                    st.session_state['LMOD'] = vals['LIMITS']['mod']
                if 'ROLLING' in vals:
                    st.session_state['ROLLING_WINDOW'] = vals['ROLLING']

        menu_ops = ["Cấu hình đã lưu (Saved)"] + list(SCORES_PRESETS.keys()) if os.path.exists(CONFIG_FILE) else list(SCORES_PRESETS.keys())
        st.selectbox("📚 Chọn bộ mẫu:", options=menu_ops, index=1, key="preset_choice", on_change=update_scores)

        ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, key="ROLLING_WINDOW")
        
        # --- CẤU HÌNH ĐỘNG THEO CHẾ ĐỘ ---
        if STRATEGY_MODE == "🛡️ V24 Cổ Điển":
            with st.expander("✂️ Cắt Top V24", expanded=True):
                L_TOP_12 = st.number_input("Top 1 & 2 lấy:", step=1, key="L12")
                L_TOP_34 = st.number_input("Top 3 & 4 lấy:", step=1, key="L34")
                L_TOP_56 = st.number_input("Top 5 & 6 lấy:", step=1, key="L56")
                LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", step=1, key="LMOD")
            MAX_TRIM_NUMS = st.slider("🛡️ Max Trim Final:", 50, 90, key="MAX_TRIM")
        else:
            with st.expander("⚔️ Cắt Gốc 3", expanded=True):
                G3_INPUT = st.slider("Input Top (Lấy vào):", 60, 100, key="G3_INPUT")
                G3_TARGET = st.slider("Target (Giữ lại):", 50, 80, key="G3_TARGET")
            L_TOP_12=0; L_TOP_34=0; L_TOP_56=0; LIMIT_MODIFIED=0; MAX_TRIM_NUMS=75

        with st.expander("🎚️ 1. Điểm & Auto Limit", expanded=False):
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.write("**GỐC**")
                for i in range(11): st.number_input(f"M{i}", key=f"std_{i}")
            with c_s2:
                st.write("**MOD**")
                for i in range(11): st.number_input(f"M{i}", key=f"mod_{i}")

        st.markdown("---")
        with st.expander("👁️ Hiển thị (Dự Đoán)", expanded=True):
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                show_goc = st.checkbox("Hiện Gốc", value=True)
                show_mod = st.checkbox("Hiện Mod", value=False)
            with c_v2:
                show_final = st.checkbox("Hiện Final", value=True)
                show_hybrid = st.checkbox("Hiện Hybrid", value=True)

        MIN_VOTES = st.number_input("Vote tối thiểu:", min_value=1, max_value=10, key="MIN_VOTES")
        USE_INVERSE = st.checkbox("Chấm Điểm Đảo (Ngược)", key="USE_INVERSE")
        
        st.markdown("---")
        st.session_state['USE_ADAPTIVE'] = st.checkbox("🧠 Kích hoạt M Động (Adaptive)", value=st.session_state.get('USE_ADAPTIVE', False))
        USE_ADAPTIVE = st.session_state['USE_ADAPTIVE']

        if st.button("💾 LƯU CẤU HÌNH", type="secondary", use_container_width=True):
            save_data = {}
            for i in range(11):
                save_data[f'std_{i}'] = st.session_state[f'std_{i}']
                save_data[f'mod_{i}'] = st.session_state[f'mod_{i}']
            save_data.update({
                'L12': st.session_state['L12'], 'L34': st.session_state['L34'],
                'L56': st.session_state['L56'], 'LMOD': st.session_state['LMOD'],
                'MAX_TRIM': st.session_state['MAX_TRIM'], 'ROLLING_WINDOW': st.session_state['ROLLING_WINDOW'],
                'MIN_VOTES': st.session_state['MIN_VOTES'], 'USE_INVERSE': st.session_state['USE_INVERSE'],
                'USE_ADAPTIVE': st.session_state['USE_ADAPTIVE'],
                'STRATEGY_MODE': st.session_state['STRATEGY_MODE'],
                'G3_INPUT': st.session_state.get('G3_INPUT', 75),
                'G3_TARGET': st.session_state.get('G3_TARGET', 70)
            })
            if save_config(save_data): st.success("Đã lưu!"); time.sleep(1); st.rerun()
        
        if st.button("🗑️ XÓA CACHE", type="primary"): st.cache_data.clear(); st.rerun()

    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        with st.expander("🕵️ Trạng thái File & Debug", expanded=False):
            for s in f_status: st.success(s)
            for e in err_logs: st.error(e)
            if data_cache:
                debug_date = st.selectbox("Kiểm tra ngày:", sorted(data_cache.keys(), reverse=True))
                if debug_date: st.dataframe(data_cache[debug_date]['df'].head(5))
        
        if data_cache:
            last_d = max(data_cache.keys())
            tab1, tab2, tab3 = st.tabs(["📊 DỰ ĐOÁN (ANALYSIS)", "🔙 BACKTEST", "🎯 MATRIX"])
            
            # --- TAB 1: PREDICTION (NÂNG CẤP HYBRID ĐỘNG) ---
            with tab1:
                st.subheader(f"Dự đoán: {STRATEGY_MODE}")
                if USE_ADAPTIVE: st.info("🧠 M Động: BẬT")
                c_d1, c_d2 = st.columns([1, 1])
                with c_d1: target = st.date_input("Ngày:", value=last_d)

                if st.button("🚀 CHẠY PHÂN TÍCH & SOI HYBRID", type="primary", use_container_width=True):
                    with st.spinner("Đang tính toán..."):
                        base_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                        base_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                        
                        if USE_ADAPTIVE:
                            curr_std = get_adaptive_weights(target, base_std, data_cache, kq_db, window=3, factor=1.5)
                            curr_mod = get_adaptive_weights(target, base_mod, data_cache, kq_db, window=3, factor=1.5)
                        else: curr_std, curr_mod = base_std, base_mod

                        # 1. Chạy cấu hình chính (Màn hình) -> Đây là "res_curr"
                        if STRATEGY_MODE == "🛡️ V24 Cổ Điển":
                            user_limits = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
                            res_curr, err_curr = calculate_v24_final(target, ROLLING_WINDOW, data_cache, kq_db, user_limits, MIN_VOTES, curr_std, curr_mod, USE_INVERSE, None, max_trim=MAX_TRIM_NUMS)
                        else: # Gốc 3
                            g3_res = calculate_goc_3_logic(target, ROLLING_WINDOW, data_cache, kq_db, st.session_state['G3_INPUT'], st.session_state['G3_TARGET'], curr_std, USE_INVERSE, MIN_VOTES)
                            if g3_res:
                                res_curr = {'dan_goc': g3_res['dan_final'], 'dan_mod': [], 'dan_final': g3_res['dan_final'], 'source_col': g3_res['source_col']}
                                err_curr = None
                            else: res_curr=None; err_curr="Lỗi"

                        # 2. Chạy Hard Core (Gốc) Cố định để làm trụ
                        s_hc, m_hc, l_hc, r_hc = get_preset_params("Hard Core (Gốc)")
                        if USE_ADAPTIVE: s_hc = get_adaptive_weights(target, s_hc, data_cache, kq_db, 3, 1.5)
                        res_hc = calculate_v24_logic_only(target, r_hc, data_cache, kq_db, l_hc, MIN_VOTES, s_hc, m_hc, USE_INVERSE, None, max_trim=MAX_TRIM_NUMS)
                        
                        # 3. TÍNH HYBRID ĐỘNG (Dynamic Intersection)
                        # Hybrid = Giao của [Hard Core] + [Dàn Hiện Tại trên Màn Hình]
                        # Điều này đảm bảo khi bạn chỉnh màn hình, Hybrid sẽ thay đổi theo.
                        hybrid_goc = []
                        hc_goc = []
                        screen_goc = []
                        
                        if res_hc and res_curr:
                            hc_goc = res_hc['dan_goc']
                            screen_goc = res_curr['dan_goc'] # Dàn bạn đang chỉnh (CH1 hoặc bất cứ gì)
                            hybrid_goc = sorted(list(set(hc_goc).intersection(set(screen_goc))))

                        st.session_state['run_result'] = {
                            'res_curr': res_curr, 'target': target, 'err': err_curr,
                            'hc_goc': hc_goc, 'screen_goc': screen_goc, 'hybrid_goc': hybrid_goc
                        }

                if 'run_result' in st.session_state and st.session_state['run_result']['target'] == target:
                    rr = st.session_state['run_result']; res = rr['res_curr']
                    if not rr['err']:
                        st.info(f"Phân nhóm nguồn: {res['source_col']}")
                        
                        cols_main = []
                        t_lbl = "Gốc 3" if STRATEGY_MODE == "⚔️ Gốc 3 Bá Đạo" else "Gốc V24 (Màn Hình)"
                        if show_goc: cols_main.append({"t": f"{t_lbl} ({len(res['dan_goc'])})", "d": res['dan_goc']})
                        if show_final: cols_main.append({"t": f"Final ({len(res['dan_final'])})", "d": res['dan_final']})
                        
                        if cols_main:
                            c_m = st.columns(len(cols_main))
                            for i, o in enumerate(cols_main):
                                with c_m[i]: st.text_area(o['t'], ",".join(o['d']), height=100)
                        
                        st.divider()
                        st.write("#### 🧬 Phân Tích Hybrid (Hard Core + Màn Hình)")
                        st.caption("Dàn Hybrid này là giao thoa giữa **Hard Core (Gốc)** và **Cấu hình bạn đang chỉnh**.")
                        
                        c_h1, c_h2, c_h3 = st.columns(3)
                        with c_h1: st.text_area(f"Hard Core (Trụ) ({len(rr['hc_goc'])})", ",".join(rr['hc_goc']), height=100)
                        with c_h2: st.text_area(f"Màn Hình (Biến) ({len(rr['screen_goc'])})", ",".join(rr['screen_goc']), height=100)
                        with c_h3: st.text_area(f"⚔️ HYBRID ĐỘNG ({len(rr['hybrid_goc'])})", ",".join(rr['hybrid_goc']), height=100)

                        if target in kq_db:
                            real = kq_db[target]
                            st.markdown("### 🏁 KẾT QUẢ")
                            c_r1, c_r2, c_r3 = st.columns(3)
                            with c_r1: st.metric("KQ", real)
                            with c_r2:
                                if real in res['dan_final']: st.success("Dàn chính: WIN")
                                else: st.error("Dàn chính: MISS")
                            with c_r3:
                                if real in rr['hybrid_goc']: st.success("Hybrid: WIN")
                                else: st.error("Hybrid: MISS")

            # --- TAB 2: BACKTEST (SINGLE MODE) ---
            with tab2:
                st.subheader("🔙 Backtest Chi Tiết (Single Mode)")
                
                c_bt_1, c_bt_2 = st.columns([1, 2])
                with c_bt_1:
                    cfg_options = ["Màn hình hiện tại"] + list(SCORES_PRESETS.keys()) + ["Gốc 3 (Test Input 75/Target 70)", "⚔️ Hybrid: HC(Gốc) + CH1(Gốc)"]
                    selected_cfg = st.selectbox("Chọn Cấu Hình Backtest:", cfg_options)
                    st.write("---")
                    use_adaptive_bt = st.checkbox("🧠 Bật M Động (Adaptive)", value=False)
                
                with c_bt_2:
                    d_start = st.date_input("Từ ngày:", value=last_d - timedelta(days=10), key="bt_d1")
                    d_end = st.date_input("Đến ngày:", value=last_d, key="bt_d2")
                    btn_run_bt = st.button("▶️ CHẠY BACKTEST", type="primary", use_container_width=True)

                if btn_run_bt:
                    if d_start > d_end: st.error("Ngày bắt đầu > Ngày kết thúc")
                    else:
                        dates_range = [d_start + timedelta(days=i) for i in range((d_end - d_start).days + 1)]
                        logs = []; bar = st.progress(0)
                        
                        def check_win(kq, arr): return "✅" if kq in arr else "❌"

                        for idx, d in enumerate(dates_range):
                            bar.progress((idx + 1) / len(dates_range))
                            if d not in kq_db: continue
                            real_kq = kq_db[d]
                            row = {"Ngày": d.strftime("%d/%m"), "KQ": real_kq}
                            
                            if selected_cfg == "⚔️ Hybrid: HC(Gốc) + CH1(Gốc)":
                                s_hc, m_hc, l_hc, r_hc = get_preset_params("Hard Core (Gốc)")
                                s_ch1, m_ch1, l_ch1, r_ch1 = get_preset_params("CH1: Bám Đuôi (Gốc)")
                                if use_adaptive_bt:
                                    s_hc = get_adaptive_weights(d, s_hc, data_cache, kq_db, 3, 1.5)
                                    s_ch1 = get_adaptive_weights(d, s_ch1, data_cache, kq_db, 3, 1.5)
                                res_hc = calculate_v24_logic_only(d, r_hc, data_cache, kq_db, l_hc, MIN_VOTES, s_hc, m_hc, USE_INVERSE, None, max_trim=MAX_TRIM_NUMS)
                                res_ch1 = calculate_v24_logic_only(d, r_ch1, data_cache, kq_db, l_ch1, MIN_VOTES, s_ch1, m_ch1, USE_INVERSE, None, max_trim=MAX_TRIM_NUMS)
                                if res_hc and res_ch1:
                                    fin_hc = res_hc['dan_goc']; fin_ch1 = res_ch1['dan_goc']
                                    fin_hyb = sorted(list(set(fin_hc).intersection(set(fin_ch1))))
                                    row.update({"HC Gốc": f"{check_win(real_kq, fin_hc)} ({len(fin_hc)})", "CH1 Gốc": f"{check_win(real_kq, fin_ch1)} ({len(fin_ch1)})", "Hybrid": f"{check_win(real_kq, fin_hyb)} ({len(fin_hyb)})"})
                                    logs.append(row)
                            else:
                                run_s = {}; run_m = {}; run_l = {}; run_r = 10; is_goc3 = False
                                if selected_cfg == "Màn hình hiện tại":
                                    run_s = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                                    run_m = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                                    if STRATEGY_MODE == "🛡️ V24 Cổ Điển": run_l = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
                                    else: is_goc3 = True; inp = st.session_state.get('G3_INPUT', 75); tar = st.session_state.get('G3_TARGET', 70)
                                    run_r = ROLLING_WINDOW
                                elif selected_cfg == "Gốc 3 (Test Input 75/Target 70)":
                                    is_goc3 = True; run_s = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}; inp = 75; tar = 70; run_r = ROLLING_WINDOW
                                elif selected_cfg in SCORES_PRESETS:
                                    run_s, run_m, run_l, run_r = get_preset_params(selected_cfg)
                                
                                if use_adaptive_bt:
                                    run_s = get_adaptive_weights(d, run_s, data_cache, kq_db, 3, 1.5)
                                    if not is_goc3: run_m = get_adaptive_weights(d, run_m, data_cache, kq_db, 3, 1.5)
                                
                                if is_goc3:
                                    res = calculate_goc_3_logic(d, run_r, data_cache, kq_db, inp, tar, run_s, USE_INVERSE, MIN_VOTES)
                                    if res:
                                        fin = res['dan_final']
                                        row.update({"Gốc 3": f"{check_win(real_kq, fin)} ({len(fin)})", "Final": f"{check_win(real_kq, fin)} ({len(fin)})"})
                                        logs.append(row)
                                else:
                                    res = calculate_v24_logic_only(d, run_r, data_cache, kq_db, run_l, MIN_VOTES, run_s, run_m, USE_INVERSE, None, max_trim=MAX_TRIM_NUMS)
                                    if res:
                                        row.update({"Gốc": f"{check_win(real_kq, res['dan_goc'])} ({len(res['dan_goc'])})", "Mod": f"{check_win(real_kq, res['dan_mod'])} ({len(res['dan_mod'])})", "Final": f"{check_win(real_kq, res['dan_final'])} ({len(res['dan_final'])})"})
                                        logs.append(row)

                        bar.empty()
                        if logs:
                            df_log = pd.DataFrame(logs)
                            st.dataframe(df_log, use_container_width=True) 
                            st.write(f"### 📊 Thống kê: {selected_cfg}")
                            cols_calc = [c for c in df_log.columns if c not in ["Ngày", "KQ"]]
                            st_cols = st.columns(len(cols_calc))
                            for i, c_name in enumerate(cols_calc):
                                wins = df_log[c_name].astype(str).apply(lambda x: 1 if "✅" in x else 0).sum()
                                sizes = df_log[c_name].astype(str).apply(lambda x: int(re.search(r'\((\d+)\)', x).group(1)) if re.search(r'\((\d+)\)', x) else 0)
                                avg_size = sizes.mean() if not sizes.empty else 0
                                with st_cols[i]:
                                    st.metric(f"{c_name}", f"{wins}/{len(df_log)} ({wins/len(df_log)*100:.1f}%)", f"TB: {avg_size:.1f}")

            # --- TAB 3: MATRIX (GIỮ NGUYÊN) ---
            with tab3:
                st.subheader("🎯 MA TRẬN CHIẾN LƯỢC: QUANT HUNTER")
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1.5, 1.5, 1])
                    with c1:
                        strategy_mode = st.selectbox("⚔️ Chọn Chiến Thuật:", ["🔥 Mid-Range Focus (Săn M6-M9)", "🛡️ Storm Shelter (Thủ M10 Only)", "💎 Elite 5 (Vip Mode)", "👥 Đại Trà (Top 10 File)"], index=0)
                    with c2:
                        if "Săn M6-M9" in strategy_mode:
                            current_weights = [0, 0, 0, 0, 0, 0, 30, 40, 50, 60, 0]
                            top_n_select = 10; filter_mode = 'score'; def_cut = 40; def_skip = 0
                            st.success("⚡ M6-M9 Focus")
                        elif "Thủ M10" in strategy_mode:
                            current_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60]
                            top_n_select = 20; filter_mode = 'score'; def_cut = 80; def_skip = 0
                            st.warning("🛡️ M10 Defense")
                        elif "Elite 5" in strategy_mode:
                            current_weights = [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60]
                            top_n_select = 5; filter_mode = 'score'; def_cut = 65; def_skip = 0
                            st.info("💎 Top 5 Elite")
                        else:
                            current_weights = [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60]
                            top_n_select = 10; filter_mode = 'stt'; def_cut = 65; def_skip = 0
                            st.info("🔹 Top 10 File")
                    with c3:
                        cut_val = st.number_input("✂️ Lấy:", value=def_cut, step=5)
                        skip_val = st.number_input("🚫 Bỏ:", value=def_skip, step=5)
                        target_matrix_date = st.date_input("Chọn ngày soi:", value=last_d, key="matrix_date")
                        btn_scan = st.button("🚀 QUÉT SỐ", type="primary", use_container_width=True)

                if btn_scan:
                    target_d = target_matrix_date
                    st.write(f"📅 Ngày: **{target_d.strftime('%d/%m/%Y')}**")
                    if target_d in data_cache:
                        df_target = data_cache[target_d]['df']
                        with st.spinner("Đang xử lý..."):
                            input_df = get_elite_members(df_target, top_n=top_n_select, sort_by=filter_mode)
                            with st.expander("📋 Danh sách Cao thủ"):
                                st.dataframe(input_df[['STT', 'MEMBER', 'SCORE_SORT'] if 'MEMBER' in input_df.columns else input_df.columns], use_container_width=True)
                            ranked_numbers = calculate_matrix_simple(input_df, current_weights)
                            start_idx = skip_val; end_idx = skip_val + cut_val
                            final_set = [n for n, score in ranked_numbers[start_idx:end_idx]]
                            final_set.sort()
                            st.divider()
                            st.text_area("👇 Dàn số:", value=",".join([f"{n:02d}" for n in final_set]), height=100)
                            col_s1, col_s2 = st.columns(2)
                            with col_s1: st.metric("Số lượng", f"{len(final_set)}")
                            with col_s2:
                                if target_d in kq_db:
                                    real = kq_db[target_d]
                                    real_int = int(real)
                                    rank = next((i+1 for i, (n,s) in enumerate(ranked_numbers) if n == real_int), 999)
                                    if start_idx < rank <= end_idx: st.success(f"WIN: {real} (Hạng {rank})")
                                    else: st.error(f"MISS: {real} (Hạng {rank})")

if __name__ == "__main__":
    main()
