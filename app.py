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
# 1. CẤU HÌNH HỆ THỐNG & PRESETS (ĐẦY ĐỦ)
# ==============================================================================
st.set_page_config(
    page_title="Ly Thong V62 - The Original Ultimate", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🛡️ Lý Thị Thông: V62 ULTIMATE (FULL OPTION)")
st.caption("🚀 Nguyên bản: Matrix | Gốc 3 | Auto-M | Vote 8x (Fix chuẩn 63 số)")

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
    },
    "Vote 8x (Chuẩn)": {
        "STD": [0]*11, "MOD": [0]*11,
        "LIMITS": {'l12': 80, 'l34': 70, 'l56': 60, 'mod': 80},
        "ROLLING": 10
    }
}

RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CORE UTILS & HELPERS
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

# --- TÍNH NĂNG CAO CẤP: AUTO-CALIBRATION ---
def calculate_auto_weights_from_data(target_date, data_cache, kq_db, lookback=10):
    m_performance = {i: 0 for i in range(11)} 
    check_date = target_date - timedelta(days=1)
    past_days = []
    while len(past_days) < lookback:
        if check_date in data_cache and check_date in kq_db:
            past_days.append(check_date)
        check_date -= timedelta(days=1)
        if (target_date - check_date).days > 60: break 
    if not past_days:
        return {f'M{i}': 10 for i in range(11)} 

    for d in past_days:
        real_kq = str(kq_db[d]).zfill(2)
        df = data_cache[d]['df']
        cols = [c for c in df.columns if re.match(r'^M\s*\d+', c) or c in ['M10', 'M 1 0']]
        m_map = {}
        for c in cols:
            clean = c.replace(' ', '').replace('M', '')
            try: idx = int(clean); m_map[c] = idx
            except: pass
        for c_name, m_idx in m_map.items():
            all_nums = []
            for val in df[c_name].dropna():
                all_nums.extend(get_nums(val))
            if real_kq in all_nums:
                m_performance[m_idx] += 1

    sorted_m = sorted(m_performance.items(), key=lambda x: x[1], reverse=True)
    ranking_scores = [60, 50, 40, 30, 25, 20, 15, 10, 5, 0, 0]
    final_weights = {}
    for rank, (m_idx, count) in enumerate(sorted_m):
        score = ranking_scores[rank] if rank < len(ranking_scores) else 0
        final_weights[f'M{m_idx}'] = score
    return final_weights

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

# ==============================================================================
# 3. CORE LOGIC FUNCTIONS (V24, 8X, MATRIX, GOC 3)
# ==============================================================================

# --- A. HÀM CHO V24 CỔ ĐIỂN & GỐC 3 (GIỮ NGUYÊN CODE CŨ) ---
def fast_get_top_nums_score(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
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
    stats['V'] = exploded.reset_index().groupby('Num')['index'].nunique()
    stats = stats[stats['V'] >= min_v].reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    # Sort by Score (P)
    stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[True, True, True] if inverse else [False, False, True])
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
    exploded['Score'] = exploded['variable'].map(p_map).fillna(0) 
    final_scores = exploded.groupby('Num')['Score'].sum().reset_index()
    final_scores = final_scores.sort_values(by='Score', ascending=False)
    return sorted(final_scores.head(int(target_size))['Num'].tolist())

# --- B. HÀM CHO VOTE 8X (MỚI - TÁCH BIỆT) ---
def get_top_nums_by_vote(df_members, col_name, limit):
    if df_members.empty: return []
    all_nums = []
    # Chỉ lấy cột 8x
    vals = df_members[col_name].dropna().astype(str).tolist()
    for val in vals:
        if any(kw in val.upper() for kw in BAD_KEYWORDS): continue
        all_nums.extend(get_nums(val))
    counts = Counter(all_nums)
    # Sort: Vote cao -> thấp, Số bé -> lớn
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], int(x[0])))
    return [n for n, c in sorted_items[:int(limit)]]

# --- 1. THUẬT TOÁN: V24 CỔ ĐIỂN (CLASSIC) ---
def calculate_v24_classic(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, max_trim=None):
    if target_date not in _cache: return None, "No data"
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
    if not col_hist: return None, "No Hist Column"

    # Backtest Classic
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
            s_p = get_col_score(col, score_std_tuple)
            if s_p > 0: d_p_map[col] = s_p
            s_s = get_col_score(col, score_mod_tuple)
            if s_s > 0: d_s_map[col] = s_s
        sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
        d_hist_col = _cache[d]['hist_map'][sorted_dates[0]] if sorted_dates else None
        if not d_hist_col: continue
        try:
            hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
            for g in groups:
                mems = d_df[hist_series_d == g.upper()]
                if mems.empty: stats_std[g]['ranks'].append(999); continue
                top80 = fast_get_top_nums_score(mems, d_p_map, d_s_map, 80, min_votes, use_inverse)
                if kq in top80: stats_std[g]['wins']+=1; stats_std[g]['ranks'].append(top80.index(kq)+1)
                else: stats_std[g]['ranks'].append(999)
                top86_mod = fast_get_top_nums_score(mems, d_s_map, d_p_map, int(limits_config['mod']), min_votes, use_inverse)
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
            res = fast_get_top_nums_score(df[hist_series==g.upper()], p_map_dict, s_map_dict, lim_dict.get(g, 80), min_votes, use_inverse)
            p.extend(res)
        return p
    
    lim_map = {top6[0]: limits_config['l12'], top6[1]: limits_config['l12'], top6[2]: limits_config['l34'], top6[3]: limits_config['l34'], top6[4]: limits_config['l56'], top6[5]: limits_config['l56']}
    s1 = {n for n, c in Counter(get_pool([top6[0], top6[4], top6[2]], lim_map)).items() if c>=2}
    s2 = {n for n, c in Counter(get_pool([top6[1], top6[3], top6[5]], lim_map)).items() if c>=2}
    dan_goc = sorted(list(s1.intersection(s2)))
    
    dan_mod = sorted(fast_get_top_nums_score(df[hist_series==best_mod.upper()], s_map_dict, p_map_dict, int(limits_config['mod']), min_votes, use_inverse))
    final = sorted(list(set(dan_goc).intersection(set(dan_mod))))
    
    if max_trim and len(final) > max_trim:
        final = smart_trim_by_score(final, df, p_map_dict, s_map_dict, max_trim)
        
    return {"top6_std": top6, "best_mod": best_mod, "dan_goc": dan_goc, "dan_final": final, "source_col": col_hist}, None

# --- 2. THUẬT TOÁN: VOTE 8X (FIX CHUẨN 63 SỐ) ---
def calculate_vote_8x_strict(target_date, rolling_window, _cache, _kq_db, limits_config):
    if target_date not in _cache: return None, "No data"
    curr_data = _cache[target_date]; df = curr_data['df']
    
    # 1. Tìm cột 8X
    col_8x = next((c for c in df.columns if re.match(r'^(8X|80|DÀN|DAN)$', c.strip().upper()) or '8X' in c.strip().upper()), None)
    if not col_8x: return None, "Thiếu cột 8X"

    # 2. Tìm cột Phân Nhóm
    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache: prev_date = target_date - timedelta(days=i); break
    
    col_group = curr_data['hist_map'].get(prev_date)
    if not col_group and prev_date in _cache: col_group = _cache[prev_date]['hist_map'].get(prev_date)
    if not col_group: return None, "Thiếu cột phân nhóm"

    # 3. BACKTEST (Tìm Top 6)
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
                        stats[g]['wins'] += 1; stats[g]['ranks'].append(top80.index(kq))
                    else: stats[g]['ranks'].append(999)
            except: continue

    final_rank = []
    for g, inf in stats.items(): final_rank.append((g, -inf['wins'], sum(inf['ranks'])))
    final_rank.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_rank[:6]]

    # 4. FINAL CUT (2 LIÊN MINH - KHÔNG MOD)
    hist_series = df[col_group].astype(str).str.upper().str.replace('S', '6').str.replace(r'[^0-9X]', '', regex=True)
    
    # LM1: Top 1, 5, 3
    pool1 = []
    pool1 += get_top_nums_by_vote(df[hist_series == top6[0].upper()], col_8x, limits_config['l12'])
    pool1 += get_top_nums_by_vote(df[hist_series == top6[4].upper()], col_8x, limits_config['l56'])
    pool1 += get_top_nums_by_vote(df[hist_series == top6[2].upper()], col_8x, limits_config['l34'])
    s1 = {n for n, c in Counter(pool1).items() if c >= 2}

    # LM2: Top 2, 4, 6
    pool2 = []
    pool2 += get_top_nums_by_vote(df[hist_series == top6[1].upper()], col_8x, limits_config['l12'])
    pool2 += get_top_nums_by_vote(df[hist_series == top6[3].upper()], col_8x, limits_config['l34'])
    pool2 += get_top_nums_by_vote(df[hist_series == top6[5].upper()], col_8x, limits_config['l56'])
    s2 = {n for n, c in Counter(pool2).items() if c >= 2}

    # Giao thoa
    final_dan = sorted(list(s1.intersection(s2)))

    return {
        "top6_std": top6,
        "dan_goc": final_dan,
        "dan_final": final_dan, 
        "source_col": col_group,
        "debug_s1": len(s1), "debug_s2": len(s2)
    }, None

# --- 3. THUẬT TOÁN: GỐC 3 ---
def calculate_goc_3_logic(target_date, rolling_window, _cache, _kq_db, input_limit, target_limit, score_std, min_votes, use_inverse):
    dummy_lim = {'l12':1, 'l34':1, 'l56':1, 'mod':1}
    res_v24, _ = calculate_v24_classic(target_date, rolling_window, _cache, _kq_db, dummy_lim, min_votes, score_std, score_std, use_inverse)
    if not res_v24: return None
    
    top3 = res_v24['top6_std'][:3]
    col_hist = res_v24['source_col']
    curr_data = _cache[target_date]; df = curr_data['df']
    p_map = {}
    score_std_tuple = tuple(score_std.items())
    for col in df.columns:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map[col] = s_p
        
    hist_series = df[col_hist].astype(str).str.upper().replace('S', '6').str.replace(r'[^0-9X]', '', regex=True)
    
    pool_sets = []
    for g in top3:
        mems = df[hist_series == g.upper()]
        res = fast_get_top_nums_score(mems, p_map, p_map, int(input_limit), min_votes, use_inverse)
        pool_sets.append(res)
    
    all_nums = []
    for s in pool_sets: all_nums.extend(s)
    overlap_nums = [n for n, c in Counter(all_nums).items() if c >= 2]
    final_set = smart_trim_by_score(overlap_nums, df, p_map, {}, target_limit)
    return {"top3": top3, "dan_final": final_set, "source_col": col_hist}

# --- 4. THUẬT TOÁN: MATRIX & ELITE ---
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
# ==============================================================================
# 4. HÀM LOAD DỮ LIỆU (ĐỌC FILE EXCEL/CSV)
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_data_v24(files):
    cache = {}; kq_db = {}; file_status = []; err_logs = []
    files = sorted(files, key=lambda x: x.name)
    
    for file in files:
        # Bỏ qua các file tạm/rác
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper(): continue
        f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
        
        try:
            dfs = []
            # --- XỬ LÝ FILE EXCEL ---
            if file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(file, engine='openpyxl')
                for sheet in xls.sheet_names:
                    s_date = None
                    try:
                        clean_s = re.sub(r'[^0-9]', ' ', sheet).strip()
                        parts = [int(x) for x in clean_s.split()]
                        if parts: 
                            d_s, m_s = parts[0], f_m
                            y_s = parts[2] if len(parts)>=3 and parts[2]>2000 else f_y
                            s_date = datetime.date(y_s, m_s, d_s)
                    except: pass
                    
                    if not s_date: s_date = date_from_name
                    
                    if s_date:
                        # Tìm dòng tiêu đề (header row)
                        preview = pd.read_excel(xls, sheet_name=sheet, nrows=30, header=None, engine='openpyxl')
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row, engine='openpyxl')
                        dfs.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")

            # --- XỬ LÝ FILE CSV ---
            elif file.name.endswith('.csv'):
                if not date_from_name: continue
                # Thử nhiều encoding để tránh lỗi phông chữ
                encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
                df_raw = None; h_row = 0
                for enc in encodings:
                    try:
                        file.seek(0)
                        preview = pd.read_csv(file, header=None, nrows=30, encoding=enc)
                        h_row = find_header_row(preview)
                        file.seek(0)
                        df_raw = pd.read_csv(file, header=None, encoding=enc)
                        break
                    except: continue
                
                if df_raw is not None:
                    # Xử lý trường hợp tên cột bị trùng (ví dụ M 1 0)
                    df = df_raw.iloc[h_row+1:].copy()
                    raw_cols = df_raw.iloc[h_row].astype(str).tolist()
                    seen = {}; final_cols = []
                    for c in raw_cols:
                        c = str(c).strip().upper().replace('M 1 0', 'M10')
                        if c in seen: seen[c] += 1; final_cols.append(f"{c}.{seen[c]}")
                        else: seen[c] = 0; final_cols.append(c)
                    df.columns = final_cols
                    dfs.append((date_from_name, df))
                    file_status.append(f"✅ CSV: {file.name}")
                else:
                    err_logs.append(f"❌ Lỗi Encoding: {file.name}")

            # --- XỬ LÝ DATAFRAME SAU KHI LOAD ---
            for t_date, df in dfs:
                # 1. Làm sạch tên cột
                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]
                
                # 2. Tạo cột Score Sort (Để xếp hạng Matrix)
                score_col = next((c for c in df.columns if 'Đ9' in c or 'DIEM' in c or 'ĐIỂM' in c), None)
                if score_col: df['SCORE_SORT'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
                else: df['SCORE_SORT'] = 0
                
                # 3. Chuẩn hóa tên các cột M0-M10
                rename_map = {}
                for c in df.columns:
                    clean_c = c.replace(" ", "")
                    if re.match(r'^M\d+$', clean_c) or clean_c == 'M10': rename_map[c] = clean_c
                if rename_map: df = df.rename(columns=rename_map)

                # 4. Tìm kết quả xổ số (KQ) và Map các cột lịch sử
                hist_map = {}
                kq_row = None
                if not df.empty:
                    # Quét 2 cột đầu để tìm dòng chứa chữ "KQ" hoặc "KẾT QUẢ"
                    for c_idx in range(min(2, len(df.columns))):
                        col_check = df.columns[c_idx]
                        if df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ').any():
                            kq_row = df[df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')].iloc[0]
                            break
                
                for col in df.columns:
                    # Bỏ qua các cột không phải ngày tháng
                    if "UNNAMED" in col or col.startswith("M") or col in ["STT", "SCORE_SORT"]: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: 
                        hist_map[d_obj] = col
                        # Lưu KQ vào database nếu tìm thấy
                        if kq_row is not None:
                            try:
                                nums = get_nums(str(kq_row[col]))
                                if nums: kq_db[d_obj] = nums[0]
                            except: pass
                
                cache[t_date] = {'df': df, 'hist_map': hist_map}
                
        except Exception as e: err_logs.append(f"Lỗi file '{file.name}': {str(e)}"); continue
        
    return cache, kq_db, file_status, err_logs

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (MAIN APP)
# ==============================================================================

def main():
    uploaded_files = st.file_uploader("📂 Tải file dữ liệu (Excel/CSV)", type=['xlsx', 'csv'], accept_multiple_files=True)
    
    # --- KHỞI TẠO SESSION STATE (Lưu cấu hình) ---
    if 'L12' not in st.session_state:
        st.session_state.update({
            'L12':80, 'L34':70, 'L56':60, 'LMOD':80, 
            'ROLLING':10, 'STRATEGY':'Vote 8x (Chuẩn)', 
            'G3_IN':75, 'G3_OUT':70,
            'USE_AUTO_WEIGHTS': False, 'AUTO_LOOKBACK': 10
        })
        for i in range(11): st.session_state[f'std_{i}'] = 0; st.session_state[f'mod_{i}'] = 0

    # --- SIDEBAR: CẤU HÌNH ---
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # 1. Chọn chiến thuật
        st.session_state['STRATEGY'] = st.radio(
            "🎯 CHIẾN THUẬT:", 
            ["Vote 8x (Chuẩn)", "V24 Cổ Điển", "Gốc 3", "Matrix"]
        )
        STRAT = st.session_state['STRATEGY']
        
        if STRAT == "Vote 8x (Chuẩn)":
            st.success("✅ 8X Mode: Lấy 8X -> Top 6 -> Giao Thoa 2 LM (Bỏ Mod).")
        
        # 2. Load bộ mẫu (Presets)
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

        st.selectbox("📚 Bộ Mẫu Cấu Hình:", list(SCORES_PRESETS.keys()), key="preset_choice", on_change=update_scores)
        
        st.markdown("---")
        
        # 3. Các tham số chi tiết
        st.session_state['ROLLING'] = st.number_input("Backtest (Ngày):", value=st.session_state['ROLLING'])
        
        # Auto-Calibration (Chỉ hiện cho V24 Classic/Gốc 3)
        if STRAT in ["V24 Cổ Điển", "Gốc 3"]:
            st.session_state['USE_AUTO_WEIGHTS'] = st.checkbox("🤖 Auto-Calibration (Tự động điểm M)", value=st.session_state['USE_AUTO_WEIGHTS'])
            if st.session_state['USE_AUTO_WEIGHTS']:
                st.session_state['AUTO_LOOKBACK'] = st.number_input("Lookback Auto:", value=10)

        # Cấu hình Cắt Số (Limits)
        with st.expander("✂️ Cấu hình Cắt Số (Limits)", expanded=True):
            st.session_state['L12'] = st.number_input("Top 1 & 2:", value=st.session_state['L12'], step=1)
            st.session_state['L34'] = st.number_input("Top 3 & 4:", value=st.session_state['L34'], step=1)
            st.session_state['L56'] = st.number_input("Top 5 & 6:", value=st.session_state['L56'], step=1)
            if STRAT == "V24 Cổ Điển":
                st.session_state['LMOD'] = st.number_input("Mod (V24):", value=st.session_state['LMOD'], step=1)

        # Cấu hình riêng cho Gốc 3
        if STRAT == "Gốc 3":
            st.session_state['G3_IN'] = st.slider("Gốc 3 Input:", 50, 100, st.session_state['G3_IN'])
            st.session_state['G3_OUT'] = st.slider("Gốc 3 Target:", 50, 80, st.session_state['G3_OUT'])

        # Cấu hình Điểm số M (nếu không dùng Auto)
        if STRAT in ["V24 Cổ Điển", "Gốc 3"] and not st.session_state['USE_AUTO_WEIGHTS']:
            with st.expander("🎚️ Điểm số M (Gốc/Mod)"):
                c1, c2 = st.columns(2)
                with c1: 
                    st.write("Gốc")
                    for i in range(11): st.number_input(f"M{i}", key=f"std_{i}")
                with c2:
                    st.write("Mod")
                    for i in range(11): st.number_input(f"M{i}", key=f"mod_{i}")

        MIN_VOTES = st.number_input("Vote tối thiểu:", 1)
        USE_INVERSE = st.checkbox("Chấm điểm Đảo (Inverse)")
        
        # Save & Clear
        if st.button("💾 LƯU CẤU HÌNH"):
            save_data = {
                'STD': [st.session_state[f'std_{i}'] for i in range(11)],
                'MOD': [st.session_state[f'mod_{i}'] for i in range(11)],
                'LIMITS': {'l12': st.session_state['L12'], 'l34': st.session_state['L34'], 'l56': st.session_state['L56'], 'mod': st.session_state['LMOD']},
                'ROLLING': st.session_state['ROLLING']
            }
            with open(CONFIG_FILE, 'w') as f: json.dump(save_data, f)
            st.success("Đã lưu!")
        
        if st.button("🗑️ XÓA CACHE"): st.cache_data.clear(); st.rerun()

    # --- MÀN HÌNH CHÍNH ---
    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        with st.expander("ℹ️ Thông tin File & Debug"):
            for s in f_status: st.write(s)
            for e in err_logs: st.error(e)
        
        if data_cache:
            last_d = max(data_cache.keys())
            tab1, tab2, tab3 = st.tabs(["📊 SOI CẦU (PREDICT)", "🔙 KIỂM CHỨNG (BACKTEST)", "🎯 MATRIX SCANNER"])
            
            # ------------------------------------------------------------------
            # TAB 1: SOI CẦU (PREDICTION)
            # ------------------------------------------------------------------
            with tab1:
                col_d, col_btn = st.columns([1, 2])
                with col_d: target_d = st.date_input("Ngày soi:", value=last_d)
                
                if st.button("🚀 CHẠY PHÂN TÍCH", type="primary"):
                    # 1. Chuẩn bị tham số
                    limits = {'l12': st.session_state['L12'], 'l34': st.session_state['L34'], 'l56': st.session_state['L56'], 'mod': st.session_state['LMOD']}
                    
                    # 2. Xử lý trọng số (Weights)
                    if st.session_state['USE_AUTO_WEIGHTS']:
                        auto_w = calculate_auto_weights_from_data(target_d, data_cache, kq_db, st.session_state['AUTO_LOOKBACK'])
                        score_std = auto_w; score_mod = auto_w
                        st.info("🤖 Đang sử dụng điểm Auto-Calibration (Tự động tối ưu).")
                    else:
                        score_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                        score_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    
                    res = None; err = None
                    strat = st.session_state['STRATEGY']

                    # 3. Chạy thuật toán tương ứng
                    if strat == "Vote 8x (Chuẩn)":
                        res, err = calculate_vote_8x_strict(target_d, st.session_state['ROLLING'], data_cache, kq_db, limits)
                    elif strat == "V24 Cổ Điển":
                        res, err = calculate_v24_classic(target_d, st.session_state['ROLLING'], data_cache, kq_db, limits, MIN_VOTES, score_std, score_mod, USE_INVERSE)
                    elif strat == "Gốc 3":
                        res = calculate_goc_3_logic(target_d, st.session_state['ROLLING'], data_cache, kq_db, st.session_state['G3_IN'], st.session_state['G3_OUT'], score_std, MIN_VOTES, USE_INVERSE)

                    # 4. Hiển thị kết quả
                    if err: st.error(err)
                    elif res:
                        st.success(f"✅ Đã phân tích xong ngày {target_d.strftime('%d/%m/%Y')}")
                        
                        # Top Groups info
                        if 'top6_std' in res: st.info(f"🏆 Top 6 Nhóm Mạnh Nhất: {', '.join(res['top6_std'])}")
                        elif 'top3' in res: st.info(f"🏆 Top 3 Nhóm Mạnh Nhất: {', '.join(res['top3'])}")
                        
                        st.divider()
                        c1, c2 = st.columns(2)
                        
                        # Cột trái: Dàn Gốc / Liên Minh
                        with c1:
                            if "dan_goc" in res:
                                label = "Dàn Giao Thoa 2 Liên Minh" if strat == "Vote 8x (Chuẩn)" else "Dàn Gốc (Chưa qua Mod)"
                                st.write(f"**{label} ({len(res['dan_goc'])})**")
                                st.text_area("dan_goc_txt", ",".join(res['dan_goc']), height=150, label_visibility="collapsed")
                                if strat == "Vote 8x (Chuẩn)":
                                    st.caption(f"Chi tiết: LM1 ({res.get('debug_s1',0)}) ∩ LM2 ({res.get('debug_s2',0)})")
                        
                        # Cột phải: FINAL
                        with c2:
                            st.write(f"**🔥 DÀN FINAL (CHỐT HẠ) ({len(res['dan_final'])})**")
                            st.text_area("dan_final_txt", ",".join(res['dan_final']), height=150, label_visibility="collapsed")
                        
                        # --- TÍNH NĂNG HYBRID (SOI CHÉO) ---
                        if strat != "Hard Core (Gốc)" and strat != "Matrix":
                            st.write("---")
                            st.write("🧬 **HYBRID CHECK (Soi chéo với Hard Core)**")
                            # Chạy ngầm Hard Core để lấy dàn Gốc
                            s_hc, m_hc, l_hc, r_hc = get_preset_params("Hard Core (Gốc)")
                            res_hc, _ = calculate_v24_classic(target_d, r_hc, data_cache, kq_db, l_hc, 1, s_hc, m_hc, False)
                            
                            if res_hc:
                                hc_goc = res_hc['dan_goc']
                                # Giao thoa Final hiện tại với HC Gốc
                                hybrid_set = sorted(list(set(res['dan_final']).intersection(set(hc_goc))))
                                st.success(f"⚔️ Dàn Hybrid ({len(hybrid_set)} số): {','.join(hybrid_set)}")
                            else:
                                st.warning("Không chạy được Hard Core để soi chéo.")

                        # --- KIỂM TRA KẾT QUẢ THỰC TẾ ---
                        if target_d in kq_db:
                            kq = kq_db[target_d]
                            st.markdown(f"### 🏁 KẾT QUẢ THỰC TẾ: `{kq}`")
                            if kq in res['dan_final']: st.success("🎉 CHÚC MỪNG! Dàn Final ĐÃ ĂN.")
                            else: st.error("❌ Rất tiếc, Dàn Final đã trượt.")

            # ------------------------------------------------------------------
            # TAB 2: BACKTEST (KIỂM CHỨNG)
            # ------------------------------------------------------------------
            with tab2:
                c1, c2 = st.columns(2)
                with c1: d_start = st.date_input("Từ ngày:", value=last_d - timedelta(days=5))
                with c2: d_end = st.date_input("Đến ngày:", value=last_d)
                
                if st.button("▶️ CHẠY BACKTEST HỆ THỐNG"):
                    logs = []; bar = st.progress(0)
                    days = [d_start + timedelta(days=x) for x in range((d_end - d_start).days + 1)]
                    
                    # Setup weights cho backtest
                    if not st.session_state['USE_AUTO_WEIGHTS']:
                        score_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                        score_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    limits = {'l12': st.session_state['L12'], 'l34': st.session_state['L34'], 'l56': st.session_state['L56'], 'mod': st.session_state['LMOD']}
                    
                    for i, d in enumerate(days):
                        bar.progress((i+1)/len(days))
                        if d not in kq_db: continue
                        
                        # Auto Weights trong quá khứ
                        if st.session_state['USE_AUTO_WEIGHTS']:
                            w = calculate_auto_weights_from_data(d, data_cache, kq_db, st.session_state['AUTO_LOOKBACK'])
                            score_std = w; score_mod = w

                        r = None
                        strat = st.session_state['STRATEGY']
                        
                        # Chạy thuật toán
                        if strat == "Vote 8x (Chuẩn)":
                            r, _ = calculate_vote_8x_strict(d, st.session_state['ROLLING'], data_cache, kq_db, limits)
                        elif strat == "V24 Cổ Điển":
                            r, _ = calculate_v24_classic(d, st.session_state['ROLLING'], data_cache, kq_db, limits, MIN_VOTES, score_std, score_mod, USE_INVERSE)
                        elif strat == "Gốc 3":
                            r = calculate_goc_3_logic(d, st.session_state['ROLLING'], data_cache, kq_db, st.session_state['G3_IN'], st.session_state['G3_OUT'], score_std, MIN_VOTES, USE_INVERSE)
                        
                        if r:
                            kq = kq_db[d]
                            win = "✅ WIN" if kq in r['dan_final'] else "❌"
                            logs.append({"Ngày": d.strftime("%d/%m"), "KQ": kq, "Kết quả": win, "Size": len(r['dan_final'])})
                    
                    if logs:
                        df_log = pd.DataFrame(logs)
                        st.dataframe(df_log, use_container_width=True)
                        wins = df_log[df_log['Kết quả'].str.contains("WIN")].shape[0]
                        st.metric("Tỷ lệ chiến thắng", f"{wins}/{len(df_log)} ({wins/len(df_log)*100:.1f}%)")

            # ------------------------------------------------------------------
            # TAB 3: MATRIX SCANNER
            # ------------------------------------------------------------------
            with tab3:
                st.subheader("🎯 MATRIX SCANNER - QUÉT SỐ CAO CẤP")
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1: 
                    mtx_d = st.date_input("Ngày soi Matrix:", value=last_d)
                    mtx_strat = st.selectbox("Chiến thuật Matrix:", ["🔥 Săn M6-M9", "🛡️ Thủ M10 (An toàn)", "💎 Elite 5 (Vip)", "👥 Top 10 File"])
                with c2: cut = st.number_input("Số lượng lấy:", 40)
                with c3: skip = st.number_input("Bỏ (Top đầu):", 0)
                
                if st.button("🚀 QUÉT MATRIX"):
                    if mtx_d in data_cache:
                        df_t = data_cache[mtx_d]['df']
                        
                        # Cấu hình Matrix
                        if "Săn M6-M9" in mtx_strat: w=[0,0,0,0,0,0,30,40,50,60,0]; top=10; s='score'
                        elif "Thủ M10" in mtx_strat: w=[0,0,0,0,0,0,0,0,0,0,60]; top=20; s='score'
                        elif "Elite 5" in mtx_strat: w=[0,0,5,10,15,25,30,35,40,50,60]; top=5; s='score'
                        else: w=[0,0,5,10,15,25,30,35,40,50,60]; top=10; s='stt'
                        
                        # Lấy danh sách thành viên ưu tú
                        elite = get_elite_members(df_t, top, s)
                        st.write(f"📋 **Danh sách Top {top} Cao Thủ được chọn:**")
                        st.dataframe(elite[['STT', 'MEMBER', 'SCORE_SORT'] if 'MEMBER' in elite.columns else elite.columns], use_container_width=True)
                        
                        # Tính Matrix
                        res = calculate_matrix_simple(elite, w)
                        fin = [f"{n:02d}" for n,sc in res[skip:skip+cut]]
                        fin.sort()
                        
                        st.divider()
                        st.text_area("👇 KẾT QUẢ MATRIX:", ",".join(fin), height=100)
                        
                        if mtx_d in kq_db:
                            k = kq_db[mtx_d]
                            try: rank = next(i+1 for i,(n,sc) in enumerate(res) if f"{n:02d}"==k)
                            except: rank = 999
                            
                            if k in fin: st.success(f"✅ WIN - Số {k} nằm ở hạng {rank}")
                            else: st.error(f"❌ MISS - Số {k} nằm ở hạng {rank}")

# Helper để lấy Preset cho Hybrid
def get_preset_params(name):
    if name not in SCORES_PRESETS: return {}, {}, {}, 10
    p = SCORES_PRESETS[name]
    s = {f'M{i}': p['STD'][i] for i in range(11)}
    m = {f'M{i}': p['MOD'][i] for i in range(11)}
    return s, m, p['LIMITS'], p['ROLLING']

if __name__ == "__main__":
    main()
