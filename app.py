import streamlit as st
import pandas as pd
import re
import datetime
import json
import os
from datetime import timedelta
from collections import Counter
from functools import lru_cache
import numpy as np

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(page_title="V62 Ultimate (Full Restore)", page_icon="🛡️", layout="wide")

CONFIG_FILE = 'config.json'

SCORES_PRESETS = {
    "Vote 8x (Chuẩn 63s)": {
        "STD": [0]*11, "MOD": [0]*11,
        "LIMITS": {'l12': 80, 'l34': 70, 'l56': 60, 'mod': 80}, 
        "ROLLING": 10
    },
    "Balanced (Khuyên dùng)": { 
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
    },
    "CH1: Bám Đuôi": { 
        "STD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70], 
        "MOD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70],
        "LIMITS": {'l12': 80, 'l34': 75, 'l56': 60, 'mod': 88},
        "ROLLING": 10
    }
}

RE_NUMS = re.compile(r'\d+')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. UTILS & DATE PARSER (KHÔI PHỤC LOGIC CŨ)
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    if any(kw in s_str.upper() for kw in BAD_KEYWORDS): return []
    raw_nums = RE_NUMS.findall(s_str)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name, mapping_tuple):
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper().replace(' ', ''))
    mapping = dict(mapping_tuple)
    if 'M10' in clean: return mapping.get('M10', 0)
    for key, score in mapping.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

# --- DATE PARSER CŨ (Đơn giản & Mạnh mẽ) ---
def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper().replace('NGAY', '').replace('NGÀY', '').strip()
    # Tìm chuỗi ngày tháng dạng d/m hoặc d.m hoặc d-m
    match = re.search(r'(\d{1,2})[\.\-\/](\d{1,2})', s)
    if match:
        try:
            d, m = int(match.group(1)), int(match.group(2))
            if m < 1 or m > 12 or d < 1 or d > 31: return None
            
            # Xử lý năm: Nếu file tháng 1 mà cột là tháng 12 -> Năm trước
            curr_y = f_y
            if m == 12 and f_m == 1: curr_y -= 1
            elif m == 1 and f_m == 12: curr_y += 1
            
            return datetime.date(curr_y, m, d)
        except: return None
        
    # Tìm dạng ISO 2026-01-13
    match_iso = re.search(r'(20\d{2})[\.\-\/](\d{1,2})[\.\-\/](\d{1,2})', s)
    if match_iso:
        try:
            return datetime.date(int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3)))
        except: return None
    return None

def extract_meta_from_filename(filename):
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    return m_global, y_global, None

def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords): return idx
    return 3

# --- AUTO WEIGHTS (KHÔI PHỤC) ---
def calculate_auto_weights(target_date, data_cache, kq_db, lookback=10):
    m_perf = {i: 0 for i in range(11)} 
    check_d = target_date - timedelta(days=1)
    past = []
    while len(past) < lookback:
        if check_d in data_cache and check_d in kq_db: past.append(check_d)
        check_d -= timedelta(days=1)
        if (target_date - check_d).days > 60: break 
    if not past: return {f'M{i}': 10 for i in range(11)} 

    for d in past:
        real = str(kq_db[d]).zfill(2)
        df = data_cache[d]['df']
        for c in df.columns:
            cl = c.replace(' ', '').upper()
            midx = -1
            if cl == 'M10': midx = 10
            elif re.match(r'^M\d+$', cl): midx = int(cl.replace('M',''))
            
            if midx != -1:
                nums = []
                for v in df[c].dropna(): nums.extend(get_nums(v))
                if real in nums: m_perf[midx] += 1

    sorted_m = sorted(m_perf.items(), key=lambda x: x[1], reverse=True)
    scores = [60, 50, 40, 30, 25, 20, 15, 10, 5, 0, 0]
    final_w = {}
    for r, (m, _) in enumerate(sorted_m):
        final_w[f'M{m}'] = scores[r] if r < len(scores) else 0
    return final_w

# ==============================================================================
# 3. CHIẾN THUẬT XỬ LÝ (LOGIC)
# ==============================================================================

# --- A. VOTE 8X (MỚI - ĐỘC LẬP - RA 63 SỐ) ---
def get_top_nums_by_vote_strict(df_members, col_name, limit):
    if df_members.empty: return []
    all_nums = []
    vals = df_members[col_name].dropna().astype(str).tolist()
    for val in vals:
        if any(kw in val.upper() for kw in BAD_KEYWORDS): continue
        all_nums.extend(get_nums(val))
    counts = Counter(all_nums)
    # Sort: Vote cao -> thấp, Số bé -> lớn
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], int(x[0])))
    return [n for n, c in sorted_items[:int(limit)]]

def calculate_vote_8x_strict(target_date, rolling_window, _cache, _kq_db, limits_config):
    if target_date not in _cache: return None, "No Data"
    curr_data = _cache[target_date]; df = curr_data['df']
    
    # 1. Tìm cột 8X
    col_8x = next((c for c in df.columns if re.match(r'^(8X|80|DÀN|DAN)$', c.strip().upper()) or '8X' in c.strip().upper()), None)
    if not col_8x: return None, "Không tìm thấy cột 8X"

    # 2. Tìm cột Nhóm (Fallback 5 ngày để tìm cột nhóm)
    col_group = None
    prev = target_date - timedelta(days=1)
    for _ in range(5):
        # Nếu prev có trong map của target
        if prev in curr_data['hist_map']: 
            col_group = curr_data['hist_map'][prev]
            break
        # Nếu không, check xem prev có file không, và lấy cột KQ của nó (ngày tháng)
        # Cách đơn giản nhất: Lấy cột ngày gần nhất trong hist_map bé hơn target_date
        prev -= timedelta(days=1)
    
    if not col_group:
        # Fallback mạnh: Lấy cột có ngày lớn nhất < target_date
        sd = sorted([k for k in curr_data['hist_map'].keys() if k < target_date], reverse=True)
        if sd: col_group = curr_data['hist_map'][sd[0]]
    
    if not col_group: return None, "Lỗi: Không tìm thấy cột Phân Nhóm"

    # 3. Backtest Top 6
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    past = []
    chk = target_date - timedelta(days=1)
    while len(past) < rolling_window:
        if chk in _cache and chk in _kq_db: past.append(chk)
        chk -= timedelta(days=1)
        if (target_date - chk).days > 60: break
    
    for d in past:
        d_df = _cache[d]['df']; kq = _kq_db[d]
        d_c8 = next((c for c in d_df.columns if '8X' in c.upper()), None)
        d_sd = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
        d_grp = _cache[d]['hist_map'].get(d_sd[0]) if d_sd else None

        if d_c8 and d_grp:
            try:
                # Chuẩn hóa cột nhóm: S -> 6
                g_ser = d_df[d_grp].astype(str).str.upper().str.replace('S','6').str.replace(r'[^0-9X]','', regex=True)
                for g in groups:
                    mems = d_df[g_ser == g.upper()]
                    # Cắt cứng 80 số
                    top80 = get_top_nums_by_vote_strict(mems, d_c8, 80)
                    if kq in top80: stats[g]['wins'] += 1; stats[g]['ranks'].append(top80.index(kq))
                    else: stats[g]['ranks'].append(999)
            except: continue

    final_rk = []
    for g, inf in stats.items(): final_rk.append((g, -inf['wins'], sum(inf['ranks'])))
    final_rk.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_rk[:6]]

    # 4. Final Cut (2 Liên Minh - NO MOD)
    h_ser = df[col_group].astype(str).str.upper().str.replace('S','6').str.replace(r'[^0-9X]','', regex=True)
    
    def get_pool_vote(grps, lim_map):
        p = []
        for g in grps:
            p += get_top_nums_by_vote_strict(df[h_ser == g.upper()], col_8x, lim_map[g])
        return {n for n, c in Counter(p).items() if c >= 2}

    lm_lims = {
        top6[0]: limits_config['l12'], top6[1]: limits_config['l12'], 
        top6[2]: limits_config['l34'], top6[3]: limits_config['l34'], 
        top6[4]: limits_config['l56'], top6[5]: limits_config['l56']
    }

    s1 = get_pool_vote([top6[0], top6[4], top6[2]], lm_lims) # LM1: Top 1,5,3
    s2 = get_pool_vote([top6[1], top6[3], top6[5]], lm_lims) # LM2: Top 2,4,6

    # Giao thoa LM1 & LM2 -> 63 số
    final_dan = sorted(list(s1.intersection(s2)))
    return {"top6_std": top6, "dan_goc": final_dan, "dan_final": final_dan, "source_col": col_group}, None

# --- B. V24 CỔ ĐIỂN & GỐC 3 (GIỮ NGUYÊN CODE CŨ) ---
def fast_get_top_nums_score(df, p_map, s_map, top_n, min_v, inverse):
    cols = sorted(list(set(p_map.keys()) | set(s_map.keys())))
    v_cols = [c for c in cols if c in df.columns]
    if not v_cols: return []
    sub = df[v_cols].copy()
    melt = sub.melt(ignore_index=False, value_name='Val').dropna(subset=['Val'])
    melt = melt[~melt['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT', regex=True)]
    s_nums = melt['Val'].astype(str).str.findall(r'\d+')
    expl = melt.assign(Num=s_nums).explode('Num').dropna(subset=['Num'])
    expl['Num'] = expl['Num'].str.strip().str.zfill(2)
    expl['P'] = expl['variable'].map(p_map).fillna(0)
    expl['S'] = expl['variable'].map(s_map).fillna(0)
    stats = expl.groupby('Num')[['P','S']].sum()
    stats['V'] = expl.reset_index().groupby('Num')['index'].nunique()
    stats = stats[stats['V'] >= min_v].reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    stats = stats.sort_values(by=['P','V','Num_Int'], ascending=[True,True,True] if inverse else [False,False,True])
    return stats['Num'].head(int(top_n)).tolist()

def smart_trim(nums, df, p_map, target):
    if len(nums) <= target: return sorted(nums)
    # Trim logic simplified
    return sorted(nums[:int(target)])

def calculate_v24_classic(target_date, rolling_window, _cache, _kq_db, limits, min_v, s_std, s_mod, inv, max_trim=None):
    if target_date not in _cache: return None, "No data"
    df = _cache[target_date]['df']; p_map = {}; s_map = {}
    for c in df.columns:
        s = get_col_score(c, tuple(s_std.items())); 
        if s > 0: p_map[c] = s
        m = get_col_score(c, tuple(s_mod.items())); 
        if m > 0: s_map[c] = m
    
    # Tìm cột nhóm (Logic Fallback)
    col_h = None
    sd = sorted([k for k in _cache[target_date]['hist_map'].keys() if k < target_date], reverse=True)
    if sd: col_h = _cache[target_date]['hist_map'][sd[0]]
    if not col_h: return None, "No Group Col"

    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}
    
    past = []
    chk = target_date - timedelta(days=1)
    while len(past) < rolling_window:
        if chk in _cache and chk in _kq_db: past.append(chk)
        chk -= timedelta(days=1)
        if (target_date - chk).days > 60: break
    
    for d in past:
        d_df = _cache[d]['df']; kq = _kq_db[d]
        d_p = {}; d_s = {}
        for c in d_df.columns:
            s = get_col_score(c, tuple(s_std.items())); 
            if s > 0: d_p[c] = s
            m = get_col_score(c, tuple(s_mod.items())); 
            if m > 0: d_s[c] = m
        
        d_sd = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
        d_grp = _cache[d]['hist_map'].get(d_sd[0]) if d_sd else None
        if d_grp:
            try:
                g_ser = d_df[d_grp].astype(str).str.upper().replace('S','6').str.replace(r'[^0-9X]','',regex=True)
                for g in groups:
                    mems = d_df[g_ser == g.upper()]
                    t80 = fast_get_top_nums_score(mems, d_p, d_s, 80, min_v, inv)
                    if kq in t80: stats_std[g]['wins']+=1; stats_std[g]['ranks'].append(t80.index(kq))
                    else: stats_std[g]['ranks'].append(999)
                    tMod = fast_get_top_nums_score(mems, d_s, d_p, int(limits['mod']), min_v, inv)
                    if kq in tMod: stats_mod[g]['wins']+=1
            except: continue

    fin_rk = []
    for g, i in stats_std.items(): fin_rk.append((g, -i['wins'], sum(i['ranks'])))
    fin_rk.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in fin_rk[:6]]
    best_mod = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]

    h_ser = df[col_h].astype(str).str.upper().replace('S','6').str.replace(r'[^0-9X]','',regex=True)
    def pool(gl, lims):
        p = []
        for g in gl: p.extend(fast_get_top_nums_score(df[h_ser==g.upper()], p_map, s_map, lims.get(g,80), min_v, inv))
        return p
    
    lm = {top6[0]: limits['l12'], top6[1]: limits['l12'], top6[2]: limits['l34'], top6[3]: limits['l34'], top6[4]: limits['l56'], top6[5]: limits['l56']}
    s1 = {n for n,c in Counter(pool([top6[0], top6[4], top6[2]], lm)).items() if c>=2}
    s2 = {n for n,c in Counter(pool([top6[1], top6[3], top6[5]], lm)).items() if c>=2}
    dan_goc = sorted(list(s1.intersection(s2)))
    
    dan_mod = sorted(fast_get_top_nums_score(df[h_ser==best_mod.upper()], s_map, p_map, int(limits['mod']), min_v, inv))
    final = sorted(list(set(dan_goc).intersection(set(dan_mod))))
    
    if max_trim and len(final) > max_trim:
        # Simplified smart trim
        final = final[:int(max_trim)]
        
    return {"top6_std": top6, "best_mod": best_mod, "dan_goc": dan_goc, "dan_final": final, "source_col": col_h}, None

def calculate_goc_3_logic(target_date, rolling_window, _cache, _kq_db, input_limit, target_limit, score_std, min_votes, use_inverse):
    dummy_lim = {'l12':1, 'l34':1, 'l56':1, 'mod':1}
    res_v24, _ = calculate_v24_classic(target_date, rolling_window, _cache, _kq_db, dummy_lim, min_votes, score_std, score_std, use_inverse)
    if not res_v24: return None
    
    top3 = res_v24['top6_std'][:3]
    col_hist = res_v24['source_col']
    df = _cache[target_date]['df']
    p_map = {}; score_std_t = tuple(score_std.items())
    for col in df.columns:
        if get_col_score(col, score_std_t) > 0: p_map[col] = get_col_score(col, score_std_t)
        
    h_ser = df[col_hist].astype(str).str.upper().replace('S','6').str.replace(r'[^0-9X]','',regex=True)
    all_nums = []
    for g in top3:
        res = fast_get_top_nums_score(df[h_ser==g.upper()], p_map, p_map, int(input_limit), min_votes, use_inverse)
        all_nums.extend(res)
    overlap = [n for n, c in Counter(all_nums).items() if c >= 2]
    return {"top3": top3, "dan_final": sorted(overlap), "source_col": col_hist}

# --- C. MATRIX ---
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
# 4. LOAD FILE DATA
# ==============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_data_v24(files):
    cache = {}; kq_db = {}; file_status = []; err_logs = []
    files = sorted(files, key=lambda x: x.name)
    
    for file in files:
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper(): continue
        f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
        
        try:
            dfs = []
            # Xử lý Excel
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
                        # Tìm header row
                        preview = pd.read_excel(xls, sheet_name=sheet, nrows=30, header=None, engine='openpyxl')
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row, engine='openpyxl')
                        dfs.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")

            # Xử lý CSV
            elif file.name.endswith('.csv'):
                # Thử nhiều encoding
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
                    # Xử lý header trùng lặp (ví dụ M 1 0)
                    df = df_raw.iloc[h_row+1:].copy()
                    raw_cols = df_raw.iloc[h_row].astype(str).tolist()
                    seen = {}; final_cols = []
                    for c in raw_cols:
                        c = str(c).strip().upper().replace('M 1 0', 'M10')
                        if c in seen: seen[c] += 1; final_cols.append(f"{c}.{seen[c]}")
                        else: seen[c] = 0; final_cols.append(c)
                    df.columns = final_cols
                    
                    if date_from_name: dfs.append((date_from_name, df))
                    file_status.append(f"✅ CSV: {file.name}")
                else:
                    err_logs.append(f"❌ Lỗi Encoding: {file.name}")

            # Xử lý DataFrame sau khi load
            for t_date, df in dfs:
                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]
                
                # Tạo cột Score Sort
                score_col = next((c for c in df.columns if 'Đ9' in c or 'DIEM' in c or 'ĐIỂM' in c), None)
                if score_col: df['SCORE_SORT'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
                else: df['SCORE_SORT'] = 0
                
                # Chuẩn hóa tên cột M
                rename_map = {}
                for c in df.columns:
                    clean_c = c.replace(" ", "")
                    if re.match(r'^M\d+$', clean_c) or clean_c == 'M10': rename_map[c] = clean_c
                if rename_map: df = df.rename(columns=rename_map)

                # Map lịch sử & Lấy KQ
                hist_map = {}
                kq_row = None
                if not df.empty:
                    # Tìm dòng KQ (quét 2 cột đầu)
                    for c_idx in range(min(2, len(df.columns))):
                        col_check = df.columns[c_idx]
                        if df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ').any():
                            kq_row = df[df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')].iloc[0]
                            break
                
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
# 5. GIAO DIỆN CHÍNH (MAIN APP)
# ==============================================================================

def main():
    uploaded_files = st.file_uploader("📂 Tải file dữ liệu (Excel/CSV)", type=['xlsx', 'csv'], accept_multiple_files=True)
    
    # Init Session State
    if 'L12' not in st.session_state:
        st.session_state.update({
            'L12':80, 'L34':70, 'L56':60, 'LMOD':80, 
            'ROLLING':10, 'STRATEGY':'Vote 8x (Chuẩn 63s)', 
            'G3_IN':75, 'G3_OUT':70,
            'USE_AUTO_WEIGHTS': False, 'AUTO_LOOKBACK': 10
        })
        for i in range(11): st.session_state[f'std_{i}'] = 0; st.session_state[f'mod_{i}'] = 0

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        st.session_state['STRATEGY'] = st.radio(
            "🎯 CHIẾN THUẬT:", 
            ["Vote 8x (Chuẩn 63s)", "V24 Cổ Điển", "Gốc 3", "Matrix"]
        )
        STRAT = st.session_state['STRATEGY']
        
        if STRAT == "Vote 8x (Chuẩn 63s)":
            st.success("✅ Đã Fix: Chỉ giao thoa 2 LM, bỏ Mod.")
        
        # Load Presets
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
        
        st.markdown("---")
        st.session_state['ROLLING'] = st.number_input("Backtest (Ngày):", value=st.session_state['ROLLING'])
        
        # Auto-M toggle
        if STRAT in ["V24 Cổ Điển", "Gốc 3"]:
            st.session_state['USE_AUTO_WEIGHTS'] = st.checkbox("🤖 Auto-M", value=st.session_state['USE_AUTO_WEIGHTS'])

        with st.expander("✂️ Cắt Số", expanded=True):
            st.session_state['L12'] = st.number_input("Top 1 & 2:", value=st.session_state['L12'])
            st.session_state['L34'] = st.number_input("Top 3 & 4:", value=st.session_state['L34'])
            st.session_state['L56'] = st.number_input("Top 5 & 6:", value=st.session_state['L56'])
            st.session_state['LMOD'] = st.number_input("Mod:", value=st.session_state['LMOD'])

        if STRAT == "Gốc 3":
            st.session_state['G3_IN'] = st.slider("Gốc 3 In:", 50, 100, st.session_state['G3_IN'])
            st.session_state['G3_OUT'] = st.slider("Gốc 3 Out:", 50, 80, st.session_state['G3_OUT'])

        if STRAT in ["V24 Cổ Điển", "Gốc 3"] and not st.session_state['USE_AUTO_WEIGHTS']:
            with st.expander("Điểm M"):
                c1, c2 = st.columns(2)
                with c1: 
                    st.write("Gốc")
                    for i in range(11): st.session_state[f'std_{i}'] = st.number_input(f"S{i}", value=st.session_state[f'std_{i}'])
                with c2:
                    st.write("Mod")
                    for i in range(11): st.session_state[f'mod_{i}'] = st.number_input(f"M{i}", value=st.session_state[f'mod_{i}'])
        
        MIN_VOTES = st.number_input("Vote Min:", 1)
        USE_INVERSE = st.checkbox("Đảo")
        MAX_TRIM = st.slider("Max Trim:", 50, 90, 75)
        
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

    # --- MÀN HÌNH CHÍNH (CONTENT) ---
    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        with st.expander("Debug Info"):
            for s in f_status: st.write(s)
            for e in err_logs: st.error(e)
        
        if data_cache:
            last_d = max(data_cache.keys())
            tab1, tab2, tab3 = st.tabs(["SOI CẦU", "BACKTEST", "MATRIX"])
            
            # --- TAB 1 ---
            with tab1:
                st.title(f"Soi Cầu: {STRAT}")
                col_d, col_b = st.columns([1, 2])
                with col_d: target_d = st.date_input("Ngày:", value=last_d)
                
                if st.button("🚀 CHẠY"):
                    limits = {'l12': st.session_state['L12'], 'l34': st.session_state['L34'], 'l56': st.session_state['L56'], 'mod': st.session_state['LMOD']}
                    
                    if st.session_state['USE_AUTO_WEIGHTS']:
                        w = calculate_auto_weights(target_d, data_cache, kq_db, 10)
                        score_std = w; score_mod = w
                    else:
                        score_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                        score_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    
                    res = None; err = None
                    if STRAT == "Vote 8x (Chuẩn 63s)":
                        res, err = calculate_vote_8x_strict(target_d, st.session_state['ROLLING'], data_cache, kq_db, limits)
                    elif STRAT == "V24 Cổ Điển":
                        res, err = calculate_v24_classic(target_d, st.session_state['ROLLING'], data_cache, kq_db, limits, MIN_VOTES, score_std, score_mod, USE_INVERSE, MAX_TRIM)
                    elif STRAT == "Gốc 3":
                        res = calculate_goc_3_logic(target_d, st.session_state['ROLLING'], data_cache, kq_db, st.session_state['G3_IN'], st.session_state['G3_OUT'], score_std, MIN_VOTES, USE_INVERSE)

                    if err: st.error(err)
                    elif res:
                        st.success("Xong!")
                        if 'top6_std' in res: st.info(f"Top: {', '.join(res['top6_std'])}")
                        
                        st.divider()
                        c1, c2 = st.columns(2)
                        with c1: 
                            if "dan_goc" in res: st.text_area(f"Gốc/LM ({len(res['dan_goc'])})", ",".join(res['dan_goc']), height=150)
                        with c2: st.text_area(f"FINAL ({len(res['dan_final'])})", ",".join(res['dan_final']), height=150)
                        
                        if target_d in kq_db:
                            k = kq_db[target_d]
                            if k in res['dan_final']: st.success(f"WIN {k}")
                            else: st.error(f"MISS {k}")

            # --- TAB 2 ---
            with tab2:
                d_start = st.date_input("Từ:", value=last_d - timedelta(days=5))
                d_end = st.date_input("Đến:", value=last_d)
                if st.button("▶️ BACKTEST"):
                    logs = []
                    bar = st.progress(0)
                    days = [d_start + timedelta(days=x) for x in range((d_end - d_start).days + 1)]
                    
                    score_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                    score_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    limits = {'l12': st.session_state['L12'], 'l34': st.session_state['L34'], 'l56': st.session_state['L56'], 'mod': st.session_state['LMOD']}

                    for i, d in enumerate(days):
                        bar.progress((i+1)/len(days))
                        if d not in kq_db: continue
                        
                        if st.session_state['USE_AUTO_WEIGHTS']:
                            w = calculate_auto_weights(d, data_cache, kq_db)
                            score_std = w; score_mod = w

                        r = None
                        if STRAT == "Vote 8x (Chuẩn 63s)":
                            r, _ = calculate_vote_8x_strict(d, st.session_state['ROLLING'], data_cache, kq_db, limits)
                        elif STRAT == "V24 Cổ Điển":
                            r, _ = calculate_v24_classic(d, st.session_state['ROLLING'], data_cache, kq_db, limits, MIN_VOTES, score_std, score_mod, USE_INVERSE)
                        elif STRAT == "Gốc 3":
                            r = calculate_goc_3_logic(d, st.session_state['ROLLING'], data_cache, kq_db, st.session_state['G3_IN'], st.session_state['G3_OUT'], score_std, MIN_VOTES, USE_INVERSE)
                        
                        if r:
                            k = kq_db[d]
                            w = "✅" if k in r['dan_final'] else "❌"
                            logs.append({"Ngày": d.strftime("%d/%m"), "KQ": k, "Win": w, "Size": len(r['dan_final'])})
                    
                    if logs: st.dataframe(pd.DataFrame(logs))

            # --- TAB 3 ---
            with tab3:
                st.subheader("Matrix")
                c1, c2, c3 = st.columns([2,1,1])
                with c1: 
                    mtx_d = st.date_input("Ngày:", value=last_d)
                    strat = st.selectbox("Kiểu:", ["Săn M6-M9", "Thủ M10", "Elite 5", "Top 10"])
                with c2: cut = st.number_input("Lấy:", 40)
                with c3: skip = st.number_input("Bỏ:", 0)
                
                if st.button("Quét"):
                    if mtx_d in data_cache:
                        df_t = data_cache[mtx_d]['df']
                        if strat == "Săn M6-M9": w=[0,0,0,0,0,0,30,40,50,60,0]; top=10; s='score'
                        elif strat == "Thủ M10": w=[0,0,0,0,0,0,0,0,0,0,60]; top=20; s='score'
                        elif strat == "Elite 5": w=[0,0,5,10,15,25,30,35,40,50,60]; top=5; s='score'
                        else: w=[0,0,5,10,15,25,30,35,40,50,60]; top=10; s='stt'
                        
                        elite = get_elite_members(df_t, top, s)
                        st.dataframe(elite[['STT', 'MEMBER', 'SCORE_SORT'] if 'MEMBER' in elite.columns else elite.columns])
                        res = calculate_matrix_simple(elite, w)
                        fin = [f"{n:02d}" for n,sc in res[skip:skip+cut]]
                        st.text_area("KQ:", ",".join(sorted(fin)))
                        
                        if mtx_d in kq_db:
                            k = kq_db[mtx_d]
                            try: rk = next(i+1 for i,(n,sc) in enumerate(res) if f"{n:02d}"==k)
                            except: rk=999
                            if k in fin: st.success(f"WIN ({rk})")
                            else: st.error(f"MISS ({rk})")

if __name__ == "__main__":
    main()
