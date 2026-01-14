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
    page_title="V62 Ultimate (Full Restore)", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🛡️ Lý Thị Thông: V62 ULTIMATE")
st.caption("🚀 Đã khôi phục: Backtest Full Option | Matrix | Gốc 3 | 8x Chuẩn")

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
    },
    "Vote 8x (Chuẩn 63s)": { # Preset mới cho 8x
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
# 2. CORE UTILS (DÙNG LẠI HÀM CŨ CỦA ANH)
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

# --- HÀM NGÀY THÁNG CŨ (CHẠY ỔN ĐỊNH) ---
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
    return m_global, y_global, None

def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords): return idx
    return 3

# --- 3. LOGIC XỬ LÝ SỐ ---

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse, sort_by_vote=False):
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
    
    stats = exploded.groupby('Num')['P'].sum().to_frame(name='P')
    votes = exploded.reset_index().groupby('Num')['index'].nunique()
    stats['V'] = votes
    stats = stats[stats['V'] >= min_v]
    if stats.empty: return []
    stats = stats.reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    
    if sort_by_vote: # Mode 8x
        if inverse: stats = stats.sort_values(by=['V', 'P', 'Num_Int'], ascending=[True, True, True])
        else: stats = stats.sort_values(by=['V', 'P', 'Num_Int'], ascending=[False, False, True])
    else: # Mode V24
        if inverse: stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[True, True, True])
        else: stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])
        
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

# --- 4. HÀM TÍNH TOÁN (V24 CŨ & 8X MỚI) ---

def calculate_v24_logic_full(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None, max_trim=None, strategy_mode="🛡️ V24 Cổ Điển"):
    if target_date not in _cache: return None
    curr_data = _cache[target_date]; df = curr_data['df']
    
    # 1. Setup Map
    p_map_dict = {}; s_map_dict = {}
    is_8x_mode = (strategy_mode == "🛡️ V24 Vote 8x")
    
    if is_8x_mode:
        col_8x = next((c for c in df.columns if re.match(r'^(8X|80|DÀN|DAN)$', c.strip().upper()) or '8X' in c.strip().upper()), None)
        if col_8x: p_map_dict = {col_8x: 10}; s_map_dict = {col_8x: 10}
        else: return None
    else:
        score_std_tuple = tuple(score_std.items())
        for col in df.columns:
            s_p = get_col_score(col, score_std_tuple)
            if s_p > 0: p_map_dict[col] = s_p
            # V24 cũ dùng s_map cho Mod
            s_m = get_col_score(col, tuple(score_mod.items()))
            if s_m > 0: s_map_dict[col] = s_m

    # 2. Tìm cột nhóm
    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache: prev_date = target_date - timedelta(days=i); break
    
    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in _cache: col_hist_used = _cache[prev_date]['hist_map'].get(prev_date)
    if not col_hist_used: return None 

    # 3. Backtest Top 6
    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups} # Chỉ dùng cho V24 cũ

    if not manual_groups:
        past_dates = []
        check_d = target_date - timedelta(days=1)
        while len(past_dates) < rolling_window:
            if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
            check_d -= timedelta(days=1)
            if (target_date - check_d).days > 60: break 
            
        for d in past_dates:
            d_df = _cache[d]['df']; kq = _kq_db[d]
            d_p_map = {}; d_s_map = {}
            
            if is_8x_mode:
                d_col_8x = next((c for c in d_df.columns if '8X' in c.upper()), None)
                if d_col_8x: d_p_map = {d_col_8x: 10}
            else:
                for col in d_df.columns:
                    s_p = get_col_score(col, tuple(score_std.items()))
                    if s_p > 0: d_p_map[col] = s_p
                    s_m = get_col_score(col, tuple(score_mod.items()))
                    if s_m > 0: d_s_map[col] = s_m
            
            d_hist_col = None
            sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
            if sorted_dates: d_hist_col = _cache[d]['hist_map'][sorted_dates[0]]
            if not d_hist_col: continue
            
            try:
                hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
                for g in groups:
                    mems = d_df[hist_series_d == g.upper()]
                    if mems.empty: stats_std[g]['ranks'].append(999); continue
                    
                    top80 = fast_get_top_nums(mems, d_p_map, {}, 80, min_votes, use_inverse, sort_by_vote=is_8x_mode)
                    if kq in top80: stats_std[g]['wins']+=1; stats_std[g]['ranks'].append(top80.index(kq)+1)
                    else: stats_std[g]['ranks'].append(999)
                    
                    # Backtest Mod (chỉ cho V24 cũ)
                    if not is_8x_mode:
                        topMod = fast_get_top_nums(mems, d_s_map, {}, int(limits_config['mod']), min_votes, use_inverse, False)
                        if kq in topMod: stats_mod[g]['wins']+=1
            except: continue

    final_std = []
    for g, inf in stats_std.items(): final_std.append((g, -inf['wins'], sum(inf['ranks'])))
    final_std.sort(key=lambda x: (x[1], x[2])) 
    top6_std = [x[0] for x in final_std[:6]]
    
    # 4. Final Cut
    hist_series = df[col_hist_used].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
    
    def get_pool(grp_list, lim_dict):
        p = []
        for g in grp_list:
            res = fast_get_top_nums(df[hist_series==g.upper()], p_map_dict, {}, lim_dict.get(g,80), min_votes, use_inverse, is_8x_mode)
            p.extend(res)
        return p
            
    limits_map = {top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']}
    
    # LM1: Top 1, 5, 3
    s1 = {n for n, c in Counter(get_pool([top6_std[0], top6_std[4], top6_std[2]], limits_map)).items() if c >= 2} 
    # LM2: Top 2, 4, 6
    s2 = {n for n, c in Counter(get_pool([top6_std[1], top6_std[3], top6_std[5]], limits_map)).items() if c >= 2}
    
    # GIAO THOA GỐC
    dan_goc = sorted(list(s1.intersection(s2)))
    
    if is_8x_mode:
        # 8X MODE: DỪNG TẠI ĐÂY (KHÔNG GIAO MOD) -> 63 SỐ
        final_intersect = dan_goc
    else:
        # V24 CỔ ĐIỂN: GIAO TIẾP VỚI MOD
        best_mod_grp = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]
        dan_mod = sorted(fast_get_top_nums(df[hist_series==best_mod_grp.upper()], s_map_dict, {}, int(limits_config['mod']), min_votes, use_inverse, False))
        final_intersect = sorted(list(set(dan_goc).intersection(set(dan_mod))))

    if max_trim and len(final_intersect) > max_trim:
        final_intersect = smart_trim_by_score(final_intersect, df, p_map_dict, {}, max_trim)
    
    return {"top6_std": top6_std, "dan_goc": dan_goc, "dan_final": final_intersect, "source_col": col_hist_used}

def calculate_goc_3_logic(target_date, rolling_window, _cache, _kq_db, input_limit, target_limit, score_std, use_inverse, min_votes):
    dummy_lim = {'l12':1, 'l34':1, 'l56':1, 'mod':1}
    res_v24 = calculate_v24_logic_full(target_date, rolling_window, _cache, _kq_db, dummy_lim, min_votes, score_std, score_std, use_inverse, strategy_mode="🛡️ V24 Cổ Điển")
    if not res_v24: return None
    top3 = res_v24['top6_std'][:3]
    col_hist = res_v24['source_col']
    df = _cache[target_date]['df']
    p_map = {}
    for col in df.columns:
        s = get_col_score(col, tuple(score_std.items()))
        if s > 0: p_map[col] = s
    hist = df[col_hist].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
    
    pool = []
    for g in top3:
        pool.extend(fast_get_top_nums(df[hist==g.upper()], p_map, {}, int(input_limit), min_votes, use_inverse, False))
    
    overlap = [n for n, c in Counter(pool).items() if c >= 2]
    fin = smart_trim_by_score(overlap, df, p_map, {}, target_limit)
    return {"top3": top3, "dan_final": fin, "source_col": col_hist}

# Matrix Logic
def get_elite_members(df, top_n=10, sort_by='score'):
    if df.empty: return df
    m_cols = [c for c in df.columns if c.startswith('M')]
    df = df.dropna(subset=m_cols, how='all')
    if sort_by == 'score': return df.sort_values(by='SCORE_SORT', ascending=False).head(top_n)
    else: return df.sort_values(by='STT', ascending=True).head(top_n)

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
# 4. LOAD FILE DATA (GIỮ NGUYÊN LOGIC CŨ)
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
            # Excel Load
            if file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(file, engine='openpyxl')
                for sheet in xls.sheet_names:
                    s_date = None
                    try:
                        parts = [int(x) for x in re.sub(r'[^0-9]', ' ', sheet).strip().split()]
                        if parts: 
                            d_s, m_s = parts[0], f_m
                            y_s = parts[2] if len(parts)>=3 and parts[2]>2000 else f_y
                            s_date = datetime.date(y_s, m_s, d_s)
                    except: pass
                    if not s_date: s_date = date_from_name
                    if s_date:
                        # Tìm header row bằng hàm cũ
                        preview = pd.read_excel(xls, sheet_name=sheet, nrows=30, header=None, engine='openpyxl')
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row, engine='openpyxl')
                        dfs.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")

            # CSV Load
            elif file.name.endswith('.csv'):
                if date_from_name:
                    encodings = ['utf-8-sig', 'latin-1', 'cp1252']
                    df = None
                    for enc in encodings:
                        try:
                            file.seek(0)
                            prev = pd.read_csv(file, header=None, nrows=30, encoding=enc)
                            hr = find_header_row(prev)
                            file.seek(0)
                            df = pd.read_csv(file, header=hr, encoding=enc)
                            break
                        except: continue
                    if df is not None:
                        # Clean duplicate headers (M 1 0 fix)
                        cols = df.columns.astype(str).tolist()
                        seen = {}; new_cols = []
                        for c in cols:
                            c = c.strip().upper().replace('M 1 0', 'M10')
                            if c in seen: seen[c]+=1; new_cols.append(f"{c}.{seen[c]}")
                            else: seen[c]=0; new_cols.append(c)
                        df.columns = new_cols
                        dfs.append((date_from_name, df))
                        file_status.append(f"✅ CSV: {file.name}")
            
            for t_date, df in dfs:
                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]
                score_col = next((c for c in df.columns if 'Đ9' in c or 'DIEM' in c or 'ĐIỂM' in c), None)
                if score_col: df['SCORE_SORT'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
                else: df['SCORE_SORT'] = 0
                
                # Rename M cols
                rename_map = {}
                for c in df.columns:
                    clean_c = c.replace(" ", "")
                    if re.match(r'^M\d+$', clean_c) or clean_c == 'M10': rename_map[c] = clean_c
                if rename_map: df = df.rename(columns=rename_map)

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
                    # Dùng parse_date_smart cũ
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: 
                        hist_map[d_obj] = col
                        if kq_row is not None:
                            try:
                                nums = get_nums(str(kq_row[col]))
                                if nums: kq_db[d_obj] = nums[0]
                            except: pass
                cache[t_date] = {'df': df, 'hist_map': hist_map}
        except: continue
    return cache, kq_db, file_status, err_logs

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (MAIN APP)
# ==============================================================================

def main():
    uploaded_files = st.file_uploader("📂 Tải file dữ liệu", type=['xlsx', 'csv'], accept_multiple_files=True)
    
    # Init State
    if 'L12' not in st.session_state:
        st.session_state.update({
            'L12':80, 'L34':70, 'L56':60, 'LMOD':80, 
            'ROLLING':10, 'STRATEGY':'Vote 8x (Chuẩn)', 
            'G3_IN':75, 'G3_OUT':70, 'USE_AUTO_WEIGHTS': False, 'AUTO_LOOKBACK': 10
        })
        for i in range(11): st.session_state[f'std_{i}'] = 0; st.session_state[f'mod_{i}'] = 0

    with st.sidebar:
        st.header("⚙️ Cài đặt")
        st.session_state['STRATEGY'] = st.radio("🎯 CHIẾN THUẬT:", ["Vote 8x (Chuẩn)", "V24 Cổ Điển", "Gốc 3", "Matrix"])
        STRAT = st.session_state['STRATEGY']
        
        if STRAT == "Vote 8x (Chuẩn)":
            st.success("✅ 8X Mode: Giao thoa 2 Liên Minh (Bỏ Mod).")
        
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

        st.selectbox("📚 Preset:", list(SCORES_PRESETS.keys()), key="preset_choice", on_change=update_scores)
        
        st.markdown("---")
        st.session_state['ROLLING'] = st.number_input("Backtest (Ngày):", value=st.session_state['ROLLING'])
        
        if STRAT in ["V24 Cổ Điển", "Gốc 3"]:
            st.session_state['USE_AUTO_WEIGHTS'] = st.checkbox("🤖 Auto-M", value=st.session_state['USE_AUTO_WEIGHTS'])

        with st.expander("✂️ Cắt Số (Custom)", expanded=True):
            st.session_state['L12'] = st.number_input("Top 1-2:", value=st.session_state['L12'])
            st.session_state['L34'] = st.number_input("Top 3-4:", value=st.session_state['L34'])
            st.session_state['L56'] = st.number_input("Top 5-6:", value=st.session_state['L56'])
            if STRAT == "V24 Cổ Điển":
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

    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        with st.expander("File Status"):
            for s in f_status: st.success(s)
            for e in err_logs: st.error(e)
        
        if data_cache:
            last_d = max(data_cache.keys())
            tab1, tab2, tab3 = st.tabs(["SOI CẦU", "BACKTEST", "MATRIX"])
            
            # --- TAB 1: SOI CẦU ---
            with tab1:
                col_d, col_b = st.columns([1, 2])
                with col_d: target_d = st.date_input("Ngày:", value=last_d)
                
                if st.button("🚀 CHẠY PHÂN TÍCH"):
                    limits = {'l12': st.session_state['L12'], 'l34': st.session_state['L34'], 'l56': st.session_state['L56'], 'mod': st.session_state['LMOD']}
                    
                    if st.session_state['USE_AUTO_WEIGHTS']:
                        w = calculate_auto_weights_from_data(target_d, data_cache, kq_db, 10)
                        score_std = w; score_mod = w
                    else:
                        score_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                        score_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    
                    res = None; err = None
                    if STRAT == "Vote 8x (Chuẩn)":
                        res, err = calculate_vote_8x_custom(target_d, st.session_state['ROLLING'], data_cache, kq_db, limits)
                    elif STRAT == "V24 Cổ Điển":
                        res, err = calculate_v24_classic(target_d, st.session_state['ROLLING'], data_cache, kq_db, limits, MIN_VOTES, score_std, score_mod, USE_INVERSE, MAX_TRIM)
                    elif STRAT == "Gốc 3":
                        res = calculate_goc_3_logic(target_d, st.session_state['ROLLING'], data_cache, kq_db, st.session_state['G3_IN'], st.session_state['G3_OUT'], score_std, MIN_VOTES, USE_INVERSE)

                    if err: st.error(err)
                    elif res:
                        st.success(f"Kết quả ngày {target_d.strftime('%d/%m/%Y')}")
                        if 'top6_std' in res: st.info(f"Top 6: {', '.join(res['top6_std'])}")
                        elif 'top3' in res: st.info(f"Top 3: {', '.join(res['top3'])}")
                        
                        st.divider()
                        c1, c2 = st.columns(2)
                        with c1: 
                            if "dan_goc" in res: 
                                lbl = "LM Giao Thoa (Vote)" if STRAT == "Vote 8x (Chuẩn)" else "Dàn Gốc"
                                st.text_area(f"{lbl} ({len(res['dan_goc'])})", ",".join(res['dan_goc']), height=150)
                        with c2: st.text_area(f"FINAL CHỐT ({len(res['dan_final'])})", ",".join(res['dan_final']), height=150)
                        
                        if target_d in kq_db:
                            k = kq_db[target_d]
                            if k in res['dan_final']: st.success(f"WIN {k}")
                            else: st.error(f"MISS {k}")

            # --- TAB 2: BACKTEST (KHÔI PHỤC MENU CHỌN CẤU HÌNH) ---
            with tab2:
                c1, c2 = st.columns([1, 2])
                with c1:
                    # Menu chọn cấu hình Backtest
                    bt_opts = ["Màn hình hiện tại"] + list(SCORES_PRESETS.keys()) + ["Gốc 3 (Custom Input/Target)"]
                    sel_bt = st.selectbox("Chọn Cấu Hình Backtest:", bt_opts)
                    use_auto_bt = st.checkbox("Auto-M Backtest", value=False)
                
                with c2:
                    d_start = st.date_input("Từ:", value=last_d - timedelta(days=5))
                    d_end = st.date_input("Đến:", value=last_d)
                    if st.button("▶️ CHẠY BACKTEST"):
                        logs = []; bar = st.progress(0)
                        days = [d_start + timedelta(days=x) for x in range((d_end - d_start).days + 1)]
                        
                        for i, d in enumerate(days):
                            bar.progress((i+1)/len(days))
                            if d not in kq_db: continue
                            
                            # Xác định tham số chạy theo lựa chọn
                            run_s = {}; run_m = {}; run_l = {}; run_strat = "V24 Cổ Điển"
                            run_roll = 10; is_goc3 = False; g3_in=75; g3_out=70

                            if sel_bt == "Màn hình hiện tại":
                                # Lấy từ session_state
                                run_s = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                                run_m = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                                run_l = {'l12': st.session_state['L12'], 'l34': st.session_state['L34'], 'l56': st.session_state['L56'], 'mod': st.session_state['LMOD']}
                                run_roll = st.session_state['ROLLING']
                                run_strat = STRAT # Dùng chiến thuật đang chọn ở Sidebar
                                if STRAT == "Gốc 3": 
                                    is_goc3 = True; g3_in = st.session_state['G3_IN']; g3_out = st.session_state['G3_OUT']
                            
                            elif sel_bt == "Gốc 3 (Custom Input/Target)":
                                is_goc3 = True; g3_in = st.session_state['G3_IN']; g3_out = st.session_state['G3_OUT']
                                run_s = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)} # Vẫn dùng điểm màn hình
                                run_roll = st.session_state['ROLLING']

                            elif sel_bt in SCORES_PRESETS:
                                p = SCORES_PRESETS[sel_bt]
                                run_s = {f'M{i}': p['STD'][i] for i in range(11)}
                                run_m = {f'M{i}': p['MOD'][i] for i in range(11)}
                                run_l = p['LIMITS']
                                run_roll = p.get('ROLLING', 10)
                                if "Vote 8x" in sel_bt: run_strat = "Vote 8x (Chuẩn)"
                                else: run_strat = "V24 Cổ Điển"

                            # Auto Weights Override
                            if use_auto_bt:
                                w = calculate_auto_weights_from_data(d, data_cache, kq_db)
                                run_s = w; run_m = w

                            # EXECUTE
                            r = None
                            if is_goc3:
                                r = calculate_goc_3_logic(d, run_roll, data_cache, kq_db, g3_in, g3_out, run_s, MIN_VOTES, USE_INVERSE)
                            elif run_strat == "Vote 8x (Chuẩn)":
                                r, _ = calculate_vote_8x_custom(d, run_roll, data_cache, kq_db, run_l)
                            else: # V24 Classic
                                r, _ = calculate_v24_classic(d, run_roll, data_cache, kq_db, run_l, MIN_VOTES, run_s, run_m, USE_INVERSE, MAX_TRIM)
                            
                            if r:
                                k = kq_db[d]
                                w = "✅" if k in r['dan_final'] else "❌"
                                logs.append({"Ngày": d.strftime("%d/%m"), "KQ": k, "Win": w, "Size": len(r['dan_final'])})
                        
                        if logs: 
                            st.dataframe(pd.DataFrame(logs), use_container_width=True)
                            wins = pd.DataFrame(logs)['Win'].str.contains("✅").sum()
                            st.metric("Win Rate", f"{wins}/{len(logs)}")

            # --- TAB 3: MATRIX ---
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
