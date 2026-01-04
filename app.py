import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta
from functools import lru_cache

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & UI
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V29 - Ultimate Final", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🎯 Quang Handsome: V29 Ultimate Custom")
st.caption("🚀 Final 4 = Gốc + Smart 58 | Core Optimized | 100% Deterministic Logic")

# Regex & Sets (Tối ưu hiệu suất xử lý chuỗi)
RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU (CORE - ĐÃ TỐI ƯU & FIX LỖI)
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    """Lấy số từ chuỗi, bỏ qua các từ khóa xấu."""
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    raw_nums = RE_NUMS.findall(s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name, mapping_tuple):
    """Lấy điểm số cấu hình cho từng cột M."""
    # Fix lỗi M 1 0 (có khoảng trắng)
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
    """Phân tích ngày tháng từ tên cột."""
    s = str(col_str).strip().upper()
    s = s.replace('NGAY', '').replace('NGÀY', '').strip()
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
        # Xử lý vắt năm (tháng 12 -> tháng 1)
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: return None
    return None

def find_header_row(df_preview):
    """Tìm dòng chứa tiêu đề cột."""
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords):
            return idx
    return 3

def extract_meta_from_filename(filename):
    """
    Trí tuệ nhân tạo: Tự động ghép Năm/Tháng từ tên file mẹ 
    với Ngày từ tên file con để ra thời gian chính xác 100%.
    """
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    
    # 1. Tìm Năm (2025, 2026...)
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    
    # 2. Tìm Tháng (THANG 12, T1...)
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    
    # 3. Tìm Ngày từ đuôi file
    # Case A: 1.12 (Ngày đầy đủ)
    full_date_match = re.search(r'[\s\-](\d{1,2})[\.\-](\d{1,2})(?:[\.\-]20\d{2})?$', clean_name)
    if full_date_match:
        try:
            d = int(full_date_match.group(1))
            m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            if m == 12 and m_global == 1: y -= 1 # Fix lùi năm
            return m, y, datetime.date(y, m, d)
        except: pass
    
    # Case B: Số đơn lẻ (20, 30, 3) -> Ghép với m_global, y_global
    single_day_match = re.search(r'-\s*(\d{1,2})$', clean_name)
    if single_day_match:
        try:
            d = int(single_day_match.group(1))
            return m_global, y_global, datetime.date(y_global, m_global, d)
        except: pass
    
    return m_global, y_global, None

@st.cache_data(ttl=600)
def load_data_v24(files):
    """Đọc dữ liệu từ file upload và lưu vào cache."""
    cache = {} 
    kq_db = {}
    err_logs = []
    file_status = []

    # --- FIX CRITICAL: KHÓA THỨ TỰ ĐỌC FILE (A-Z) ---
    files = sorted(files, key=lambda x: x.name)

    for file in files:
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper(): continue
        f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
        try:
            dfs_to_process = []
            if file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(file)
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
                    if s_date:
                        preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=20)
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row)
                        # Fix M 1 0
                        df.columns = [str(c).strip().upper().replace('M 1 0', 'M10') for c in df.columns]
                        dfs_to_process.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")
            elif file.name.endswith('.csv'):
                if not date_from_name: continue
                try:
                    preview = pd.read_csv(file, header=None, nrows=20, encoding='utf-8')
                    file.seek(0)
                    df_raw = pd.read_csv(file, header=None, encoding='utf-8')
                except:
                    file.seek(0)
                    try:
                        preview = pd.read_csv(file, header=None, nrows=20, encoding='latin-1')
                        file.seek(0)
                        df_raw = pd.read_csv(file, header=None, encoding='latin-1')
                    except: continue
                h_row = find_header_row(preview)
                df = df_raw.iloc[h_row+1:].copy()
                df.columns = df_raw.iloc[h_row]
                # Fix M 1 0
                df.columns = [str(c).strip().upper().replace('M 1 0', 'M10') for c in df.columns]
                dfs_to_process.append((date_from_name, df))
                file_status.append(f"✅ CSV: {file.name}")

            for t_date, df in dfs_to_process:
                df.columns = [str(c).strip().upper() for c in df.columns]
                hist_map = {}
                for col in df.columns:
                    if "UNNAMED" in col: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: hist_map[d_obj] = col
                kq_row = None
                if not df.empty:
                    for c_idx in range(min(2, len(df.columns))):
                        col_check = df.columns[c_idx]
                        mask_kq = df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')
                        if mask_kq.any():
                            kq_row = df[mask_kq].iloc[0]
                            break
                if kq_row is not None:
                    for d_val, c_name in hist_map.items():
                        val = str(kq_row[c_name])
                        nums = get_nums(val)
                        if nums: kq_db[d_val] = nums[0]
                cache[t_date] = {'df': df, 'hist_map': hist_map}
        except Exception as e:
            err_logs.append(f"Lỗi '{file.name}': {str(e)}")
            continue
    return cache, kq_db, file_status, err_logs

# ==============================================================================
# 3. MODULE K55 & SMART 58 (LOGIC BỔ TRỢ)
# ==============================================================================
def k55_parse_numbers(val):
    if pd.isna(val): return []
    nums = re.findall(r'\d+', str(val))
    return [f"{int(n):02d}" for n in nums]

def get_v12_scores(df):
    elite_scores = {}
    count = 0
    data_cols = [c for c in df.columns if c.startswith('M') and len(c) < 5]
    for idx, row in df.iterrows():
        row_str = str(row.values)
        if len(re.findall(r'\b\d{2}\b', row_str)) < 10: continue
        count += 1
        weight = 0
        if count <= 5: weight = 10    
        elif count <= 20: weight = 5 
        else: break 
        for c in data_cols:
            if c not in row: continue
            nums = k55_parse_numbers(row[c])
            for n in nums:
                elite_scores[n] = elite_scores.get(n, 0) + weight
    return elite_scores

def get_k55_scores(df):
    heat_scores = {}
    col_map = {}
    for c in df.columns:
        w = 0
        if c == 'M10': w = 100
        elif c == 'M9': w = 90
        elif c == 'M8': w = 80
        elif c == 'M7': w = 70
        elif c == 'M6': w = 60
        elif c == 'M5': w = 50
        elif c == 'M4': w = 40
        elif c == 'M3': w = 30
        elif c == 'M2': w = 20
        elif c == 'M1': w = 10
        if w > 0: col_map[c] = w
    score_col = next((c for c in df.columns if '9X0X' in c and 'TV' not in c), None)
    for idx, row in df.iterrows():
        player_power = 1.0
        if score_col and pd.notna(row[score_col]):
            try: player_power = float(row[score_col]) / 100 
            except: player_power = 1.0
        for col_name, weight in col_map.items():
            if col_name not in row: continue
            nums = k55_parse_numbers(row[col_name])
            for n in nums:
                heat_scores[n] = heat_scores.get(n, 0) + (weight * player_power)
    return heat_scores

def calculate_k55_integrated(target_date, cache, kq_db, k55_limit):
    if target_date not in cache: return [], "No Data"
    df = cache[target_date]['df']
    last_res_val = None
    prev_date = target_date - timedelta(days=1)
    if prev_date not in kq_db:
         for i in range(2, 4):
             if (target_date - timedelta(days=i)) in kq_db:
                 prev_date = target_date - timedelta(days=i); break
    if prev_date in kq_db: last_res_val = kq_db[prev_date]
    scores_v12 = get_v12_scores(df)
    scores_k55 = get_k55_scores(df)
    max_v12 = max(scores_v12.values()) if scores_v12 else 1
    max_k55 = max(scores_k55.values()) if scores_k55 else 1
    final_scores = {}
    all_nums = set(scores_v12.keys()) | set(scores_k55.keys())
    for n in all_nums:
        s1 = (scores_v12.get(n, 0) / max_v12) * 100 
        s2 = (scores_k55.get(n, 0) / max_k55) * 100 
        base_score = (s1 * 0.4) + (s2 * 0.6)
        bonus = 0
        if s1 > 0 and s2 > 0: bonus = 20 
        final_scores[n] = base_score + bonus
    ranked_final = sorted(final_scores.keys(), key=lambda x: (-final_scores[x], x))
    hybrid_set = ranked_final[:k55_limit]
    if last_res_val and last_res_val not in hybrid_set and hybrid_set:
        hybrid_set[-1] = last_res_val 
    hybrid_set.sort()
    return hybrid_set, None

def smart58_get_weighted_numbers(row):
    weighted_nums = {}
    m_cols = [c for c in row.index if str(c).strip().upper().startswith('M') and len(str(c)) < 5]
    for col in m_cols:
        col_name = str(col).strip().upper()
        digits = re.findall(r'\d+', col_name)
        if not digits: continue
        k = int(digits[0])
        weight = k
        if k == 10: weight = 50
        elif k == 9: weight = 25
        elif k == 8: weight = 15
        val = row[col]
        if pd.notna(val):
            nums = re.split(r'[,\s;]+', str(val))
            for n in nums:
                n = n.strip()
                if n.isdigit():
                    fn = f"{int(n):02d}"
                    weighted_nums[fn] = max(weighted_nums.get(fn, 0), weight)
    return weighted_nums

def smart58_calc_streak(row, hist_map_sorted):
    streak = 0
    for col_name in hist_map_sorted:
        if col_name not in row: continue
        val = str(row[col_name]).strip().lower()
        if 'x' == val: streak += 1
        elif 'xịt' in val or val in ['nan', '', '0', 'miss']: break
        else: break
    return streak

def calculate_smart58(target_date, cache, limit):
    if target_date not in cache: return [], "No Data"
    data = cache[target_date]
    df = data['df']
    hist_map = data['hist_map']
    past_dates = sorted([d for d in hist_map.keys() if d < target_date], reverse=True)
    past_cols = [hist_map[d] for d in past_dates]
    score_col = next((c for c in df.columns if 'Đ' in c.upper() or '9X' in c.upper()), None)
    player_list = []
    for idx, row in df.iterrows():
        name_val = str(row.values[0]) if len(row.values) > 0 else ""
        if len(name_val) < 2 or name_val.isdigit() or any(x in name_val.upper() for x in ['KQ', 'TV TOP', 'THỐNG KÊ', 'NAN']):
            continue
        streak = smart58_calc_streak(row, past_cols)
        f_score = 0
        if score_col and pd.notna(row[score_col]):
            try: f_score = float(str(row[score_col]).replace(',', '').replace(' ', ''))
            except: f_score = 0
        player_list.append({'row': row, 'streak': streak, 'f_score': f_score})
    if not player_list: return [], "No valid players"
    ranked = sorted(player_list, key=lambda x: (-x['streak'], -x['f_score']))
    top_15 = ranked[:15]
    agg_scores = {}
    for entry in top_15:
        wn = smart58_get_weighted_numbers(entry['row'])
        for n, w in wn.items():
            agg_scores[n] = agg_scores.get(n, 0) + w
    final_items = sorted(agg_scores.items(), key=lambda x: (-x[1], int(x[0])))[:limit]
    final_set = sorted([x[0] for x in final_items])
    return final_set, None

# ==============================================================================
# 4. CORE LOGIC V25 (VECTORIZED & DETERMINISTIC)
# ==============================================================================

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
    """
    Vectorized Function: Faster 50x
    Fix: .zfill(2) for 05/5
    """
    cols_in_scope = list(set(p_map_dict.keys()) | set(s_map_dict.keys()))
    valid_cols = [c for c in cols_in_scope if c in df.columns]
    
    if not valid_cols or df.empty: return []

    sub_df = df[valid_cols].copy()
    melted = sub_df.melt(ignore_index=False, var_name='Col', value_name='Val')
    
    melted = melted.dropna(subset=['Val'])
    bad_pattern = r'N|NGHI|SX|XIT|MISS|TRUOT|NGHỈ|LỖI'
    mask_valid = ~melted['Val'].astype(str).str.upper().str.contains(bad_pattern, regex=True)
    melted = melted[mask_valid]
    if melted.empty: return []

    s_nums = melted['Val'].astype(str).str.findall(r'\d+')
    exploded = melted.assign(Num=s_nums).explode('Num')
    exploded = exploded.dropna(subset=['Num'])
    
    # --- FIX 05 vs 5 ---
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].str.len() <= 2]

    exploded['P'] = exploded['Col'].map(p_map_dict).fillna(0)
    exploded['S'] = exploded['Col'].map(s_map_dict).fillna(0)

    stats = exploded.groupby('Num')[['P', 'S']].sum()
    votes = exploded.reset_index().groupby('Num')['index'].nunique()
    stats['V'] = votes

    stats = stats[stats['V'] >= min_v]
    if stats.empty: return []

    # Deterministic Sort
    stats = stats.reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    
    if inverse:
        stats = stats.sort_values(by=['P', 'S', 'Num_Int'], ascending=[False, False, True])
    else:
        stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])

    return stats['Num'].head(top_n).tolist()

def calculate_v24_final(target_date, rolling_window, cache, kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None):
    if target_date not in cache: return None, "Chưa có dữ liệu ngày này."
    curr_data = cache[target_date]
    df = curr_data['df']
    
    real_cols = df.columns
    p_map_dict = {}
    s_map_dict = {}
    score_std_tuple = tuple(score_std.items())
    score_mod_tuple = tuple(score_mod.items())
    
    for col in real_cols:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map_dict[col] = s_p
        s_s = get_col_score(col, score_mod_tuple)
        if s_s > 0: s_map_dict[col] = s_s

    prev_date = target_date - timedelta(days=1)
    if prev_date not in cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in cache:
                prev_date = target_date - timedelta(days=i); break
    
    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in cache:
        col_hist_used = cache[prev_date]['hist_map'].get(prev_date)
    if not col_hist_used: 
        return None, f"Không tìm thấy cột dữ liệu ngày {prev_date.strftime('%d/%m')}."

    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}

    # --- PHASE 1: Backtest ---
    if not manual_groups:
        past_dates = []
        check_d = target_date - timedelta(days=1)
        while len(past_dates) < rolling_window:
            if check_d in cache and check_d in kq_db: past_dates.append(check_d)
            check_d -= timedelta(days=1)
            if (target_date - check_d).days > 40: break

        for d in past_dates:
            d_df = cache[d]['df']
            kq = kq_db[d]
            d_p_map = {}; d_s_map = {}
            for col in d_df.columns:
                s_p = get_col_score(col, score_std_tuple)
                if s_p > 0: d_p_map[col] = s_p
                s_s = get_col_score(col, score_mod_tuple)
                if s_s > 0: d_s_map[col] = s_s
            
            d_hist_col = None
            sorted_dates = sorted([k for k in cache[d]['hist_map'].keys() if k < d], reverse=True)
            if sorted_dates: d_hist_col = cache[d]['hist_map'][sorted_dates[0]]
            if not d_hist_col: continue
            
            try:
                hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False)
                hist_series_d = hist_series_d.str.replace(r'[^0-9X]', '', regex=True)
            except: continue

            for g in groups:
                mask = hist_series_d == g.upper()
                mems = d_df[mask]
                if mems.empty:
                    stats_std[g]['ranks'].append(999); continue
                
                top80_std = fast_get_top_nums(mems, d_p_map, d_s_map, 80, min_votes, use_inverse)
                if kq in top80_std:
                    stats_std[g]['wins'] += 1
                    stats_std[g]['ranks'].append(top80_std.index(kq) + 1)
                else: stats_std[g]['ranks'].append(999)
                
                top86_mod = fast_get_top_nums(mems, d_s_map, d_p_map, limits_config['mod'], min_votes, use_inverse)
                if kq in top86_mod: stats_mod[g]['wins'] += 1

    top6_std = []
    best_mod_grp = ""
    
    if not manual_groups:
        final_std = []
        for g, inf in stats_std.items(): 
            final_std.append((g, -inf['wins'], sum(inf['ranks']), sorted(inf['ranks'])))
        final_std.sort(key=lambda x: (x[1], x[2], x[3], x[0])) 
        top6_std = [x[0] for x in final_std[:6]]
        best_mod_grp = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]
    
    # --- PHASE 2: Predict ---
    hist_series = df[col_hist_used].astype(str).str.upper().replace('S', '6', regex=False)
    hist_series = hist_series.str.replace(r'[^0-9X]', '', regex=True)
    
    def get_final_pool(group_list, limit_dict, p_map, s_map):
        pool = []
        for g in group_list:
            mask = hist_series == g.upper()
            valid_mems = df[mask]
            lim = limit_dict.get(g, limit_dict.get('default', 80))
            res = fast_get_top_nums(valid_mems, p_map, s_map, lim, min_votes, use_inverse)
            pool.extend(res)
        return pool

    final_original = []
    final_modified = []
    
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
        final_modified = sorted(fast_get_top_nums(df[mask_mod], s_map_dict, p_map_dict, limits_config['mod'], min_votes, use_inverse))

    final_intersect = sorted(list(set(final_original).intersection(set(final_modified))))
    
    return {
        "top6_std": top6_std, "best_mod": best_mod_grp,
        "dan_goc": final_original, "dan_mod": final_modified,
        "dan_final": final_intersect, "source_col": col_hist_used,
        "stats_groups_std": stats_std, "stats_groups_mod": stats_mod
    }, None

def analyze_group_performance(start_date, end_date, cut_limit, score_map, data_cache, kq_db, min_v, inverse):
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    score_map_tuple = tuple(score_map.items())
    grp_stats = {f"{i}x": {'wins': 0, 'ranks': [], 'history': [], 'last_pred': []} for i in range(10)}
    detailed_rows = [] 
    
    # Pre-calc map
    score_map_dict = {f"M{i}": v for i, v in enumerate(list(score_map.values()))} 

    for d in reversed(dates):
        day_record = {"Ngày": d.strftime("%d/%m"), "KQ": kq_db.get(d, "N/A")}
        if d not in kq_db or d not in data_cache: 
            for g in grp_stats: 
                grp_stats[g]['history'].append(None); grp_stats[g]['ranks'].append(999); day_record[g] = "-"
            detailed_rows.append(day_record); continue
        
        curr_data = data_cache[d]
        df = curr_data['df']
        prev_date = d - timedelta(days=1)
        if prev_date not in data_cache: 
            for k in range(2, 4):
                if (d - timedelta(days=k)) in data_cache: 
                     prev_date = d - timedelta(days=k); break
        
        hist_col_name = curr_data['hist_map'].get(prev_date) if prev_date in curr_data['hist_map'] else None
        if not hist_col_name:
             for g in grp_stats: 
                 grp_stats[g]['history'].append(None); grp_stats[g]['ranks'].append(999); day_record[g] = "-"
             detailed_rows.append(day_record); continue
        
        try:
            hist_series = df[hist_col_name].astype(str).str.upper().replace('S', '6', regex=False)
            hist_series = hist_series.str.replace(r'[^0-9X]', '', regex=True)
        except: continue
        
        kq = kq_db[d]
        d_p_map = {}; d_s_map = {} 
        for col in df.columns:
            s_p = get_col_score(col, score_map_tuple)
            if s_p > 0: d_p_map[col] = s_p

        for g in grp_stats:
            mask = hist_series == g.upper()
            valid_mems = df[mask]
            
            top_list = fast_get_top_nums(valid_mems, d_p_map, d_p_map, cut_limit, min_v, inverse)
            top_set = set(top_list)
            grp_stats[g]['last_pred'] = sorted(top_list)
            
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
        
    final_report = []
    for g, info in grp_stats.items():
        hist = info['history']
        valid_days = len([x for x in hist if x is not None])
        wins = info['wins']
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

# ==============================================================================
# 5. GIAO DIỆN CHÍNH
# ==============================================================================

SCORES_PRESETS = {
    "Gốc (V24 Standard)": {
        "STD": [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    },
    "Tối ưu (Big Data 2026)": {
        "STD": [0, 1, 2, 3, 4, 8, 10, 15, 25, 40, 60],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    },
    "Lai tạo (Hybrid - Thực chiến)": {
        "STD": [0, 2, 4, 6, 12, 16, 20, 25, 30, 32, 35],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    },
    "Miền Nam (Theo Ảnh)": {
        "STD": [50, 8, 9, 10, 10, 30, 40, 30, 25, 30, 30],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    }
}

def main():
    uploaded_files = st.file_uploader("📂 Tải file CSV/Excel", type=['xlsx', 'csv'], accept_multiple_files=True)

    if 'std_0' not in st.session_state:
        def_vals = SCORES_PRESETS["Gốc (V24 Standard)"]
        for i in range(11):
            st.session_state[f'std_{i}'] = def_vals["STD"][i]
            st.session_state[f'mod_{i}'] = def_vals["MOD"][i]

    with st.sidebar:
        st.header("⚙️ Cài đặt")
        ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
        
        with st.expander("🎚️ 1. Điểm M0-M10 (Cấu hình)", expanded=False):
            def update_scores():
                choice = st.session_state.preset_choice
                if choice in SCORES_PRESETS:
                    vals = SCORES_PRESETS[choice]
                    for i in range(11):
                        st.session_state[f'std_{i}'] = vals["STD"][i]
                        st.session_state[f'mod_{i}'] = vals["MOD"][i]

            st.selectbox(
                "📚 Chọn bộ tham số mẫu:",
                options=["Tùy chỉnh"] + list(SCORES_PRESETS.keys()),
                index=3, 
                key="preset_choice",
                on_change=update_scores
            )
            st.markdown("---")
            c_s1, c_s2 = st.columns(2)
            custom_std = {}
            custom_mod = {}
            with c_s1:
                st.write("**GỐC (Std)**")
                for i in range(11): 
                    custom_std[f'M{i}'] = st.number_input(f"M{i}", key=f"std_{i}")
            with c_s2:
                st.write("**MOD**")
                for i in range(11): 
                    custom_mod[f'M{i}'] = st.number_input(f"M{i}", key=f"mod_{i}")

        st.markdown("---")
        st.header("⚖️ Lọc & Cắt")
        MIN_VOTES = st.number_input("Vote tối thiểu:", min_value=1, max_value=10, value=1)
        USE_INVERSE = st.checkbox("Chấm Điểm Đảo (Ngược)", value=False)
        
        with st.expander("✂️ Chi tiết cắt Top (V25)", expanded=False):
            L_TOP_12 = st.number_input("Top 1 & 2 lấy:", value=80)
            L_TOP_34 = st.number_input("Top 3 & 4 lấy:", value=65)
            L_TOP_56 = st.number_input("Top 5 & 6 lấy:", value=60)
            LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", value=86)

        st.markdown("---")
        with st.expander("🔥 Cấu hình Số lượng", expanded=True):
            K55_LIMIT = st.number_input("K55 lấy bao nhiêu số?", min_value=10, max_value=90, value=56)
            SMART58_LIMIT = st.number_input("Smart 58 lấy bao nhiêu số?", min_value=10, max_value=90, value=58)
        
        st.markdown("---")
        with st.expander("👁️ Tùy chọn hiển thị", expanded=True):
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                show_goc = st.checkbox("Dàn Gốc", value=True)
                show_mod = st.checkbox("Dàn Mod", value=False)
                show_f1 = st.checkbox("Final 1 (V25)", value=True)
                show_f2 = st.checkbox("Final 2 (G+K55)", value=True)
            with c_v2:
                show_k55 = st.checkbox("K55 Heart", value=True)
                show_s58 = st.checkbox("Smart 58", value=True)
                show_f3 = st.checkbox("Final 3 (G+S58)", value=True)

        st.markdown("---")
        if st.button("🗑️ XÓA CACHE", type="primary"):
            st.cache_data.clear(); st.rerun()

    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        
        with st.expander("🕵️ Trạng thái File", expanded=False):
            for s in f_status:
                if "✅" in s: st.success(s)
                else: st.error(s)
            for e in err_logs: st.error(e)
        
        if data_cache:
            limit_cfg = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
            last_d = max(data_cache.keys())
            
            tab1, tab2, tab3 = st.tabs(["📊 DỰ ĐOÁN", "🔙 BACKTEST", "🔍 PHÂN TÍCH"])
            
            with tab1:
                st.subheader("Dự đoán hàng ngày")
                c_d1, c_d2 = st.columns([1, 1])
                with c_d1: target = st.date_input("Ngày:", value=last_d)
                
                with c_d2:
                    manual_mode = st.checkbox("Thủ Công (Manual)", value=False)
                    manual_selection = []
                    manual_score_opt = "Giao thoa"
                    if manual_mode:
                        manual_selection = st.multiselect("Chọn nhóm:", options=[f"{i}x" for i in range(10)], default=["0x", "1x"])
                        manual_score_opt = st.radio("Chế độ:", ["Giao thoa", "Chỉ Gốc", "Chỉ Mod"], horizontal=True)
                
                if st.button("🚀 CHẠY", type="primary", use_container_width=True):
                    with st.spinner("Đang tính toán..."):
                        grps = manual_selection if manual_mode else None
                        res, err = calculate_v24_final(target, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, grps)
                        k55_res, k55_err = calculate_k55_integrated(target, data_cache, kq_db, K55_LIMIT)
                        s58_res, s58_err = calculate_smart58(target, data_cache, SMART58_LIMIT)

                        if err: st.error(err)
                        else:
                            st.info(f"Phân nhóm theo ngày: {res['source_col']}")
                            dan_goc = res['dan_goc']
                            dan_mod = res['dan_mod']
                            dan_f1 = res['dan_final']
                            if manual_mode:
                                if manual_score_opt == "Chỉ Gốc": dan_f1 = dan_goc
                                elif manual_score_opt == "Chỉ Mod": dan_f1 = dan_mod
                            dan_f2 = sorted(list(set(dan_goc).intersection(set(k55_res)))) if not k55_err else []
                            dan_f3 = sorted(list(set(dan_goc).intersection(set(s58_res)))) if not s58_err else []

                            cols_to_show = []
                            if show_goc: cols_to_show.append({"t": f"Gốc ({len(dan_goc)})", "d": dan_goc, "k": "Goc"})
                            if show_mod: cols_to_show.append({"t": f"Mod ({len(dan_mod)})", "d": dan_mod, "k": "Mod"})
                            if show_f1:  cols_to_show.append({"t": f"Final 1 ({len(dan_f1)})", "d": dan_f1, "k": "F1"})
                            if show_k55: cols_to_show.append({"t": f"K55 ({len(k55_res)})", "d": k55_res, "k": "K55"})
                            if show_s58: cols_to_show.append({"t": f"Smart 58 ({len(s58_res)})", "d": s58_res, "k": "S58"})
                            if show_f2:  cols_to_show.append({"t": f"F2 (G+K55): {len(dan_f2)}", "d": dan_f2, "k": "F2"})
                            if show_f3:  cols_to_show.append({"t": f"F3 (G+S58): {len(dan_f3)}", "d": dan_f3, "k": "F3"})

                            if cols_to_show:
                                cols = st.columns(len(cols_to_show))
                                for i, c_obj in enumerate(cols_to_show):
                                    with cols[i]:
                                        if "F2" in c_obj['t'] or "F3" in c_obj['t']: st.error(c_obj['t'])
                                        else: st.caption(c_obj['t'])
                                        st.text_area(c_obj['k'], ",".join(c_obj['d']), height=120, label_visibility="collapsed")
                                        
                                        # --- CẬP NHẬT MỚI: HIỂN THỊ TOP 6 & BEST MOD ---
                                        if c_obj['k'] == "Goc":
                                            t6 = ", ".join(res['top6_std'])
                                            bm = res['best_mod']
                                            st.info(f"🏆 Top 6: {t6}\n\n🌟 Best Mod: {bm}")
                                        # -----------------------------------------------
                            else: st.warning("Bạn đã tắt hết các bảng hiển thị!")

                            if target in kq_db:
                                real = kq_db[target]
                                st.markdown("---")
                                msg_map = {}
                                if show_f1: msg_map["V25"] = "WIN" if real in dan_f1 else "MISS"
                                if show_k55: msg_map["K55"] = "WIN" if (not k55_err and real in k55_res) else "MISS"
                                if show_s58: msg_map["Smart58"] = "WIN" if (not s58_err and real in s58_res) else "MISS"
                                if show_f2: msg_map["Final 2"] = "WIN" if real in dan_f2 else "MISS"
                                if show_f3: msg_map["Final 3"] = "WIN" if real in dan_f3 else "MISS"

                                if "WIN" in msg_map.values(): st.balloons()
                                r_cols = st.columns(len(msg_map))
                                for idx, (name, status) in enumerate(msg_map.items()):
                                    with r_cols[idx]:
                                        if status == "WIN": st.success(f"{name}: **{real}** WIN")
                                        else: st.error(f"{name}: **{real}** MISS")

            with tab2:
                st.subheader("Kiểm thử Backtest")
                with st.expander("⚙️ Cấu hình Backtest", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1: date_range = st.date_input("Khoảng ngày:", [last_d - timedelta(days=7), last_d])
                    with c2: bt_mode = st.selectbox("Chế độ:", [
                        "Final 1 (Giao thoa Gốc+Mod)", 
                        "Final 2 (Giao thoa Gốc+K55)", 
                        "Final 3 (Giao thoa Gốc+Smart58)",
                        "Smart 58 (Mới)", 
                        "K55 Hybrid", 
                        "Dàn Gốc"
                    ])
                    btn_backtest = st.button("🔄 CHẠY BACKTEST", use_container_width=True, type="primary")

                if btn_backtest:
                    if len(date_range) < 2: st.warning("Chọn đủ ngày.")
                    else:
                        start, end = date_range[0], date_range[1]
                        logs = []
                        bar = st.progress(0, text="Đang chạy...")
                        delta = (end - start).days + 1
                        
                        for i in range(delta):
                            d = start + timedelta(days=i)
                            bar.progress((i + 1) / delta, text=f"Đang tính: {d.strftime('%d/%m')}")
                            if d not in kq_db: continue
                            
                            t_set = []
                            if bt_mode == "Smart 58 (Mới)":
                                s58_res, s58_err = calculate_smart58(d, data_cache, SMART58_LIMIT)
                                if not s58_err: t_set = s58_res
                            elif bt_mode == "K55 Hybrid":
                                k_res, k_err = calculate_k55_integrated(d, data_cache, kq_db, K55_LIMIT)
                                if not k_err: t_set = k_res
                            elif bt_mode == "Final 2 (Giao thoa Gốc+K55)":
                                res_v25, err_v25 = calculate_v24_final(d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                                k_res, k_err = calculate_k55_integrated(d, data_cache, kq_db, K55_LIMIT)
                                if not err_v25 and not k_err: t_set = sorted(list(set(res_v25['dan_goc']).intersection(set(k_res))))
                            elif bt_mode == "Final 3 (Giao thoa Gốc+Smart58)":
                                res_v25, err_v25 = calculate_v24_final(d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                                s58_res, s58_err = calculate_smart58(d, data_cache, SMART58_LIMIT)
                                if not err_v25 and not s58_err: t_set = sorted(list(set(res_v25['dan_goc']).intersection(set(s58_res))))
                            else:
                                res, err = calculate_v24_final(d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                                if not err:
                                    if "Final 1" in bt_mode: t_set = res['dan_final']
                                    elif "Gốc" in bt_mode: t_set = res['dan_goc']
                            
                            real = kq_db[d]
                            logs.append({
                                "Ngày": d.strftime("%d/%m"), "KQ": real, 
                                "TT": "WIN" if real in t_set else "MISS", "Số số": len(t_set)
                            })
                        bar.empty()
                        
                        if logs:
                            df_log = pd.DataFrame(logs)
                            wins = df_log[df_log["TT"] == "WIN"].shape[0]
                            st.metric(label=f"Kết quả {bt_mode}", value=f"{wins}/{df_log.shape[0]} (Ngày ăn)", delta=f"{(wins/df_log.shape[0])*100:.1f}%")
                            st.dataframe(df_log, use_container_width=True, height=500, hide_index=True)

            with tab3:
                st.subheader("Phân Tích Nhóm (Matrix)")
                with st.expander("⚙️ Cấu hình Phân tích", expanded=False):
                    c_a1, c_a2 = st.columns(2)
                    with c_a1: d_range_a = st.date_input("Thời gian:", [last_d - timedelta(days=15), last_d], key="dr_a")
                    with c_a2: 
                        cut_val = st.number_input("Cắt Top:", value=60, step=5)
                        score_mode = st.radio("Hệ điểm:", ["Gốc (Std)", "Modified"], horizontal=True)
                    btn_scan = st.button("🔎 QUÉT MATRIX", use_container_width=True)
                
                if btn_scan:
                    if len(d_range_a) < 2: st.warning("Chọn đủ ngày.")
                    else:
                        with st.spinner("Đang xử lý..."):
                            s_map = custom_std if score_mode == "Gốc (Std)" else custom_mod
                            df_report, df_detail = analyze_group_performance(d_range_a[0], d_range_a[1], cut_val, s_map, data_cache, kq_db, MIN_VOTES, USE_INVERSE)
                            st.write("📊 **Thống kê tổng hợp**"); st.dataframe(df_report, use_container_width=True)
                            st.write("📅 **Chi tiết từng ngày**")
                            def color_matrix(val):
                                if val == "MISS": return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;'
                                elif val == "WIN": return 'background-color: #ccffcc; color: #006600; font-weight: bold;'
                                return ''
                            st.dataframe(df_detail.style.map(color_matrix), use_container_width=True, height=600)

if __name__ == "__main__":
    main()
