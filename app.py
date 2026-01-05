import streamlit as st
import pandas as pd
import re
import datetime
import time
import random
from datetime import timedelta
from collections import Counter
from functools import lru_cache

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V45.0 - Final Simulation", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🧬 Quang Handsome: V45.0 Simulation Core")
st.caption("🚀 Matrix Engine | Auto-Learning Backtest | Super Final Logic")

# Regex & Constants
RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CORE FUNCTIONS (V24 LOGIC - GIỮ NGUYÊN)
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

def parse_date_smart(col_str, f_m, f_y):
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
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: return None
    return None

def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords):
            return idx
    return 3

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
            d = int(full_date_match.group(1))
            m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            if m == 12 and m_global == 1: y -= 1 
            return m, y, datetime.date(y, m, d)
        except: pass
    single_day_match = re.findall(r'(\d{1,2})$', clean_name)
    if single_day_match:
        try:
            d = int(single_day_match[-1])
            return m_global, y_global, datetime.date(y_global, m_global, d)
        except: pass
    return m_global, y_global, None

@st.cache_data(ttl=600, show_spinner=False)
def load_data_v24(files):
    cache = {} 
    kq_db = {}
    err_logs = []
    file_status = []
    
    files = sorted(files, key=lambda x: x.name)
    IGNORE_KEYWORDS = ['N.CSV', 'BPĐ', 'BPD', 'BANG PHU', '~$', 'DS.CSV']

    for file in files:
        f_name_upper = file.name.upper()
        if any(kw in f_name_upper for kw in IGNORE_KEYWORDS): continue
            
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
                    
                    if s_date:
                        preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=20, engine='openpyxl')
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row, engine='openpyxl')
                        df.columns = [str(c).strip().upper().replace('M 1 0', 'M10') for c in df.columns]
                        dfs_to_process.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")
            
            elif file.name.endswith('.csv'):
                if not date_from_name: 
                    err_logs.append(f"⚠️ Bỏ qua '{file.name}': Không tìm thấy ngày")
                    continue
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
                    except: 
                        err_logs.append(f"❌ Lỗi encoding '{file.name}'")
                        continue
                
                h_row = find_header_row(preview)
                df = df_raw.iloc[h_row+1:].copy()
                df.columns = df_raw.iloc[h_row]
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
            err_logs.append(f"❌ Lỗi '{file.name}': {str(e)}")
            continue
    return cache, kq_db, file_status, err_logs

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
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
    
    if inverse:
        stats = stats.sort_values(by=['P', 'S', 'Num_Int'], ascending=[False, False, True])
    else:
        stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])

    return stats['Num'].head(top_n).tolist()

def calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None):
    if target_date not in _cache: return None
    curr_data = _cache[target_date]
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
    if prev_date not in _cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache:
                prev_date = target_date - timedelta(days=i); break
    
    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in _cache:
        col_hist_used = _cache[prev_date]['hist_map'].get(prev_date)
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
            d_df = _cache[d]['df']
            kq = _kq_db[d]
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
        "top6_std": top6_std, 
        "best_mod": best_mod_grp,
        "dan_goc": final_original,
        "dan_mod": final_modified,
        "dan_final": final_intersect, 
        "source_col": col_hist_used
    }

@st.cache_data(show_spinner=False)
def calculate_v24_final(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None):
    res = calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups)
    if not res: return None, "Lỗi dữ liệu"
    return res, None

def analyze_group_performance(start_date, end_date, cut_limit, score_map, _cache, _kq_db, min_v, inverse):
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    score_map_tuple = tuple(score_map.items())
    grp_stats = {f"{i}x": {'wins': 0, 'ranks': [], 'history': [], 'last_pred': []} for i in range(10)}
    detailed_rows = [] 
    for d in reversed(dates):
        day_record = {"Ngày": d.strftime("%d/%m"), "KQ": _kq_db.get(d, "N/A")}
        if d not in _kq_db or d not in _cache: 
             detailed_rows.append(day_record); continue
        curr_data = _cache[d]
        df = curr_data['df']
        prev_date = d - timedelta(days=1)
        if prev_date not in _cache: 
            for k in range(2, 4):
                if (d - timedelta(days=k)) in _cache: prev_date = d - timedelta(days=k); break
        hist_col_name = curr_data['hist_map'].get(prev_date) if prev_date in curr_data['hist_map'] else None
        if not hist_col_name: detailed_rows.append(day_record); continue
        try:
            hist_series = df[hist_col_name].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
        except: continue
        kq = _kq_db[d]
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
# 3. AI HUNTER & MATRIX ENGINE
# ==============================================================================

def prepare_hunter_data(test_dates, _cache, _kq_db, rolling_window):
    """Chuẩn bị dữ liệu Matrix tần suất cho AI."""
    prepared_days = []
    for d in test_dates:
        # Rolling: Lấy quá khứ của d
        past_dates = []
        check_d = d - timedelta(days=1)
        while len(past_dates) < rolling_window:
            if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
            check_d -= timedelta(days=1)
            if (d - check_d).days > 45: break
            
        day_matrix = pd.DataFrame(0, index=[f"{x:02d}" for x in range(100)], columns=[f"M{i}" for i in range(11)])
        has_data = False
        
        for pd_date in past_dates:
            df = _cache[pd_date]['df']
            for col in df.columns:
                clean_col = str(col).upper().replace(' ', '').replace('M10', 'MX')
                m_idx = -1
                if 'MX' in clean_col: m_idx = 10
                else:
                    for k in range(10): 
                        if f"M{k}" in clean_col: m_idx = k; break
                if m_idx == -1: continue
                
                vals = df[col].astype(str).str.upper()
                mask_bad = vals.str.contains(r'N|NGHI|SX|XIT|MISS|TRUOT|NGHỈ|LỖI', regex=True)
                nums = vals[~mask_bad].str.findall(r'\d+').explode().dropna()
                nums = nums.str.strip().str.zfill(2)
                nums = nums[nums.str.len() == 2]
                
                counts = nums.value_counts()
                if not counts.empty:
                    day_matrix[f"M{m_idx}"] = day_matrix[f"M{m_idx}"].add(counts, fill_value=0)
                    has_data = True
        
        kq = _kq_db.get(d, None)
        if has_data:
            prepared_days.append({'date': d, 'kq': kq, 'matrix': day_matrix})
    return prepared_days

def get_hunter_nums_for_day(matrix, genome, top_n):
    """Tính điểm Matrix cho 1 ngày cụ thể và lấy Top N số."""
    score_vec = pd.Series([genome.get(f"M{i}", 0) for i in range(11)], index=[f"M{i}" for i in range(11)])
    scores = matrix.dot(score_vec)
    scores = scores[scores > 0]
    if scores.empty: return []
    
    df_res = scores.to_frame(name='S')
    df_res['N_Int'] = df_res.index.astype(int)
    
    # Sắp xếp: Điểm cao -> thấp, Số bé -> lớn
    df_res = df_res.sort_values(by=['S', 'N_Int'], ascending=[False, True])
    return df_res.index[:top_n].tolist()

def evaluate_fitness_optimized(genome, prepared_days, max_nums):
    """Đánh giá bộ Gen (AI Loop)."""
    wins = 0; total_nums = 0; valid_days = 0; history = []
    for day_obj in prepared_days:
        kq = day_obj['kq']
        if kq is None: continue 
        top_nums = get_hunter_nums_for_day(day_obj['matrix'], genome, max_nums)
        is_win = kq in top_nums
        history.append("W" if is_win else "L")
        if is_win: wins += 1
        total_nums += len(top_nums)
        valid_days += 1
        
    if valid_days == 0: return 0, 0, 999, history
    
    avg_nums = total_nums / valid_days
    win_rate = (wins / valid_days) * 100
    fitness = win_rate * 10 - avg_nums 
    return fitness, win_rate, avg_nums, history

def generate_random_genome():
    possible_values = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
    return {f"M{i}": random.choice(possible_values) for i in range(11)}

def mutate_genome(genome, mutation_rate=0.2):
    new_genome = genome.copy()
    possible_values = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
    for k in new_genome:
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                new_genome[k] = random.choice(possible_values)
            else:
                curr_idx = possible_values.index(new_genome[k]) if new_genome[k] in possible_values else 0
                step = random.choice([-1, 1])
                new_idx = max(0, min(len(possible_values)-1, curr_idx + step))
                new_genome[k] = possible_values[new_idx]
    return new_genome

def crossover_genome(parent1, parent2):
    child = {}
    for i in range(11):
        key = f"M{i}"
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

def run_genetic_search(target_date, _cache, _kq_db, max_allowed_nums, test_days_limit, generations, population_size, progress_bar=None, status_text=None):
    """Chạy thuật toán di truyền."""
    test_dates = []
    check = target_date - timedelta(days=1)
    
    while len(test_dates) < test_days_limit: 
        if check in _kq_db and check in _cache: test_dates.append(check)
        check -= timedelta(days=1)
        if (target_date - check).days > (test_days_limit * 2 + 40): break
    
    if not test_dates: return []

    if status_text: status_text.text("⚙️ Đang mã hóa dữ liệu (Matrix)...")
    prepared_data = prepare_hunter_data(test_dates, _cache, _kq_db, rolling_window=10)
    
    population = [generate_random_genome() for _ in range(population_size)]
    population[0] = {f"M{i}": 0 for i in range(11)}; population[0]['M10']=50 
    
    best_solution = None
    history_best = []

    for gen in range(generations):
        scored_pop = []
        for genome in population:
            fit, wr, avg, hist = evaluate_fitness_optimized(genome, prepared_data, max_allowed_nums)
            scored_pop.append({'genome': genome, 'fitness': fit, 'wr': wr, 'avg': avg, 'hist': hist})
        
        scored_pop.sort(key=lambda x: x['fitness'], reverse=True)
        current_best = scored_pop[0]
        
        if best_solution is None or current_best['fitness'] > best_solution['fitness']:
            best_solution = current_best
        history_best.append(best_solution)

        if status_text and gen % 5 == 0:
            status_text.markdown(f"🏃 Gen {gen+1}/{generations} | Best: {current_best['wr']:.0f}%")
        if progress_bar: progress_bar.progress((gen + 1) / generations)

        elite_count = int(population_size * 0.3)
        new_pop = [x['genome'] for x in scored_pop[:elite_count]]
        while len(new_pop) < population_size:
            p1 = random.choice(scored_pop[:10])['genome']
            p2 = random.choice(scored_pop[:10])['genome']
            child = crossover_genome(p1, p2)
            child = mutate_genome(child, mutation_rate=0.3)
            new_pop.append(child)
        population = new_pop

    unique_solutions = []
    seen = set()
    date_labels = [d.strftime("%d/%m") for d in test_dates] 
    
    for sol in sorted(history_best, key=lambda x: x['fitness'], reverse=True):
        s_str = str(sol['genome'])
        if s_str not in seen:
            raw_hist = sol['hist']
            icon_hist = ["✅" if h == 'W' else "❌" for h in raw_hist]
            display_hist_str = " | ".join(reversed(icon_hist))
            display_date_str = " | ".join(reversed(date_labels))
            unique_solutions.append({
                "Name": f"AI Hunter-{random.randint(100,999)}",
                "WinRate": sol['wr'],
                "AvgNums": sol['avg'],
                "Scores": sol['genome'],
                "HistoryIcons": display_hist_str,
                "HistoryDates": display_date_str,
                "CutQty": max_allowed_nums
            })
            seen.add(s_str)
            if len(unique_solutions) >= 5: break
            
    return unique_solutions

# ==============================================================================
# 4. GIAO DIỆN CHÍNH & LOGIC KẾT HỢP
# ==============================================================================

def apply_hunter_as_filter_callback(scores, cut_qty):
    st.session_state['hunter_genome'] = scores
    st.session_state['hunter_cut_sync'] = cut_qty 
    st.session_state['hunter_msg'] = f"✅ Đã nạp cấu hình! Tự động set Top AI = {cut_qty}"

SCORES_PRESETS = {
    "Gốc (V24 Standard)": {
        "STD": [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    },
    "Miền Trung": {
        "STD": [60, 8, 9, 10, 10, 30, 70, 30, 30, 30, 30],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    },
    "🔥 CH1: Bám Đuôi (An Toàn)": {
        "STD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70],
        "MOD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70]
    },
    "⚡ CH2: Đột Biến (Săn Số Ít)": {
        "STD": [60, 0, 0, 10, 10, 30, 50, 30, 0, 30, 20],
        "MOD": [60, 0, 0, 10, 10, 30, 50, 30, 0, 30, 20]
    },
    "⚖️ CH3: Cân Bằng": {
        "STD": [30, 25, 20, 20, 20, 30, 40, 30, 20, 25, 50],
        "MOD": [30, 25, 20, 20, 20, 30, 40, 30, 20, 25, 50]
    }
}

def main():
    uploaded_files = st.file_uploader("📂 Tải file CSV/Excel", type=['xlsx', 'csv'], accept_multiple_files=True)

    if 'std_0' not in st.session_state:
        def_vals = SCORES_PRESETS["Gốc (V24 Standard)"]
        for i in range(11):
            st.session_state[f'std_{i}'] = def_vals["STD"][i]
            st.session_state[f'mod_{i}'] = def_vals["MOD"][i]
    
    if 'hunter_genome' not in st.session_state:
        st.session_state['hunter_genome'] = None

    with st.sidebar:
        st.header("⚙️ Cài đặt V24")
        ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
        with st.expander("🎚️ Điểm M0-M10 (Cấu hình)", expanded=False):
            def update_scores():
                choice = st.session_state.preset_choice
                if choice in SCORES_PRESETS:
                    vals = SCORES_PRESETS[choice]
                    for i in range(11):
                        st.session_state[f'std_{i}'] = vals["STD"][i]
                        st.session_state[f'mod_{i}'] = vals["MOD"][i]
            st.selectbox("📚 Chọn bộ mẫu:", options=["Tùy chỉnh"] + list(SCORES_PRESETS.keys()), index=1, key="preset_choice", on_change=update_scores)
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.write("**GỐC**")
                for i in range(11): st.number_input(f"M{i}", key=f"std_{i}")
            with c_s2:
                st.write("**MOD**")
                for i in range(11): st.number_input(f"M{i}", key=f"mod_{i}")

        st.markdown("---")
        st.header("⚖️ Lọc & Cắt")
        MIN_VOTES = st.number_input("Vote tối thiểu:", min_value=1, max_value=10, value=1)
        USE_INVERSE = st.checkbox("Chấm Điểm Đảo (Ngược)", value=False)
        with st.expander("✂️ Chi tiết cắt Top", expanded=True):
            st.caption("--- Cắt V24 ---")
            L_TOP_12 = st.number_input("Top 1 & 2 lấy:", value=80, key="L12")
            L_TOP_34 = st.number_input("Top 3 & 4 lấy:", value=65, key="L34")
            L_TOP_56 = st.number_input("Top 5 & 6 lấy:", value=60, key="L56")
            LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", value=86, key="LMOD")
            
            st.caption("--- Cắt AI Filter ---")
            default_hunter_val = 60
            if 'hunter_cut_sync' in st.session_state:
                default_hunter_val = st.session_state['hunter_cut_sync']
            L_HUNTER_CUT = st.number_input("Top AI Hunter lấy:", value=default_hunter_val, key="L_HUNTER_CUT_INPUT", help="Số lượng tự động cập nhật khi áp dụng Hunter")

        if st.button("🗑️ XÓA CACHE", type="primary"):
            st.cache_data.clear(); st.rerun()

    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        with st.expander("🕵️ Trạng thái File", expanded=False):
            for s in f_status: st.success(s)
            for e in err_logs: st.error(e)
        
        if 'hunter_msg' in st.session_state:
            st.toast(st.session_state['hunter_msg'], icon="🔥")
            del st.session_state['hunter_msg']

        if data_cache:
            limit_cfg = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
            last_d = max(data_cache.keys())
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 DỰ ĐOÁN", "🔙 BACKTEST", "🔍 MATRIX", "🧬 AI HUNTER"])
            
            # --- TAB 1: DỰ ĐOÁN ---
            with tab1:
                st.subheader("Dự đoán & Giao thoa")
                c_opt1, c_opt2, c_opt3, c_opt4, c_opt5 = st.columns(5)
                with c_opt1: show_goc = st.checkbox("Gốc", value=True)
                with c_opt2: show_mod = st.checkbox("Mod", value=True)
                with c_opt3: show_final = st.checkbox("Final 1", value=True)
                with c_opt4: show_raw = st.checkbox("AI Raw", value=False)
                with c_opt5: show_super = st.checkbox("Super Final", value=True)

                c_d1, c_d2 = st.columns([1, 1.5])
                with c_d1: target = st.date_input("Ngày:", value=last_d)
                with c_d2:
                    st.write("")
                    if st.button("🚀 CHẠY PHÂN TÍCH", type="primary"):
                        with st.spinner("Đang tính toán..."):
                            custom_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                            custom_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                            res_v24, err = calculate_v24_final(target, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                            
                            res_hunter = []
                            if st.session_state['hunter_genome']:
                                p_data = prepare_hunter_data([target], data_cache, kq_db, 10)
                                if p_data:
                                    res_hunter = get_hunter_nums_for_day(p_data[0]['matrix'], st.session_state['hunter_genome'], L_HUNTER_CUT)
                            
                            st.session_state['run_result'] = {'v24': res_v24, 'hunter': res_hunter, 'target': target}
                
                if st.session_state['hunter_genome']:
                    st.info(f"🔥 AI Filter đang bật: Lấy Top {L_HUNTER_CUT} | Đã được đồng bộ từ bên Săn Gen.")

                if 'run_result' in st.session_state and st.session_state['run_result']['target'] == target:
                    rr = st.session_state['run_result']
                    res = rr['v24']
                    dan_hunter = rr['hunter']
                    
                    if res:
                        st.success(f"Phân nhóm nguồn: {res['source_col']}")
                        
                        dan_super_final = []
                        if dan_hunter:
                            dan_super_final = sorted(list(set(res['dan_goc']).intersection(set(dan_hunter))))
                        
                        cols_to_show = []
                        if show_goc: cols_to_show.append({"t": f"Gốc ({len(res['dan_goc'])})", "d": res['dan_goc']})
                        if show_mod: cols_to_show.append({"t": f"Mod ({len(res['dan_mod'])})", "d": res['dan_mod']})
                        if show_final: cols_to_show.append({"t": f"Final 1 ({len(res['dan_final'])})", "d": res['dan_final']})
                        if show_raw and dan_hunter:
                            cols_to_show.append({"t": f"AI Raw ({len(dan_hunter)})", "d": dan_hunter})
                        if show_super and dan_hunter: 
                             cols_to_show.append({"t": f"⚡ Super Final ({len(dan_super_final)})", "d": dan_super_final})

                        if cols_to_show:
                            cols = st.columns(len(cols_to_show))
                            for i, item in enumerate(cols_to_show):
                                with cols[i]:
                                    st.caption(item['t'])
                                    st.text_area(f"area_{i}", ",".join(item['d']), height=120)

                        if target in kq_db:
                            real = kq_db[target]
                            msg = []
                            if real in res['dan_goc']: msg.append("Gốc")
                            if real in res['dan_mod']: msg.append("Mod")
                            if real in res['dan_final']: msg.append("F1")
                            if dan_hunter and real in dan_super_final: msg.append("SUPER")
                            
                            if msg: st.balloons(); st.success(f"WIN {real} tại: {', '.join(msg)}")
                            else: st.error(f"MISS {real}")

            # --- TAB 2: BACKTEST SIMULATION ---
            with tab2:
                st.subheader("Backtest Đa Chế Độ (Có Simulation)")
                c_b1, c_b2, c_b3 = st.columns(3)
                with c_b1: d_start = st.date_input("Từ ngày:", value=last_d - timedelta(days=7))
                with c_b2: d_end = st.date_input("Đến ngày:", value=last_d)
                with c_b3: 
                    mode = st.selectbox("Chế độ Test", 
                                        ["Dàn Gốc", "Dàn Mod", "Final 1 (V24)", 
                                         "🤖 AI Hunter (Simulation)", "⚡ Super Final (Simulation)"])
                
                # Logic ẩn hiện Input cho Simulation
                learn_days = 5
                cut_qty_sim = 60
                
                if "Simulation" in mode:
                    st.info("ℹ️ Chế độ Thực Chiến: Hệ thống sẽ tự động Săn Gen lại cho TỪNG NGÀY (Sẽ chạy chậm hơn).")
                    c_sim1, c_sim2 = st.columns(2)
                    with c_sim1: learn_days = st.number_input("Số ngày Học (Learning Days):", value=5, min_value=3)
                    with c_sim2: cut_qty_sim = st.number_input("Số lượng cắt (AI Cut):", value=60)

                if st.button("Chạy Backtest"):
                    custom_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                    custom_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    dates_range = [d_start + timedelta(days=i) for i in range((d_end - d_start).days + 1)]
                    
                    logs = []
                    bar = st.progress(0)
                    status_text = st.empty()

                    for idx, d in enumerate(dates_range):
                        bar.progress((idx + 1) / len(dates_range))
                        if d not in kq_db: continue
                        
                        # 1. Tính V24 (Luôn cần)
                        res_v24 = calculate_v24_logic_only(d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                        
                        # 2. Tính Simulation AI (Nếu chọn)
                        dan_hunter_temp = []
                        if "Simulation" in mode:
                            status_text.text(f"🛠️ Đang Săn Gen cho ngày {d.strftime('%d/%m')}...")
                            # Tự động chạy săn gen mini (15 gen, pop 20) cho nhanh
                            best_sols = run_genetic_search(d, data_cache, kq_db, cut_qty_sim, learn_days, 15, 20)
                            if best_sols:
                                best_genome = best_sols[0]['Scores']
                                # Áp dụng ngay genome đó cho ngày d
                                p_data = prepare_hunter_data([d], data_cache, kq_db, 10)
                                if p_data:
                                    dan_hunter_temp = get_hunter_nums_for_day(p_data[0]['matrix'], best_genome, cut_qty_sim)
                        
                        # 3. Kết quả
                        final_set = []
                        if mode == "Dàn Gốc": final_set = res_v24['dan_goc'] if res_v24 else []
                        elif mode == "Dàn Mod": final_set = res_v24['dan_mod'] if res_v24 else []
                        elif mode == "Final 1 (V24)": final_set = res_v24['dan_final'] if res_v24 else []
                        elif mode == "🤖 AI Hunter (Simulation)": final_set = dan_hunter_temp
                        elif mode == "⚡ Super Final (Simulation)":
                            if res_v24 and dan_hunter_temp:
                                final_set = sorted(list(set(res_v24['dan_goc']).intersection(set(dan_hunter_temp))))

                        status = "WIN" if kq_db[d] in final_set else "MISS"
                        logs.append({"Ngày": d.strftime("%d/%m"), "KQ": kq_db[d], "TT": status, "Số lượng": len(final_set)})
                    
                    bar.empty()
                    status_text.empty()
                    if logs:
                        df_log = pd.DataFrame(logs)
                        wins = df_log[df_log["TT"]=="WIN"].shape[0]
                        st.metric(f"WinRate ({mode})", f"{wins}/{len(df_log)}", delta=f"{(wins/len(df_log))*100:.1f}%")
                        st.dataframe(df_log, use_container_width=True)

            # --- TAB 3: MATRIX ---
            with tab3:
                st.subheader("Phân Tích Matrix")
                c_a1, c_a2 = st.columns(2)
                with c_a1: d_range_a = st.date_input("Thời gian:", [last_d - timedelta(days=15), last_d], key="dr_a")
                with c_a2: 
                    cut_val = st.number_input("Cắt Top:", value=60, step=5, key="cut_mtx")
                    score_mode = st.radio("Hệ điểm:", ["Gốc (Std)", "Modified"], horizontal=True)
                
                if st.button("🔎 QUÉT MATRIX", use_container_width=True):
                    s_map_vals = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)} if score_mode == "Gốc (Std)" else {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    with st.spinner("Đang xử lý..."):
                        df_report, df_detail = analyze_group_performance(d_range_a[0], d_range_a[1], cut_val, s_map_vals, data_cache, kq_db, MIN_VOTES, USE_INVERSE)
                        st.dataframe(df_report, use_container_width=True)
                        st.dataframe(df_detail, use_container_width=True)

            # --- TAB 4: AI HUNTER GENETIC ---
            with tab4:
                st.subheader("🧬 AI GENETIC HUNTER")
                
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    target_hunter = st.date_input("Ngày dự đoán:", value=last_d, key="t_hunter")
                    days_backtest = st.number_input("Số ngày Học (Learning Days):", min_value=3, max_value=20, value=5)
                    max_nums_hunter = st.slider("Max Số Lượng (Để tính điểm):", 40, 100, 65, key="mx_hunter")
                    
                    if st.button("🚀 CHẠY SĂN GEN", type="primary"):
                        st.toast(f"🚀 Đang học {days_backtest} ngày...", icon="⚡") 
                        prog_bar = st.progress(0)
                        status_txt = st.empty()
                        
                        best_scenarios = run_genetic_search(
                            target_hunter, data_cache, kq_db, max_nums_hunter,
                            days_backtest, 20, 30, prog_bar, status_txt
                        )
                        prog_bar.empty()
                        if best_scenarios:
                            st.session_state['best_scenarios'] = best_scenarios
                            st.success(f"✅ Tìm thấy {len(best_scenarios)} bộ gen.")
                
                with c2:
                    if 'best_scenarios' in st.session_state:
                        scenarios = st.session_state['best_scenarios']
                        st.write(f"🎉 **Kết quả Săn (Tự động lưu Max Số Lượng):**")
                        for idx, sc in enumerate(scenarios):
                            with st.expander(f"🏅 Rank {idx+1} | Win {sc['WinRate']:.0f}% | {sc['AvgNums']:.1f} số", expanded=(idx==0)):
                                st.write("Biểu đồ nhịp cầu (Trái: Cũ -> Phải: Mới):")
                                st.code(f"{sc['HistoryDates']}\n{sc['HistoryIcons']}")
                                st.json(sc['Scores'])
                                st.button(
                                    f"👉 Áp dụng Ngay (Top {sc['CutQty']})", 
                                    key=f"apply_gen_{idx}",
                                    on_click=apply_hunter_as_filter_callback,
                                    args=(sc['Scores'], sc['CutQty'])
                                )

if __name__ == "__main__":
    main()
