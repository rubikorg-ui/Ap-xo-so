import streamlit as st
import pandas as pd
import re
import datetime
import time
from datetime import timedelta
from collections import Counter
from functools import lru_cache

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V42.1 - Stable Core", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🛡️ Quang Handsome: V42.1 Stable Core")
st.caption("🚀 Fix lỗi StreamlitAPIException | Tối ưu Data Loader | Cấu hình Miền Trung")

# Regex & Sets
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
    
    # Sắp xếp file để đảm bảo thứ tự
    files = sorted(files, key=lambda x: x.name)

    # DANH SÁCH FILE CẦN BỎ QUA (IGNORE LIST)
    IGNORE_KEYWORDS = ['N.CSV', 'BPĐ', 'BPD', 'BANG PHU', '~$', 'DS.CSV']

    for file in files:
        f_name_upper = file.name.upper()
        
        # 1. BỘ LỌC FILE RÁC/KHÔNG CẦN THIẾT
        if any(kw in f_name_upper for kw in IGNORE_KEYWORDS): 
            continue
            
        f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
        
        try:
            dfs_to_process = []
            
            # XỬ LÝ FILE EXCEL
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
                        # Fix lỗi cột M 1 0 do merge cell
                        df.columns = [str(c).strip().upper().replace('M 1 0', 'M10') for c in df.columns]
                        dfs_to_process.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")
            
            # XỬ LÝ FILE CSV (Dạng data của bạn)
            elif file.name.endswith('.csv'):
                if not date_from_name: 
                    err_logs.append(f"⚠️ Bỏ qua '{file.name}': Không tìm thấy ngày trong tên file")
                    continue
                
                try:
                    preview = pd.read_csv(file, header=None, nrows=20, encoding='utf-8')
                    file.seek(0)
                    df_raw = pd.read_csv(file, header=None, encoding='utf-8')
                except:
                    file.seek(0)
                    try:
                        # Thử encoding khác cho file tiếng Việt cũ
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

            # XỬ LÝ DATAFRAME SAU KHI LOAD
            for t_date, df in dfs_to_process:
                df.columns = [str(c).strip().upper() for c in df.columns]
                hist_map = {}
                
                # Quét các cột ngày tháng trong file (nếu có cột lịch sử)
                for col in df.columns:
                    if "UNNAMED" in col: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: hist_map[d_obj] = col
                
                # Tìm KQ (Kết quả) để lưu vào DB
                kq_row = None
                if not df.empty:
                    # Chỉ quét 2 cột đầu để tìm chữ KQ, tránh quét toàn bảng gây chậm
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
            err_logs.append(f"❌ Lỗi nghiêm trọng '{file.name}': {str(e)}")
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
# 3. AUTO-HUNTER
# ==============================================================================

def analyze_column_ranks(target_date, lookback, _cache, _kq_db):
    past_dates = []
    check_d = target_date - timedelta(days=1)
    while len(past_dates) < lookback:
        if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
        check_d -= timedelta(days=1)
        if (target_date - check_d).days > 60: break
    col_hits = Counter()
    if not past_dates: return []
    for d in past_dates:
        df = _cache[d]['df']
        kq = _kq_db[d]
        for col in df.columns:
            clean_name = str(col).upper().replace(" ", "")
            if re.match(r'^M\d+$', clean_name):
                all_vals = " ".join(df[col].astype(str).tolist())
                if kq in get_nums(all_vals): col_hits[clean_name] += 1
    return col_hits.most_common()

def generate_smart_scenarios(ranked_cols):
    scenarios = []
    if not ranked_cols:
        s_def = {f"M{i}": 0 for i in range(11)}; s_def["M10"] = 50; s_def["M9"] = 30
        scenarios.append(("Mặc định (Backup)", s_def))
        return scenarios
    
    s1 = {f"M{i}": 0 for i in range(11)}
    if len(ranked_cols) >= 2:
        s1[ranked_cols[0][0]] = 60
        s1[ranked_cols[1][0]] = 40
    scenarios.append(("Bám Top 2 (Hot)", s1))
    
    s2 = {f"M{i}": 0 for i in range(11)}
    weights = [50, 40, 30, 20, 10]
    for idx, (col, _) in enumerate(ranked_cols[:5]):
        if idx < len(weights): s2[col] = weights[idx]
    scenarios.append(("Top 5 Phân Bố", s2))
    
    s3 = {f"M{i}": 0 for i in range(11)}
    for idx, (col, _) in enumerate(ranked_cols[:8]):
        s3[col] = 20
    scenarios.append(("Top 8 An Toàn", s3))
    
    s4 = {f"M{i}": 0 for i in range(11)}
    if len(ranked_cols) >= 1: s4[ranked_cols[0][0]] = 100
    scenarios.append(("Top 1 Gánh Team", s4))
    return scenarios

def hunt_best_scenario(target_date, _cache, _kq_db, fixed_limits, min_v, use_inv, max_allowed_nums, progress_bar=None, status_text=None):
    if status_text: status_text.text("🤖 Bước 1/3: Đang phân tích xu hướng...")
    ranked = analyze_column_ranks(target_date, 15, _cache, _kq_db)
    
    if not ranked and status_text:
        status_text.text("⚠️ Không đọc được cột M nào. Dùng mặc định.")
        ranked = [('M10', 10), ('M9', 8), ('M8', 6), ('M5', 5), ('M6', 4)]

    scenarios = generate_smart_scenarios(ranked)
    total_steps = len(scenarios)
    results = []
    test_dates = []
    check = target_date - timedelta(days=1)
    while len(test_dates) < 5:
        if check in _kq_db: test_dates.append(check)
        check -= timedelta(days=1)
        if (target_date - check).days > 30: break
    
    for idx, (name, score_set) in enumerate(scenarios):
        if progress_bar: progress_bar.progress((idx + 1) / total_steps)
        if status_text: status_text.text(f"🤖 Bước 2/3: Đang đấu giải '{name}'...")
        wins = 0
        total_nums = 0
        valid = 0
        for d in test_dates:
            res = calculate_v24_logic_only(d, 3, _cache, _kq_db, fixed_limits, min_v, score_set, score_set, use_inv, None)
            if res:
                t = res['dan_final']
                if _kq_db[d] in t: wins += 1
                total_nums += len(t)
                valid += 1
        if valid > 0:
            avg = total_nums / valid
            wr = (wins / valid) * 100
            if avg <= max_allowed_nums:
                results.append({"Name": name, "WinRate": wr, "AvgNums": avg, "Scores": score_set})
    
    if status_text: status_text.text("✅ Hoàn tất!")
    results.sort(key=lambda x: (-x['WinRate'], x['AvgNums']))
    return results

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================

# Callback cập nhật điểm để tránh lỗi StreamlitAPIException
def apply_hunter_callback(scores):
    for k, v in scores.items():
        key_suffix = k[1:] 
        st.session_state[f'std_{key_suffix}'] = v
        st.session_state[f'mod_{key_suffix}'] = v
    st.session_state['applied_success'] = True

SCORES_PRESETS = {
    "Gốc (V24 Standard)": {
        "STD": [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    },
    "Miền Nam (Theo Ảnh)": {
        "STD": [50, 8, 9, 10, 10, 30, 40, 30, 25, 30, 30],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
    },
    "Miền Trung": {
        "STD": [60, 8, 9, 10, 10, 30, 70, 30, 30, 30, 30],
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
        with st.expander("✂️ Chi tiết cắt Top (V25)", expanded=True):
            L_TOP_12 = st.number_input("Top 1 & 2 lấy:", value=80, key="L12")
            L_TOP_34 = st.number_input("Top 3 & 4 lấy:", value=65, key="L34")
            L_TOP_56 = st.number_input("Top 5 & 6 lấy:", value=60, key="L56")
            LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", value=86, key="LMOD")

        st.markdown("---")
        with st.expander("👁️ Hiển thị (Dự Đoán)", expanded=True):
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                show_goc = st.checkbox("Hiện Gốc", value=True)
                show_mod = st.checkbox("Hiện Mod", value=True)
            with c_v2:
                show_final = st.checkbox("Hiện Final", value=True)

        if st.button("🗑️ XÓA CACHE", type="primary"):
            st.cache_data.clear(); st.rerun()

    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        with st.expander("🕵️ Trạng thái File", expanded=False):
            for s in f_status: st.success(s)
            for e in err_logs: st.error(e)
        
        # Check success flag
        if st.session_state.get('applied_success'):
            st.toast("✅ Đã áp dụng cấu hình thành công!", icon="🎉")
            st.session_state['applied_success'] = False

        if data_cache:
            limit_cfg = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
            last_d = max(data_cache.keys())
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 DỰ ĐOÁN", "🔙 BACKTEST", "🔍 MATRIX", "🏹 SĂN KỊCH BẢN"])
            
            with tab1:
                st.subheader("Dự đoán thủ công (3 Bảng)")
                c_d1, c_d2 = st.columns([1, 1])
                with c_d1: target = st.date_input("Ngày:", value=last_d)
                
                if st.button("🚀 CHẠY PHÂN TÍCH", type="primary", use_container_width=True):
                    with st.spinner("Đang tính toán..."):
                        # Lấy values từ session state
                        custom_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                        custom_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                        res, err = calculate_v24_final(target, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                        st.session_state['run_result'] = {'res': res, 'err': err, 'target': target}

                if 'run_result' in st.session_state and st.session_state['run_result']['target'] == target:
                    rr = st.session_state['run_result']
                    res = rr['res']
                    if not rr['err']:
                        st.success(f"Phân nhóm nguồn: {res['source_col']}")
                        cols_to_show = []
                        if show_goc: cols_to_show.append({"t": f"Gốc ({len(res['dan_goc'])})", "d": res['dan_goc'], "k": "Goc"})
                        if show_mod: cols_to_show.append({"t": f"Mod ({len(res['dan_mod'])})", "d": res['dan_mod'], "k": "Mod"})
                        if show_final: cols_to_show.append({"t": f"Final 1 ({len(res['dan_final'])})", "d": res['dan_final'], "k": "F1"})
                        
                        if cols_to_show:
                            cols = st.columns(len(cols_to_show))
                            for i, c_obj in enumerate(cols_to_show):
                                with cols[i]:
                                    st.caption(c_obj['t'])
                                    st.text_area(c_obj['k'], ",".join(c_obj['d']), height=120)
                                    if c_obj['k'] == "Goc":
                                        st.info(f"🏆 Top 6 Gốc: {', '.join(res['top6_std'])}\n\n🌟 Best Mod: {res['best_mod']}")
                        
                        if target in kq_db:
                            real = kq_db[target]
                            if real in res['dan_final']: st.balloons(); st.success(f"WIN {real}")
                            else: st.error(f"MISS {real}")

            with tab2:
                st.subheader("Backtest (Đã khôi phục)")
                c_b1, c_b2 = st.columns(2)
                with c_b1: d_start = st.date_input("Từ ngày:", value=last_d - timedelta(days=7))
                with c_b2: d_end = st.date_input("Đến ngày:", value=last_d)
                
                if st.button("Chạy Backtest (Final 1)"):
                    custom_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                    custom_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    if d_start > d_end: st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")
                    else:
                        dates_range = [d_start + timedelta(days=i) for i in range((d_end - d_start).days + 1)]
                        logs = []
                        bar = st.progress(0)
                        for idx, d in enumerate(dates_range):
                            bar.progress((idx + 1) / len(dates_range))
                            if d not in kq_db: continue
                            res = calculate_v24_logic_only(d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                            if res:
                                t = res['dan_final']
                                status = "WIN" if kq_db[d] in t else "MISS"
                                logs.append({"Ngày": d.strftime("%d/%m"), "KQ": kq_db[d], "TT": status, "Số lượng": len(t)})
                        bar.empty()
                        if logs:
                            df_log = pd.DataFrame(logs)
                            wins = df_log[df_log["TT"]=="WIN"].shape[0]
                            st.metric("WinRate (Final 1)", f"{wins}/{len(df_log)}", delta=f"{(wins/len(df_log))*100:.1f}%")
                            st.dataframe(df_log, use_container_width=True)

            with tab3:
                st.subheader("Phân Tích Matrix")
                with st.expander("⚙️ Cấu hình", expanded=True):
                    c_a1, c_a2 = st.columns(2)
                    with c_a1: d_range_a = st.date_input("Thời gian:", [last_d - timedelta(days=15), last_d], key="dr_a")
                    with c_a2: 
                        cut_val = st.number_input("Cắt Top:", value=60, step=5, key="cut_mtx")
                        score_mode = st.radio("Hệ điểm:", ["Gốc (Std)", "Modified"], horizontal=True)
                    btn_scan = st.button("🔎 QUÉT MATRIX", use_container_width=True)
                
                if btn_scan:
                    s_map_vals = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)} if score_mode == "Gốc (Std)" else {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                    with st.spinner("Đang xử lý..."):
                        df_report, df_detail = analyze_group_performance(d_range_a[0], d_range_a[1], cut_val, s_map_vals, data_cache, kq_db, MIN_VOTES, USE_INVERSE)
                        st.dataframe(df_report, use_container_width=True)
                        st.dataframe(df_detail, use_container_width=True)

            with tab4:
                st.subheader("🏹 Săn Kịch Bản Tối Ưu")
                st.info("AI tự động phân tích tần suất trúng, tạo ra các chiến thuật và chọn cái tốt nhất.")
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    target_hunter = st.date_input("Ngày:", value=last_d, key="t_hunter")
                    max_nums_hunter = st.slider("Max Số Lượng:", 40, 80, 65, key="mx_hunter")
                    
                    if st.button("🏹 BẮT ĐẦU SĂN", type="primary"):
                        st.toast("🚀 Đã nhận lệnh! AI đang khởi động...") 
                        prog_bar = st.progress(0)
                        status_txt = st.empty()
                        
                        best_scenarios = hunt_best_scenario(
                            target_hunter, data_cache, kq_db, limit_cfg, 
                            MIN_VOTES, USE_INVERSE, max_nums_hunter, 
                            progress_bar=prog_bar, status_text=status_txt
                        )
                        prog_bar.empty()
                        status_txt.empty()
                        st.session_state['best_scenarios'] = best_scenarios
                
                with c2:
                    if 'best_scenarios' in st.session_state:
                        scenarios = st.session_state['best_scenarios']
                        if not scenarios:
                            st.warning("⚠️ Không tìm thấy kịch bản nào phù hợp tiêu chí số lượng. Hãy tăng 'Max Số Lượng' lên 70 hoặc 75.")
                        else:
                            st.success(f"🎉 Tìm thấy {len(scenarios)} kịch bản tiềm năng!")
                            for idx, sc in enumerate(scenarios):
                                with st.expander(f"🏅 #{idx+1}: {sc['Name']} | Win {sc['WinRate']:.0f}% | TB {sc['AvgNums']:.1f} số", expanded=(idx==0)):
                                    st.write("**Bộ điểm đề xuất:**")
                                    st.json(sc['Scores'])
                                    st.button(f"👉 Áp dụng #{idx+1}", key=f"apply_{idx}", on_click=apply_hunter_callback, args=(sc['Scores'],))

if __name__ == "__main__":
    main()
