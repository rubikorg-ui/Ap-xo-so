import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta
import io
from functools import lru_cache

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V24", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🎯 Quang Handsome: Matrix Edition")
st.caption("🚀 Mobile Optimized | V24 Core Logic")

# Pre-compile Regex & Sets
RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    """Tách số, lọc từ khóa xấu và chuẩn hóa 01-99."""
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    
    raw_nums = RE_NUMS.findall(s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name, mapping_tuple):
    """Logic lấy điểm cột - Ưu tiên M10."""
    clean = RE_CLEAN_SCORE.sub('', str(col_name).upper())
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
    y_match = re.search(r'20\d{2}', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12

    full_date_match = re.search(r'-\s*(\d{1,2})[\.\-](\d{1,2})[\.\-](20\d{2})$', clean_name)
    if full_date_match:
        try:
            d, m, y = int(full_date_match.group(1)), int(full_date_match.group(2)), int(full_date_match.group(3))
            return m, y, datetime.date(y, m, d)
        except: pass
    return m_global, y_global, None

@st.cache_data(ttl=600)
def load_data_v24(files):
    cache = {} 
    kq_db = {}
    err_logs = []
    file_status = []

    for file in files:
        if file.name.upper() == 'N.CSV' or file.name.startswith('~$'): continue
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
                        for idx, val in df[col_check].astype(str).items():
                            if "KQ" in val.upper() or "KẾT QUẢ" in val.upper():
                                kq_row = df.loc[idx]
                                break
                        if kq_row is not None: break
            
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
# 3. CORE LOGIC (GIỮ NGUYÊN 100%)
# ==============================================================================

def calculate_v24_final(target_date, rolling_window, cache, kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None):
    if target_date not in cache: return None, "Chưa có dữ liệu ngày này."
    curr_data = cache[target_date]
    df = curr_data['df']
    
    score_std_tuple = tuple(score_std.items())
    score_mod_tuple = tuple(score_mod.items())
    
    prev_date = target_date - timedelta(days=1)
    if prev_date not in cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in cache:
                prev_date = target_date - timedelta(days=i)
                break

    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in cache:
        col_hist_used = cache[prev_date]['hist_map'].get(prev_date)
        
    if not col_hist_used:
        return None, f"Không tìm thấy cột dữ liệu ngày {prev_date.strftime('%d/%m')}."

    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}

    if not manual_groups:
        past_dates = []
        check_d = target_date - timedelta(days=1)
        while len(past_dates) < rolling_window:
            if check_d in cache and check_d in kq_db:
                past_dates.append(check_d)
            check_d -= timedelta(days=1)
            if (target_date - check_d).days > 35: break 

        for d in past_dates:
            d_df = cache[d]['df']
            d_hist_col = None
            sorted_dates = sorted([k for k in cache[d]['hist_map'].keys() if k < d], reverse=True)
            if sorted_dates: d_hist_col = cache[d]['hist_map'][sorted_dates[0]]
            if not d_hist_col: continue
            
            kq = kq_db[d]
            
            d_score_std = {c: get_col_score(c, score_std_tuple) for c in d_df.columns if get_col_score(c, score_std_tuple) > 0}
            d_score_mod = {c: get_col_score(c, score_mod_tuple) for c in d_df.columns if get_col_score(c, score_mod_tuple) > 0}

            for g in groups:
                try:
                    mask = d_df[d_hist_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6'))) == g.upper()
                    mems = d_df[mask]
                except: continue
                
                if mems.empty: 
                    stats_std[g]['ranks'].append(999)
                    continue
                
                def get_top_nums_bt(members_df, pre_calc_p_map, pre_calc_s_map, top_n, min_v, inverse):
                    num_stats = {}
                    cols_in_scope = list(set(pre_calc_p_map.keys()) | set(pre_calc_s_map.keys()))
                    
                    for _, r in members_df.iterrows():
                        processed_nums = set()
                        for col in cols_in_scope:
                            if col not in r: continue
                            val = r[col]
                            nums = get_nums(val)
                            for n in nums:
                                if n not in num_stats: num_stats[n] = {'p': 0, 's': 0, 'v': 0}
                                if n in processed_nums: continue 
                                if col in pre_calc_p_map: num_stats[n]['p'] += pre_calc_p_map[col]
                                if col in pre_calc_s_map: num_stats[n]['s'] += pre_calc_s_map[col]
                            processed_nums.update(nums)
                    
                    for _, r in members_df.iterrows():
                        found_in_row = set()
                        for col in pre_calc_p_map:
                            if col in r:
                                for n in get_nums(r[col]): 
                                    if n in num_stats: found_in_row.add(n)
                        for n in found_in_row: num_stats[n]['v'] += 1

                    filtered = [n for n, s in num_stats.items() if s['v'] >= min_v]
                    
                    if inverse:
                        return sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['s'], int(n)))[:top_n]
                    else:
                        return sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['v'], int(n)))[:top_n]

                top80_std = get_top_nums_bt(mems, d_score_std, d_score_mod, 80, min_votes, use_inverse)
                if kq in top80_std:
                    stats_std[g]['wins'] += 1
                    stats_std[g]['ranks'].append(top80_std.index(kq) + 1)
                else: stats_std[g]['ranks'].append(999)
                
                top86_mod = get_top_nums_bt(mems, d_score_mod, d_score_std, limits_config['mod'], min_votes, use_inverse)
                if kq in top86_mod: stats_mod[g]['wins'] += 1

    top6_std = []
    best_mod_grp = ""
    
    if not manual_groups:
        final_std = []
        for g, inf in stats_std.items(): 
            sorted_rank_list = sorted(inf['ranks'])
            final_std.append((g, -inf['wins'], sum(inf['ranks']), sorted_rank_list))
        final_std.sort(key=lambda x: (x[1], x[2], x[3], x[0])) 
        top6_std = [x[0] for x in final_std[:6]]
        best_mod_grp = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]
    
    hist_series = df[col_hist_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6')))
    
    def get_group_set_final(group_name, p_map_in, s_map_in, limit, min_v, inverse):
        p_map_t = tuple(p_map_in.items()) if isinstance(p_map_in, dict) else p_map_in
        s_map_t = tuple(s_map_in.items()) if isinstance(s_map_in, dict) else s_map_in

        mask = hist_series == group_name.upper()
        valid_mems = df[mask]
        
        num_stats = {}
        p_cols_dict = {c: get_col_score(c, p_map_t) for c in df.columns if get_col_score(c, p_map_t) > 0}
        s_cols_dict = {c: get_col_score(c, s_map_t) for c in df.columns if get_col_score(c, s_map_t) > 0}
        
        for _, r in valid_mems.iterrows():
            all_cols = set(p_cols_dict.keys()).union(set(s_cols_dict.keys()))
            processed = set()
            for col in all_cols:
                if col not in valid_mems.columns: continue
                val = r[col]
                for n in get_nums(val): 
                    if n not in num_stats: num_stats[n] = {'p':0, 's':0, 'v':0}
                    if n in processed: continue
                    if col in p_cols_dict: num_stats[n]['p'] += p_cols_dict[col]
                    if col in s_cols_dict: num_stats[n]['s'] += s_cols_dict[col]
                processed.update(get_nums(val))
        
        for n in num_stats: num_stats[n]['v'] = 0
        for _, r in valid_mems.iterrows():
            found = set()
            for col in p_cols_dict:
                if col in r:
                    for n in get_nums(r[col]): 
                        if n in num_stats: found.add(n)
            for n in found: num_stats[n]['v'] += 1
            
        filtered = [n for n, s in num_stats.items() if s['v'] >= min_v]
        
        if inverse:
            sorted_res = sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['s'], int(n)))
        else:
            sorted_res = sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['v'], int(n)))
            
        return set(sorted_res[:limit])

    final_original = []
    final_modified = []

    if manual_groups:
        pool_std = []
        for g in manual_groups:
            pool_std.extend(list(get_group_set_final(g, score_std_tuple, score_mod_tuple, limits_config['l12'], min_votes, use_inverse)))
        final_original = sorted(list(set(pool_std))) 
        
        pool_mod = []
        for g in manual_groups:
             pool_mod.extend(list(get_group_set_final(g, score_mod_tuple, score_std_tuple, limits_config['mod'], min_votes, use_inverse)))
        final_modified = sorted(list(set(pool_mod)))
        
    else:
        limits_std = {
            top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], 
            top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], 
            top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']
        }
        
        pool1 = []
        for g in [top6_std[0], top6_std[5], top6_std[3]]: 
            pool1.extend(list(get_group_set_final(g, score_std_tuple, score_mod_tuple, limits_std[g], min_votes, use_inverse)))
        s1 = {n for n, c in Counter(pool1).items() if c >= 2}
        
        pool2 = []
        for g in [top6_std[1], top6_std[4], top6_std[2]]: 
            pool2.extend(list(get_group_set_final(g, score_std_tuple, score_mod_tuple, limits_std[g], min_votes, use_inverse)))
        s2 = {n for n, c in Counter(pool2).items() if c >= 2}
        
        final_original = sorted(list(s1.intersection(s2)))
        final_modified = sorted(list(get_group_set_final(best_mod_grp, score_mod_tuple, score_std_tuple, limits_config['mod'], min_votes, use_inverse)))

    final_intersect = sorted(list(set(final_original).intersection(set(final_modified))))

    return {
        "top6_std": top6_std,
        "best_mod": best_mod_grp,
        "dan_goc": final_original,
        "dan_mod": final_modified,
        "dan_final": final_intersect,
        "source_col": col_hist_used,
        "stats_groups_std": stats_std,
        "stats_groups_mod": stats_mod
    }, None

def analyze_group_performance(start_date, end_date, cut_limit, score_map, data_cache, kq_db, min_v, inverse):
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    score_map_tuple = tuple(score_map.items())

    grp_stats = {f"{i}x": {'wins': 0, 'ranks': [], 'history': [], 'last_pred': []} for i in range(10)}
    detailed_rows = [] 
    
    for d in dates:
        day_record = {"Ngày": d.strftime("%d/%m"), "KQ": kq_db.get(d, "N/A")}
        
        if d not in kq_db or d not in data_cache: 
            for g in grp_stats: 
                grp_stats[g]['history'].append(None)
                grp_stats[g]['ranks'].append(999) 
                day_record[g] = "-"
            detailed_rows.append(day_record)
            continue
            
        curr_data = data_cache[d]
        df = curr_data['df']
        
        prev_date = d - timedelta(days=1)
        if prev_date not in data_cache: 
            for k in range(2, 4):
                 if (d - timedelta(days=k)) in data_cache: 
                     prev_date = d - timedelta(days=k)
                     break
        
        hist_col_name = None
        if prev_date in data_cache:
             hist_col_name = data_cache[d]['hist_map'].get(prev_date)
        
        if not hist_col_name:
             for g in grp_stats: 
                 grp_stats[g]['history'].append(None)
                 grp_stats[g]['ranks'].append(999)
                 day_record[g] = "-"
             detailed_rows.append(day_record)
             continue
             
        hist_series = df[hist_col_name].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6')))
        kq = kq_db[d]
        p_cols_dict = {c: get_col_score(c, score_map_tuple) for c in df.columns if get_col_score(c, score_map_tuple) > 0}

        for g in grp_stats:
            mask = hist_series == g.upper()
            valid_mems = df[mask]
            
            num_stats = {}
            for _, r in valid_mems.iterrows():
                processed = set()
                for col, pts in p_cols_dict.items():
                    if col not in valid_mems.columns: continue
                    val = r[col]
                    for n in get_nums(val):
                        if n not in num_stats: num_stats[n] = {'p':0, 'v':0}
                        if n in processed: continue
                        num_stats[n]['p'] += pts
                    processed.update(get_nums(val))
            
            for n in num_stats: num_stats[n]['v'] = 0
            for _, r in valid_mems.iterrows():
                found = set()
                for col in p_cols_dict:
                    if col in r:
                         for n in get_nums(r[col]): 
                            if n in num_stats: found.add(n)
                for n in found: num_stats[n]['v'] += 1
            
            filtered = [n for n, s in num_stats.items() if s['v'] >= min_v]
            if inverse: sorted_res = sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['p'], int(n)))
            else: sorted_res = sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['v'], int(n)))

            top_list = sorted_res[:cut_limit]
            top_set = set(top_list)
            
            grp_stats[g]['last_pred'] = sorted(top_list)
            
            if kq in top_set:
                grp_stats[g]['wins'] += 1
                rank = top_list.index(kq) + 1
                grp_stats[g]['ranks'].append(rank)
                grp_stats[g]['history'].append("W")
                day_record[g] = f"✅" 
            else:
                grp_stats[g]['ranks'].append(999) 
                grp_stats[g]['history'].append("L")
                day_record[g] = "░"
            
        detailed_rows.append(day_record)
            
    final_report = []
    for g, info in grp_stats.items():
        hist = info['history']
        valid_days = len([x for x in hist if x is not None])
        wins = info['wins']
        
        max_lose = 0
        curr_lose = 0
        temp_lose = 0
        for x in reversed(hist):
            if x == "L": curr_lose += 1
            elif x == "W": break
        for x in hist:
            if x == "L": temp_lose += 1
            else:
                max_lose = max(max_lose, temp_lose)
                temp_lose = 0
        max_lose = max(max_lose, temp_lose)
        
        final_report.append({
            "Nhóm": g,
            "Số ngày trúng": wins,
            "Tỉ lệ": f"{(wins/valid_days)*100:.1f}%" if valid_days > 0 else "0%",
            "Gãy thông": max_lose,
            "Gãy hiện tại": curr_lose
        })
        
    df_rep = pd.DataFrame(final_report)
    if not df_rep.empty:
        df_rep = df_rep.sort_values(by="Số ngày trúng", ascending=False)
        
    return df_rep, pd.DataFrame(detailed_rows)

# ==============================================================================
# 4. GIAO DIỆN NGƯỜI DÙNG
# ==============================================================================

uploaded_files = st.file_uploader("📂 Tải file CSV/Excel", type=['xlsx', 'csv'], accept_multiple_files=True)

with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
    
    with st.expander("🎚️ 1. Điểm M0-M10", expanded=False):
        c_s1, c_s2 = st.columns(2)
        DEF_STD = [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50] 
        DEF_MOD = [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40]
        
        custom_std = {}
        custom_mod = {}
        with c_s1:
            st.write("**GỐC (Std)**")
            for i in range(11): custom_std[f'M{i}'] = st.number_input(f"M{i}", value=DEF_STD[i], key=f"std_{i}")
        with c_s2:
            st.write("**MOD**")
            for i in range(11): custom_mod[f'M{i}'] = st.number_input(f"M{i}", value=DEF_MOD[i], key=f"mod_{i}")

    st.markdown("---")
    st.header("⚖️ Lọc & Cắt")
    MIN_VOTES = st.number_input("Vote tối thiểu:", min_value=1, max_value=10, value=1)
    USE_INVERSE = st.checkbox("Chấm Điểm Đảo (Ngược)", value=False)
    
    with st.expander("✂️ Chi tiết cắt Top", expanded=False):
        L_TOP_12 = st.number_input("Top 1 & 2 lấy:", value=80)
        L_TOP_34 = st.number_input("Top 3 & 4 lấy:", value=65)
        L_TOP_56 = st.number_input("Top 5 & 6 lấy:", value=60)
        LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", value=86)

    st.markdown("---")
    if st.button("🗑️ XÓA CACHE", type="primary"):
        st.cache_data.clear()
        st.rerun()

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
                    if err: st.error(err)
                    else:
                        st.info(f"Phân nhóm theo ngày: {res['source_col']}")
                        
                        final_res_set = res['dan_final']
                        if manual_mode:
                            if manual_score_opt == "Chỉ Gốc": final_res_set = res['dan_goc']
                            elif manual_score_opt == "Chỉ Mod": final_res_set = res['dan_mod']

                        c1, c2, c3 = st.columns(3)
                        
                        with c1:
                            st.success(f"Gốc ({len(res['dan_goc'])})")
                            st.text_area("Gốc", ",".join(res['dan_goc']), height=150, label_visibility="collapsed")
                            if not manual_mode: 
                                st.caption(f"Top 6: {', '.join(res['top6_std'])}")

                        with c2:
                            st.warning(f"Mod ({len(res['dan_mod'])})")
                            st.text_area("Mod", ",".join(res['dan_mod']), height=150, label_visibility="collapsed")
                            if not manual_mode: 
                                st.caption(f"Best: {res['best_mod']}")
                        
                        with c3:
                            st.error(f"FINAL ({len(final_res_set)})")
                            st.text_area("Final", ",".join(final_res_set), height=150, label_visibility="collapsed")
                        
                        if target in kq_db:
                            real = kq_db[target]
                            st.markdown("---")
                            if real in final_res_set: 
                                st.balloons()
                                st.success(f"🎉 KQ **{real}** WIN!")
                            else: 
                                st.error(f"❌ KQ **{real}** MISS.")

        with tab2:
            st.subheader("Kiểm thử Backtest")
            c1, c2 = st.columns(2)
            with c1: date_range = st.date_input("Khoảng ngày:", [last_d - timedelta(days=7), last_d])
            with c2: bt_mode = st.selectbox("Chế độ:", ["FINAL (Giao thoa)", "Dàn Gốc", "Dàn Mod"])

            if st.button("🔄 CHẠY BACKTEST", use_container_width=True):
                if len(date_range) < 2: st.warning("Chọn đủ ngày.")
                else:
                    start, end = date_range[0], date_range[1]
                    logs = []
                    bar = st.progress(0)
                    delta = (end - start).days + 1
                    
                    for i in range(delta):
                        d = start + timedelta(days=i)
                        bar.progress((i+1)/delta)
                        if d not in kq_db: continue
                        res, err = calculate_v24_final(d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                        if err: continue
                        real = kq_db[d]
                        t_set = res['dan_final'] if "FINAL" in bt_mode else (res['dan_goc'] if "Gốc" in bt_mode else res['dan_mod'])
                        is_win = real in t_set
                        logs.append({
                            "Ngày": d.strftime("%d/%m"), 
                            "KQ": real, 
                            "TT": "✅ WIN" if is_win else "❌ MISS", 
                            "Số lượng": len(t_set),
                            "Chi tiết": ",".join(t_set)
                        })

                    bar.empty()
                    if logs:
                        df_log = pd.DataFrame(logs)
                        wins = df_log[df_log["TT"].str.contains("WIN")].shape[0]
                        st.metric(f"Tỉ lệ Win ({bt_mode})", f"{wins}/{df_log.shape[0]}", f"{(wins/df_log.shape[0])*100:.1f}%")
                        st.dataframe(df_log, use_container_width=True)

        with tab3:
            st.subheader("Phân Tích Nhóm (Matrix)")
            c_a1, c_a2 = st.columns(2)
            with c_a1: d_range_a = st.date_input("Thời gian:", [last_d - timedelta(days=15), last_d], key="dr_a")
            with c_a2: 
                cut_val = st.number_input("Cắt Top:", value=60, step=5)
                score_mode = st.radio("Hệ điểm:", ["Gốc (Std)", "Modified"], horizontal=True)
            
            if st.button("🔎 QUÉT MATRIX", use_container_width=True):
                if len(d_range_a) < 2: st.warning("Chọn đủ ngày.")
                else:
                    with st.spinner("Đang xử lý..."):
                        s_map = custom_std if score_mode == "Gốc (Std)" else custom_mod
                        df_report, df_detail = analyze_group_performance(d_range_a[0], d_range_a[1], cut_val, s_map, data_cache, kq_db, MIN_VOTES, USE_INVERSE)
                        
                        st.write("📊 **Thống kê tổng hợp**")
                        st.dataframe(df_report, use_container_width=True)
                        
                        st.write("📅 **Chi tiết từng ngày**")
                        st.dataframe(df_detail, use_container_width=True, height=500)
