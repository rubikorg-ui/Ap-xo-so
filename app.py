import streamlit as st
import pandas as pd
import re
import random
import datetime
from datetime import timedelta
from collections import Counter
from functools import lru_cache

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V32 - Final 1 Focus", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🎯 Quang Handsome: V32 Final 1 Specialist")
st.caption("🚀 Chỉ tập trung tối ưu hóa Dàn Giao Thoa (Final 1)")

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
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    full_date_match = re.search(r'[\s\-](\d{1,2})[\.\-](\d{1,2})(?:[\.\-]20\d{2})?$', clean_name)
    if full_date_match:
        try:
            d = int(full_date_match.group(1))
            m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            if m == 12 and m_global == 1: y -= 1 
            return m, y, datetime.date(y, m, d)
        except: pass
    single_day_match = re.search(r'-\s*(\d{1,2})$', clean_name)
    if single_day_match:
        try:
            d = int(single_day_match.group(1))
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
                    if s_date:
                        preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=20, engine='openpyxl')
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row, engine='openpyxl')
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

    # --- PHASE 1: Backtest ---
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
    
    # CHỈ TRẢ VỀ FINAL ĐỂ TỐI ƯU
    return {
        "top6_std": top6_std, 
        "best_mod": best_mod_grp,
        "dan_final": final_intersect, 
        "source_col": col_hist_used
    }

@st.cache_data(show_spinner=False)
def calculate_v24_final(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups=None):
    res = calculate_v24_logic_only(target_date, rolling_window, _cache, _kq_db, limits_config, min_votes, score_std, score_mod, use_inverse, manual_groups)
    if not res: return None, "Lỗi dữ liệu"
    return res, None

# ==============================================================================
# 3. AUTO-OPTIMIZER (FINAL 1 SPECIALIST)
# ==============================================================================

def random_score_set():
    base_pool = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    s = {}
    for i in range(11):
        if i == 10: s[f'M{i}'] = random.choice([30, 40, 50, 60, 80, 100])
        else: s[f'M{i}'] = random.choice(base_pool)
    return s

def random_limits():
    # Mở rộng giới hạn để tìm ra điểm giao thoa tốt nhất
    return {
        'l12': random.choice([75, 80, 85, 90]), 
        'l34': random.choice([60, 65, 70, 75]),
        'l56': random.choice([55, 60, 65, 70]),
        'mod': random.choice([80, 85, 88, 92])
    }

def run_optimization(trials, start_d, end_d, _cache, _kq_db, min_v, use_inv, max_allowed_nums):
    best_results = []
    delta = (end_d - start_d).days + 1
    dates_to_test = [start_d + timedelta(days=i) for i in range(delta)]
    dates_to_test = [d for d in dates_to_test if d in _kq_db and d in _cache]
    
    if not dates_to_test: return []

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(trials):
        r_std = random_score_set()
        r_mod = random_score_set()
        r_lim = random_limits()
        
        wins = 0
        total_nums = 0
        valid_days_count = 0
        
        for d in dates_to_test:
            # Chạy nhanh với rolling=5
            res = calculate_v24_logic_only(d, 5, _cache, _kq_db, r_lim, min_v, r_std, r_mod, use_inv, None)
            if res:
                final_set = res['dan_final'] # CHỈ QUAN TÂM FINAL 1
                real = _kq_db[d]
                if real in final_set: wins += 1
                total_nums += len(final_set)
                valid_days_count += 1
        
        if valid_days_count > 0:
            avg_nums = total_nums / valid_days_count
            win_rate = (wins / valid_days_count) * 100
            
            # Smart Filter cho Final 1:
            # 1. Trung bình số phải <= Max
            # 2. Trung bình số phải >= 5 (để tránh trường hợp dàn rỗng liên tục mà vẫn tính là OK)
            if 5 <= avg_nums <= max_allowed_nums:
                best_results.append({
                    "WinRate": win_rate,
                    "Wins": wins,
                    "AvgNums": avg_nums,
                    "STD": r_std,
                    "MOD": r_mod,
                    "LIMITS": r_lim
                })
        
        if i % 25 == 0:
            progress_bar.progress((i + 1) / trials)
            curr_best = max([x['WinRate'] for x in best_results] + [0])
            status_text.text(f"Đang tối ưu Final 1: {i+1}/{trials} | Max Win: {curr_best:.1f}%")

    progress_bar.empty()
    status_text.empty()
    
    # Sắp xếp: Ưu tiên Tỷ lệ thắng -> Sau đó ưu tiên ít số
    best_results.sort(key=lambda x: (-x['WinRate'], x['AvgNums']))
    return best_results[:5]

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================

SCORES_PRESETS = {
    "Gốc (V24 Standard)": {
        "STD": [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50],
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
            L_TOP_12 = st.number_input("Top 1 & 2 lấy:", value=80, key="L12")
            L_TOP_34 = st.number_input("Top 3 & 4 lấy:", value=65, key="L34")
            L_TOP_56 = st.number_input("Top 5 & 6 lấy:", value=60, key="L56")
            LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", value=86, key="LMOD")

        st.markdown("---")
        if st.button("🗑️ XÓA CACHE", type="primary"):
            st.cache_data.clear(); st.rerun()

    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        
        with st.expander("🕵️ Trạng thái File", expanded=False):
            for s in f_status: st.success(s)
            for e in err_logs: st.error(e)
        
        if data_cache:
            limit_cfg = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
            last_d = max(data_cache.keys())
            
            tab1, tab2, tab3 = st.tabs(["📊 DỰ ĐOÁN", "🔙 BACKTEST", "🎯 TỐI ƯU FINAL 1"])
            
            with tab1:
                st.subheader("Dự đoán hàng ngày (Chỉ Final 1)")
                c_d1, c_d2 = st.columns([1, 1])
                with c_d1: target = st.date_input("Ngày:", value=last_d)
                
                if st.button("🚀 CHẠY PHÂN TÍCH", type="primary", use_container_width=True):
                    with st.spinner("Đang tính toán..."):
                        res, err = calculate_v24_final(target, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                        st.session_state['run_result'] = {'res': res, 'err': err, 'target': target}

                if 'run_result' in st.session_state and st.session_state['run_result']['target'] == target:
                    rr = st.session_state['run_result']
                    res = rr['res']
                    if not rr['err']:
                        st.success(f"Phân nhóm nguồn: {res['source_col']}")
                        st.caption(f"Số lượng dàn: {len(res['dan_final'])} số")
                        
                        st.text_area("Final 1 (Giao thoa Gốc + Mod)", ",".join(res['dan_final']), height=100)
                        
                        # Chỉ hiển thị info cần thiết
                        st.info(f"🏆 Top 6 Gốc: {', '.join(res['top6_std'])}\n\n🌟 Best Mod: {res['best_mod']}")

                        if target in kq_db:
                            real = kq_db[target]
                            if real in res['dan_final']: st.balloons(); st.success(f"WIN: {real}")
                            else: st.error(f"MISS: {real}")

            with tab2:
                st.subheader("Backtest Nhanh (Final 1)")
                d_range = st.date_input("Khoảng ngày:", [last_d - timedelta(days=7), last_d])
                if st.button("Chạy Backtest"):
                     if len(d_range) < 2: st.warning("Chọn đủ ngày.")
                     else:
                        start, end = d_range[0], d_range[1]
                        logs = []
                        delta = (end - start).days + 1
                        for i in range(delta):
                            d = start + timedelta(days=i)
                            if d not in kq_db: continue
                            res = calculate_v24_logic_only(d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg, MIN_VOTES, custom_std, custom_mod, USE_INVERSE, None)
                            if res:
                                t_set = res['dan_final'] # CHỈ LẤY FINAL
                                real = kq_db[d]
                                logs.append({"Ngày": d.strftime("%d/%m"), "KQ": real, "TT": "WIN" if real in t_set else "MISS", "Số số": len(t_set)})
                        
                        if logs:
                            df_log = pd.DataFrame(logs)
                            wins = df_log[df_log["TT"] == "WIN"].shape[0]
                            st.metric("Kết quả Final 1", f"{wins}/{len(df_log)}", delta=f"{(wins/len(df_log))*100:.1f}%")
                            st.dataframe(df_log, use_container_width=True)

            with tab3:
                st.subheader("🎯 Tối ưu hóa Dàn Giao Thoa (Final 1)")
                st.info("Hệ thống sẽ tìm điểm Gốc & Mod sao cho phần Giao Nhau (Intersection) có tỷ lệ thắng cao nhất.")
                
                c_o1, c_o2, c_o3 = st.columns(3)
                with c_o1:
                    opt_days = st.slider("Số ngày Backtest:", 5, 20, 10)
                with c_o2:
                    n_trials = st.selectbox("Số lần thử nghiệm:", [500, 1000, 2000], index=1)
                with c_o3:
                    max_allowed_nums = st.slider("Max Số Lượng (Final 1):", 30, 80, 60)
                
                opt_end_date = st.date_input("Ngày kết thúc xét duyệt:", value=last_d)
                start_opt_date = opt_end_date - timedelta(days=opt_days)
                
                if st.button("🔥 BẮT ĐẦU DÒ TÌM", type="primary", use_container_width=True):
                    with st.spinner("Đang tìm kiếm giao điểm vàng..."):
                        best_configs = run_optimization(n_trials, start_opt_date, opt_end_date, data_cache, kq_db, MIN_VOTES, USE_INVERSE, max_allowed_nums)
                        
                        if not best_configs:
                            st.warning(f"Không tìm thấy cấu hình nào có trung bình số <= {max_allowed_nums}. Hãy nới lỏng giới hạn.")
                        else:
                            st.success("🎉 Tìm thấy cấu hình tối ưu cho Final 1!")
                            for idx, cfg in enumerate(best_configs):
                                with st.expander(f"🏆 TOP {idx+1}: Win {cfg['WinRate']:.1f}% - TB {cfg['AvgNums']:.1f} số", expanded=(idx==0)):
                                    c1, c2, c3 = st.columns(3)
                                    with c1: st.write("GỐC"); st.json(cfg['STD'])
                                    with c2: st.write("MOD"); st.json(cfg['MOD'])
                                    with c3: st.write("CUT"); st.json(cfg['LIMITS'])
                                    
                                    if st.button(f"👉 Áp dụng Top {idx+1}", key=f"apply_{idx}"):
                                        for k, v in cfg['STD'].items(): st.session_state[f'std_{k[1:]}'] = v
                                        for k, v in cfg['MOD'].items(): st.session_state[f'mod_{k[1:]}'] = v
                                        st.session_state['L12'] = cfg['LIMITS']['l12']
                                        st.session_state['L34'] = cfg['LIMITS']['l34']
                                        st.session_state['L56'] = cfg['LIMITS']['l56']
                                        st.session_state['LMOD'] = cfg['LIMITS']['mod']
                                        st.success("Đã áp dụng!")
                                        st.rerun()

if __name__ == "__main__":
    main()
