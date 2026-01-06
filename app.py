import streamlit as st
import pandas as pd
import re
import datetime
import time
from datetime import timedelta
from collections import Counter
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Set, Any

# ==============================================================================
# 1. CONSTANTS & CONFIG (CẤU HÌNH HỆ THỐNG)
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V55 - Clean Core", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- GLOBAL CONSTANTS ---
RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')

BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])
HEADER_KEYWORDS = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
GROUPS_10X = [f"{i}x" for i in range(10)]

SCORES_PRESETS = {
    "Hard Core (Khuyên dùng)": { 
        "STD": [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60], 
        "MOD": [0, 5, 10, 20, 25, 45, 50, 40, 30, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88}
    },
    "CH1: Bám Đuôi (An Toàn)": { 
        "STD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70], 
        "MOD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70],
        "LIMITS": {'l12': 80, 'l34': 75, 'l56': 60, 'mod': 88}
    },
    "Gốc (V24 Standard)": {
        "STD": [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88}
    }
}

# ==============================================================================
# 2. UTILITY FUNCTIONS (HÀM HỖ TRỢ)
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s: str) -> List[str]:
    """Trích xuất số từ chuỗi, bỏ qua các từ khóa xấu."""
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    raw_nums = RE_NUMS.findall(s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name: str, mapping_tuple: Tuple) -> int:
    """Tính điểm cột dựa trên tên cột (M1, M2...)."""
    clean = RE_CLEAN_SCORE.sub('', str(col_name).upper().replace(' ', ''))
    mapping = dict(mapping_tuple)
    if 'M10' in clean: return mapping.get('M10', 0)
    for key, score in mapping.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def find_header_row(df_preview: pd.DataFrame) -> int:
    """Tìm dòng tiêu đề thật sự trong file Excel/CSV."""
    for idx, row in df_preview.iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in HEADER_KEYWORDS):
            return idx
    return 3

def extract_meta_from_filename(filename: str) -> Tuple[int, int, Optional[datetime.date]]:
    """Lấy ngày tháng từ tên file."""
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    clean_name = re.sub(r'\s*-\s*', '-', clean_name)
    
    # 1. Tìm năm
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    
    # 2. Tìm tháng (nếu có chữ THANG)
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    
    # 3. Tìm ngày đầy đủ (dd.mm.yyyy hoặc dd.mm)
    full_date_match = re.search(r'(\d{1,2})[\.\-](\d{1,2})(?:[\.\-]20\d{2})?', clean_name)
    if full_date_match:
        try:
            d = int(full_date_match.group(1))
            m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            # Xử lý chuyển năm (Tháng 12 -> Tháng 1)
            if m == 12 and m_global == 1: y -= 1
            if m == 1 and m_global == 12: y += 1
            return m, y, datetime.date(y, m, d)
        except: pass
        
    # 4. Tìm ngày lẻ (số cuối cùng)
    # Ví dụ: "... - 1.12" (ngày 1 tháng 12) hoặc "... - 2" (ngày 2)
    # Logic cải tiến cho V55:
    single_day_match = re.findall(r'(\d{1,2})$', clean_name)
    if single_day_match:
        try:
            d = int(single_day_match[-1])
            return m_global, y_global, datetime.date(y_global, m_global, d)
        except: pass
        
    return m_global, y_global, None

def parse_date_smart(col_str, f_m, f_y):
    """Parse ngày từ tên cột trong file."""
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

# ==============================================================================
# 3. DATA LOADING (XỬ LÝ FILE)
# ==============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_data_v24(files) -> Tuple[Dict, Dict, List[str], List[str]]:
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
            # --- XỬ LÝ EXCEL ---
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
                        dfs_to_process.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")
            
            # --- XỬ LÝ CSV ---
            elif file.name.endswith('.csv'):
                if not date_from_name: continue
                encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
                df_raw = None
                preview = None
                for enc in encodings:
                    try:
                        file.seek(0)
                        preview = pd.read_csv(file, header=None, nrows=20, encoding=enc)
                        file.seek(0)
                        df_raw = pd.read_csv(file, header=None, encoding=enc)
                        break
                    except: continue
                
                if df_raw is None:
                    err_logs.append(f"❌ Lỗi encoding: {file.name}")
                    continue

                h_row = find_header_row(preview)
                df = df_raw.iloc[h_row+1:].copy()
                df.columns = df_raw.iloc[h_row]
                dfs_to_process.append((date_from_name, df))
                file_status.append(f"✅ CSV: {file.name}")

            # --- CHUẨN HÓA DỮ LIỆU ---
            for t_date, df in dfs_to_process:
                # Clean columns
                df.columns = [str(c).strip().upper().replace('M 1 0', 'M10') for c in df.columns]
                
                # Map ngày tháng vào cột
                hist_map = {}
                for col in df.columns:
                    if "UNNAMED" in col: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: hist_map[d_obj] = col
                
                # Tìm KQ
                kq_row = None
                if not df.empty:
                    for c_idx in range(min(2, len(df.columns))):
                        col_check = df.columns[c_idx]
                        if df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ').any():
                            kq_row = df[df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')].iloc[0]
                            break
                
                if kq_row is not None:
                    for d_val, c_name in hist_map.items():
                        val = str(kq_row[c_name])
                        nums = get_nums(val)
                        if nums: kq_db[d_val] = nums[0]
                        
                cache[t_date] = {'df': df, 'hist_map': hist_map}
                
        except Exception as e:
            err_logs.append(f"⚠️ Lỗi xử lý '{file.name}': {str(e)}")
            continue
            
    return cache, kq_db, file_status, err_logs

# ==============================================================================
# 4. CORE LOGIC (V55 REFACTORED)
# ==============================================================================

def fast_get_top_nums(df: pd.DataFrame, p_map: Dict, s_map: Dict, top_n: int, min_v: int, inverse: bool) -> List[str]:
    """Lấy danh sách số Top từ DataFrame đã lọc."""
    cols = list(set(p_map.keys()) | set(s_map.keys()))
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols or df.empty: return []

    sub_df = df[valid_cols].copy()
    melted = sub_df.melt(ignore_index=False, var_name='Col', value_name='Val').dropna(subset=['Val'])
    
    # Filter bad keywords
    mask_valid = ~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT|MISS|TRUOT|NGHỈ|LỖI', regex=True)
    melted = melted[mask_valid]
    if melted.empty: return []

    # Extract numbers
    exploded = melted.assign(Num=melted['Val'].astype(str).str.findall(r'\d+')).explode('Num')
    exploded = exploded.dropna(subset=['Num'])
    exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
    exploded = exploded[exploded['Num'].str.len() <= 2]

    # Map scores
    exploded['P'] = exploded['Col'].map(p_map).fillna(0)
    exploded['S'] = exploded['Col'].map(s_map).fillna(0)

    # Aggregation
    stats = exploded.groupby('Num')[['P', 'S']].sum()
    stats['V'] = exploded.reset_index().groupby('Num')['index'].nunique()

    stats = stats[stats['V'] >= min_v]
    if stats.empty: return []

    stats = stats.reset_index()
    stats['Num_Int'] = stats['Num'].astype(int)
    
    # Sorting
    if inverse:
        stats = stats.sort_values(by=['P', 'S', 'Num_Int'], ascending=[False, False, True])
    else:
        stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])

    return stats['Num'].head(int(top_n)).tolist()

def analyze_past_performance(
    target_date, 
    rolling_window, 
    _cache, 
    _kq_db, 
    score_std_tuple, 
    score_mod_tuple, 
    min_votes, 
    use_inverse,
    limits_mod_val
) -> Tuple[List[str], str]:
    """Phân tích quá khứ để tìm Top 6 nhóm Std và nhóm Mod tốt nhất."""
    
    groups = GROUPS_10X
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}

    # Tìm danh sách ngày quá khứ hợp lệ
    past_dates = []
    check_d = target_date - timedelta(days=1)
    while len(past_dates) < rolling_window:
        if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
        check_d -= timedelta(days=1)
        if (target_date - check_d).days > 40: break # Safety break

    for d in past_dates:
        d_df = _cache[d]['df']
        kq = _kq_db[d]
        
        # Mapping score cho ngày d
        d_p_map, d_s_map = {}, {}
        for col in d_df.columns:
            s_p = get_col_score(col, score_std_tuple)
            if s_p > 0: d_p_map[col] = s_p
            s_s = get_col_score(col, score_mod_tuple)
            if s_s > 0: d_s_map[col] = s_s
        
        # Tìm cột lịch sử của ngày d
        d_hist_col = None
        sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
        if sorted_dates: d_hist_col = _cache[d]['hist_map'][sorted_dates[0]]
        if not d_hist_col: continue

        # Lấy data nhóm từ cột lịch sử
        try:
            hist_series_d = d_df[d_hist_col].astype(str).str.upper().replace('S', '6', regex=False)
            hist_series_d = hist_series_d.str.replace(r'[^0-9X]', '', regex=True)
        except: continue

        for g in groups:
            mask = hist_series_d == g.upper()
            mems = d_df[mask]
            if mems.empty:
                stats_std[g]['ranks'].append(999); continue
            
            # Check Win Std (Top 80 default for ranking check)
            top80_std = fast_get_top_nums(mems, d_p_map, d_s_map, 80, min_votes, use_inverse)
            if kq in top80_std:
                stats_std[g]['wins'] += 1
                stats_std[g]['ranks'].append(top80_std.index(kq) + 1)
            else: stats_std[g]['ranks'].append(999)
            
            # Check Win Mod
            top_mod = fast_get_top_nums(mems, d_s_map, d_p_map, int(limits_mod_val), min_votes, use_inverse)
            if kq in top_mod: stats_mod[g]['wins'] += 1

    # Ranking Std
    final_std = []
    for g, inf in stats_std.items(): 
        # Sort: Nhiều win nhất -> Tổng rank thấp nhất -> Rank chi tiết tốt -> Tên nhóm
        final_std.append((g, -inf['wins'], sum(inf['ranks']), sorted(inf['ranks'])))
    final_std.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
    top6_std = [x[0] for x in final_std[:6]]
    
    # Ranking Mod
    best_mod_grp = sorted(stats_mod.keys(), key=lambda g: (-stats_mod[g]['wins'], g))[0]
    
    return top6_std, best_mod_grp

def calculate_v24_clean(
    target_date, 
    rolling_window, 
    _cache, 
    _kq_db, 
    limits_config, 
    min_votes, 
    score_std, 
    score_mod, 
    use_inverse, 
    manual_groups=None, 
    max_trim=None
):
    """
    Hàm tính toán chính (V55 Clean Version).
    Thay thế hàm cũ khổng lồ bằng logic rõ ràng hơn.
    """
    if target_date not in _cache: return None, "Không có dữ liệu ngày này"
    curr_data = _cache[target_date]
    df = curr_data['df']
    
    # Chuẩn bị Score Map
    p_map, s_map = {}, {}
    score_std_tuple = tuple(score_std.items())
    score_mod_tuple = tuple(score_mod.items())
    
    for col in df.columns:
        s_p = get_col_score(col, score_std_tuple)
        if s_p > 0: p_map[col] = s_p
        s_s = get_col_score(col, score_mod_tuple)
        if s_s > 0: s_map[col] = s_s

    # Tìm cột phân nhóm (lịch sử)
    # Logic: Lấy ngày hôm qua, hoặc hôm kia...
    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache:
                prev_date = target_date - timedelta(days=i); break
    
    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in _cache:
        col_hist_used = _cache[prev_date]['hist_map'].get(prev_date)
    
    if not col_hist_used: return None, "Không tìm thấy cột lịch sử phân nhóm"

    # Chuẩn bị Series Nhóm
    try:
        hist_series = df[col_hist_used].astype(str).str.upper().replace('S', '6', regex=False)
        hist_series = hist_series.str.replace(r'[^0-9X]', '', regex=True)
    except: return None, "Lỗi định dạng cột nhóm"

    # Helper nội bộ để lấy số từ nhóm
    def get_pool_from_groups(group_list, limit_dict, main_map, sub_map):
        pool = []
        for g in group_list:
            mask = hist_series == g.upper()
            valid_mems = df[mask]
            lim = limit_dict.get(g, limit_dict.get('default', 80))
            res = fast_get_top_nums(valid_mems, main_map, sub_map, int(lim), min_votes, use_inverse)
            pool.extend(res)
        return pool

    # --- MAIN FLOW ---
    top6_std = []
    best_mod = ""
    
    if manual_groups:
        # Chế độ thủ công (ít dùng, nhưng giữ để tương thích)
        limit_map = {'default': limits_config['l12']}
        final_original = sorted(list(set(get_pool_from_groups(manual_groups, limit_map, p_map, s_map))))
        final_modified = sorted(list(set(get_pool_from_groups(manual_groups, {'default': limits_config['mod']}, s_map, p_map))))
    else:
        # 1. Phân tích quá khứ để tìm Top Groups
        top6_std, best_mod = analyze_past_performance(
            target_date, rolling_window, _cache, _kq_db,
            score_std_tuple, score_mod_tuple, min_votes, use_inverse, limits_config['mod']
        )
        
        # 2. Tạo Dàn Gốc (Std) - Logic Giao 2 Tập
        limits_std = {
            top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], 
            top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], 
            top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']
        }
        
        # Set 1: Top 1, 6, 4 (theo index 0, 5, 3)
        g_set1 = [top6_std[0], top6_std[5], top6_std[3]]
        pool1 = get_pool_from_groups(g_set1, limits_std, p_map, s_map)
        s1 = {n for n, c in Counter(pool1).items() if c >= 2} 
        
        # Set 2: Top 2, 5, 3 (theo index 1, 4, 2)
        g_set2 = [top6_std[1], top6_std[4], top6_std[2]]
        pool2 = get_pool_from_groups(g_set2, limits_std, p_map, s_map)
        s2 = {n for n, c in Counter(pool2).items() if c >= 2}
        
        final_original = sorted(list(s1.intersection(s2)))

        # 3. Tạo Dàn Mod (Modified)
        # Chỉ lấy từ nhóm Best Mod Group
        mask_mod = hist_series == best_mod.upper()
        final_modified = sorted(fast_get_top_nums(
            df[mask_mod], s_map, p_map, int(limits_config['mod']), min_votes, use_inverse
        ))

    # 4. Giao thoa Final & Smart Trim
    intersect_list = list(set(final_original).intersection(set(final_modified)))

    final_intersect = sorted(intersect_list)
    if max_trim and len(intersect_list) > max_trim:
        # Logic Trim: Tính lại tổng điểm (P+S) của các số trong list giao
        # Để code gọn, ta dùng lại logic lọc đơn giản
        temp_df = df.copy()
        valid_cols = list(set(p_map.keys()) | set(s_map.keys()))
        sub_df = temp_df[valid_cols]
        melted = sub_df.melt(value_name='Val').dropna(subset=['Val'])
        mask_bad = ~melted['Val'].astype(str).str.upper().str.contains(r'N|NGHI|SX|XIT', regex=True)
        melted = melted[mask_bad]
        
        exploded = melted.assign(Num=melted['Val'].astype(str).str.findall(r'\d+')).explode('Num')
        exploded = exploded.dropna(subset=['Num'])
        exploded['Num'] = exploded['Num'].str.strip().str.zfill(2)
        
        # Chỉ giữ lại các số nằm trong intersection
        exploded = exploded[exploded['Num'].isin(intersect_list)]
        
        # Tính điểm
        exploded['Score'] = exploded['variable'].map(p_map).fillna(0) + exploded['variable'].map(s_map).fillna(0)
        final_scores = exploded.groupby('Num')['Score'].sum().reset_index().sort_values(by='Score', ascending=False)
        
        final_intersect = sorted(final_scores.head(int(max_trim))['Num'].tolist())

    return {
        "top6_std": top6_std, 
        "best_mod": best_mod,
        "dan_goc": final_original,
        "dan_mod": final_modified,
        "dan_final": final_intersect, 
        "source_col": col_hist_used
    }, None

# ==============================================================================
# 5. UI & APP FLOW
# ==============================================================================

def main():
    st.title("🛡️ Quang Handsome: V55 Full Stats (Refactored)")
    st.caption("🚀 Optimized Codebase | Logic V54 Hybrid | Dữ liệu Tháng 12/25 - 01/26")

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        uploaded_files = st.file_uploader("📂 Tải file CSV/Excel", type=['xlsx', 'csv'], accept_multiple_files=True)
        
        # Init Session State
        if 'std_0' not in st.session_state:
            def_vals = SCORES_PRESETS["Hard Core (Khuyên dùng)"]
            for i in range(11):
                st.session_state[f'std_{i}'] = def_vals["STD"][i]
                st.session_state[f'mod_{i}'] = def_vals["MOD"][i]
            st.session_state['L12'] = def_vals['LIMITS']['l12']
            st.session_state['L34'] = def_vals['LIMITS']['l34']
            st.session_state['L56'] = def_vals['LIMITS']['l56']
            st.session_state['LMOD'] = def_vals['LIMITS']['mod']

        MAX_TRIM_NUMS = st.slider("🛡️ Phanh An Toàn (Max số):", 50, 90, 65)
        ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
        
        with st.expander("🎚️ Presets & Điểm", expanded=False):
            def update_scores():
                choice = st.session_state.preset_choice
                if choice in SCORES_PRESETS:
                    vals = SCORES_PRESETS[choice]
                    for i in range(11):
                        st.session_state[f'std_{i}'] = vals["STD"][i]
                        st.session_state[f'mod_{i}'] = vals["MOD"][i]
                    if 'LIMITS' in vals:
                        st.session_state['L12'] = vals['LIMITS']['l12']
                        st.session_state['L34'] = vals['LIMITS']['l34']
                        st.session_state['L56'] = vals['LIMITS']['l56']
                        st.session_state['LMOD'] = vals['LIMITS']['mod']
                        
            st.selectbox("📚 Bộ mẫu:", ["Tùy chỉnh"] + list(SCORES_PRESETS.keys()), key="preset_choice", on_change=update_scores)
            c1, c2 = st.columns(2)
            with c1: 
                for i in range(11): st.number_input(f"STD M{i}", key=f"std_{i}")
            with c2: 
                for i in range(11): st.number_input(f"MOD M{i}", key=f"mod_{i}")

        LIMIT_CFG = {
            'l12': st.session_state['L12'], 
            'l34': st.session_state['L34'], 
            'l56': st.session_state['L56'], 
            'mod': st.session_state['LMOD']
        }
        
        MIN_VOTES = st.number_input("Min Votes:", 1, 10, 1)
        USE_INVERSE = st.checkbox("Chấm điểm đảo", False)
        if st.button("🗑️ Xóa Cache"): st.cache_data.clear(); st.rerun()

    # --- MAIN CONTENT ---
    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        
        if f_status:
            with st.expander(f"✅ Đã tải {len(f_status)} file", expanded=False):
                st.write(f_status)
        if err_logs:
            st.error(f"Có {len(err_logs)} lỗi file")
            with st.expander("Chi tiết lỗi"): st.write(err_logs)

        if data_cache:
            last_d = max(data_cache.keys())
            
            tab1, tab2 = st.tabs(["📊 DỰ ĐOÁN HYBRID", "🔙 BACKTEST NHANH"])
            
            # --- TAB 1: DỰ ĐOÁN ---
            with tab1:
                col_ctrl, col_act = st.columns([2, 1])
                with col_ctrl: target = st.date_input("Ngày dự đoán:", value=last_d)
                with col_act: 
                    btn_run = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)

                if btn_run:
                    with st.spinner("Đang tính toán đa luồng..."):
                        # 1. Luồng hiện tại (Current UI Settings)
                        curr_std = {f'M{i}': st.session_state[f'std_{i}'] for i in range(11)}
                        curr_mod = {f'M{i}': st.session_state[f'mod_{i}'] for i in range(11)}
                        
                        res_curr, err = calculate_v24_clean(
                            target, ROLLING_WINDOW, data_cache, kq_db, LIMIT_CFG, 
                            MIN_VOTES, curr_std, curr_mod, USE_INVERSE, None, MAX_TRIM_NUMS
                        )
                        
                        # 2. Luồng CH1 (Cố định)
                        ch1_p = SCORES_PRESETS["CH1: Bám Đuôi (An Toàn)"]
                        ch1_std = {f'M{i}': ch1_p['STD'][i] for i in range(11)}
                        ch1_mod = {f'M{i}': ch1_p['MOD'][i] for i in range(11)}
                        res_ch1, _ = calculate_v24_clean(
                            target, ROLLING_WINDOW, data_cache, kq_db, ch1_p['LIMITS'], 
                            MIN_VOTES, ch1_std, ch1_mod, USE_INVERSE, None, MAX_TRIM_NUMS
                        )

                        # 3. Luồng HardCore (Cố định)
                        hc_p = SCORES_PRESETS["Hard Core (Khuyên dùng)"]
                        hc_std = {f'M{i}': hc_p['STD'][i] for i in range(11)}
                        hc_mod = {f'M{i}': hc_p['MOD'][i] for i in range(11)}
                        res_hc, _ = calculate_v24_clean(
                            target, ROLLING_WINDOW, data_cache, kq_db, hc_p['LIMITS'], 
                            MIN_VOTES, hc_std, hc_mod, USE_INVERSE, None, MAX_TRIM_NUMS
                        )
                        
                        # Hybrid Logic
                        hybrid_goc = []
                        if res_ch1 and res_hc:
                            hybrid_goc = sorted(list(set(res_ch1['dan_goc']).intersection(set(res_hc['dan_goc']))))

                        st.session_state['res'] = {
                            'curr': res_curr, 'ch1': res_ch1, 'hc': res_hc, 'hybrid': hybrid_goc, 'target': target, 'err': err
                        }

                if 'res' in st.session_state and st.session_state['res']['target'] == target:
                    res = st.session_state['res']
                    if res['err']: st.error(res['err'])
                    else:
                        cur = res['curr']
                        st.info(f"Nguồn phân nhóm: {cur['source_col']}")
                        
                        cols = st.columns(4)
                        with cols[0]:
                            st.caption(f"Gốc Current ({len(cur['dan_goc'])})")
                            st.text_area("G", ",".join(cur['dan_goc']), height=120)
                        with cols[1]:
                            st.caption(f"Mod Current ({len(cur['dan_mod'])})")
                            st.text_area("M", ",".join(cur['dan_mod']), height=120)
                        with cols[2]:
                            st.caption(f"Final ({len(cur['dan_final'])})")
                            st.text_area("F", ",".join(cur['dan_final']), height=120)
                        with cols[3]:
                            st.caption(f"💎 Hybrid Gốc ({len(res['hybrid'])})")
                            st.text_area("H", ",".join(res['hybrid']), height=120)
                        
                        # Check Result
                        if target in kq_db:
                            real = kq_db[target]
                            st.markdown(f"### 🏁 Kết quả: **{real}**")
                            c1, c2 = st.columns(2)
                            is_win_f = real in cur['dan_final']
                            is_win_h = real in res['hybrid']
                            c1.success("Final WIN") if is_win_f else c1.error("Final MISS")
                            c2.success("Hybrid WIN") if is_win_h else c2.error("Hybrid MISS")

            # --- TAB 2: BACKTEST ---
            with tab2:
                c_d1, c_d2 = st.columns(2)
                d_start = c_d1.date_input("Từ ngày", value=last_d - timedelta(days=7))
                d_end = c_d2.date_input("Đến ngày", value=last_d)
                
                if st.button("▶️ CHẠY BACKTEST HYBRID"):
                    dates = [d_start + timedelta(days=i) for i in range((d_end - d_start).days + 1)]
                    logs = []
                    
                    # Pre-load Configs
                    p1 = SCORES_PRESETS["CH1: Bám Đuôi (An Toàn)"]
                    s1, m1 = {f'M{i}':p1['STD'][i] for i in range(11)}, {f'M{i}':p1['MOD'][i] for i in range(11)}
                    
                    p2 = SCORES_PRESETS["Hard Core (Khuyên dùng)"]
                    s2, m2 = {f'M{i}':p2['STD'][i] for i in range(11)}, {f'M{i}':p2['MOD'][i] for i in range(11)}

                    my_bar = st.progress(0)
                    for i, d in enumerate(dates):
                        my_bar.progress((i+1)/len(dates))
                        if d not in kq_db: continue
                        
                        r1, _ = calculate_v24_clean(d, ROLLING_WINDOW, data_cache, kq_db, p1['LIMITS'], MIN_VOTES, s1, m1, USE_INVERSE, None, MAX_TRIM_NUMS)
                        r2, _ = calculate_v24_clean(d, ROLLING_WINDOW, data_cache, kq_db, p2['LIMITS'], MIN_VOTES, s2, m2, USE_INVERSE, None, MAX_TRIM_NUMS)
                        
                        if r1 and r2:
                            g1 = r1['dan_goc']
                            g2 = r2['dan_goc']
                            hb = sorted(list(set(g1).intersection(set(g2))))
                            kq = kq_db[d]
                            
                            logs.append({
                                "Ngày": d.strftime("%d/%m"),
                                "KQ": kq,
                                "CH1": f"{'✅' if kq in g1 else '❌'} ({len(g1)})",
                                "HC": f"{'✅' if kq in g2 else '❌'} ({len(g2)})",
                                "Hybrid": f"{'✅' if kq in hb else '❌'} ({len(hb)})"
                            })
                    my_bar.empty()
                    if logs:
                        st.dataframe(pd.DataFrame(logs), use_container_width=True)

if __name__ == "__main__":
    main()
