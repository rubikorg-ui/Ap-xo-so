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

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & PRESETS (NGUYÊN BẢN CODE 1)
# ==============================================================================
st.set_page_config(
    page_title="Quang Pro V54 - Full Stats (Fix Data)", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# CSS FIX UI (Giữ bảng không nhảy)
st.markdown("""
<style>
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 5px; }
    thead tr th:first-child { display:none }
    tbody th { display:none }
    .stButton>button { width: 100%; height: 50px; border-radius: 8px; font-weight: bold; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ QUANG HANDSOME: V54 FULL STATS (FIX DATA)")
st.caption("🚀 Logic: Nguyên bản Code 1 (Chi tiết nhóm, Limits) | Data: Smart Loader (Code 2)")

# --- CÁC CẤU HÌNH MẪU (PRESETS) ---
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
    },
    "Miền Nam (Experimental)": {
        "STD": [60, 8, 9, 10, 10, 30, 70, 30, 30, 30, 30],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40],
        "LIMITS": {'l12': 85, 'l34': 80, 'l56': 75, 'mod': 90}
    }
}

RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# Init Session
if 'std_0' not in st.session_state:
    preset = SCORES_PRESETS["Hard Core (Khuyên dùng)"]
    for i in range(11):
        st.session_state[f'std_{i}'] = preset["STD"][i]
        st.session_state[f'mod_{i}'] = preset["MOD"][i]

# ==============================================================================
# 2. CORE FUNCTIONS (GIỮ NGUYÊN 100% TỪ CODE 1)
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
        try: return datetime.date(f_y, m, d)
        except: return None
    return None

def extract_meta_from_filename(fname):
    fname = fname.upper()
    match_m = re.search(r'TH[AÁ]NG\s*(\d{1,2})', fname)
    f_m = int(match_m.group(1)) if match_m else datetime.date.today().month
    match_y = re.search(r'20\d{2}', fname)
    f_y = int(match_y.group(0)) if match_y else datetime.date.today().year
    return f_m, f_y, None

# ==============================================================================
# 3. DATA LOADER MỚI (THAY THẾ LOAD_DATA_V24 CŨ)
# ==============================================================================
# Đây là phần DUY NHẤT được sửa đổi để fix lỗi đọc file rác

def find_header_row_smart(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        count = sum(1 for k in keywords if k in row_str)
        if count >= 1: return idx
    return 0

@st.cache_data(ttl=600, show_spinner=False)
def load_data_smart(files):
    cache = {} 
    kq_db = {}
    err_logs = []
    files = sorted(files, key=lambda x: x.name)

    for file in files:
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper() or 'BPĐ' in file.name.upper(): continue
        f_m, f_y, _ = extract_meta_from_filename(file.name)
        
        try:
            # 1. Auto Detect Header
            try:
                df_raw = pd.read_csv(file, header=None, encoding='utf-8', on_bad_lines='skip')
            except:
                df_raw = pd.read_excel(file, header=None) if file.name.endswith('.xlsx') else pd.DataFrame()
                
            if df_raw.empty: continue
            
            header_idx = find_header_row_smart(df_raw)
            
            # Đọc lại với header đúng
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, header=header_idx, encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(file, header=header_idx)
            
            # 2. Fix Trùng Cột Thành Viên
            tv_cols = [c for c in df.columns if "THÀNH VIÊN" in str(c).upper() or "MEMBER" in str(c).upper()]
            valid_mem_col = None
            if len(tv_cols) > 0:
                for col in tv_cols:
                    sample = df[col].iloc[1:6].astype(str)
                    if sample.str.contains(r'[a-zA-Z]').any():
                        valid_mem_col = col; break
                if valid_mem_col: df.rename(columns={valid_mem_col: 'MEMBER'}, inplace=True)
            
            if 'MEMBER' not in df.columns:
                stt_cols = [c for c in df.columns if "STT" in str(c).upper()]
                if stt_cols:
                    stt_idx = df.columns.get_loc(stt_cols[0])
                    if stt_idx + 1 < len(df.columns): df.rename(columns={df.columns[stt_idx+1]: 'MEMBER'}, inplace=True)

            if 'MEMBER' not in df.columns: continue

            # 3. Lọc Rác
            df = df[df['MEMBER'].notna()]
            df = df[~df['MEMBER'].astype(str).str.contains("THÀNH VIÊN|STT|MEMBER|DANH SÁCH", case=False)]
            
            # 4. Map Ngày & History Map (Logic Code 1 CẦN cái này)
            col_map_date = {}
            for col in df.columns:
                d_obj = parse_date_smart(col, f_m, f_y)
                if d_obj: col_map_date[col] = d_obj
            
            # Trích xuất KQ
            kq_rows = df[df.iloc[:, 0].astype(str).str.contains("KQ", case=False, na=False)]
            if not kq_rows.empty:
                kq_row = kq_rows.iloc[0]
                for col, d_obj in col_map_date.items():
                    try:
                        val = str(kq_row[col])
                        if val.isdigit(): kq_db[d_obj] = int(val)
                    except: pass
            
            # Tạo Hist Map cho file này (Ngày -> Cột hôm qua)
            hist_map = {}
            sorted_dates = sorted(col_map_date.values())
            date_to_col = {v: k for k, v in col_map_date.items()}
            
            for i in range(1, len(sorted_dates)):
                curr_d = sorted_dates[i]
                prev_d = sorted_dates[i-1]
                hist_map[curr_d] = date_to_col[prev_d]
            
            # Lưu Cache (Structure Code 1 yêu cầu)
            for col, d_obj in col_map_date.items():
                cache[d_obj] = {'df': df, 'col_name': col, 'hist_map': hist_map}
                
        except Exception: continue
            
    return cache, kq_db, err_logs

# ==============================================================================
# 4. LOGIC TÍNH TOÁN CHI TIẾT (NGUYÊN BẢN CODE 1)
# ==============================================================================

def fast_get_top_nums(df, p_map_dict, s_map_dict, top_n, min_v, inverse):
    """Hàm tính Top nguyên bản, có xử lý sort để tránh random"""
    cols_in_scope = sorted(list(set(p_map_dict.keys()) | set(s_map_dict.keys())))
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
    
    if inverse: stats = stats.sort_values(by=['P', 'S', 'Num_Int'], ascending=[False, False, True])
    else: stats = stats.sort_values(by=['P', 'V', 'Num_Int'], ascending=[False, False, True])

    return stats['Num'].head(int(top_n)).tolist()

def analyze_group_performance(start_date, end_date, cut_limit, score_map, _cache, _kq_db, min_v, inverse):
    """
    HÀM CỐT LÕI TẠO RA BẢNG CHI TIẾT NHÓM MÀ MÀY CẦN
    Trả về df_rep (Bảng thống kê) và detailed_rows (Chi tiết từng ngày)
    """
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    score_map_tuple = tuple(score_map.items())
    
    # Stats container
    grp_stats = {f"{i}x": {'wins': 0, 'ranks': [], 'history': [], 'last_pred': []} for i in range(10)}
    detailed_rows = [] 
    
    for d in reversed(dates):
        day_record = {"Ngày": d.strftime("%d/%m"), "KQ": _kq_db.get(d, "N/A")}
        if d not in _kq_db or d not in _cache: 
             detailed_rows.append(day_record); continue
        
        curr_data = _cache[d]
        df = curr_data['df']
        
        # Tìm cột hôm qua để phân loại nhóm M
        prev_date = d - timedelta(days=1)
        if prev_date not in curr_data['hist_map']: 
            for k in range(2, 4):
                if (d - timedelta(days=k)) in curr_data['hist_map']: prev_date = d - timedelta(days=k); break
        
        hist_col_name = curr_data['hist_map'].get(prev_date) if prev_date in curr_data['hist_map'] else None
        
        # Nếu không có lịch sử -> Không phân nhóm được
        if not hist_col_name: detailed_rows.append(day_record); continue
        
        # Lấy series M của hôm qua
        try:
            hist_series = df[hist_col_name].astype(str).str.upper().replace('S', '6', regex=False).str.replace(r'[^0-9X]', '', regex=True)
        except: continue
        
        kq = _kq_db[d]
        d_p_map = {}; d_s_map = {} 
        for col in df.columns:
            s_p = get_col_score(col, score_map_tuple)
            if s_p > 0: d_p_map[col] = s_p
            
        # LOOP TÍNH CHI TIẾT TỪNG NHÓM (CÁI MÀY CẦN)
        for g in grp_stats: # g = "0x", "1x"...
            # Logic Code 1: Lọc thành viên dựa vào kết quả hôm qua (hist_series)
            # Nếu g="0X", tìm những người hôm qua có số trúng (khớp 1 phần logic Code 1 cũ)
            # Ở đây Code 1 dùng logic: Check chuỗi hist_series có khớp g.upper() không
            # Vd: Nếu hist_series là "1X" thì thuộc nhóm 1x.
            
            # NOTE: Với file tĩnh mới, cột M có thể tên là M0, M1. 
            # Ta cần map logic này: Nếu file có cột M0..M9, dùng cột đó.
            # Nếu không, dùng logic hist_series (fallback).
            
            valid_mems = pd.DataFrame()
            
            # Ưu tiên tìm cột M có sẵn trong file (Fix cho file mới)
            m_idx = int(g[0])
            m_col_real = None
            for c in df.columns:
                if f"M{m_idx}" == c.upper() or f"M {m_idx}" in c.upper(): m_col_real = c; break
            
            if m_col_real:
                mask = df[m_col_real].astype(str).str.contains('1', na=False)
                valid_mems = df[mask]
            else:
                # Fallback logic cũ (Check hist_series)
                mask = hist_series == g.upper()
                valid_mems = df[mask]
            
            if valid_mems.empty:
                grp_stats[g]['ranks'].append(999); grp_stats[g]['history'].append(None); continue

            # Tính Top số cho nhóm này
            top_list = fast_get_top_nums(valid_mems, d_p_map, d_p_map, int(cut_limit), min_v, inverse)
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
        
    # Tạo Báo Cáo
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

def calculate_matrix_v54(df, target_col, score_map, alliance_report, limits, cut_top, is_mod):
    """
    LOGIC TÍNH MATRIX CỦA CODE 1 (CÓ XỬ LÝ LIÊN MINH LIMITS)
    """
    # Nếu chạy MOD, cần xác định Liên Minh từ alliance_report
    l12, l34, l56 = [], [], []
    if is_mod and not alliance_report.empty:
        top_grps = alliance_report['Nhóm'].head(6).tolist()
        # Parse "0x" -> 0
        try:
            l12 = [int(g[0]) for g in top_grps[:2]]
            l34 = [int(g[0]) for g in top_grps[2:4]]
            l56 = [int(g[0]) for g in top_grps[4:6]]
        except: pass # Fallback
    
    # Fallback default
    if is_mod and not l12: l12, l34, l56 = [0,1,5], [2,3,4], [6,7]

    matrix = np.zeros(100)
    
    for idx, row in df.iterrows():
        if "KQ" in str(row.iloc[0]): continue
        if pd.isna(row['MEMBER']): continue
        
        nums = get_nums(row[target_col])
        if not nums: continue
        
        # Xác định M hiện tại
        m_curr = 10
        for m in range(10):
            # Check cột M0, M1...
            c_name = None
            for c in df.columns:
                if f"M{m}" == c.upper() or f"M {m}" in c.upper(): c_name = c; break
            if c_name and str(row[c_name]) == '1':
                m_curr = m; break
        
        # Tính điểm
        sc = 0
        if is_mod:
            if m_curr in l12: sc = limits['l12']
            elif m_curr in l34: sc = limits['l34']
            elif m_curr in l56: sc = limits['l56']
            else: sc = score_map.get(f'M{m_curr}', 0)
        else:
            sc = score_map.get(f'M{m_curr}', 0)
            
        for n_str in nums:
            n = int(n_str)
            if 0 <= n <= 99: matrix[n] += sc

    ranked = [(i, matrix[i]) for i in range(100)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    final_set = [x[0] for x in ranked[:cut_top]]
    final_set.sort()
    
    return final_set, ranked, (l12, l34, l56)

# ==============================================================================
# 5. GIAO DIỆN CHÍNH (FULL TÍNH NĂNG NHƯ CODE 1)
# ==============================================================================

def main():
    with st.sidebar:
        st.header("📂 Dữ Liệu")
        uploaded_files = st.file_uploader("Upload CSV/Excel:", accept_multiple_files=True)
        st.divider()
        st.header("⚙️ Cấu Hình")
        preset_name = st.selectbox("Preset:", list(SCORES_PRESETS.keys()))
        if st.button("Load Preset"):
            p = SCORES_PRESETS[preset_name]
            for i in range(11):
                st.session_state[f'std_{i}'] = p["STD"][i]
                st.session_state[f'mod_{i}'] = p["MOD"][i]
            st.success("Loaded!")
        
        with st.expander("Chỉnh điểm chi tiết"):
            c1, c2 = st.columns(2)
            with c1: 
                for i in range(11): st.session_state[f'std_{i}'] = st.number_input(f"S-M{i}", value=st.session_state[f'std_{i}'], key=f"s{i}")
            with c2:
                for i in range(11): st.session_state[f'mod_{i}'] = st.number_input(f"M-M{i}", value=st.session_state[f'mod_{i}'], key=f"m{i}")

    if not uploaded_files: st.info("👈 Upload file để bắt đầu."); return

    # LOAD DATA (SMART)
    with st.spinner("Đang xử lý dữ liệu..."):
        cache, kq_db, errs = load_data_smart(uploaded_files)
    
    if errs: 
        for e in errs: st.warning(e)
    if not cache: st.error("Không có dữ liệu."); return
    
    sorted_dates = sorted(cache.keys())
    last_d = sorted_dates[-1]

    # MAIN UI
    tab1, tab2, tab3 = st.tabs(["🔎 PHÂN TÍCH MATRIX", "📊 BACKTEST", "📈 CHI TIẾT NHÓM"])

    # TAB 1: PHÂN TÍCH MATRIX
    with tab1:
        st.subheader(f"Ngày: {last_d.strftime('%d/%m/%Y')}")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1: target_d = st.selectbox("Chọn ngày:", sorted_dates, index=len(sorted_dates)-1, format_func=lambda x: x.strftime("%d/%m/%Y"))
        with c2: cut_val = st.number_input("Cắt Top:", 10, 90, 60)
        with c3: mode = st.radio("Chế độ:", ["STD", "MOD"])
        is_mod = (mode == "MOD")
        
        if st.button("🚀 QUÉT MATRIX", type="primary"):
            # Lấy Score Map
            s_map = {f"M{m}": st.session_state[f'mod_{m}' if is_mod else f'std_{m}'] for m in range(11)}
            limits = SCORES_PRESETS["Hard Core (Khuyên dùng)"]["LIMITS"]
            
            # 1. Chạy Analyze Group (Để lấy thông tin Liên Minh)
            df_rep, _ = analyze_group_performance(target_d - timedelta(days=15), target_d, cut_val, s_map, cache, kq_db, 1, False)
            
            # 2. Tính Matrix Final
            curr_data = cache[target_d]
            f_set, ranked, (l12, l34, l56) = calculate_matrix_v54(curr_data['df'], curr_data['col_name'], s_map, df_rep, limits, cut_val, is_mod)
            
            # Hiển thị thông tin Liên Minh (Cái mày cần)
            if is_mod:
                st.info(f"🏆 LIÊN MINH 1 (Mạnh nhất): Nhóm {l12} (Điểm {limits['l12']})")
                st.success(f"🥈 LIÊN MINH 2: Nhóm {l34} (Điểm {limits['l34']}) | 🥉 LIÊN MINH 3: Nhóm {l56}")
            
            st.divider()
            val_str = ",".join([f"{n:02d}" for n in f_set])
            st.text_area("👇 DÀN SỐ:", value=val_str, height=80)
            
            if target_d in kq_db:
                real = kq_db[target_d]
                rnk = 999
                for i, (n, s) in enumerate(ranked):
                    if n == real: rnk = i + 1; break
                m1, m2 = st.columns(2)
                if real in f_set: m1.metric("KẾT QUẢ", f"WIN {real}", delta=f"Hạng {rnk}")
                else: m1.metric("KẾT QUẢ", f"MISS {real}", delta_color="inverse")
                m2.metric("Tổng số", len(f_set))
            
            # Bảng xếp hạng chi tiết
            rank_df = pd.DataFrame(ranked, columns=["Số", "Điểm"])
            rank_df["Số"] = rank_df["Số"].apply(lambda x: f"{x:02d}")
            rank_df["Trạng Thái"] = ["LẤY" if i < cut_val else "LOẠI" for i in range(100)]
            st.dataframe(rank_df, use_container_width=True, height=500, hide_index=True)

    # TAB 2: BACKTEST
    with tab2:
        st.subheader("Backtest (Roll 10 ngày)")
        days_bt = st.slider("Số ngày:", 5, 30, 10)
        if st.button("Chạy Backtest"):
            # Chạy Loop Matrix cho các ngày trước
            dates_bt = [d for d in sorted_dates if d <= target_d][-days_bt:]
            stats = []
            bar = st.progress(0)
            s_map = {f"M{m}": st.session_state[f'std_{m}'] for m in range(11)}
            limits = SCORES_PRESETS["Hard Core (Khuyên dùng)"]["LIMITS"]
            
            for i, d in enumerate(dates_bt):
                if d not in kq_db: continue
                # Chạy STD để test nhanh
                curr_data = cache[d]
                f_set, rk, _ = calculate_matrix_v54(curr_data['df'], curr_data['col_name'], s_map, pd.DataFrame(), limits, cut_val, False)
                real = kq_db[d]
                rnk = 999
                for idx, (n, s) in enumerate(rk):
                    if n == real: rnk = idx + 1; break
                stats.append({"Ngày": d.strftime("%d/%m"), "KQ": real, "Status": "WIN" if real in f_set else "MISS", "Hạng": rnk})
                bar.progress((i+1)/len(dates_bt))
            st.dataframe(pd.DataFrame(stats), use_container_width=True)

    # TAB 3: CHI TIẾT NHÓM (CÁI MÀY CẦN)
    with tab3:
        st.subheader("Hiệu Suất Nhóm 0x-9x")
        if st.button("Phân Tích Nhóm"):
            s_map = {f"M{m}": st.session_state[f'mod_{m}' if is_mod else f'std_{m}'] for m in range(11)}
            df_rep, df_detail = analyze_group_performance(target_d - timedelta(days=15), target_d, cut_val, s_map, cache, kq_db, 1, False)
            
            c1, c2 = st.columns([1, 1])
            with c1: 
                st.write("Bảng Tổng Hợp")
                st.dataframe(df_rep, use_container_width=True)
            with c2: 
                st.write("Chi Tiết Từng Ngày")
                st.dataframe(df_detail, use_container_width=True)

if __name__ == "__main__":
    main()
