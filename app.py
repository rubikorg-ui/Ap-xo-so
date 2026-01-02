import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V24 (Pro)", page_icon="🎯", layout="wide") # Chuyển layout sang wide cho thoáng
st.title("🎯 V24: Logic Gốc + Modified (Tùy Chỉnh Cắt Số)")

# --- 1. TẢI FILE ---
uploaded_files = st.file_uploader("Tải file Excel (T12, T1...):", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Cài đặt chung")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
    
    st.markdown("---")
    st.header("✂️ Tùy chọn cắt số")
    
    st.subheader("1. Dàn Gốc (Original)")
    st.caption("Số lượng số lấy cho các nhóm Top:")
    L_TOP_12 = st.number_input("Top 1 & 2 lấy:", min_value=10, max_value=90, value=80, help="Mặc định cũ: 80")
    L_TOP_34 = st.number_input("Top 3 & 4 lấy:", min_value=10, max_value=90, value=65, help="Mặc định cũ: 65")
    L_TOP_56 = st.number_input("Top 5 & 6 lấy:", min_value=10, max_value=90, value=60, help="Mặc định cũ: 60")
    
    st.markdown("---")
    st.subheader("2. Dàn Modified")
    LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", min_value=50, value=86, help="Mặc định: 86")

# ==============================================================================
# ⚠️ CẤU HÌNH ĐIỂM SỐ - TÁCH BIỆT RÕ RÀNG
# ==============================================================================

# 1. BẢNG ĐIỂM CŨ (Dùng cho Dàn Gốc - Logic V3 cũ)
SCORE_MAPPING_STD = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}

# 2. BẢNG ĐIỂM MỚI (Dùng riêng cho Dàn Modified - Ưu tiên M6)
SCORE_MAPPING_MOD = {
    'M6': 50, 'M10': 40, 'M7': 35,
    'M4': 30, 'M5': 30, 'M8': 25, 'M9': 25,
    'M3': 15, 'M2': 10, 'M1': 5, 'M0': 0
}

# ==============================================================================

# --- HÀM XỬ LÝ TEXT & NGÀY ---
def get_nums(s):
    if pd.isna(s): return []
    # Lọc từ khóa rác
    BAD_KEYWORDS = ['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI']
    s_upper = str(s).upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    
    raw_nums = re.findall(r'\d+', s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

def get_col_score(col_name, mapping):
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return mapping['M10']
    for key, score in mapping.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper()
    match_iso = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        curr_y = f_y
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: pass
    return None

def get_file_meta(filename):
    y_match = re.search(r'20\d{2}', filename)
    y = int(y_match.group(0)) if y_match else 2025
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m = int(m_match.group(1)) if m_match else 1
    return m, y

def get_sheet_date(sheet_name, f_m, f_y):
    s_clean = re.sub(r'[^0-9]', ' ', sheet_name).strip()
    try:
        parts = [int(x) for x in s_clean.split()]
        if not parts: return None
        d = parts[0]
        m, y = f_m, f_y
        if len(parts) >= 3 and parts[2] > 2000: y = parts[2]; m = parts[1]
        return datetime.date(y, m, d)
    except: return None

@st.cache_data(ttl=600)
def load_data_v24(files):
    cache = {} 
    kq_db = {}
    for file in files:
        f_m, f_y = get_file_meta(file.name)
        try:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                try:
                    t_date = get_sheet_date(sheet, f_m, f_y)
                    if not t_date: continue
                    
                    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    h_row = 3
                    for idx, row in preview.iterrows():
                        if "TV TOP" in str(row.values).upper() or "THÀNH VIÊN" in str(row.values).upper():
                            h_row = idx; break
                    df = pd.read_excel(xls, sheet_name=sheet, header=h_row)
                    
                    hist_map = {}
                    for col in df.columns:
                        d_obj = parse_date_smart(col, f_m, f_y)
                        if d_obj: hist_map[d_obj] = col
                    
                    data_map = {}
                    for col in df.columns:
                        c_upper = str(col).strip().upper()
                        if "SX" in c_upper: c_upper = c_upper.replace("SX", "6X")
                        if re.match(r'^\d+X$', c_upper):
                            data_map[c_upper.lower()] = col

                    kq_row = None
                    for idx, row in df.iterrows():
                        if str(row.values[0]).strip().upper() == "KQ": kq_row = row; break
                    if kq_row is not None:
                        for d_val, c_name in hist_map.items():
                            val = str(kq_row[c_name])
                            nums = get_nums(val)
                            if nums: kq_db[d_val] = nums[0]

                    cache[t_date] = {'df': df, 'hist_map': hist_map, 'data_map': data_map}
                except: continue
        except: continue
    return cache, kq_db

# --- HÀM TÍNH TOÁN CORE (NÂNG CẤP VỚI THAM SỐ CẮT) ---
def calculate_v24_upgrade(target_date, rolling_window, cache, kq_db, limits_config):
    if target_date not in cache: return None, "Chưa có Sheet dữ liệu."
    
    curr_data = cache[target_date]
    df = curr_data['df']
    
    prev_date = target_date - timedelta(days=1)
    col_hist_used = curr_data['hist_map'].get(prev_date)
    df_source = df
    
    if not col_hist_used and prev_date in cache:
        col_hist_used = cache[prev_date]['hist_map'].get(prev_date)
        df_source = cache[prev_date]['df']
        
    if not col_hist_used:
        return None, f"Không tìm thấy cột dữ liệu ngày {prev_date.strftime('%d/%m')}."

    # --- PHẦN BACKTEST ---
    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups} # Dàn Gốc
    stats_mod = {g: {'wins': 0} for g in groups}              # Dàn Modified

    past_dates = [target_date - timedelta(days=i) for i in range(1, rolling_window + 1)]
    
    for d in past_dates:
        if d not in kq_db or d not in cache: continue
        d_df = cache[d]['df']
        d_prev = d - timedelta(days=1)
        d_hist_col = cache[d]['hist_map'].get(d_prev)
        if not d_hist_col: continue
        kq = kq_db[d]
        
        d_score_std = {c: get_col_score(c, SCORE_MAPPING_STD) for c in d_df.columns if get_col_score(c, SCORE_MAPPING_STD) > 0}
        d_score_mod = {c: get_col_score(c, SCORE_MAPPING_MOD) for c in d_df.columns if get_col_score(c, SCORE_MAPPING_MOD) > 0}

        for g in groups:
            mask = d_df[d_hist_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6'))) == g.upper()
            mems = d_df[mask]
            if mems.empty: 
                stats_std[g]['ranks'].append(999)
                continue
            
            def get_top_nums(members_df, score_dict, top_n):
                num_stats = {}
                for _, r in members_df.iterrows():
                    p_votes = set()
                    for sc_col, pts in score_dict.items():
                        if sc_col not in members_df.columns: continue
                        for n in get_nums(r[sc_col]):
                            if n not in num_stats: num_stats[n] = {'score':0, 'votes':0}
                            num_stats[n]['score'] += pts
                            if n not in p_votes:
                                num_stats[n]['votes'] += 1
                                p_votes.add(n)
                return sorted(num_stats.keys(), key=lambda n: (-num_stats[n]['score'], -num_stats[n]['votes'], int(n)))[:top_n]

            # 1. Backtest Standard (Top 80 - Để xếp hạng nhóm chuẩn nhất)
            top80_std = get_top_nums(mems, d_score_std, 80)
            if kq in top80_std:
                stats_std[g]['wins'] += 1
                stats_std[g]['ranks'].append(top80_std.index(kq) + 1)
            else:
                stats_std[g]['ranks'].append(999)
            
            # 2. Backtest Modified (Lấy theo LIMIT_MODIFIED)
            top86_mod = get_top_nums(mems, d_score_mod, limits_config['mod'])
            if kq in top86_mod:
                stats_mod[g]['wins'] += 1

    # --- TỔNG HỢP KẾT QUẢ BACKTEST ---
    final_std = []
    for g, inf in stats_std.items(): final_std.append((g, -inf['wins'], sum(inf['ranks'])))
    final_std.sort(key=lambda x: (x[1], x[2]))
    top6_std = [x[0] for x in final_std[:6]]

    best_mod_grp = sorted(stats_mod.keys(), key=lambda g: -stats_mod[g]['wins'])[0]
    
    # --- DỰ ĐOÁN CHO NGÀY TARGET ---
    def get_target_group_nums(group_list, score_mapping, limit_dict=None, fixed_limit=None):
        sets_of_nums = []
        hist_series = df_source[col_hist_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6')))
        L = min(len(df), len(hist_series))
        
        curr_score_cols = {c: get_col_score(c, score_mapping) for c in df.columns if get_col_score(c, score_mapping) > 0}

        for g in group_list:
            mask = hist_series.iloc[:L] == g.upper()
            valid_mems = df.iloc[:L][mask.values]
            
            local_stats = {}
            for _, r in valid_mems.iterrows():
                p_votes = set()
                for sc_col, pts in curr_score_cols.items():
                    if sc_col not in valid_mems.columns: continue
                    for n in get_nums(r[sc_col]):
                        if n not in local_stats: local_stats[n] = {'score':0, 'votes':0}
                        local_stats[n]['score'] += pts
                        if n not in p_votes:
                            local_stats[n]['votes'] += 1
                            p_votes.add(n)
            
            sorted_nums = sorted(local_stats.keys(), key=lambda n: (-local_stats[n]['score'], -local_stats[n]['votes'], int(n)))
            
            limit = 60
            if fixed_limit: limit = fixed_limit
            elif limit_dict: limit = limit_dict.get(g, 60)
            
            sets_of_nums.append(set(sorted_nums[:limit]))
        return sets_of_nums

    # --- ÁP DỤNG CẤU HÌNH CẮT SỐ MỚI CHO DÀN GỐC ---
    # Thay vì fix cứng 80, 65, 60, ta lấy từ limits_config
    limits_std = {
        top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], 
        top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], 
        top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']
    }
    
    s1_list = get_target_group_nums([top6_std[0], top6_std[5], top6_std[3]], SCORE_MAPPING_STD, limit_dict=limits_std)
    s2_list = get_target_group_nums([top6_std[1], top6_std[4], top6_std[2]], SCORE_MAPPING_STD, limit_dict=limits_std)
    
    def pool_alliance(sets):
        pool = []
        for s in sets: pool.extend(list(s))
        return {n for n, c in Counter(pool).items() if c >= 2}

    set_alliance_1 = pool_alliance(s1_list)
    set_alliance_2 = pool_alliance(s2_list)
    final_original_set = sorted(list(set_alliance_1.intersection(set_alliance_2)))

    # --- DÀN MODIFIED (Lấy theo cấu hình) ---
    mod_list = get_target_group_nums([best_mod_grp], SCORE_MAPPING_MOD, fixed_limit=limits_config['mod'])
    final_modified_set = sorted(list(mod_list[0])) if mod_list else []

    # --- DÀN GIAO THOA (FINAL) ---
    final_intersection = sorted(list(set(final_original_set).intersection(set(final_modified_set))))

    results = {
        "top6_std": top6_std,
        "best_mod": best_mod_grp,
        "dan_goc": final_original_set,
        "dan_mod": final_modified_set,
        "dan_final": final_intersection,
        "source_col": col_hist_used
    }
    return results, None

# --- UI ---
if uploaded_files:
    data_cache, kq_db = load_data_v24(uploaded_files)
    st.success(f"✅ Đã đọc {len(data_cache)} ngày dữ liệu.")
    
    last_d = max(data_cache.keys()) if data_cache else datetime.date.today()
    target = st.date_input("📅 Chọn ngày dự đoán:", value=last_d)
    
    # Đóng gói cấu hình giới hạn
    limit_cfg = {
        'l12': L_TOP_12,
        'l34': L_TOP_34,
        'l56': L_TOP_56,
        'mod': LIMIT_MODIFIED
    }
    
    if st.button("🚀 CHẠY PHÂN TÍCH", type="primary"):
        with st.spinner("Đang tính toán với cấu hình tùy chỉnh..."):
            res, err = calculate_v24_upgrade(target, ROLLING_WINDOW, data_cache, kq_db, limit_cfg)
            
        if err: st.error(err)
        else:
            st.info(f"Dữ liệu phân nhóm dựa trên cột: {res['source_col']}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("1️⃣ Dàn Gốc")
                st.caption(f"Top 6: {', '.join(res['top6_std'])}")
                st.text_area(f"SL: {len(res['dan_goc'])} số", ",".join(res['dan_goc']), height=150)
                st.markdown(f"*Cắt: {L_TOP_12} - {L_TOP_34} - {L_TOP_56}*")
            
            with c2:
                st.subheader("2️⃣ Modified")
                st.caption(f"Best Group: {res['best_mod']}")
                st.text_area(f"SL: {len(res['dan_mod'])} số", ",".join(res['dan_mod']), height=150)
                st.markdown(f"*Cắt: {LIMIT_MODIFIED}*")
                
            with c3:
                st.subheader("3️⃣ FINAL (Chốt)")
                st.caption("Giao thoa (1) & (2)")
                st.warning(f"**{len(res['dan_final'])} Số:**")
                st.code(",".join(res['dan_final']), language="text")

            # Check KQ
            if target in kq_db:
                real = kq_db[target]
                is_win = real in res['dan_final']
                st.markdown("---")
                st.markdown(f"### 🎲 Kết quả thực tế: **{real}**")
                if is_win: st.balloons(); st.success(f"🎉 CHÚC MỪNG! ĂN DÀN FINAL ({len(res['dan_final'])} số)")
                else:
                    win_goc = real in res['dan_goc']
                    win_mod = real in res['dan_mod']
                    msg = []
                    if win_goc: msg.append("Ăn Dàn Gốc")
                    if win_mod: msg.append("Ăn Dàn Mod")
                    if msg: st.warning(f"❌ Trượt Final, nhưng {' & '.join(msg)}")
                    else: st.error("❌ TRƯỢT TẤT CẢ CÁC DÀN")
