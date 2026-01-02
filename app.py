import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta
import io

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V24 (Full Option)", page_icon="🎯", layout="wide")
st.title("🎯 V24: Backtest Đa Năng (Xuyên Năm)")

# --- 1. TẢI FILE ---
uploaded_files = st.file_uploader("Tải tất cả file CSV (Chọn cả T12 và T1 cùng lúc):", type=['xlsx', 'csv'], accept_multiple_files=True)

# --- CẤU HÌNH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Cài đặt chung")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
    
    st.markdown("---")
    st.header("✂️ Tùy chọn cắt số")
    L_TOP_12 = st.number_input("Top 1 & 2 lấy:", min_value=10, max_value=90, value=80)
    L_TOP_34 = st.number_input("Top 3 & 4 lấy:", min_value=10, max_value=90, value=65)
    L_TOP_56 = st.number_input("Top 5 & 6 lấy:", min_value=10, max_value=90, value=60)
    LIMIT_MODIFIED = st.number_input("Top 1 Modified lấy:", min_value=50, value=86)

# ==============================================================================
# CẤU HÌNH ĐIỂM SỐ
# ==============================================================================
SCORE_MAPPING_STD = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}

SCORE_MAPPING_MOD = {
    'M6': 50, 'M10': 40, 'M7': 35,
    'M4': 30, 'M5': 30, 'M8': 25, 'M9': 25,
    'M3': 15, 'M2': 10, 'M1': 5, 'M0': 0
}

# ==============================================================================
# HÀM XỬ LÝ DỮ LIỆU (Đã update regex cho file 1.1.2026)
# ==============================================================================

def get_nums(s):
    if pd.isna(s): return []
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
    s = re.sub(r'NGAY|NGÀY', '', s).strip()
    match_iso = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)
    match_slash = re.search(r'(\d{1,2})[\.\-/](\d{1,2})', s)
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
    # Lấy Năm
    y_match = re.search(r'20\d{2}', filename)
    y_global = int(y_match.group(0)) if y_match else 2025
    
    # Lấy Tháng
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m_global = int(m_match.group(1)) if m_match else 12

    # Lấy Ngày cụ thể từ đuôi file (hỗ trợ cả 1.1.2026.csv)
    # Regex này bắt: "- 1.12.csv", "- 2.csv", "- 1.1.2026.csv"
    specific_match = re.search(r'-\s*(\d{1,2})(?:[\.\-](\d{1,2}))?(?:[\.\-]\d{4})?\.csv$', filename, re.IGNORECASE)
    
    target_date = None
    if specific_match:
        d = int(specific_match.group(1))
        # Nếu có tháng đi kèm ở đuôi thì lấy, không thì lấy tháng global
        m = int(specific_match.group(2)) if specific_match.group(2) else m_global
        try:
            target_date = datetime.date(y_global, m, d)
        except: pass
            
    return m_global, y_global, target_date

@st.cache_data(ttl=600)
def load_data_v24(files):
    cache = {} 
    kq_db = {}
    err_logs = []

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

            for t_date, df in dfs_to_process:
                df.columns = [str(c).strip().upper() for c in df.columns]
                hist_map = {}
                for col in df.columns:
                    if "UNNAMED" in col: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: hist_map[d_obj] = col
                
                data_map = {}
                for col in df.columns:
                    c_clean = col.replace(" ", "").replace("SX", "6X")
                    if re.match(r'^\d+X$', c_clean):
                        data_map[c_clean.lower()] = col

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

                cache[t_date] = {'df': df, 'hist_map': hist_map, 'data_map': data_map}

        except Exception as e:
            err_logs.append(f"Lỗi file '{file.name}': {str(e)}")
            continue
            
    if err_logs:
        with st.expander(f"⚠️ Đã bỏ qua {len(err_logs)} file lỗi", expanded=False):
            for e in err_logs: st.write(e)
            
    return cache, kq_db

# --- HÀM TÍNH TOÁN CORE ---
def calculate_v24_final(target_date, rolling_window, cache, kq_db, limits_config):
    if target_date not in cache: return None, "Chưa có dữ liệu."
    
    curr_data = cache[target_date]
    df = curr_data['df']
    
    prev_date = target_date - timedelta(days=1)
    if prev_date not in cache and (target_date - timedelta(days=2)) in cache:
         prev_date = target_date - timedelta(days=2)

    col_hist_used = curr_data['hist_map'].get(prev_date)
    if not col_hist_used and prev_date in cache:
        col_hist_used = cache[prev_date]['hist_map'].get(prev_date)
        
    if not col_hist_used:
        return None, f"Không tìm thấy cột dữ liệu ngày {prev_date.strftime('%d/%m')}."

    groups = [f"{i}x" for i in range(10)]
    stats_std = {g: {'wins': 0, 'ranks': []} for g in groups}
    stats_mod = {g: {'wins': 0} for g in groups}

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
        
        d_score_std = {c: get_col_score(c, SCORE_MAPPING_STD) for c in d_df.columns if get_col_score(c, SCORE_MAPPING_STD) > 0}
        d_score_mod = {c: get_col_score(c, SCORE_MAPPING_MOD) for c in d_df.columns if get_col_score(c, SCORE_MAPPING_MOD) > 0}

        for g in groups:
            try:
                mask = d_df[d_hist_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6'))) == g.upper()
                mems = d_df[mask]
            except: continue
            
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

            top80_std = get_top_nums(mems, d_score_std, 80)
            if kq in top80_std:
                stats_std[g]['wins'] += 1
                stats_std[g]['ranks'].append(top80_std.index(kq) + 1)
            else: stats_std[g]['ranks'].append(999)
            
            top86_mod = get_top_nums(mems, d_score_mod, limits_config['mod'])
            if kq in top86_mod: stats_mod[g]['wins'] += 1

    final_std = []
    for g, inf in stats_std.items(): final_std.append((g, -inf['wins'], sum(inf['ranks'])))
    final_std.sort(key=lambda x: (x[1], x[2]))
    top6_std = [x[0] for x in final_std[:6]]

    best_mod_grp = sorted(stats_mod.keys(), key=lambda g: -stats_mod[g]['wins'])[0]
    
    # Dự đoán
    hist_series = df[col_hist_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6')))
    
    def get_group_set(group_name, score_map, limit):
        mask = hist_series == group_name.upper()
        valid_mems = df[mask]
        curr_scores = {c: get_col_score(c, score_map) for c in df.columns if get_col_score(c, score_map) > 0}
        
        local_stats = {}
        for _, r in valid_mems.iterrows():
            p_votes = set()
            for sc_col, pts in curr_scores.items():
                if sc_col not in valid_mems.columns: continue
                for n in get_nums(r[sc_col]):
                    if n not in local_stats: local_stats[n] = {'score':0, 'votes':0}
                    local_stats[n]['score'] += pts
                    if n not in p_votes:
                        local_stats[n]['votes'] += 1
                        p_votes.add(n)
        return set(sorted(local_stats.keys(), key=lambda n: (-local_stats[n]['score'], -local_stats[n]['votes'], int(n)))[:limit])

    limits_std = {
        top6_std[0]: limits_config['l12'], top6_std[1]: limits_config['l12'], 
        top6_std[2]: limits_config['l34'], top6_std[3]: limits_config['l34'], 
        top6_std[4]: limits_config['l56'], top6_std[5]: limits_config['l56']
    }
    
    pool1 = []
    for g in [top6_std[0], top6_std[5], top6_std[3]]: pool1.extend(list(get_group_set(g, SCORE_MAPPING_STD, limits_std[g])))
    s1 = {n for n, c in Counter(pool1).items() if c >= 2}
    
    pool2 = []
    for g in [top6_std[1], top6_std[4], top6_std[2]]: pool2.extend(list(get_group_set(g, SCORE_MAPPING_STD, limits_std[g])))
    s2 = {n for n, c in Counter(pool2).items() if c >= 2}
    
    final_original = sorted(list(s1.intersection(s2)))
    final_modified = sorted(list(get_group_set(best_mod_grp, SCORE_MAPPING_MOD, limits_config['mod'])))
    final_intersect = sorted(list(set(final_original).intersection(set(final_modified))))

    return {
        "top6_std": top6_std,
        "best_mod": best_mod_grp,
        "dan_goc": final_original,
        "dan_mod": final_modified,
        "dan_final": final_intersect,
        "source_col": col_hist_used
    }, None

# --- UI ---
if uploaded_files:
    data_cache, kq_db = load_data_v24(uploaded_files)
    
    if data_cache:
        limit_cfg = {'l12': L_TOP_12, 'l34': L_TOP_34, 'l56': L_TOP_56, 'mod': LIMIT_MODIFIED}
        last_d = max(data_cache.keys())
        
        # --- TAB CHUYỂN ĐỔI ---
        tab1, tab2 = st.tabs(["📊 DỰ ĐOÁN HÔM NAY", "🔙 BACKTEST QUÁ KHỨ"])
        
        # TAB 1: DỰ ĐOÁN
        with tab1:
            st.subheader("Dự đoán hàng ngày")
            target = st.date_input("📅 Ngày dự đoán:", value=last_d)
            if st.button("🚀 CHẠY DỰ ĐOÁN", type="primary"):
                with st.spinner("Đang tính toán..."):
                    res, err = calculate_v24_final(target, ROLLING_WINDOW, data_cache, kq_db, limit_cfg)
                    if err: st.error(err)
                    else:
                        st.info(f"Phân nhóm dựa trên ngày: {res['source_col']}")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.subheader("1️⃣ Dàn Gốc")
                            st.caption(f"Top 6: {', '.join(res['top6_std'])}")
                            st.text_area(f"Original ({len(res['dan_goc'])} số):", ",".join(res['dan_goc']), height=150)
                        with c2:
                            st.subheader("2️⃣ Modified")
                            st.caption(f"Best: {res['best_mod']}")
                            st.text_area(f"Modified ({len(res['dan_mod'])} số):", ",".join(res['dan_mod']), height=150)
                        with c3:
                            st.subheader("3️⃣ FINAL")
                            st.caption("Giao thoa (1) & (2)")
                            st.code(",".join(res['dan_final']), language="text")
                            st.metric("Số lượng", f"{len(res['dan_final'])} số")
                        
                        if target in kq_db:
                            real = kq_db[target]
                            st.markdown("---")
                            if real in res['dan_final']:
                                st.success(f"🎉 CHÚC MỪNG! KQ **{real}** TRÚNG FINAL!")
                            else:
                                st.error(f"❌ Kết quả **{real}** trượt Final.")

        # TAB 2: BACKTEST
        with tab2:
            st.subheader("Kiểm thử quá khứ (Backtest)")
            
            c_bt_1, c_bt_2 = st.columns(2)
            with c_bt_1:
                date_range = st.date_input("Chọn khoảng ngày test:", [last_d - timedelta(days=7), last_d])
            with c_bt_2:
                # TÙY CHỌN DÀN ĐỂ BACKTEST
                bt_mode = st.radio("Chọn dàn để kiểm tra:", ["FINAL (Giao thoa)", "Dàn Gốc (Original)", "Dàn Modified"], horizontal=True)

            if st.button("🔄 CHẠY BACKTEST"):
                if len(date_range) < 2:
                    st.warning("Vui lòng chọn đủ Ngày bắt đầu và Ngày kết thúc.")
                else:
                    start_date, end_date = date_range[0], date_range[1]
                    logs = []
                    
                    progress_bar = st.progress(0)
                    total_days = (end_date - start_date).days + 1
                    
                    for i in range(total_days):
                        current_d = start_date + timedelta(days=i)
                        progress_bar.progress((i + 1) / total_days)
                        
                        if current_d not in kq_db: continue # Chỉ test ngày có KQ
                        
                        try:
                            res, err = calculate_v24_final(current_d, ROLLING_WINDOW, data_cache, kq_db, limit_cfg)
                            if err: continue
                            
                            real_val = kq_db[current_d]
                            
                            # Lấy dàn số tương ứng với tùy chọn
                            if bt_mode == "FINAL (Giao thoa)":
                                target_set = res['dan_final']
                            elif bt_mode == "Dàn Gốc (Original)":
                                target_set = res['dan_goc']
                            else:
                                target_set = res['dan_mod']
                            
                            is_win = real_val in target_set
                            logs.append({
                                "Ngày": current_d.strftime("%d/%m"),
                                "KQ": real_val,
                                "Trạng thái": "WIN 🎉" if is_win else "MISS ❌",
                                "Số lượng": len(target_set),
                                "Chi tiết số": ",".join(target_set) if len(target_set) < 20 else f"{len(target_set)} số (Bấm xem)"
                            })
                        except: pass
                    
                    progress_bar.empty()
                    
                    # Hiển thị kết quả
                    if logs:
                        df_logs = pd.DataFrame(logs)
                        wins = df_logs[df_logs["Trạng thái"].str.contains("WIN")].shape[0]
                        total = df_logs.shape[0]
                        rate = (wins/total)*100 if total > 0 else 0
                        
                        st.metric(f"Tỉ lệ trúng ({bt_mode})", f"{wins}/{total}", f"{rate:.1f}%")
                        st.dataframe(df_logs, use_container_width=True)
                    else:
                        st.warning("Không có dữ liệu hoặc không tính toán được trong khoảng thời gian này.")

    else:
        st.error("Không đọc được dữ liệu nào hợp lệ. Vui lòng kiểm tra file.")
