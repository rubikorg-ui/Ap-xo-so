import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="App Xổ Số Đa Năng", page_icon="🛠️", layout="centered")
st.title("🛠️ Dự Đoán & Backtest (V11 - Fix Lỗi Ngày)")

# --- 1. TẢI FILE ---
st.info("Bước 1: Tải các file Excel (T12.2025, T1.2026...)")
uploaded_files = st.file_uploader("Chọn file:", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)

# --- CÁC HÀM XỬ LÝ ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}
RE_CLEAN = re.compile(r'[^A-Z0-9\/]')
RE_FIND_NUMS = re.compile(r'\d{1,2}') 

def clean_text(s):
    if pd.isna(s): return ""
    s_str = str(s).upper().replace('.', '/').replace('-', '/').replace('_', '/')
    return RE_CLEAN.sub('', s_str)

def get_nums(s):
    if pd.isna(s): return []
    raw_nums = RE_FIND_NUMS.findall(str(s))
    return [n.zfill(2) for n in raw_nums]

def get_col_score(col_name):
    clean = col_name 
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

def get_header_row_index(df_raw):
    for i, row in df_raw.head(10).iterrows():
        row_str = clean_text("".join(row.values.astype(str)))
        if "THANHVIEN" in row_str and "STT" in row_str: return i
    return 3

# --- HÀM ĐỌC NGÀY THÔNG MINH HƠN ---
def parse_date_from_sheet(sheet_name, filename):
    # 1. Thử lấy Tháng/Năm từ tên File trước
    year_match = re.search(r'(20\d{2})', filename)
    year = int(year_match.group(1)) if year_match else None
    
    month_match = re.search(r'(?:THANG|THÁNG|TH|T|M)[^0-9]*(\d+)', filename, re.IGNORECASE)
    if not month_match:
        # Tìm kiểu 1.2026 hoặc 12-2025
        alt_match = re.search(r'(\d+)[\.\-_/]+' + str(year), filename) if year else None
        month = int(alt_match.group(1)) if alt_match else None
    else:
        month = int(month_match.group(1))

    # 2. Lấy Ngày từ tên Sheet (Chấp nhận: "1", "01", "Ngày 1", "Day 1")
    day_match = re.search(r'(\d+)', sheet_name)
    day = int(day_match.group(1)) if day_match else None
    
    # 3. Kết hợp
    if day and month and year:
        try:
            return datetime.date(year, month, day)
        except ValueError: return None
    return None

@st.cache_data(ttl=600)
def load_data_with_calendar(files):
    data_cache = {}
    kq_db = {} 
    debug_logs = []
    
    for file in files:
        debug_logs.append(f"📂 Đang đọc file: {file.name}")
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                try:
                    # Sử dụng hàm đọc ngày mới
                    current_date = parse_date_from_sheet(sheet_name, file.name)
                    
                    if not current_date:
                        continue 

                    temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=10)
                    h = get_header_row_index(temp)
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h)
                    df.columns = [clean_text(c) for c in df.columns]
                    
                    data_cache[current_date] = df
                    
                    # Tìm KQ
                    mask_kq = df.iloc[:, 0].astype(str).apply(clean_text).str.contains("KQ", na=False)
                    if mask_kq.any():
                        kq_row = df[mask_kq].iloc[0]
                        # Patterns tìm cột ngày KQ
                        day, month = current_date.day, current_date.month
                        target_col_patterns = [
                            f"{day}/{month}", f"{day:02d}/{month}", 
                            f"{day}/{month:02d}", f"{day:02d}/{month:02d}", str(day)
                        ]
                        found_kq = None
                        for c in df.columns:
                            for p in target_col_patterns:
                                if p in c: # So sánh chuỗi chứa
                                    try: 
                                        val = str(kq_row[c])
                                        nums = get_nums(val)
                                        if nums: found_kq = nums[0]
                                    except: pass
                            if found_kq: break
                        if found_kq: kq_db[current_date] = found_kq
                except: continue
        except: 
            debug_logs.append(f"❌ Lỗi đọc file Excel này.")
            continue
            
    return data_cache, kq_db, debug_logs

def get_group_top_n_stable(df, group_name, grp_col, limit=80):
    target_group = clean_text(group_name)
    try:
        mask = df[grp_col].astype(str).apply(clean_text) == target_group
        members = df[mask]
    except KeyError: return []
    if members.empty: return []

    col_scores = {}
    valid_cols = []
    for c in sorted(df.columns):
        s = get_col_score(c)
        if s > 0: col_scores[c] = s; valid_cols.append(c)

    total_scores = Counter()
    vote_counts = Counter()
    subset = members[valid_cols]
    for row in subset.itertuples(index=False):
        person_votes = set()
        for val, col_name in zip(row, valid_cols):
            if pd.notna(val):
                nums = get_nums(val)
                score = col_scores[col_name]
                for n in nums:
                    total_scores[n] += score
                    if n not in person_votes: vote_counts[n] += 1
                    person_votes.add(n)
    all_nums = list(total_scores.keys())
    all_nums.sort(key=lambda n: (-total_scores[n], -vote_counts[n], int(n)))
    return all_nums[:limit]

def calculate_by_date(target_date, rolling_window, data_cache, kq_db):
    past_dates = []
    for i in range(1, rolling_window + 1):
        past_dates.append(target_date - timedelta(days=i))
    past_dates.reverse()
    
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    for d_obj in past_dates:
        if d_obj not in data_cache or d_obj not in kq_db: continue
        
        df = data_cache[d_obj]
        prev_date = d_obj - timedelta(days=1)
        
        patterns = [
            f"{prev_date.day}/{prev_date.month}", 
            f"{prev_date.day:02d}/{prev_date.month}",
            str(prev_date.day)
        ]
        
        grp_col = None
        for c in sorted(df.columns):
            for p in patterns:
                if p in clean_text(c): 
                    grp_col = c; break
            if grp_col: break
        if not grp_col: continue
        
        kq = kq_db[d_obj]
        for g in groups:
            top80_list = get_group_top_n_stable(df, g, grp_col, limit=80)
            if kq in top80_list:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top80_list.index(kq) + 1)
            else: stats[g]['ranks'].append(999)

    ranked_items = []
    for g in sorted(stats.keys()):
        data = stats[g]
        ranked_items.append((g, (-data['wins'], sum(data['ranks']), sorted(data['ranks']), g)))
    ranked_items.sort(key=lambda x: x[1])
    top6 = [item[0] for item in ranked_items[:6]]
    
    final_result = []
    if target_date in data_cache:
        df_target = data_cache[target_date]
        prev_date_target = target_date - timedelta(days=1)
        patterns = [f"{prev_date_target.day}/{prev_date_target.month}", str(prev_date_target.day)]
        
        grp_col_target = None
        for c in sorted(df_target.columns):
            for p in patterns:
                if p in clean_text(c): 
                    grp_col_target = c; break
            if grp_col_target: break
            
        if grp_col_target:
            limit_map = {top6[0]: 80, top6[1]: 80, top6[2]: 65, top6[3]: 65, top6[4]: 60, top6[5]: 60}
            alliance_1 = [top6[0], top6[5], top6[3]]
            alliance_2 = [top6[1], top6[4], top6[2]]
            def process_alliance(alist, df, col, l_map):
                sets = []
                for g in alist:
                    lst = get_group_top_n_stable(df, g, col, limit=l_map.get(g, 80))
                    sets.append(set(lst)) 
                all_n = []
                for s in sets: all_n.extend(sorted(list(s)))
                return {n for n, c in Counter(all_n).items() if c >= 2}
            set_1 = process_alliance(alliance_1, df_target, grp_col_target, limit_map)
            set_2 = process_alliance(alliance_2, df_target, grp_col_target, limit_map)
            final_result = sorted(list(set_1.intersection(set_2)))
    return top6, final_result

# --- MAIN SCREEN ---
if uploaded_files:
    with st.spinner("⏳ Đang phân tích..."):
        data_cache, kq_db, debug_logs = load_data_with_calendar(uploaded_files)
    
    # Hiển thị kết quả đọc file
    with st.expander("ℹ️ Trạng thái đọc file (Bấm để xem chi tiết)", expanded=True):
        st.write(f"Tổng số ngày đã nhận diện được: **{len(data_cache)}** ngày")
        if len(data_cache) == 0:
            st.error("⚠️ Chưa đọc được ngày nào! Hãy kiểm tra tên File và tên Sheet.")
        else:
            # Liệt kê vài ngày đọc được để user kiểm tra
            sample_dates = [d.strftime('%d/%m/%Y') for d in sorted(list(data_cache.keys()))[:5]]
            st.write(f"Ví dụ các ngày đã đọc: {', '.join(sample_dates)} ...")

    if data_cache:
        tab1, tab2 = st.tabs(["🎯 DỰ ĐOÁN", "🛠️ BACKTEST"])
        
        with tab1:
            st.write("### Chọn ngày:")
            default_date = max(data_cache.keys()) if data_cache else datetime.date.today()
            selected_date = st.date_input("Ngày:", value=default_date)
            
            if st.button("🚀 DỰ ĐOÁN", use_container_width=True):
                top6, result = calculate_by_date(selected_date, ROLLING_WINDOW, data_cache, kq_db)
                st.info(f"🏆 TOP 6: {', '.join(top6)}")
                st.success(f"KẾT QUẢ: {len(result)} số")
                st.code(",".join(result), language="text")
                if selected_date in kq_db:
                    st.write(f"Kết quả thực tế: **{kq_db[selected_date]}**")

        with tab2:
            st.write("### Backtest:")
            c1, c2 = st.columns(2)
            with c1: d_start = st.date_input("Từ:", value=default_date - timedelta(days=5))
            with c2: d_end = st.date_input("Đến:", value=default_date)
            
            if st.button("⚡ CHẠY", use_container_width=True):
                delta = d_end - d_start
                days_list = [d_start + timedelta(days=i) for i in range(delta.days + 1)]
                logs = []
                bar = st.progress(0)
                for i, d in enumerate(days_list):
                    bar.progress((i+1)/len(days_list))
                    if d not in data_cache: continue
                    try:
                        _, res = calculate_by_date(d, ROLLING_WINDOW, data_cache, kq_db)
                        act = kq_db.get(d, "N/A")
                        stt = "WIN ✅" if act in res else "LOSS ❌"
                        if act == "N/A": stt = "Waiting"
                        logs.append({"Ngày": d.strftime('%d/%m'), "KQ": act, "TT": stt})
                    except: pass
                bar.empty()
                if logs:
                    st.dataframe(pd.DataFrame(logs), use_container_width=True)
                else: st.warning("Không có dữ liệu.")
