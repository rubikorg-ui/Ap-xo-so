import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="App Xổ Số Lịch Vạn Niên", page_icon="📅", layout="centered")
st.title("📅 Dự Đoán Theo Lịch (V10)")
st.write("---")

# --- 1. KHU VỰC TẢI FILE ---
st.info("Bước 1: Tải các file Excel")
uploaded_files = st.file_uploader("Chọn file (Ví dụ: THÁNG 12.2025, THÁNG 1.2026)", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
    st.caption("Ví dụ: Chọn 10 ngày thì khi dự đoán ngày 2/1, máy sẽ xem lại từ 23/12 đến 1/1.")

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

# --- [FIX] HÀM ĐỌC TÊN FILE THÔNG MINH HƠN ---
def parse_month_year_from_filename(filename):
    # 1. Tìm Năm: Tìm số có 4 chữ số (2025, 2026...)
    year_match = re.search(r'(20\d{2})', filename)
    year = int(year_match.group(1)) if year_match else None
    
    # 2. Tìm Tháng: Tìm số nằm sau chữ "THÁNG", "THANG", "T", hoặc "M"
    # Regex này chấp nhận: "THÁNG 1", "THANG.1", "T12", "T 05" v.v.
    # re.IGNORECASE giúp không phân biệt hoa thường
    month_match = re.search(r'(?:THANG|THÁNG|TH|T|M)[^0-9]*(\d+)', filename, re.IGNORECASE)
    
    if month_match:
        month = int(month_match.group(1))
    else:
        # Nếu không thấy chữ "THÁNG", thử tìm dạng "1.2026" hoặc "12-2025"
        # Tìm số đứng ngay trước Năm
        alt_match = re.search(r'(\d+)[\.\-_/]+' + str(year), filename)
        month = int(alt_match.group(1)) if alt_match else None

    return month, year

@st.cache_data(ttl=600)
def load_data_with_calendar(files):
    data_cache = {}
    kq_db = {} 
    debug_logs = []
    
    for file in files:
        # Sử dụng hàm đọc tên file mới đã sửa lỗi
        month_file, year_file = parse_month_year_from_filename(file.name)
        
        # Nếu vẫn không đọc được, thử đoán: 
        # Nếu chỉ có 1 file và không đọc được, có thể gán tạm thời gian hiện tại (nhưng rủi ro).
        # Ở đây ta sẽ báo lỗi cụ thể để user biết.
        if not month_file or not year_file:
            debug_logs.append(f"❌ LỖI: Không đọc được ngày tháng file '{file.name}'")
            continue
            
        debug_logs.append(f"✅ Đã nhận diện file '{file.name}' là: Tháng {month_file} / Năm {year_file}")
        
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                try:
                    match = re.search(r'(\d+)', sheet_name)
                    if not match: continue
                    day = int(match.group(1))
                    
                    try: current_date = datetime.date(year_file, month_file, day)
                    except ValueError: continue 

                    temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=10)
                    h = get_header_row_index(temp)
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h)
                    df.columns = [clean_text(c) for c in df.columns]
                    
                    data_cache[current_date] = df
                    
                    mask_kq = df.iloc[:, 0].astype(str).apply(clean_text).str.contains("KQ", na=False)
                    if mask_kq.any():
                        kq_row = df[mask_kq].iloc[0]
                        target_col_patterns = [
                            f"{day}/{month_file}", f"{day:02d}/{month_file}", 
                            f"{day}/{month_file:02d}", str(day)
                        ]
                        found_kq = None
                        for c in df.columns:
                            for p in target_col_patterns:
                                if p in c: 
                                    try: 
                                        val = str(kq_row[c])
                                        nums = get_nums(val)
                                        if nums: found_kq = nums[0]
                                    except: pass
                            if found_kq: break
                        if found_kq: kq_db[current_date] = found_kq
                except: continue
        except: continue
        
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
        prev_day = prev_date.day
        prev_month = prev_date.month
        
        patterns = [
            f"{prev_day}/{prev_month}", f"{prev_day:02d}/{prev_month}",
            f"{prev_day}/{prev_month:02d}", f"{prev_day:02d}/{prev_month:02d}",
            str(prev_day)
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
        p_day = prev_date_target.day
        p_month = prev_date_target.month
        
        patterns = [f"{p_day}/{p_month}", f"{p_day:02d}/{p_month}", str(p_day)]
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
    with st.spinner("⏳ Đang phân tích file..."):
        data_cache, kq_db, debug_logs = load_data_with_calendar(uploaded_files)
    
    # Hiển thị log trạng thái file ngay đầu trang để kiểm tra
    with st.expander("📝 Bấm vào đây để xem máy có đọc đúng tên file không?", expanded=True):
        if not debug_logs:
            st.write("Chưa đọc được file nào.")
        else:
            for log in debug_logs:
                if "❌" in log: st.error(log)
                else: st.success(log)

    if data_cache:
        tab1, tab2 = st.tabs(["🎯 DỰ ĐOÁN", "🛠️ BACKTEST"])
        
        with tab1:
            st.write("### Chọn ngày trên lịch:")
            default_date = max(data_cache.keys()) if data_cache else datetime.date.today()
            selected_date = st.date_input("Ngày dự đoán:", value=default_date)
            
            if st.button("🚀 XEM KẾT QUẢ", use_container_width=True):
                top6, result = calculate_by_date(selected_date, ROLLING_WINDOW, data_cache, kq_db)
                st.info(f"🏆 **TOP 6:** {', '.join(top6)}")
                st.success(f"**KẾT QUẢ ({len(result)} số):**")
                st.code(",".join(result), language="text")
                if selected_date in kq_db:
                    real = kq_db[selected_date]
                    if real in result: st.success(f"🎉 TRÚNG: {real}")
                    else: st.error(f"❌ TRƯỢT: {real}")
                else: st.warning("⚠️ Chưa có KQ.")

        with tab2:
            st.write("### Backtest lịch sử:")
            c1, c2 = st.columns(2)
            with c1: d_start = st.date_input("Từ ngày:", value=default_date - timedelta(days=5))
            with c2: d_end = st.date_input("Đến ngày:", value=default_date)
            
            if st.button("⚡ CHẠY KIỂM CHỨNG", use_container_width=True):
                if d_start > d_end: st.error("Ngày sai!")
                else:
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
                            logs.append({"Ngày": d.strftime('%d/%m/%Y'), "KQ": act, "Trạng thái": stt, "Số lượng": len(res)})
                        except: pass
                    bar.empty()
                    if logs:
                        df_res = pd.DataFrame(logs)
                        wins = df_res[df_res["Trạng thái"] == "WIN ✅"].shape[0]
                        total = df_res[df_res["KQ"] != "N/A"].shape[0]
                        if total > 0: st.metric("Tỷ lệ thắng", f"{wins}/{total} ({round(wins/total*100)}%)")
                        st.dataframe(df_res, use_container_width=True)
                    else: st.warning("Không có dữ liệu.")
else:
    st.info("👈 Hãy tải file Excel lên.")
