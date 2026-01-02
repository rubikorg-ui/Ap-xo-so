import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="App Xổ Số Lịch Vạn Niên", page_icon="📅", layout="centered")
st.title("📅 Dự Đoán Theo Lịch (Chính Xác 100%)")
st.write("---")

# --- 1. KHU VỰC TẢI FILE ---
st.info("Bước 1: Tải các file Excel (Code sẽ tự đọc Tháng/Năm trong tên file)")
uploaded_files = st.file_uploader("Chọn file (Ví dụ: File T12.2025 và T1.2026)", type=['xlsx'], accept_multiple_files=True)

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

# --- HÀM THÔNG MINH: ĐỌC NGÀY THÁNG TỪ TÊN FILE ---
def parse_month_year_from_filename(filename):
    # Tìm năm (4 chữ số, vd 2025, 2026)
    year_match = re.search(r'(20\d{2})', filename)
    year = int(year_match.group(1)) if year_match else None
    
    # Tìm tháng (Chữ THÁNG hoặc T theo sau là số)
    # Ví dụ: THANG 12, THÁNG 1, T12, T01
    name_clean = clean_text(filename)
    month_match = re.search(r'(?:THANG|T)(\d+)', name_clean)
    month = int(month_match.group(1)) if month_match else None
    
    return month, year

@st.cache_data(ttl=600)
def load_data_with_calendar(files):
    # Dữ liệu sẽ lưu theo dạng: key = datetime.date(2025, 12, 1) -> value = dataframe
    data_cache = {}
    kq_db = {} # key = datetime.date -> value = "KQ"
    
    debug_logs = []
    
    for file in files:
        # 1. Tự động nhận diện Tháng/Năm từ tên file
        month_file, year_file = parse_month_year_from_filename(file.name)
        
        if not month_file or not year_file:
            st.warning(f"⚠️ Không nhận diện được Tháng/Năm trong tên file: {file.name}. (Hãy đặt tên file kiểu 'THANG 12 2025')")
            continue
            
        debug_logs.append(f"Đọc file: {file.name} (Hiểu là: T{month_file}/{year_file})")
        
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                try:
                    # Lấy ngày từ tên Sheet (1, 2, ..., 31)
                    match = re.search(r'(\d+)', sheet_name)
                    if not match: continue
                    day = int(match.group(1))
                    
                    # Tạo đối tượng ngày chuẩn xác
                    try:
                        current_date = datetime.date(year_file, month_file, day)
                    except ValueError: continue # Bỏ qua ngày không hợp lệ (ví dụ 31/2)

                    # Đọc dữ liệu
                    temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=10)
                    h = get_header_row_index(temp)
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h)
                    df.columns = [clean_text(c) for c in df.columns]
                    
                    # Lưu vào cache với Key là NGÀY CỤ THỂ
                    data_cache[current_date] = df
                    
                    # Tìm KQ
                    mask_kq = df.iloc[:, 0].astype(str).apply(clean_text).str.contains("KQ", na=False)
                    if mask_kq.any():
                        kq_row = df[mask_kq].iloc[0]
                        # Logic tìm cột chứa ngày hiện tại trong bảng
                        # Cột thường có dạng: "1/12", "01/12"
                        target_col_patterns = [
                            f"{day}/{month_file}", 
                            f"{day:02d}/{month_file}", 
                            f"{day}/{month_file:02d}", 
                            str(day)
                        ]
                        
                        found_kq = None
                        for c in df.columns:
                            # Tìm cột khớp với ngày tháng
                            for p in target_col_patterns:
                                if p in c: 
                                    try: 
                                        val = str(kq_row[c])
                                        nums = get_nums(val)
                                        if nums: found_kq = nums[0]
                                    except: pass
                            if found_kq: break
                        
                        if found_kq:
                            kq_db[current_date] = found_kq
                            
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
    # Lấy danh sách các ngày quá khứ cần thiết
    # Ví dụ: Target = 2/1/2026, Window = 2 => Cần ngày 1/1/2026 và 31/12/2025
    past_dates = []
    for i in range(1, rolling_window + 1):
        past_dates.append(target_date - timedelta(days=i))
    
    # Đảo ngược để tính từ xa đến gần
    past_dates.reverse()
    
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    for d_obj in past_dates:
        if d_obj not in data_cache or d_obj not in kq_db: continue
        
        df = data_cache[d_obj]
        prev_date = d_obj - timedelta(days=1)
        
        # Tạo các pattern để tìm cột của ngày hôm trước trong file ngày hôm nay
        # Ví dụ: Đang xét ngày 1/1/2026, cần tìm cột kết quả của ngày 31/12
        prev_day = prev_date.day
        prev_month = prev_date.month
        
        patterns = [
            f"{prev_day}/{prev_month}",
            f"{prev_day:02d}/{prev_month}",
            f"{prev_day}/{prev_month:02d}",
            f"{prev_day:02d}/{prev_month:02d}",
            str(prev_day)
        ]
        
        grp_col = None
        for c in sorted(df.columns):
            for p in patterns:
                # Tìm cột chứa chuỗi ngày tháng (Clean text để so sánh chính xác)
                if p in clean_text(c): 
                    grp_col = c
                    break
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
    
    if not data_cache:
        st.error("❌ Không đọc được dữ liệu! Hãy chắc chắn tên file có chứa Tháng và Năm (VD: T12 2025)")
    else:
        # Hiển thị log để user biết máy đã hiểu đúng
        with st.expander("ℹ️ Xem chi tiết các file đã nhận diện"):
            for log in debug_logs: st.text(log)
            st.text(f"Tổng số ngày dữ liệu: {len(data_cache)}")
        
        tab1, tab2 = st.tabs(["🎯 DỰ ĐOÁN", "🛠️ BACKTEST"])
        
        # --- TAB 1: DỰ ĐOÁN ---
        with tab1:
            st.write("### Chọn ngày trên lịch:")
            
            # Tự động chọn ngày hôm nay hoặc ngày cuối cùng có dữ liệu
            default_date = max(data_cache.keys()) if data_cache else datetime.date.today()
            
            selected_date = st.date_input("Ngày dự đoán:", value=default_date)
            
            if st.button("🚀 XEM KẾT QUẢ", use_container_width=True):
                top6, result = calculate_by_date(selected_date, ROLLING_WINDOW, data_cache, kq_db)
                
                st.info(f"🏆 **TOP 6 GROUP:** {', '.join(top6)}")
                st.success(f"**KẾT QUẢ DỰ ĐOÁN NGÀY {selected_date.strftime('%d/%m/%Y')} ({len(result)} số):**")
                st.code(",".join(result), language="text")
                
                # Check Win/Loss
                if selected_date in kq_db:
                    real = kq_db[selected_date]
                    if real in result: st.success(f"🎉 TRÚNG RỒI! Về: {real}")
                    else: st.error(f"❌ TRƯỢT! Về: {real}")
                else:
                    st.warning("⚠️ Ngày này chưa có kết quả để đối chiếu.")

        # --- TAB 2: BACKTEST ---
        with tab2:
            st.write("### Kiểm tra lịch sử:")
            c1, c2 = st.columns(2)
            with c1: d_start = st.date_input("Từ ngày:", value=default_date - timedelta(days=10))
            with c2: d_end = st.date_input("Đến ngày:", value=default_date)
            
            if st.button("⚡ CHẠY KIỂM CHỨNG", use_container_width=True):
                if d_start > d_end:
                    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc")
                else:
                    # Tạo danh sách các ngày liên tục
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
                            logs.append({
                                "Ngày": d.strftime('%d/%m/%Y'),
                                "KQ": act, "Trạng thái": stt, 
                                "Số lượng": len(res)
                            })
                        except: pass
                    bar.empty()
                    
                    if logs:
                        df_res = pd.DataFrame(logs)
                        wins = df_res[df_res["Trạng thái"] == "WIN ✅"].shape[0]
                        total = df_res[df_res["KQ"] != "N/A"].shape[0]
                        if total > 0: st.metric("Tỷ lệ thắng", f"{wins}/{total} ({round(wins/total*100)}%)")
                        st.dataframe(df_res, use_container_width=True)
                    else: st.warning("Không có dữ liệu trong khoảng này.")

else:
    st.info("👈 Hãy tải file Excel lên (đặt tên file có Tháng và Năm, VD: 'Data T12 2025.xlsx')")
