import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Dự Đoán Xổ Số V13", page_icon="🔥", layout="centered")
st.title("🔥 Dự Đoán & Backtest (V13 - Fix Lỗi Định Dạng)")

# --- 1. TẢI FILE ---
st.info("Bước 1: Tải tất cả file Excel (Tháng 12, Tháng 1...)")
uploaded_files = st.file_uploader("Chọn file:", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)

# --- HÀM XỬ LÝ SỐ LIỆU ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}
RE_FIND_NUMS = re.compile(r'\d{1,2}') 

def get_nums(s):
    if pd.isna(s): return []
    # Lấy tất cả số, lọc số > 100 để tránh lấy nhầm Năm
    raw_nums = re.findall(r'\d+', str(s))
    valid_nums = [n.zfill(2) for n in raw_nums if len(n) <= 2]
    return valid_nums

def get_col_score(col_name):
    # Làm sạch tên cột để check M1...M10
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

# --- [QUAN TRỌNG] HÀM ĐỌC NGÀY THÔNG MINH ---
def try_parse_date_column(col_name, file_month, file_year):
    """
    Hàm này chuyên trị các thể loại ngày tháng 'dị' trong file Excel
    """
    s = str(col_name).strip().upper()
    
    # 1. Định dạng chuẩn DD/MM (VD: 30/11, 1/12)
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        # Xử lý năm: Nếu tháng cột > tháng file (VD file T1, cột 30/11) => Năm trước
        y = file_year
        if m > file_month and (file_month < 6): y -= 1
        elif m < file_month and (file_month > 6): y += 1
        try: return datetime.date(y, m, d)
        except: pass

    # 2. Định dạng YYYY-MM-DD hoặc YYYY-DD-MM (Cái lỗi bạn gặp nằm ở đây)
    # Tìm chuỗi có 4 số đầu (Năm)
    match_iso = re.search(r'(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        
        # Logic phân biệt: 
        # Nếu file là tháng 12, mà thấy 2025-01-12 => p2=01(Ngày), p3=12(Tháng)
        # Nếu file là tháng 1, mà thấy 2026-01-01 => p2=01(Tháng), p3=01(Ngày)
        
        # Ưu tiên 1: Nếu p3 khớp với tháng của file => p2 là Ngày
        if p3 == file_month:
            try: return datetime.date(y, p3, p2) # YYYY-MM-DD (Đảo p2 p3)
            except: pass
        
        # Ưu tiên 2: Chuẩn quốc tế YYYY-MM-DD
        try: return datetime.date(y, p1, p2)
        except: pass
        
    return None

def parse_sheet_date(sheet_name, filename):
    # Lấy năm/tháng từ tên file
    y_match = re.search(r'20\d{2}', filename)
    y_file = int(y_match.group(0)) if y_match else 2025
    
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    if not m_match:
         m_match = re.search(r'(\d+)\.20\d{2}', filename) # Tìm kiểu 12.2025
    m_file = int(m_match.group(1)) if m_match else 1

    # Lấy ngày từ tên sheet
    # Sheet có thể là: "1.12", "1", "01", "1.1.2026"
    
    # Mẹo: Lấy số đầu tiên tìm thấy
    s_clean = re.sub(r'[^0-9]', ' ', sheet_name).strip()
    try:
        parts = [int(x) for x in s_clean.split()]
        if not parts: return None, None, None
        
        d = parts[0]
        # Nếu sheet có dạng 1.12 (2 số), số sau có thể là tháng
        if len(parts) >= 2 and parts[1] == m_file:
            pass 
        elif len(parts) >= 3: # Dạng 1 1 2026
            if parts[2] > 2000: y_file = parts[2]
            if parts[1] <= 12: m_file = parts[1]
            
        return datetime.date(y_file, m_file, d), m_file, y_file
    except: return None, m_file, y_file


@st.cache_data(ttl=600)
def load_data_v13(files):
    data_cache = {} # Key: Date, Value: DataFrame (Cleaned columns)
    kq_db = {}      # Key: Date, Value: String KQ
    
    debug_logs = []
    
    for file in files:
        try:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                try:
                    target_date, f_m, f_y = parse_sheet_date(sheet, file.name)
                    if not target_date: continue

                    # Đọc file
                    # Tìm dòng header chứa "THÀNH VIÊN" hoặc "TV"
                    temp = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    h_idx = 3
                    for i, row in temp.iterrows():
                        row_s = str(row.values).upper()
                        if "THÀNH VIÊN" in row_s or "TV TOP" in row_s:
                            h_idx = i; break
                    
                    df = pd.read_excel(xls, sheet_name=sheet, header=h_idx)
                    
                    # --- BƯỚC QUAN TRỌNG: CHUẨN HÓA TÊN CỘT ---
                    # Đổi tên các cột ngày tháng về dạng Date Object để dễ tìm
                    new_cols = {}
                    for col in df.columns:
                        parsed_d = try_parse_date_column(col, f_m, f_y)
                        if parsed_d:
                            new_cols[col] = parsed_d # Map tên cũ -> Date object
                        else:
                            new_cols[col] = str(col).strip() # Giữ nguyên nếu ko phải ngày
                    
                    # Lưu bảng đã map cột (để thuật toán dùng sau)
                    # Ta sẽ giữ nguyên df gốc nhưng tạo một index phụ để tra cứu
                    
                    data_cache[target_date] = {
                        'df': df,
                        'col_map': new_cols # Dict: { "2025-01-12": date(2025,12,1), "30/11": date(2025,11,30) }
                    }

                    # --- TÌM KẾT QUẢ (KQ) TRONG SHEET NÀY ---
                    # Tìm dòng KQ
                    kq_row = None
                    for idx, row in df.iterrows():
                        if "KQ" in str(row.values[0]).upper():
                            kq_row = row; break
                    
                    if kq_row is not None:
                        # Duyệt qua các cột đã nhận diện là ngày
                        for col_name, col_val in new_cols.items():
                            if isinstance(col_val, datetime.date):
                                try:
                                    val = str(kq_row[col_name])
                                    nums = get_nums(val)
                                    if nums:
                                        kq_db[col_val] = nums[0]
                                        # Log check lỗi
                                        # if col_val.day == 1 and col_val.month == 1:
                                        #     debug_logs.append(f"Tìm thấy KQ 1/1 trong sheet {sheet}: {nums[0]}")
                                except: pass

                except Exception as e: continue
        except: continue
        
    return data_cache, kq_db, debug_logs

def calculate_v13(target_date, rolling_window, data_cache, kq_db):
    past_dates = [target_date - timedelta(days=i) for i in range(1, rolling_window + 1)]
    past_dates.reverse()
    
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    # Cần tìm cột của ngày hôm trước (prev_date) trong file của ngày hôm nay (d_obj)
    # Hoặc file nào đó chứa dữ liệu ngày hôm trước
    
    for d_obj in past_dates:
        if d_obj not in data_cache or d_obj not in kq_db: continue
        
        sheet_data = data_cache[d_obj]
        df = sheet_data['df']
        col_map = sheet_data['col_map']
        
        prev_date = d_obj - timedelta(days=1)
        
        # Tìm cột tương ứng với prev_date
        grp_col = None
        for orig_col, parsed_val in col_map.items():
            if parsed_val == prev_date:
                grp_col = orig_col; break
        
        if not grp_col: continue # Không có cột ngày hôm trước -> Bỏ qua
        
        kq = kq_db[d_obj]
        
        # Logic tính điểm (Giữ nguyên)
        target_group_vals = df[grp_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper()))
        
        col_scores = {}
        valid_cols = []
        for c in df.columns:
            s = get_col_score(c)
            if s > 0: col_scores[c] = s; valid_cols.append(c)

        for g in groups:
            members = df[target_group_vals == g.upper()]
            if members.empty:
                stats[g]['ranks'].append(999); continue

            total_scores = Counter()
            for _, row in members.iterrows():
                for c in valid_cols:
                    nums = get_nums(row[c])
                    score = col_scores[c]
                    for n in nums: total_scores[n] += score
            
            top_nums = [n for n, s in total_scores.most_common()]
            # Sort lại cho chắc: điểm cao -> số nhỏ
            top_nums.sort(key=lambda x: (-total_scores[x], int(x)))
            top80 = top_nums[:80]
            
            if kq in top80:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top80.index(kq) + 1)
            else: stats[g]['ranks'].append(999)

    # Tổng hợp Top 6
    ranked = []
    for g, info in stats.items():
        ranked.append((g, -info['wins'], sum(info['ranks'])))
    ranked.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in ranked[:6]]
    
    # Dự đoán cho Target Date
    final_res = []
    if target_date in data_cache:
        s_data = data_cache[target_date]
        df_t = s_data['df']
        c_map = s_data['col_map']
        prev_d = target_date - timedelta(days=1)
        
        grp_col_t = None
        for k, v in c_map.items():
            if v == prev_d: grp_col_t = k; break
            
        if grp_col_t:
            # Hàm con lấy list số
            def get_set(g_list, limit_dict):
                pool = []
                # Lọc thành viên
                col_vals = df_t[grp_col_t].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper()))
                
                # Tính điểm cột
                c_scores = {c: get_col_score(c) for c in df_t.columns if get_col_score(c) > 0}
                
                for grp in g_list:
                    mems = df_t[col_vals == grp.upper()]
                    scores = Counter()
                    for _, r in mems.iterrows():
                        for c, sc in c_scores.items():
                            for n in get_nums(r[c]): scores[n] += sc
                    
                    sorted_n = [n for n, s in scores.most_common()]
                    sorted_n.sort(key=lambda x: (-scores[x], int(x)))
                    pool.extend(sorted_n[:limit_dict.get(grp, 80)])
                return pool

            limit_map = {top6[0]: 80, top6[1]: 80, top6[2]: 65, top6[3]: 65, top6[4]: 60, top6[5]: 60}
            pool1 = get_set([top6[0], top6[5], top6[3]], limit_map)
            pool2 = get_set([top6[1], top6[4], top6[2]], limit_map)
            
            # Giao nhau của 2 liên minh (số xuất hiện >= 2 lần trong tổng hợp)
            # Logic cũ: set_1.intersection(set_2)
            # Logic chính xác hơn: pool1 và pool2 là danh sách top.
            s1 = set(pool1); s2 = set(pool2)
            final_res = sorted(list(s1.intersection(s2)))
            return top6, final_result, grp_col_t

    return top6, final_res, None

# --- MAIN ---
if uploaded_files:
    with st.spinner("Đang đọc và sửa lỗi ngày tháng..."):
        data_cache, kq_db, logs = load_data_v13(uploaded_files)
    
    with st.expander("Kiểm tra dữ liệu (Bấm để xem)", expanded=True):
        if not data_cache:
            st.error("Không đọc được dữ liệu nào.")
        else:
            st.success(f"Đã đọc {len(data_cache)} ngày.")
            # Kiểm tra nhanh ngày 1/1
            check_date = datetime.date(2026, 1, 1)
            if check_date in kq_db:
                st.write(f"✅ Đã tìm thấy KQ ngày 01/01/2026: **{kq_db[check_date]}**")
            else:
                st.warning("⚠️ Chưa tìm thấy KQ ngày 01/01/2026 (Có thể do lỗi cột 2026-01-01)")

    if data_cache:
        tab1, tab2 = st.tabs(["DỰ ĐOÁN", "BACKTEST"])
        
        with tab1:
            d_max = max(data_cache.keys())
            sel_date = st.date_input("Chọn ngày:", value=d_max)
            if st.button("CHẠY DỰ ĐOÁN", use_container_width=True):
                top6, res, col_used = calculate_v13(sel_date, ROLLING_WINDOW, data_cache, kq_db)
                
                if not col_used:
                    st.error(f"Không tìm thấy cột dữ liệu ngày hôm trước ({sel_date - timedelta(days=1)}) trong sheet này.")
                else:
                    st.info(f"Dữ liệu lấy từ cột: **{col_used}**")
                    st.success(f"TOP 6: {', '.join(top6)}")
                    st.code(",".join(res))
                    if sel_date in kq_db:
                        st.write(f"KQ Thực: **{kq_db[sel_date]}**")

        with tab2:
            c1, c2 = st.columns(2)
            with c1: start = st.date_input("Từ:", value=d_max - timedelta(days=5))
            with c2: end = st.date_input("Đến:", value=d_max)
            if st.button("CHẠY BACKTEST", use_container_width=True):
                delta = (end - start).days
                logs = []
                bar = st.progress(0)
                for i in range(delta + 1):
                    d = start + timedelta(days=i)
                    bar.progress((i+1)/(delta+1))
                    try:
                        _, res, _ = calculate_v13(d, ROLLING_WINDOW, data_cache, kq_db)
                        real = kq_db.get(d, "N/A")
                        stt = "WIN" if real in res else "LOSS"
                        if real == "N/A": stt = "-"
                        logs.append({"Ngày": d.strftime("%d/%m"), "KQ": real, "TT": stt, "Số": len(res)})
                    except: pass
                bar.empty()
                st.dataframe(pd.DataFrame(logs), use_container_width=True)
