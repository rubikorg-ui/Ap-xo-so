import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V17 (Logic Gốc)", page_icon="💎", layout="centered")
st.title("💎 V17: Khôi Phục Logic Gốc (Số Tinh Gọn)")

# --- 1. TẢI FILE ---
uploaded_files = st.file_uploader("Tải file Excel (T12, T1...):", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH BÊN ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)

# --- HÀM XỬ LÝ (CORE) ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}

def get_nums(s):
    if pd.isna(s): return []
    # Chỉ lấy số 2 chữ số chuẩn
    raw_nums = re.findall(r'\d+', str(s))
    return [n.zfill(2) for n in raw_nums if len(n) == 2]

def get_col_score(col_name):
    # Làm sạch tên cột để tìm M0..M10
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            # Tránh nhầm M1 với M10
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

# --- XỬ LÝ NGÀY THÁNG (GIỮ NGUYÊN PHẦN ĐÃ FIX) ---
def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper()
    # 1. Dạng YYYY-MM-DD (Excel hay tự đổi)
    match_iso = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        # Logic fix lỗi đảo: Nếu p1!=tháng file mà p2==tháng file -> Đảo
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)

    # 2. Dạng DD/MM
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        curr_y = f_y
        # Xử lý qua năm (File T1 có cột 31/12)
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        return datetime.date(curr_y, m, d)
    return None

def get_file_meta(filename):
    y_match = re.search(r'20\d{2}', filename)
    y = int(y_match.group(0)) if y_match else 2025
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m = int(m_match.group(1)) if m_match else 1
    return m, y

def get_sheet_date(sheet_name, f_m, f_y):
    # Lấy số đầu tiên trong tên sheet làm ngày
    s_clean = re.sub(r'[^0-9]', ' ', sheet_name).strip()
    try:
        parts = [int(x) for x in s_clean.split()]
        if not parts: return None
        d = parts[0]
        m, y = f_m, f_y
        # Case sheet "1.1.2026"
        if len(parts) >= 3 and parts[2] > 2000: y = parts[2]; m = parts[1]
        return datetime.date(y, m, d)
    except: return None

@st.cache_data(ttl=600)
def load_data_v17(files):
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

                    # Tìm header
                    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    h_row = 3
                    for idx, row in preview.iterrows():
                        if "TV TOP" in str(row.values).upper() or "THÀNH VIÊN" in str(row.values).upper():
                            h_row = idx; break
                    
                    df = pd.read_excel(xls, sheet_name=sheet, header=h_row)
                    
                    # Map cột
                    col_map = {}
                    for col in df.columns:
                        d_obj = parse_date_smart(col, f_m, f_y)
                        if d_obj: col_map[d_obj] = col # Date -> Col Name

                    # Tìm KQ
                    kq_row = None
                    for idx, row in df.iterrows():
                        if str(row.values[0]).strip().upper() == "KQ": kq_row = row; break
                    
                    if kq_row is not None:
                        for d_val, c_name in col_map.items():
                            val = str(kq_row[c_name])
                            nums = get_nums(val)
                            if nums: kq_db[d_val] = nums[0]

                    cache[t_date] = {'df': df, 'map': col_map}
                except: continue
        except: continue
    return cache, kq_db

# --- PHẦN TÍNH TOÁN (LOGIC GỐC ĐƯỢC KHÔI PHỤC) ---
def calculate_v17(target_date, rolling_window, cache, kq_db):
    if target_date not in cache:
        return [], [], None, "Không tìm thấy Sheet dữ liệu."

    data = cache[target_date]
    df = data['df']
    date_to_col = data['map']
    
    # 1. Tìm dữ liệu ngày hôm trước (Quan trọng: Để phân nhóm)
    prev_date = target_date - timedelta(days=1)
    col_used = date_to_col.get(prev_date)
    
    if not col_used:
        return [], [], None, f"Trong sheet ngày {target_date}, không tìm thấy cột dữ liệu ngày {prev_date}."

    # 2. Identify Score Columns
    score_cols = {}
    for c in df.columns:
        s = get_col_score(c)
        if s > 0: score_cols[c] = s

    # 3. Backtest để tìm Top Group
    past_dates = [target_date - timedelta(days=i) for i in range(1, rolling_window + 1)]
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    for d in past_dates:
        if d not in kq_db or d not in date_to_col: continue
        hist_col = date_to_col[d]
        kq = kq_db[d]
        
        for g in groups:
            # Lọc thành viên
            mask = df[hist_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
            mems = df[mask]
            if mems.empty: stats[g]['ranks'].append(999); continue
            
            # Tính điểm
            scores = Counter()
            for _, r in mems.iterrows():
                for sc, pts in score_cols.items():
                    for n in get_nums(r[sc]): scores[n] += pts
            
            # Lấy Top 80 để check rank
            rnk = [n for n, s in scores.most_common()]
            rnk.sort(key=lambda x: (-scores[x], int(x)))
            top_check = rnk[:80]
            
            if kq in top_check:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top_check.index(kq) + 1)
            else: stats[g]['ranks'].append(999)

    # Xếp hạng Group
    final = []
    for g, inf in stats.items():
        final.append((g, -inf['wins'], sum(inf['ranks'])))
    final.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final[:6]]
    
    # 4. DỰ ĐOÁN (QUAY VỀ LOGIC CHẶT CHẼ)
    # Alliance 1: Top 1, 6, 3 (Ví dụ: 0x, 5x, 2x)
    # Alliance 2: Top 2, 5, 4 (Ví dụ: 1x, 4x, 3x)
    
    def get_alliance_numbers(grp_list):
        pool = []
        for g in grp_list:
            mask = df[col_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
            mems = df[mask]
            
            scores = Counter()
            for _, r in mems.iterrows():
                for sc, pts in score_cols.items():
                    for n in get_nums(r[sc]): scores[n] += pts
            
            rnk = [n for n, s in scores.most_common()]
            rnk.sort(key=lambda x: (-scores[x], int(x)))
            
            # GIỚI HẠN SỐ LƯỢNG (Limit cũ)
            limit = 80 # Mặc định
            # Có thể chỉnh: Top 1,2 lấy 80. Top 3,4 lấy 65. Top 5,6 lấy 60
            if g in top6[:2]: limit = 80
            elif g in top6[2:4]: limit = 65
            else: limit = 60
            
            pool.extend(rnk[:limit])
        return pool

    alliance_1 = [top6[0], top6[5], top6[3]]
    alliance_2 = [top6[1], top6[4], top6[2]]
    
    nums_1 = get_alliance_numbers(alliance_1)
    nums_2 = get_alliance_numbers(alliance_2)
    
    # GIAO NHAU (Intersection) -> Chỉ lấy số xuất hiện ở CẢ 2 LIÊN MINH
    # Đây là chìa khóa để giảm số lượng số
    final_set = set(nums_1).intersection(set(nums_2))
    res = sorted(list(final_set))
    
    return top6, res, col_used, None

# --- UI ---
if uploaded_files:
    data_cache, kq_db = load_data_v17(uploaded_files)
    st.success(f"Đã đọc {len(data_cache)} ngày. Tìm thấy {len(kq_db)} KQ lịch sử.")
    
    # KIỂM TRA NGÀY 1/1 (Để chắc chắn dự đoán được ngày 2/1)
    d_check = datetime.date(2026, 1, 1)
    if d_check in kq_db:
        st.caption(f"✅ Đã có dữ liệu KQ ngày 1/1: {kq_db[d_check]}")
    else:
        st.caption("⚠️ Chưa thấy KQ ngày 1/1 (Có thể ảnh hưởng Backtest ngày 2/1)")

    tab1, tab2 = st.tabs(["DỰ ĐOÁN", "BACKTEST"])
    
    with tab1:
        # Chọn ngày: Mặc định là ngày tiếp theo
        last_d = max(data_cache.keys()) if data_cache else datetime.date.today()
        target = st.date_input("Ngày:", value=last_d + timedelta(days=1))
        
        if st.button("PHÂN TÍCH"):
            top6, res, col, err = calculate_v17(target, ROLLING_WINDOW, data_cache, kq_db)
            if err:
                st.error(err)
            else:
                st.info(f"Dữ liệu phân nhóm lấy từ cột: {col}")
                st.success(f"TOP 6: {', '.join(top6)}")
                st.text_area(f"KẾT QUẢ ({len(res)} số):", ",".join(res))
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1: start = st.date_input("Từ:", value=last_d - timedelta(days=5))
        with c2: end = st.date_input("Đến:", value=last_d)
        if st.button("CHẠY BACKTEST"):
            logs = []
            delta = (end - start).days
            bar = st.progress(0)
            for i in range(delta + 1):
                d = start + timedelta(days=i)
                bar.progress((i+1)/(delta+1))
                try:
                    _, res, _, err = calculate_v17(d, ROLLING_WINDOW, data_cache, kq_db)
                    if err: continue
                    real = kq_db.get(d, "N/A")
                    stt = "WIN" if real in res else "LOSS"
                    if real == "N/A": stt = "-"
                    logs.append({"Ngày": d.strftime("%d/%m"), "KQ": real, "TT": stt, "Số": len(res)})
                except: pass
            bar.empty()
            st.dataframe(pd.DataFrame(logs))
