import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V16 (Final)", page_icon="🎯", layout="centered")
st.title("🎯 Dự Đoán & Backtest (V16)")

# --- 1. TẢI FILE ---
st.info("Bước 1: Tải các file Excel (T12.2025, T1.2026...)")
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

def get_nums(s):
    if pd.isna(s): return []
    # Chỉ lấy số có 2 chữ số (00-99)
    raw_nums = re.findall(r'\d+', str(s))
    return [n.zfill(2) for n in raw_nums if len(n) == 2]

def get_col_score(col_name):
    # Tìm cột M0..M10
    # Cần xử lý trường hợp "M 1 0" (có cách)
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

# --- HÀM PARSE NGÀY TỪ TÊN FILE & SHEET (QUAN TRỌNG) ---
def get_file_month_year(filename):
    y_match = re.search(r'20\d{2}', filename)
    y = int(y_match.group(0)) if y_match else 2025
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m = int(m_match.group(1)) if m_match else 1
    return m, y

def get_date_from_sheet_name(sheet_name, f_m, f_y):
    # Sheet có thể là "2", "02", "1.12", "1.1.2026"
    # Ưu tiên lấy số đầu tiên làm Ngày
    s_clean = re.sub(r'[^0-9]', ' ', sheet_name).strip()
    try:
        parts = [int(x) for x in s_clean.split()]
        if not parts: return None
        d = parts[0]
        
        # Xử lý trường hợp sheet "1.1.2026"
        m = f_m
        y = f_y
        if len(parts) >= 3 and parts[2] > 2000: y = parts[2]; m = parts[1]
        
        return datetime.date(y, m, d)
    except: return None

# --- HÀM PARSE CỘT NGÀY (XỬ LÝ LỖI 2025-01-12) ---
def parse_col_date(col_str, file_month, file_year):
    s = str(col_str).strip().upper()
    
    # 1. Dạng YYYY-MM-DD (Bị lỗi đảo)
    match_iso = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        # Nếu p1 != file_month nhưng p2 == file_month -> ĐẢO
        if p1 != file_month and p2 == file_month:
            try: return datetime.date(y, p2, p1)
            except: pass
        # Mặc định
        try: return datetime.date(y, p1, p2)
        except: pass

    # 2. Dạng DD/MM
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        curr_y = file_year
        # Xử lý qua năm
        if m == 12 and file_month == 1: curr_y -= 1
        elif m == 1 and file_month == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: pass
    return None

@st.cache_data(ttl=600)
def load_data_v16(files):
    data_cache = {} # Key: Date (Ngày dự đoán), Value: {df, col_map}
    kq_db = {}      # Key: Date (Ngày có KQ), Value: KQ String
    logs = []
    
    for file in files:
        f_m, f_y = get_file_month_year(file.name)
        logs.append(f"📂 File: {file.name} (Tháng {f_m}/{f_y})")
        
        try:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                try:
                    # 1. Xác định Ngày Dự Đoán của Sheet này
                    target_date = get_date_from_sheet_name(sheet, f_m, f_y)
                    if not target_date: continue

                    # 2. Đọc Sheet
                    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    h_row = 3
                    for idx, row in preview.iterrows():
                        r_s = str(row.values).upper()
                        if "TV TOP" in r_s or "THÀNH VIÊN" in r_s:
                            h_row = idx; break
                    
                    df = pd.read_excel(xls, sheet_name=sheet, header=h_row)
                    
                    # 3. Map các cột Ngày trong Sheet (Cột Lịch Sử)
                    col_map = {} # Key: Date, Value: Col Name
                    for col in df.columns:
                        d_obj = parse_col_date(col, f_m, f_y)
                        if d_obj:
                            col_map[d_obj] = col # Lưu ngược lại để tra cứu: Ngày -> Tên Cột
                    
                    # 4. Tìm KQ trong Sheet (để xây dựng database KQ)
                    kq_row = None
                    for idx, row in df.iterrows():
                        if str(row.values[0]).strip().upper() == "KQ":
                            kq_row = row; break
                    
                    if kq_row is not None:
                        for d_val, col_name in col_map.items():
                            val = str(kq_row[col_name])
                            nums = get_nums(val)
                            if nums: kq_db[d_val] = nums[0]

                    # 5. Lưu vào Cache: KEY LÀ NGÀY CỦA SHEET (target_date)
                    data_cache[target_date] = {'df': df, 'date_to_col': col_map}

                except Exception as e: continue
        except: continue
    
    return data_cache, kq_db, logs

def calculate_v16(target_date, rolling_window, data_cache, kq_db):
    # Lấy dữ liệu của chính ngày target_date
    if target_date not in data_cache:
        return [], [], None, "Không tìm thấy Sheet dữ liệu cho ngày này."
    
    sheet_data = data_cache[target_date]
    df = sheet_data['df']
    date_to_col = sheet_data['date_to_col']
    
    # 1. Tìm cột dữ liệu ngày hôm trước (prev_date) để phân nhóm
    prev_date = target_date - timedelta(days=1)
    col_used = date_to_col.get(prev_date)
    
    if not col_used:
        # Thử tìm lùi thêm 1 ngày (phòng trường hợp nghỉ tết/lễ)
        # prev_date = target_date - timedelta(days=2)
        # col_used = date_to_col.get(prev_date)
        return [], [], None, f"Trong Sheet '{target_date.strftime('%d/%m')}' không tìm thấy cột dữ liệu của ngày hôm trước ({prev_date.strftime('%d/%m')})."

    # 2. Xác định các cột điểm (M0..M10)
    score_cols = {}
    for c in df.columns:
        s = get_col_score(c)
        if s > 0: score_cols[c] = s

    # 3. Tính Top 6 Group dựa trên quá khứ
    past_dates = [target_date - timedelta(days=i) for i in range(1, rolling_window + 1)]
    past_dates.reverse()
    
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    for d in past_dates:
        if d not in kq_db or d not in date_to_col: continue
        
        # Cột dữ liệu quá khứ
        hist_col = date_to_col[d]
        kq = kq_db[d]
        
        for g in groups:
            # Lọc thành viên thuộc nhóm g vào ngày d
            mask = df[hist_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
            members = df[mask]
            
            if members.empty:
                stats[g]['ranks'].append(999); continue
            
            # Tính tổng điểm các số do nhóm này dự đoán
            total_scores = Counter()
            for _, row in members.iterrows():
                for sc_col, score in score_cols.items():
                    for n in get_nums(row[sc_col]): total_scores[n] += score
            
            # Lấy Top số của nhóm
            # [FIX SỐ LƯỢNG LỚN]: Chỉ lấy Top 40 số mạnh nhất của nhóm để so sánh
            ranked_nums = [n for n, s in total_scores.most_common()]
            ranked_nums.sort(key=lambda x: (-total_scores[x], int(x)))
            
            # Siết chặt limit khi check lịch sử
            top_check = ranked_nums[:60] 
            
            if kq in top_check:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top_check.index(kq) + 1)
            else: stats[g]['ranks'].append(999)

    # Xếp hạng Group
    final_ranks = []
    for g, info in stats.items():
        final_ranks.append((g, -info['wins'], sum(info['ranks'])))
    final_ranks.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_ranks[:6]]
    
    # 4. Dự đoán (Intersection)
    def get_pool(grp_list):
        pool = []
        for g in grp_list:
            # Lọc thành viên thuộc nhóm g vào ngày hôm qua (col_used)
            mask = df[col_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
            members = df[mask]
            
            scores = Counter()
            for _, row in members.iterrows():
                for sc_col, score in score_cols.items():
                    for n in get_nums(row[sc_col]): scores[n] += score
            
            r_n = [n for n, s in scores.most_common()]
            r_n.sort(key=lambda x: (-scores[x], int(x)))
            
            # [FIX SỐ LƯỢNG LỚN]: Siết limit tùy theo độ mạnh của Group
            limit = 60 # Mặc định lấy 60 số
            if g == top6[0] or g == top6[1]: limit = 70 # Top 1,2 lấy nhiều hơn chút
            
            pool.extend(r_n[:limit])
        return pool

    # Liên minh 1: Top 1, 6, 4
    s1 = set(get_pool([top6[0], top6[5], top6[3]]))
    # Liên minh 2: Top 2, 5, 3
    s2 = set(get_pool([top6[1], top6[4], top6[2]]))
    
    # Giao nhau
    final_res = sorted(list(s1.intersection(s2)))
    
    return top6, final_res, col_used, None

# --- MAIN ---
if uploaded_files:
    with st.spinner("Đang đọc dữ liệu..."):
        data_cache, kq_db, logs = load_data_v16(uploaded_files)
    
    with st.expander("LOGS ĐỌC FILE (Bấm để xem)", expanded=True):
        if not data_cache:
            st.error("❌ Không đọc được dữ liệu nào.")
            for l in logs: st.text(l)
        else:
            st.success(f"✅ Đã đọc {len(data_cache)} sheet dự đoán.")
            # Show list ngày có thể dự đoán
            avail_dates = sorted([d.strftime('%d/%m') for d in data_cache.keys()])
            st.write(f"Có thể dự đoán các ngày: {', '.join(avail_dates)}")
            
            # Check KQ
            st.write(f"Đã tìm thấy {len(kq_db)} kết quả lịch sử.")

    if data_cache:
        st.write("---")
        tab1, tab2 = st.tabs(["🔮 DỰ ĐOÁN", "🏅 BACKTEST"])
        
        with tab1:
            # Mặc định chọn ngày 2/1 nếu có
            def_date = datetime.date(2026, 1, 2)
            if def_date not in data_cache:
                def_date = max(data_cache.keys())
            
            target = st.date_input("Chọn ngày:", value=def_date)
            
            if st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True):
                top6, res, col, err = calculate_v16(target, ROLLING_WINDOW, data_cache, kq_
