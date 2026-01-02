import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V15 (Fix Hiển Thị)", page_icon="🚨", layout="centered")
st.title("🚨 Dự Đoán & Backtest (V15)")

# --- 1. TẢI FILE ---
st.info("Bước 1: Tải các file Excel")
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
    raw_nums = re.findall(r'\d+', str(s))
    # Chỉ lấy số có 1-2 chữ số
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

def get_col_score(col_name):
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

# --- HÀM XỬ LÝ NGÀY THÔNG MINH ---
def parse_date_magic(col_str, file_month, file_year):
    s = str(col_str).strip().upper()
    
    # Case 1: 30/11, 1/12 (Dạng thường)
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        y = file_year
        # Xử lý giao thừa (File T1 có cột 31/12)
        if m == 12 and file_month == 1: y -= 1
        elif m == 1 and file_month == 12: y += 1
        try: return datetime.date(y, m, d)
        except: pass

    # Case 2: 2025-01-12 (Lỗi đảo ngày tháng trong file của bạn)
    # File tháng 12 mà lại hiện 2025-01-12 -> Thực ra là ngày 01/12
    match_iso = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        
        # Logic sửa lỗi:
        # Nếu p1 (vị trí tháng) != file_month, mà p2 (vị trí ngày) == file_month
        # => ĐẢO NGƯỢC
        if p1 != file_month and p2 == file_month:
            try: return datetime.date(y, p2, p1) # p2 là Tháng, p1 là Ngày
            except: pass
            
        # Nếu thuận:
        if p1 == file_month:
            try: return datetime.date(y, p1, p2)
            except: pass
            
        # Nếu cả 2 không khớp, thử ưu tiên p1 là tháng
        try: return datetime.date(y, p1, p2)
        except: pass
        
    return None

def get_file_info(filename):
    y_match = re.search(r'20\d{2}', filename)
    y = int(y_match.group(0)) if y_match else 2025
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m = int(m_match.group(1)) if m_match else 1
    return m, y

@st.cache_data(ttl=600)
def load_data_v15(files):
    data_cache = {}
    kq_db = {}
    logs = []
    
    for file in files:
        f_m, f_y = get_file_info(file.name)
        logs.append(f"📂 Đọc file: {file.name} (Hiểu là T{f_m}/{f_y})")
        
        try:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                try:
                    # Tìm dòng Header (chứa TV TOP)
                    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    header_row = 3
                    for idx, row in preview.iterrows():
                        r_s = str(row.values).upper()
                        if "TV TOP" in r_s or "THÀNH VIÊN" in r_s:
                            header_row = idx; break
                    
                    df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
                    
                    # Map Cột Ngày
                    col_map = {}
                    found_dates = []
                    
                    for col in df.columns:
                        d_obj = parse_date_magic(col, f_m, f_y)
                        if d_obj:
                            col_map[col] = d_obj
                            found_dates.append(d_obj)
                            
                    # Tìm KQ
                    kq_row = None
                    for idx, row in df.iterrows():
                        if str(row.values[0]).strip().upper() == "KQ":
                            kq_row = row; break
                    
                    if kq_row is not None:
                        for col_name, d_val in col_map.items():
                            val = str(kq_row[col_name])
                            nums = get_nums(val)
                            if nums: kq_db[d_val] = nums[0]
                    
                    if found_dates:
                        # Lưu cache (lấy ngày lớn nhất trong sheet làm đại diện)
                        max_d = max(found_dates)
                        data_cache[max_d] = {'df': df, 'map': col_map}
                        
                except: continue
        except: continue
        
    return data_cache, kq_db, logs

def calculate_v15(target_date, rolling_window, data_cache, kq_db):
    # Tìm dữ liệu phù hợp (Sheet chứa ngày target hoặc tương lai gần nhất)
    sel_data = None
    if target_date in data_cache: sel_data = data_cache[target_date]
    else:
        futures = [d for d in data_cache.keys() if d >= target_date]
        if futures: sel_data = data_cache[min(futures)]
        
    if not sel_data: return [], [], None
    
    df = sel_data['df']
    col_map = sel_data['map']
    date_to_col = {v: k for k, v in col_map.items()}
    
    # 1. Tìm Top 6 Group
    past_dates = [target_date - timedelta(days=i) for i in range(1, rolling_window + 1)]
    past_dates.reverse()
    
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    # Xác định các cột điểm
    valid_cols_score = {}
    for c in df.columns:
        s = get_col_score(c)
        if s > 0: valid_cols_score[c] = s

    for d in past_dates:
        if d not in kq_db or d not in date_to_col: continue
        
        col_name = date_to_col[d]
        kq = kq_db[d]
        
        for g in groups:
            # Lọc thành viên
            mask = df[col_name].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
            members = df[mask]
            
            if members.empty:
                stats[g]['ranks'].append(999); continue
                
            total_scores = Counter()
            for _, row in members.iterrows():
                for sc_col, score in valid_cols_score.items():
                    for n in get_nums(row[sc_col]): total_scores[n] += score
            
            rank_n = [n for n, s in total_scores.most_common()]
            rank_n.sort(key=lambda x: (-total_scores[x], int(x)))
            
            top80 = rank_n[:80]
            if kq in top80:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top80.index(kq) + 1)
            else: stats[g]['ranks'].append(999)

    final_ranks = []
    for g, info in stats.items():
        final_ranks.append((g, -info['wins'], sum(info['ranks'])))
    final_ranks.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_ranks[:6]]
    
    # 2. Dự đoán
    prev_date = target_date - timedelta(days=1)
    res = []
    col_used = None
    
    if prev_date in date_to_col:
        col_used = date_to_col[prev_date]
        def get_pool(grp_list):
            pool = []
            for g in grp_list:
                mask = df[col_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
                mems = df[mask]
                scores = Counter()
                for _, row in mems.iterrows():
                    for sc_col, score in valid_cols_score.items():
                        for n in get_nums(row[sc_col]): scores[n] += score
                r_n = [n for n, s in scores.most_common()]
                r_n.sort(key=lambda x: (-scores[x], int(x)))
                
                limit = 80
                if g in [top6[2], top6[3]]: limit = 65
                if g in [top6[4], top6[5]]: limit = 60
                pool.extend(r_n[:limit])
            return pool

        s1 = set(get_pool([top6[0], top6[5], top6[3]]))
        s2 = set(get_pool([top6[1], top6[4], top6[2]]))
        res = sorted(list(s1.intersection(s2)))
        
    return top6, res, col_used

# --- GIAO DIỆN CHÍNH ---
if uploaded_files:
    with st.spinner("Đang soi dữ liệu..."):
        data_cache, kq_db, logs = load_data_v15(uploaded_files)
    
    # KHU VỰC HIỂN THỊ TRẠNG THÁI (LUÔN HIỆN)
    with st.expander("🧐 TRẠNG THÁI ĐỌC FILE (Quan trọng)", expanded=True):
        if not data_cache:
            st.error("❌ Không đọc được ngày nào! Hãy kiểm tra lại file.")
            for l in logs: st.text(l)
        else:
            st.success(f"✅ Đã đọc {len(data_cache)} ngày.")
            st.caption("Các ngày có Kết quả:")
            # Show KQ để user kiểm tra
            kq_list = [{"Ngày": k, "KQ": v} for k, v in kq_db.items()]
            if kq_list:
                st.dataframe(pd.DataFrame(kq_list).sort_values("Ngày"), height=150)
            else:
                st.warning("Đọc được ngày nhưng chưa tìm thấy dòng 'KQ'.")

    # KHU VỰC NÚT BẤM (LUÔN HIỆN NẾU CÓ DỮ LIỆU)
    if data_cache:
        st.write("---")
        tab1, tab2 = st.tabs(["🔮 DỰ ĐOÁN", "️u001f3c5 BACKTEST"])
        
        with tab1:
            # Tự động chọn ngày tiếp theo của ngày cuối cùng có dữ liệu
            last_d = max(data_cache.keys())
            target = st.date_input("Chọn ngày dự đoán:", value=last_d + timedelta(days=1))
            
            if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
                top6, res, col = calculate_v15(target, ROLLING_WINDOW, data_cache, kq_db)
                
                if not col:
                    st.error(f"❌ Không thể dự đoán! Lý do: Không tìm thấy dữ liệu của ngày hôm trước ({target - timedelta(days=1)}) trong file.")
                else:
                    st.success(f"Dữ liệu lấy từ cột: {col}")
                    st.info(f"🏆 TOP 6: {', '.join(top6)}")
                    st.text_area("KẾT QUẢ:", ",".join(res))
        
        with tab2:
            st.write("Chạy kiểm chứng quá khứ:")
            c1, c2 = st.columns(2)
            with c1: start = st.date_input("Từ:", value=last_d - timedelta(days=5))
            with c2: end = st.date_input("Đến:", value=last_d)
            
            if st.button("⚡ CHẠY BACKTEST", use_container_width=True):
                delta = (end - start).days
                logs = []
                bar = st.progress(0)
                for i in range(delta + 1):
                    d = start + timedelta(days=i)
                    bar.progress((i+1)/(delta+1))
                    try:
                        _, res, _ = calculate_v15(d, ROLLING_WINDOW, data_cache, kq_db)
                        real = kq_db.get(d, "N/A")
                        stt = "WIN" if real in res else "LOSS"
                        if real == "N/A": stt = "-"
                        logs.append({"Ngày": d.strftime("%d/%m"), "KQ": real, "TT": stt, "Số lượng": len(res)})
                    except: pass
                bar.empty()
                st.dataframe(pd.DataFrame(logs), use_container_width=True)

