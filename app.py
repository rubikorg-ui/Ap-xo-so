import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V12 Final", page_icon="💎", layout="centered")
st.title("💎 Dự Đoán (V12 - Fix Lỗi Cột Ngày)")

# --- 1. TẢI FILE ---
st.info("Bước 1: Tải file Excel (T12.2025, T1.2026...)")
uploaded_files = st.file_uploader("Chọn file:", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)

# --- HÀM XỬ LÝ ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}
RE_CLEAN = re.compile(r'[^A-Z0-9\/]')
RE_FIND_NUMS = re.compile(r'\d{1,2}') 

def clean_text(s):
    if pd.isna(s): return ""
    s_str = str(s).upper().replace('.', '/').replace('-', '/').replace('_', '/')
    # Giữ lại các ký tự số, chữ và dấu / để so sánh ngày
    return s_str

def get_nums(s):
    if pd.isna(s): return []
    raw_nums = re.findall(r'\d+', str(s)) # Lấy mọi con số
    # Lọc số có 1-2 chữ số (để tránh lấy nhầm năm 2026)
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

def get_header_row_index(df_raw):
    for i, row in df_raw.head(10).iterrows():
        row_str = str(row.values).upper()
        if "THÀNH VIÊN" in row_str or "THANH VIEN" in row_str: return i
    return 3

# --- [FIX] HÀM ĐỌC NGÀY TỪ SHEET ---
def parse_date_from_sheet(sheet_name, filename):
    # 1. Lấy Năm/Tháng từ Tên File
    year_match = re.search(r'(20\d{2})', filename)
    year = int(year_match.group(1)) if year_match else None
    
    month_match = re.search(r'(?:THANG|THÁNG|TH|T|M)[^0-9]*(\d+)', filename, re.IGNORECASE)
    if not month_match:
        # Fallback: tìm cụm d-yyyy hoặc d.yyyy
        alt_match = re.search(r'(\d+)[\.\-_]+' + str(year), filename) if year else None
        month = int(alt_match.group(1)) if alt_match else None
    else:
        month = int(month_match.group(1))

    # 2. Lấy Ngày từ Tên Sheet
    # Regex lấy số đầu tiên trong tên sheet (VD: "1.12" -> lấy 1, "2" -> lấy 2)
    day_match = re.search(r'^(\d+)', sheet_name.strip())
    day = int(day_match.group(1)) if day_match else None
    
    if day and month and year:
        try: return datetime.date(year, month, day)
        except: return None
    return None

@st.cache_data(ttl=600)
def load_data_v12(files):
    data_cache = {}
    kq_db = {} 
    debug_logs = []
    
    for file in files:
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                try:
                    current_date = parse_date_from_sheet(sheet_name, file.name)
                    if not current_date: continue 

                    # Đọc file
                    temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=15)
                    h = get_header_row_index(temp)
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h)
                    
                    # Quan trọng: Chuyển tên cột về String hết để dễ tìm
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    data_cache[current_date] = df
                    
                    # TÌM KẾT QUẢ (KQ)
                    # Tìm dòng chứa chữ "KQ"
                    kq_row_idx = None
                    for idx, row in df.iterrows():
                        row_s = str(row.values).upper()
                        if "KQ" in row_s and ("9X" not in row_s): # Tránh nhầm tiêu đề
                             kq_row_idx = idx; break
                    
                    if kq_row_idx is not None:
                        kq_row = df.iloc[kq_row_idx]
                        
                        # [FIX MẠNH] Tạo mọi định dạng ngày có thể để tìm cột
                        d, m, y = current_date.day, current_date.month, current_date.year
                        possible_cols = [
                            f"{d}/{m}", f"{d:02d}/{m}", f"{d}/{m:02d}", f"{d:02d}/{m:02d}", # 1/1, 01/01
                            str(d), # 1
                            f"{y}-{m:02d}-{d:02d}", # 2026-01-01 (Định dạng Excel hay dùng)
                            f"{y}-{m}-{d}", # 2026-1-1
                            f"{d}-{m}-{y}", # 01-01-2026
                        ]
                        
                        found_val = None
                        found_col_name = ""
                        
                        # Duyệt qua các cột trong file
                        for col in df.columns:
                            # So sánh: cột trong file có CHỨA một trong các pattern không?
                            col_upper = str(col).upper()
                            for p in possible_cols:
                                if p in col_upper:
                                    # Kiểm tra kỹ hơn: Nếu tìm số 1, tránh nhầm số 10, 11...
                                    # Nhưng với định dạng ngày tháng (có dấu / hoặc -) thì khá an toàn
                                    try:
                                        val = str(kq_row[col])
                                        nums = get_nums(val)
                                        if nums: 
                                            found_val = nums[0]
                                            found_col_name = col
                                    except: pass
                            if found_val: break
                            
                        if found_val: 
                            kq_db[current_date] = found_val
                            # Log kiểm tra cho ngày đầu tháng (dễ lỗi nhất)
                            if d <= 3:
                                debug_logs.append(f"✅ Ngày {current_date}: Lấy được KQ '{found_val}' từ cột '{found_col_name}'")

                except Exception as e: continue
        except: continue
            
    return data_cache, kq_db, debug_logs

def get_group_top_n_stable(df, group_name, grp_col, limit=80):
    # Lọc Group (0x, 1x...)
    # Cần clean_text nhẹ nhàng
    target = str(group_name).upper()
    try:
        # Tìm cột 9x, 8x... để lọc
        filter_col = None
        for c in df.columns:
            if "THÀNH VIÊN" in str(c).upper() and "9X" not in str(c).upper(): # Cột tên
                pass
            if target in str(df[c].astype(str).head().values).upper(): # Cách tìm cột chứa nhóm
                # Cách đơn giản hơn: Tìm cột có tên khớp pattern
                pass
                
        # Logic cũ ổn định hơn cho việc lọc hàng
        # Ta dùng lại logic tìm cột chứa nhóm
        mask = df.iloc[:, 0].astype(str).str.contains(".*", na=False) # Placeholder
        
        # Tìm cột định danh nhóm (Thường là cột TV TOP 9X0X hoặc tương tự)
        # Trong file user: Cột đầu tiên chứa tên Group? Không, cột đầu chứa tên user
        # Cột quy định nhóm (1x, 2x...) nằm ngay trong dữ liệu.
        
        # [FIX] Logic lấy Top:
        # Duyệt qua từng dòng, xem cột Group (ví dụ cột '9X') có phải là group_name không
        # Nhưng code cũ dùng 'grp_col' là cột NGÀY HÔM TRƯỚC.
        # Ý nghĩa: Lấy những người mà NGÀY HÔM TRƯỚC thuộc nhóm group_name (VD: 0x)
        
        col_vals = df[grp_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper()))
        members = df[col_vals == target.upper()]
        
    except: return []
    if members.empty: return []

    # Tính điểm
    total_scores = Counter()
    
    # Chỉ xét các cột M0...M10
    valid_cols = []
    col_scores = {}
    for c in df.columns:
        s = get_col_score(c)
        if s > 0:
            col_scores[c] = s
            valid_cols.append(c)
            
    # Cộng điểm
    for idx, row in members.iterrows():
        for col in valid_cols:
            val = str(row[col])
            score = col_scores[col]
            nums = get_nums(val)
            for n in nums:
                total_scores[n] += score
                
    all_nums = list(total_scores.keys())
    # Sort: Điểm cao -> Số nhỏ
    all_nums.sort(key=lambda n: (-total_scores[n], int(n)))
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
        
        # Tìm cột dữ liệu của ngày hôm trước (để phân nhóm)
        # Cột này chứa: 0x, 1x...
        d, m, y = prev_date.day, prev_date.month, prev_date.year
        patterns = [
            f"{d}/{m}", f"{d:02d}/{m}", str(d),
            f"{y}-{m:02d}-{d:02d}", f"{y}-{m}-{d}"
        ]
        
        grp_col = None
        for c in df.columns:
            for p in patterns:
                if p in str(c).upper(): 
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
        prev_date = target_date - timedelta(days=1)
        d, m, y = prev_date.day, prev_date.month, prev_date.year
        patterns = [f"{d}/{m}", f"{d:02d}/{m}", str(d), f"{y}-{m:02d}-{d:02d}"]
        
        grp_col_target = None
        for c in df_target.columns:
            for p in patterns:
                if p in str(c).upper(): 
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
            return top6, final_result, grp_col_target
    return top6, final_result, None

# --- MAIN ---
if uploaded_files:
    with st.spinner("⏳ Đang soi kỹ từng cột trong file..."):
        data_cache, kq_db, debug_logs = load_data_v12(uploaded_files)
    
    with st.expander("🔍 KIỂM TRA DỮ LIỆU ĐÃ ĐỌC (Quan trọng!)", expanded=True):
        if not data_cache:
            st.error("❌ Không đọc được ngày nào!")
        else:
            st.success(f"✅ Đã đọc thành công {len(data_cache)} ngày.")
            st.write("Nhật ký đọc các ngày đầu tháng (để kiểm tra cột 2026-01-01):")
            for log in debug_logs: st.text(log)

    if data_cache:
        tab1, tab2 = st.tabs(["🎯 DỰ ĐOÁN", "🛠️ BACKTEST"])
        
        with tab1:
            st.write("### Chọn ngày:")
            default_date = max(data_cache.keys()) if data_cache else datetime.date.today()
            selected_date = st.date_input("Ngày:", value=default_date)
            
            if st.button("🚀 DỰ ĐOÁN NGAY", use_container_width=True):
                top6, result, found_col = calculate_by_date(selected_date, ROLLING_WINDOW, data_cache, kq_db)
                
                if not found_col:
                    st.warning(f"⚠️ Không tìm thấy cột dữ liệu của ngày hôm trước ({selected_date - timedelta(days=1)}) để dựa vào đó dự đoán.")
                    st.caption("Gợi ý: Kiểm tra xem trong Sheet của ngày dự đoán có cột ngày hôm trước không.")
                else:
                    st.info(f"Dữ liệu dựa trên cột: **{found_col}**")
                    st.success(f"🏆 TOP 6 GROUP: {', '.join(top6)}")
                    st.code(",".join(result), language="text")
                    if selected_date in kq_db:
                        st.write(f"Kết quả thực tế ngày này: **{kq_db[selected_date]}**")

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
                        _, res, _ = calculate_by_date(d, ROLLING_WINDOW, data_cache, kq_db)
                        act = kq_db.get(d, "N/A")
                        stt = "WIN ✅" if act in res else "LOSS ❌"
                        if act == "N/A": stt = "Waiting"
                        logs.append({"Ngày": d.strftime('%d/%m'), "KQ": act, "TT": stt, "Số lượng": len(res)})
                    except: pass
                bar.empty()
                if logs: st.dataframe(pd.DataFrame(logs), use_container_width=True)
                else: st.warning("Không có dữ liệu.")
