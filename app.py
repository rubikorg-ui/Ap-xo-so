import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V14 Fix Lỗi Đảo", page_icon="🔧", layout="centered")
st.title("🔧 V14: Fix Lỗi Ngày Tháng Bị Đảo")

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
    # Chỉ lấy số có 1 hoặc 2 chữ số (tránh lấy nhầm năm 2025)
    raw_nums = re.findall(r'\d+', str(s))
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

# --- HÀM XỬ LÝ NGÀY THÔNG MINH (TRỌNG TÂM V14) ---
def smart_parse_date(col_name, file_month, file_year):
    """
    Hàm này chuyên trị lỗi 2025-01-12 (hiểu nhầm là 12/1 thay vì 1/12)
    """
    s = str(col_name).strip().upper()
    
    # 1. Thử parse theo chuẩn YYYY-MM-DD hoặc YYYY-DD-MM
    # Regex tìm: Năm (4 số) - Số A - Số B
    match = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match:
        y, p1, p2 = int(match.group(1)), int(match.group(2)), int(match.group(3))
        
        # LOGIC SỬA LỖI ĐẢO NGÀY:
        # Nếu p1 (vị trí thường là Tháng) không khớp file_month
        # Nhưng p2 (vị trí thường là Ngày) lại bằng file_month
        # => Nó bị đảo! (Dạng YYYY-DD-MM)
        if p1 != file_month and p2 == file_month:
            try: return datetime.date(y, p2, p1) # Đảo lại: p2 là Tháng, p1 là Ngày
            except: pass
            
        # Nếu p1 khớp file_month => Chuẩn (YYYY-MM-DD)
        if p1 == file_month:
            try: return datetime.date(y, p1, p2)
            except: pass
            
    # 2. Thử parse dạng DD/MM (30/11)
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        # Xử lý năm chuyển giao (vd file T1 có cột 31/12)
        curr_y = file_year
        if m == 12 and file_month == 1: curr_y -= 1
        elif m == 1 and file_month == 12: curr_y += 1
        
        try: return datetime.date(curr_y, m, d)
        except: pass
        
    return None

def get_file_info(filename):
    # Lấy Năm
    y_match = re.search(r'20\d{2}', filename)
    y = int(y_match.group(0)) if y_match else 2025
    # Lấy Tháng
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m = int(m_match.group(1)) if m_match else 1
    return m, y

@st.cache_data(ttl=600)
def load_data_v14(files):
    data_cache = {} # Key: Date, Value: {df, map}
    kq_db = {}      # Key: Date, Value: KQ String
    debug_list = [] # List các ngày đã tìm thấy để show cho user
    
    for file in files:
        f_m, f_y = get_file_info(file.name)
        
        try:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                try:
                    # Đọc Sheet, tìm dòng Header chứa "TV TOP" hoặc "THÀNH VIÊN"
                    # Đọc thử 10 dòng đầu
                    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    header_row = 3 # Mặc định
                    for idx, row in preview.iterrows():
                        row_str = str(row.values).upper()
                        if "TV TOP" in row_str or "THÀNH VIÊN" in row_str:
                            header_row = idx; break
                    
                    df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
                    
                    # --- MAP CỘT THÀNH NGÀY ---
                    col_mapping = {}
                    found_dates = []
                    
                    for col in df.columns:
                        d_obj = smart_parse_date(col, f_m, f_y)
                        if d_obj:
                            col_mapping[col] = d_obj
                            found_dates.append(d_obj)
                            
                    # --- TÌM KẾT QUẢ (KQ) ---
                    # Tìm dòng bắt đầu bằng KQ
                    kq_row = None
                    for idx, row in df.iterrows():
                        if str(row.values[0]).strip().upper() == "KQ":
                            kq_row = row; break
                            
                    if kq_row is not None:
                        for col_name, date_val in col_mapping.items():
                            val = str(kq_row[col_name])
                            nums = get_nums(val)
                            if nums:
                                kq_db[date_val] = nums[0]
                                
                    # --- LƯU CACHE (Lưu theo Sheet đại diện cho ngày nào đó) ---
                    # Mẹo: Một sheet thường đại diện cho ngày trong tên Sheet, 
                    # nhưng dữ liệu quan trọng là các cột lịch sử.
                    # Ta lưu sheet này vào tất cả các ngày mà nó chứa dữ liệu
                    if found_dates:
                        # Lấy ngày mới nhất trong sheet làm key chính (để dự đoán)
                        max_date = max(found_dates)
                        data_cache[max_date] = {
                            'df': df,
                            'col_map': col_mapping
                        }
                        # Debug info
                        if len(debug_list) < 20: # Chỉ lưu vài cái mẫu
                             debug_list.append(f"Sheet '{sheet}': Đọc được {len(found_dates)} ngày (Max: {max_date})")

                except Exception as e: continue
        except: continue
    
    # Sắp xếp kq_db để hiển thị đẹp
    sorted_kq = sorted(kq_db.items())
    return data_cache, dict(sorted_kq), debug_list

# --- LOGIC DỰ ĐOÁN ---
def calculate_v14(target_date, rolling_window, data_cache, kq_db):
    # Lấy dữ liệu từ file gần nhất chứa target_date
    # Vì file Excel của user: Sheet ngày 2 chứa dữ liệu ngày 1, 31, 30...
    # Nên ta cần tìm Sheet có chứa target_date (hoặc ngày ngay trước nó)
    
    # 1. Tìm Sheet phù hợp nhất: Sheet có ngày target_date hoặc sheet ngày hôm sau
    # Thực tế: User muốn dự đoán ngày 2/1, thì cần dữ liệu của ngày 1/1, 31/12...
    # Dữ liệu này nằm trong Sheet ngày 2/1 (hoặc mới hơn).
    
    selected_data = None
    # Tìm trong cache xem có key nào trùng target_date không
    if target_date in data_cache:
        selected_data = data_cache[target_date]
    else:
        # Nếu không, tìm ngày gần nhất trong tương lai (VD user chọn 2/1 nhưng chỉ có sheet 3/1)
        future_dates = [d for d in data_cache.keys() if d >= target_date]
        if future_dates:
            selected_data = data_cache[min(future_dates)]
    
    if not selected_data:
        return [], [], None

    df = selected_data['df']
    col_map = selected_data['col_map']
    
    # Đảo ngược mapping để tìm tên cột từ ngày
    date_to_col = {v: k for k, v in col_map.items()}
    
    # Lấy danh sách ngày quá khứ
    past_dates = [target_date - timedelta(days=i) for i in range(1, rolling_window + 1)]
    past_dates.reverse() # Xa đến gần
    
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    valid_cols_score = {}
    for c in df.columns:
        s = get_col_score(c)
        if s > 0: valid_cols_score[c] = s

    for d in past_dates:
        if d not in kq_db: continue # Không có KQ thì không tính được Rank
        if d not in date_to_col: continue # Không có cột dữ liệu thì chịu
        
        # Cột dữ liệu của ngày d (chứa thông tin phân nhóm của ngày đó)
        # LƯU Ý: Trong file user, cột ngày 1/1 chứa dữ liệu của ngày 1/1 (nhóm gì, điểm bao nhiêu)
        # Và KQ ngày 1/1 dùng để check win/loss.
        
        col_name = date_to_col[d]
        kq = kq_db[d]
        
        # Lấy thông tin Group của từng người trong ngày d
        # Cột Group thường là cột dữ liệu đó luôn (chứa 1x, 2x...)
        
        # Duyệt từng Group
        for g in groups:
            # Lọc những người thuộc Group g trong ngày d
            # Giá trị ô phải chứa "g" (VD "1x")
            # Clean giá trị ô: bỏ hết ký tự lạ, chỉ lấy số + x
            mask = df[col_name].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
            members = df[mask]
            
            if members.empty:
                stats[g]['ranks'].append(999); continue
                
            # Tính điểm cho Group này
            total_scores = Counter()
            for _, row in members.iterrows():
                for sc_col, score in valid_cols_score.items():
                    # Chỉ cộng điểm từ các cột M0..M10
                    # Lấy số từ ô đó
                    nums = get_nums(row[sc_col])
                    for n in nums: total_scores[n] += score
            
            # Xếp hạng số
            ranked_nums = [n for n, s in total_scores.most_common()]
            # Sort phụ theo giá trị số
            ranked_nums.sort(key=lambda x: (-total_scores[x], int(x)))
            
            top80 = ranked_nums[:80]
            if kq in top80:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top80.index(kq) + 1)
            else:
                stats[g]['ranks'].append(999)

    # Tổng hợp Top 6
    final_ranks = []
    for g, info in stats.items():
        # Ưu tiên: Số trận thắng nhiều nhất -> Tổng hạng nhỏ nhất
        final_ranks.append((g, -info['wins'], sum(info['ranks'])))
    
    final_ranks.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_ranks[:6]]
    
    # DỰ ĐOÁN CHO NGÀY TARGET
    # Cần tìm cột dữ liệu của ngày hôm trước (target - 1) để lấy Group
    prev_date = target_date - timedelta(days=1)
    
    result_nums = []
    col_used = None
    
    if prev_date in date_to_col:
        col_used = date_to_col[prev_date]
        
        def get_pool(alliance):
            pool = []
            # Lấy 3 group trong liên minh
            for g in alliance:
                # Lọc thành viên thuộc group g vào ngày hôm qua
                mask = df[col_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
                mems = df[mask]
                
                scores = Counter()
                for _, row in mems.iterrows():
                    for sc_col, score in valid_cols_score.items():
                        for n in get_nums(row[sc_col]): scores[n] += score
                
                rnk = [n for n, s in scores.most_common()]
                rnk.sort(key=lambda x: (-scores[x], int(x)))
                
                limit = 80
                if g == top6[2] or g == top6[3]: limit = 65
                if g == top6[4] or g == top6[5]: limit = 60
                
                pool.extend(rnk[:limit])
            return pool

        p1 = get_pool([top6[0], top6[5], top6[3]])
        p2 = get_pool([top6[1], top6[4], top6[2]])
        
        # Giao 2 tập hợp
        s
