import streamlit as st
import pandas as pd
import re
import datetime
import time
from collections import Counter
from functools import lru_cache

# ==============================================================================
# 1. CẤU HÌNH & GIAO DIỆN CHUNG
# ==============================================================================
st.set_page_config(
    page_title="Soi Cầu Xổ Số Pro - Data V24.2",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Tùy chỉnh giao diện gọn gàng
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
    .highlight-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. GLOBAL CONSTANTS & REGEX
# ==============================================================================
# Các từ khóa rác cần loại bỏ
BAD_KEYWORDS = ['WEB', 'WWW', 'COM', 'VN', 'NET', 'ZALO', 'LOTO', 'BACHTHU', 'SOICAU', 'CHUYEN', 'TOP', 'VIP', 'MB']

# Regex Patterns
RE_NUMS = re.compile(r'\d+')
RE_ISO_DATE = re.compile(r'(\d{4})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')

# ==============================================================================
# 3. HELPER FUNCTIONS (Hàm bổ trợ cần thiết cho Core Functions)
# ==============================================================================
def find_header_row(df_raw):
    """Tìm dòng tiêu đề chứa tên các cao thủ/ngày tháng"""
    for idx, row in df_raw.iterrows():
        row_str = " ".join([str(x).upper() for x in row.values if pd.notna(x)])
        # Dấu hiệu nhận biết dòng header: có chứa STT, TÊN, hoặc có nhiều cột dữ liệu
        if 'STT' in row_str or 'TÊN' in row_str or 'MÃ' in row_str or len(row_str) > 50:
            return idx
    return 0

# ==============================================================================
# 4. CORE FUNCTIONS (NEW V24.2 - OPTIMIZED)
# ==============================================================================
# --- BẮT ĐẦU PHẦN CODE THAY THẾ ---

# 1. Hàm trích xuất số liệu (Fix lỗi nhập liệu)
@lru_cache(maxsize=10000)
def get_nums(s):
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): return []
    raw_nums = RE_NUMS.findall(s_upper)
    final = []
    for n in raw_nums:
        if len(n) > 2: n = n[-2:]
        final.append(n.zfill(2))
    return final

# 2. Hàm đọc thông tin file (Fix lỗi chuyển năm 2025-2026)
def extract_meta_from_filename(filename):
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    full_date_match = re.search(r'(\d{1,2})[\.\-](\d{1,2})(?:[\.\-]20\d{2})?', clean_name)
    specific_date = None
    if full_date_match:
        try:
            d = int(full_date_match.group(1)); m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            if m > 12 and d <= 12: d, m = m, d
            specific_date = datetime.date(y, m, d)
        except: pass
    return m_global, y_global, specific_date

# 3. Hàm phân tích ngày (Logic thông minh)
def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper().replace('NGAY', '').replace('NGÀY', '').strip()
    match_iso = RE_ISO_DATE.search(s)
    if match_iso:
        p1, p2, p3 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 > 1000: return datetime.date(p1, p2, p3)
        else: return datetime.date(p3, p2, p1)
    match_slash = RE_SLASH_DATE.search(s)
    if match_slash:
        n1, n2 = int(match_slash.group(1)), int(match_slash.group(2))
        d, m = n1, n2
        if m > 12 and d <= 12: d, m = n2, n1
        if m < 1 or m > 12 or d < 1 or d > 31: return None
        curr_y = f_y
        if f_m == 12 and m == 1: curr_y += 1
        elif f_m == 1 and m == 12: curr_y -= 1
        try: return datetime.date(curr_y, m, d)
        except: return None
    return None

# 4. Hàm Load Data V24.2 (Fix Đa Sheet, Quét toàn bộ tìm KQ)
@st.cache_data(ttl=600, show_spinner=False)
def load_data_v24(files):
    cache = {}; kq_db = {}; err_logs = []; file_status = []
    files = sorted(files, key=lambda x: x.name)
    for file in files:
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper(): continue
        f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
        try:
            raw_data_list = []
            if file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(file, engine='openpyxl')
                for sheet_name in xls.sheet_names:
                    s_nums = re.findall(r'\d+', sheet_name)
                    if not s_nums: continue 
                    try:
                        d_val = int(s_nums[0]); curr_date = datetime.date(f_y, f_m, d_val)
                        df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine='openpyxl')
                        if not df_raw.empty: raw_data_list.append((curr_date, df_raw))
                    except: continue
                if raw_data_list: file_status.append(f"✅ Excel: {file.name} ({len(raw_data_list)} sheets)")
            elif file.name.endswith('.csv'):
                if not date_from_name: continue
                encodings = ['utf-8', 'utf-16', 'cp1258', 'latin-1']
                df_raw = None
                for enc in encodings:
                    try: file.seek(0); df_raw = pd.read_csv(file, header=None, encoding=enc); break
                    except: continue
                if df_raw is not None: raw_data_list.append((date_from_name, df_raw)); file_status.append(f"✅ CSV: {file.name}")

            for t_date, df_raw in raw_data_list:
                h_row_idx = find_header_row(df_raw)
                header_vals = df_raw.iloc[h_row_idx].astype(str).str.strip().str.upper()
                df = df_raw.iloc[h_row_idx+1:].copy()
                df.columns = header_vals
                df.columns = [str(c).replace('M 1 0', 'M10').replace(' ', '') for c in df.columns]
                if 'STT' in df.columns: df = df[pd.to_numeric(df['STT'], errors='coerce').notna()]
                if 'Đ9X0X' in df.columns: df['SCORE_SORT'] = pd.to_numeric(df['Đ9X0X'], errors='coerce').fillna(0)
                else: df['SCORE_SORT'] = 0
                
                hist_map = {}; col_date_map = {}
                for idx, col_name in enumerate(header_vals):
                    d_obj = parse_date_smart(col_name, t_date.month, t_date.year)
                    if d_obj: hist_map[d_obj] = df.columns[idx]; col_date_map[idx] = d_obj
                
                for r_idx, row in df_raw.iterrows():
                    first_cols = [str(x).upper() for x in row.values[:3]]
                    if any(re.search(r'(KQ|KẾT|QUẢ|RESULT)', s) for s in first_cols):
                        for c_idx, d_val in col_date_map.items():
                            val = str(row.values[c_idx])
                            nums = get_nums(val)
                            if nums: kq_db[d_val] = nums[0]
                cache[t_date] = {'df': df, 'hist_map': hist_map}
        except Exception as e: err_logs.append(f"Lỗi '{file.name}': {str(e)}"); continue
    return cache, kq_db, file_status, err_logs

# --- KẾT THÚC PHẦN CODE THAY THẾ ---

# ==============================================================================
# 5. LOGIC CHIẾN THUẬT (STRATEGY E - OVERLAP BOOST)
# ==============================================================================

def calculate_momentum(player_row, hist_map, kq_db, target_date, days_lookback=5):
    """Tính chuỗi thắng (Momentum) và hiệu suất"""
    wins = 0
    streak = 0
    current_streak_active = True
    
    # Lấy danh sách ngày trong quá khứ, sort giảm dần (gần nhất trước)
    sorted_dates = sorted([d for d in hist_map.keys() if d < target_date], reverse=True)
    check_dates = sorted_dates[:days_lookback]
    
    for d in check_dates:
        col = hist_map[d]
        if col in player_row and pd.notna(player_row[col]):
            nums = get_nums(player_row[col])
            if d in kq_db and kq_db[d] in nums:
                wins += 1
                if current_streak_active: streak += 1
            else:
                current_streak_active = False
    
    return streak, wins

def execute_strategy_e(df, current_hist_col, kq_db, target_date, hist_map):
    """
    Thực hiện Chiến thuật E (Overlap Boost):
    - Dàn Đại Tướng: Top 7 Momentum -> Lọc 75s theo vị trí.
    - Dàn Hộ Vệ: Top 5-14 (Overlap) -> Lọc 80s theo vị trí.
    - Dàn Tinh Hoa: Giao thoa.
    """
    player_stats = []
    
    for idx, row in df.iterrows():
        streak, wins = calculate_momentum(row, hist_map, kq_db, target_date)
        # Điểm vị trí (giả lập lấy từ cột SCORE_SORT hoặc mặc định)
        pos_score = row.get('SCORE_SORT', 0)
        
        # Lấy số dự đoán cho ngày hiện tại
        current_nums = []
        if current_hist_col in row and pd.notna(row[current_hist_col]):
            current_nums = get_nums(row[current_hist_col])
            
        if current_nums:
            player_stats.append({
                'name': row.get('TÊN', f'Player_{idx}'),
                'streak': streak,
                'wins': wins,
                'pos_score': pos_score,
                'nums': current_nums
            })
    
    # Sắp xếp theo Momentum (Chuỗi thắng) giảm dần, sau đó là Pos Score
    player_stats.sort(key=lambda x: (x['streak'], x['pos_score']), reverse=True)
    
    # 1. Nhóm Đại Tướng (Top 7 Momentum)
    dai_tuong_players = player_stats[:7]
    all_dai_tuong_nums = []
    for p in dai_tuong_players:
        all_dai_tuong_nums.extend(p['nums'])
    # Lọc Top 75 số xuất hiện nhiều nhất (theo tần suất/vị trí)
    c_dt = Counter(all_dai_tuong_nums)
    dan_dai_tuong = [n for n, c in c_dt.most_common(75)]
    
    # 2. Nhóm Hộ Vệ (Top 5 đến 14 - Overlap)
    ho_ve_players = player_stats[4:14] # Lấy từ index 4 (Top 5) đến 13 (Top 14)
    all_ho_ve_nums = []
    for p in ho_ve_players:
        all_ho_ve_nums.extend(p['nums'])
    # Lọc Top 80 số
    c_hv = Counter(all_ho_ve_nums)
    dan_ho_ve = [n for n, c in c_hv.most_common(80)]
    
    # 3. Dàn Tinh Hoa (Giao thoa)
    dan_tinh_hoa = sorted(list(set(dan_dai_tuong) & set(dan_ho_ve)))
    
    return {
        'tinh_hoa': dan_tinh_hoa,
        'dai_tuong': sorted(dan_dai_tuong),
        'ho_ve': sorted(dan_ho_ve),
        'top_players': dai_tuong_players
    }

# ==============================================================================
# 6. MAIN APPLICATION
# ==============================================================================
def main():
    st.title("🎲 Soi Cầu Xổ Số Pro - Phiên Bản Overlap Boost (Fix 2026)")
    
    # Sidebar: Upload & Config
    with st.sidebar:
        st.header("1. Dữ Liệu Đầu Vào")
        uploaded_files = st.file_uploader("Chọn file Excel/CSV (Hỗ trợ 2025-2026)", 
                                        type=['xlsx', 'csv'], accept_multiple_files=True)
        
        if not uploaded_files:
            st.info("Vui lòng tải lên ít nhất một file dữ liệu.")
            return

        with st.spinner("Đang xử lý dữ liệu lõi..."):
            cache, kq_db, status_logs, err_logs = load_data_v24(uploaded_files)
        
        # Hiển thị trạng thái file
        with st.expander("Trạng thái tải file"):
            for log in status_logs: st.success(log)
            for err in err_logs: st.error(err)
            
        if not cache:
            st.error("Không đọc được dữ liệu nào hợp lệ.")
            return

        # Chọn ngày phân tích
        available_dates = sorted(cache.keys(), reverse=True)
        target_date = st.selectbox("Chọn ngày soi cầu:", available_dates)
        
        st.divider()
        st.write("Cấu hình: **Chiến thuật E (Overlap Boost)**")
        st.caption("✅ Đại Tướng: Top 7 Momentum (75s)")
        st.caption("✅ Hộ Vệ: Top 5-14 (80s)")
        st.caption("✅ Tinh Hoa: Giao Thoa An Toàn")

    # Lấy dữ liệu ngày được chọn
    day_data = cache[target_date]
    df = day_data['df']
    hist_map = day_data['hist_map']
    
    # Xác định cột dữ liệu của ngày hôm đó (để lấy số dự đoán)
    # Lưu ý: Trong file excel, ngày hiện tại thường là cột cuối cùng hoặc cột được định danh cụ thể
    # Ở đây ta lấy cột tương ứng với target_date trong hist_map
    current_col = hist_map.get(target_date)
    
    if not current_col:
        st.warning(f"Không tìm thấy cột dữ liệu cho ngày {target_date} trong file.")
        return

    # Tabs chính
    tab1, tab2, tab3 = st.tabs(["📊 PHÂN TÍCH HÔM NAY", "🛠 BACKTEST QUÁ KHỨ", "📂 DỮ LIỆU GỐC"])

    # --- TAB 1: PHÂN TÍCH ---
    with tab1:
        st.header(f"Kết Quả Soi Cầu Ngày {target_date.strftime('%d/%m/%Y')}")
        
        # Thực thi chiến thuật
        results = execute_strategy_e(df, current_col, kq_db, target_date, hist_map)
        
        # Hiển thị kết quả chính (Tinh Hoa)
        st.subheader("🏆 DÀN TINH HOA (Mục tiêu chính)")
        if results['tinh_hoa']:
            nums_str = ", ".join(results['tinh_hoa'])
            st.success(f"**Số lượng:** {len(results['tinh_hoa'])} số")
            st.text_area("Copy dàn số:", value=nums_str, height=100)
            
            # Kiểm tra trúng trượt nếu đã có KQ
            if target_date in kq_db:
                kq = kq_db[target_date]
                is_win = kq in results['tinh_hoa']
                st.metric("Kết quả thực tế", f"Đề về: {kq}", delta="TRÚNG" if is_win else "TRƯỢT", 
                         delta_color="normal" if is_win else "inverse")
        else:
            st.warning("Không tìm thấy số giao thoa đủ điều kiện.")

        col_dt, col_hv = st.columns(2)
        with col_dt:
            st.info(f"**Dàn Đại Tướng ({len(results['dai_tuong'])}s)**: Top 7 Momentum")
            with st.expander("Xem chi tiết"):
                st.write(", ".join(results['dai_tuong']))
        with col_hv:
            st.warning(f"**Dàn Hộ Vệ ({len(results['ho_ve'])}s)**: Top 5-14 Overlap")
            with st.expander("Xem chi tiết"):
                st.write(", ".join(results['ho_ve']))
        
        st.write("---")
        st.subheader("🎖 Top Cao Thủ Dẫn Đầu (Momentum)")
        top_df = pd.DataFrame(results['top_players'])
        if not top_df.empty:
            st.dataframe(top_df[['name', 'streak', 'wins', 'pos_score']].head(10), use_container_width=True)

    # --- TAB 2: BACKTEST ---
    with tab2:
        st.subheader("Kiểm Tra Hiệu Quả Chiến Thuật E")
        days_to_test = st.slider("Số ngày test ngược về quá khứ:", 1, 30, 7)
        
        if st.button("Chạy Backtest"):
            report = []
            progress_bar = st.progress(0)
            
            test_dates = [d for d in available_dates if d <= target_date][:days_to_test]
            
            for i, d in enumerate(test_dates):
                # Data ngày đó
                d_data = cache[d]
                d_df = d_data['df']
                d_hist = d_data['hist_map']
                d_col = d_hist.get(d)
                
                if d in kq_db and d_col:
                    res = execute_strategy_e(d_df, d_col, kq_db, d, d_hist)
                    kq = kq_db[d]
                    win = kq in res['tinh_hoa']
                    report.append({
                        'Ngày': d.strftime('%d/%m'),
                        'KQ': kq,
                        'Dàn Tinh Hoa': len(res['tinh_hoa']),
                        'Trúng': '✅' if win else '❌'
                    })
                progress_bar.progress((i + 1) / len(test_dates))
            
            res_df = pd.DataFrame(report)
            if not res_df.empty:
                st.table(res_df)
                win_rate = res_df['Trúng'].value_counts(normalize=True).get('✅', 0) * 100
                st.metric("Tỷ lệ trúng Backtest", f"{win_rate:.1f}%")
            else:
                st.warning("Chưa đủ dữ liệu kết quả để backtest.")

    # --- TAB 3: DATA RAW ---
    with tab3:
        st.dataframe(df.head(100))

if __name__ == "__main__":
    main()
