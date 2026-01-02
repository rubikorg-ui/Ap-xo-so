import streamlit as st
import pandas as pd
import re
from collections import Counter
import io

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="App Xổ Số Mobile", page_icon="📱", layout="centered")

# --- HEADER ---
st.title("📱 Dự Đoán & Backtest")
st.write("---")

# --- 1. KHU VỰC TẢI FILE (ĐƯA RA GIỮA MÀN HÌNH) ---
st.info("Bước 1: Tải file dữ liệu (.xlsx)")
uploaded_files = st.file_uploader("Chọn file Excel (Tháng 12, Tháng 1...)", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH PHỤ (ẨN TRONG MENU ĐỂ GỌN) ---
with st.sidebar:
    st.header("⚙️ Cài đặt nâng cao")
    TARGET_MONTH = st.text_input("Tháng mục tiêu (Ví dụ: 01)", value="01")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
    st.write("---")
    st.caption("Các cài đặt này ít khi phải đổi.")

# --- HÀM XỬ LÝ (GIỮ NGUYÊN LOGIC) ---
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

@st.cache_data(ttl=600)
def load_data_multifile(files, target_month):
    data_cache = {}
    kq_db = {}
    sorted_files = sorted(files, key=lambda x: x.name)
    
    for file in sorted_files:
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                try:
                    match = re.search(r'(\d+)', sheet_name)
                    if not match: continue
                    day = int(match.group(1))
                    
                    temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=10)
                    h = get_header_row_index(temp)
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h)
                    df.columns = [clean_text(c) for c in df.columns]
                    
                    data_cache[day] = df
                    
                    mask_kq = df.iloc[:, 0].astype(str).apply(clean_text).str.contains("KQ", na=False)
                    if mask_kq.any():
                        kq_row = df[mask_kq].iloc[0]
                        for c in sorted(df.columns):
                            d_val = None
                            if f"/{target_month}" in c: 
                                try: d_val = int(c.split("/")[0])
                                except: pass
                            elif c.isdigit() and 1 <= int(c) <= 31: d_val = int(c)
                            if d_val and 1 <= d_val <= 31:
                                val = str(kq_row[c])
                                nums = get_nums(val)
                                if nums: kq_db[d_val] = nums[0]
                except: continue
        except: continue
    return data_cache, kq_db

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

def calculate_for_one_day(target_day, target_month, rolling_window, data_cache, kq_db):
    start_hist = max(1, target_day - rolling_window)
    end_hist = target_day - 1
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    for d in range(start_hist, end_hist + 1):
        if d not in data_cache or d not in kq_db: continue
        df = data_cache[d]
        prev = d - 1
        raw_patterns = [str(prev), f"{prev:02d}", f"{prev}/{target_month}", f"{prev:02d}/{target_month}"]
        if prev == 0: raw_patterns.extend(["30/11", "29/11", "31/12"])
        patterns = [clean_text(p) for p in raw_patterns]
        grp_col = None
        for c in sorted(df.columns):
            if c in patterns: grp_col = c; break
        if not grp_col: continue
        kq = kq_db[d]
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
    if target_day in data_cache:
        df_target = data_cache[target_day]
        prev = target_day - 1
        raw_patterns = [str(prev), f"{prev:02d}", f"{prev}/{target_month}", f"{prev:02d}/{target_month}"]
        patterns = [clean_text(p) for p in raw_patterns]
        grp_col_target = None
        for c in sorted(df_target.columns):
            if c in patterns: grp_col_target = c; break
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

# --- PHẦN CHÍNH (MAIN SCREEN) ---

if uploaded_files:
    # 1. Load dữ liệu ngay khi có file
    with st.spinner("⏳ Đang xử lý file..."):
        data_cache, kq_db = load_data_multifile(uploaded_files, TARGET_MONTH)
    
    if not data_cache:
        st.error("❌ Không đọc được dữ liệu! Vui lòng kiểm tra file Excel.")
    else:
        st.success(f"✅ Đã tải xong! Có {len(data_cache)} ngày dữ liệu.")
        
        # 2. Chọn chế độ bằng TAB (Dễ bấm hơn)
        tab1, tab2 = st.tabs(["🎯 DỰ ĐOÁN (Hôm nay)", "🛠️ BACKTEST (Lịch sử)"])
        
        # --- TAB 1: DỰ ĐOÁN ---
        with tab1:
            st.write("### Chọn ngày cần dự đoán:")
            # Nút chọn ngày to rõ
            col_date, col_btn = st.columns([2, 1])
            with col_date:
                max_d = len(data_cache) if data_cache else 31
                target_day = st.number_input("Ngày:", min_value=1, max_value=31, value=1, key="day_pred")
            with col_btn:
                st.write("") # Spacer
                st.write("") 
                run_pred = st.button("🚀 CHẠY", key="btn_pred", use_container_width=True)
            
            if run_pred:
                top6, result = calculate_for_one_day(target_day, TARGET_MONTH, ROLLING_WINDOW, data_cache, kq_db)
                st.info(f"🏆 **TOP 6 GROUP:** {', '.join(top6)}")
                st.success(f"**KẾT QUẢ ({len(result)} số):**")
                st.code(",".join(result), language="text")
                
                # Check kết quả ngay tại chỗ
                if target_day in kq_db:
                    real = kq_db[target_day]
                    if real in result: st.success(f"🎉 TRÚNG RỒI: {real}")
                    else: st.error(f"❌ TRƯỢT: Về {real}")
                else: st.warning("⚠️ Chưa có KQ ngày này.")

        # --- TAB 2: BACKTEST ---
        with tab2:
            st.write("### Kiểm tra khoảng thời gian:")
            c1, c2 = st.columns(2)
            with c1: start_d = st.number_input("Từ ngày:", min_value=1, value=1, key="bt_start")
            with c2: end_d = st.number_input("Đến ngày:", min_value=1, value=max_d, key="bt_end")
            
            if st.button("⚡ CHẠY KIỂM CHỨNG", use_container_width=True):
                if start_d > end_d: st.error("Ngày sai!")
                else:
                    logs = []
                    bar = st.progress(0)
                    days = range(start_d, end_d + 1)
                    for i, d in enumerate(days):
                        bar.progress((i+1)/len(days))
                        try:
                            _, res = calculate_for_one_day(d, TARGET_MONTH, ROLLING_WINDOW, data_cache, kq_db)
                            act = kq_db.get(d, "N/A")
                            stt = "WIN ✅" if act in res else "LOSS ❌"
                            if act == "N/A": stt = "Waiting"
                            logs.append({"Ngày": d, "KQ": act, "Trạng thái": stt, "Số lượng": len(res)})
                        except: pass
                    bar.empty()
                    
                    df_res = pd.DataFrame(logs)
                    wins = df_res[df_res["Trạng thái"] == "WIN ✅"].shape[0]
                    total = df_res[df_res["KQ"] != "N/A"].shape[0]
                    st.metric("Tỷ lệ thắng", f"{wins}/{total} ({round(wins/total*100 if total else 0)}%)")
                    st.dataframe(df_res, use_container_width=True)

else:
    # Màn hình chờ khi chưa có file
    st.warning("👈 Vui lòng tải file Excel (.xlsx) để bắt đầu.")
    st.write("Lưu ý: File phải là Excel, có nhiều Sheet.")
