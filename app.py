import streamlit as st
import pandas as pd
import re
from collections import Counter
import io

# --- CẤU HÌNH ---
st.set_page_config(page_title="Siêu Backtest Đa File", page_icon="📈", layout="wide")
st.title("📈 Phân Tích & Gộp Nhiều File")

# --- HÀM XỬ LÝ (CORE LOGIC) ---
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

# --- HÀM ĐỌC NHIỀU FILE ---
@st.cache_data(ttl=600)
def load_data_multifile(uploaded_files, target_month):
    data_cache = {}
    kq_db = {}
    
    # Sắp xếp file để đảm bảo file tháng cũ nạp trước, file tháng mới nạp sau (ghi đè)
    # Logic: Dữ liệu tháng hiện tại (Target) sẽ được ưu tiên nhất
    sorted_files = sorted(uploaded_files, key=lambda x: x.name)
    
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
                    
                    # Lưu vào bộ nhớ (Nếu ngày trùng nhau, file nạp sau sẽ ghi đè - đúng tính chất nối tháng)
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
                except Exception: continue
        except Exception as e:
            st.error(f"Lỗi đọc file {file.name}: {e}")
            
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
        # Mẹo: Nếu d = 31 mà data_cache[31] là của tháng 12, code vẫn lấy đúng!
        # Vì tháng 1 (hiện tại) chưa có ngày 31 để ghi đè lên.
        if d not in data_cache or d not in kq_db: continue
        df = data_cache[d]
        prev = d - 1
        raw_patterns = [str(prev), f"{prev:02d}", f"{prev}/{target_month}", f"{prev:02d}/{target_month}"]
        if prev == 0: raw_patterns.extend(["30/11", "29/11", "31/12"]) # Xử lý ngày cuối năm
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

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    APP_MODE = st.radio("Chọn chế độ:", ["🎯 Dự đoán 1 ngày", "🛠️ Kiểm chứng (Backtest)"])
    st.divider()
    TARGET_MONTH = st.text_input("Tháng dữ liệu (Ví dụ: 01)", value="01")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)
    
    # --- CẬP NHẬT QUAN TRỌNG: CHO PHÉP CHỌN NHIỀU FILE ---
    uploaded_files = st.file_uploader("📂 Tải tất cả file Excel (.xlsx)", type=['xlsx'], accept_multiple_files=True)

if uploaded_files:
    # Load data
    with st.spinner("Đang gộp dữ liệu từ các file..."):
        data_cache, kq_db = load_data_multifile(uploaded_files, TARGET_MONTH)
    
    # Hiển thị thông báo thông minh
    total_days = len(data_cache)
    st.sidebar.success(f"✅ Đã tải xong {len(uploaded_files)} file.")
    st.sidebar.info(f"Tổng số ngày trong bộ nhớ: {total_days}")

    if APP_MODE == "🎯 Dự đoán 1 ngày":
        st.subheader("🎯 Dự đoán cho một ngày cụ thể")
        target_day = st.number_input("Chọn ngày muốn dự đoán:", min_value=1, max_value=31, value=1)
        
        if st.button("🚀 Phân Tích Ngay"):
            # Kiểm tra xem có đủ dữ liệu quá khứ không
            if target_day == 1 and 31 not in data_cache:
                st.warning("⚠️ Cảnh báo: Bạn đang dự đoán ngày 1 nhưng chưa tải file tháng trước (thiếu ngày 31). Kết quả có thể không chính xác.")
            
            with st.spinner("Đang tính toán..."):
                top6, result = calculate_for_one_day(target_day, TARGET_MONTH, ROLLING_WINDOW, data_cache, kq_db)
            
            st.info(f"🏆 **TOP 6 GROUP:** {', '.join(top6)}")
            st.success(f"**KẾT QUẢ DỰ ĐOÁN ({len(result)} số):**")
            st.text_area("Copy dàn số:", ",".join(result))
            
            if target_day in kq_db:
                real_kq = kq_db[target_day]
                if real_kq in result:
                    st.balloons(); st.success(f"🎉 CHÚC MỪNG! KQ **{real_kq}** ĐÃ TRÚNG.")
                else: st.error(f"❌ Rất tiếc. KQ **{real_kq}** không có trong dàn.")

    elif APP_MODE == "🛠️ Kiểm chứng (Backtest)":
        st.subheader("🛠️ Chạy thử nghiệm Lịch sử")
        c1, c2 = st.columns(2)
        with c1: start_d = st.number_input("Từ ngày:", min_value=1, value=1)
        with c2: end_d = st.number_input("Đến ngày:", min_value=1, value=total_days if total_days < 31 else 10)
        
        if st.button("⚡ Chạy Kiểm Chứng"):
            if start_d > end_d: st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")
            else:
                results_log = []
                progress_bar = st.progress(0)
                days_list = range(start_d, end_d + 1)
                
                for idx, d in enumerate(days_list):
                    progress_bar.progress((idx + 1) / len(days_list))
                    try:
                        _, pred_nums = calculate_for_one_day(d, TARGET_MONTH, ROLLING_WINDOW, data_cache, kq_db)
                        actual = kq_db.get(d, "N/A")
                        is_win = actual in pred_nums if actual != "N/A" else False
                        status = "WIN ✅" if is_win else ("LOSS ❌" if actual != "N/A" else "Chưa có KQ")
                        
                        results_log.append({
                            "Ngày": d, "KQ Thực": actual, "Trạng thái": status,
                            "Số lượng": len(pred_nums), "Dàn số": ",".join(pred_nums)
                        })
                    except: pass
                progress_bar.empty()
                
                df_res = pd.DataFrame(results_log)
                wins = df_res[df_res["Trạng thái"] == "WIN ✅"].shape[0]
                total = df_res[df_res["KQ Thực"] != "N/A"].shape[0]
                
                m1, m2 = st.columns(2)
                m1.metric("Số ngày Trúng", f"{wins}/{total}")
                if total > 0: m2.metric("Tỷ lệ", f"{round((wins/total)*100, 1)}%")
                
                def color_rows(row):
                    return ['background-color: #d4edda; color: black' if row["Trạng thái"] == "WIN ✅" else ('background-color: #f8d7da; color: black' if row["Trạng thái"] == "LOSS ❌" else '')] * len(row)
                st.dataframe(df_res.style.apply(color_rows, axis=1), use_container_width=True)

else:
    st.info("👈 Hãy tải CẢ 2 FILE Excel (Tháng 12 & Tháng 1) vào ô bên trái.")
