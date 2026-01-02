import streamlit as st
import pandas as pd
import re
from collections import Counter
import io

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Dự Đoán Xổ Số", page_icon="🚀")
st.title("🚀 Ứng Dụng Phân Tích Dữ Liệu")

# --- SIDEBAR (THANH BÊN) ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    TARGET_DAY = st.number_input("Chọn Ngày (Target Day)", min_value=1, max_value=31, value=1)
    TARGET_MONTH = st.text_input("Tháng (Ví dụ: 01)", value="01")
    TARGET_YEAR_PREFIX = st.text_input("Năm (Ví dụ: 2026)", value="2026")
    ROLLING_WINDOW = st.number_input("Chu kỳ (Rolling Window)", min_value=1, value=10)
    uploaded_files = st.file_uploader("📂 Tải file CSV vào đây", accept_multiple_files=True, type=['csv'])

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

@st.cache_data(ttl=600)
def load_data_from_upload(files):
    data_cache = {}
    kq_db = {}
    for uploaded_file in files:
        try:
            base = uploaded_file.name
            match = re.search(r'(\d+)', base)
            if not match: continue
            day = int(match.group(1))
            
            try: temp = pd.read_csv(uploaded_file, header=None, nrows=10, encoding='utf-8-sig')
            except: 
                uploaded_file.seek(0)
                temp = pd.read_csv(uploaded_file, header=None, nrows=10, encoding='latin-1')
            
            h = get_header_row_index(temp)
            uploaded_file.seek(0)
            try: df = pd.read_csv(uploaded_file, header=h, encoding='utf-8-sig')
            except: 
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=h, encoding='latin-1')
                
            df.columns = [clean_text(c) for c in df.columns]
            data_cache[day] = df
            
            mask_kq = df.iloc[:, 0].astype(str).apply(clean_text).str.contains("KQ", na=False)
            if mask_kq.any():
                kq_row = df[mask_kq].iloc[0]
                for c in sorted(df.columns):
                    d_val = None
                    if f"/{TARGET_MONTH}" in c: 
                        try: d_val = int(c.split("/")[0])
                        except: pass
                    elif c.isdigit() and 1 <= int(c) <= 31: d_val = int(c)
                    if d_val and 1 <= d_val <= 31:
                        val = str(kq_row[c])
                        nums = get_nums(val)
                        if nums: kq_db[d_val] = nums[0]
        except Exception: continue
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
        if s > 0: 
            col_scores[c] = s; valid_cols.append(c)

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

# --- MAIN ---
if uploaded_files:
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH"):
        data_cache, kq_db = load_data_from_upload(uploaded_files)
        st.success(f"✅ Đã tải {len(data_cache)} ngày.")
        
        start_hist = max(1, TARGET_DAY - ROLLING_WINDOW)
        end_hist = TARGET_DAY - 1
        groups = [f"{i}x" for i in range(10)]
        stats = {g: {'wins': 0, 'ranks': []} for g in groups}

        for d in range(start_hist, end_hist + 1):
            if d not in data_cache or d not in kq_db: continue
            df = data_cache[d]
            prev = d - 1
            raw_patterns = [str(prev), f"{prev:02d}", f"{prev}/{TARGET_MONTH}", f"{prev:02d}/{TARGET_MONTH}"]
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
        st.info(f"🏆 TOP 6: {', '.join(top6)}")
        
        limit_map = {top6[0]: 80, top6[1]: 80, top6[2]: 65, top6[3]: 65, top6[4]: 60, top6[5]: 60}
        alliance_1 = [top6[0], top6[5], top6[3]]
        alliance_2 = [top6[1], top6[4], top6[2]]
        
        if TARGET_DAY in data_cache:
            df_target = data_cache[TARGET_DAY]
            prev = TARGET_DAY - 1
            raw_patterns = [str(prev), f"{prev:02d}", f"{prev}/{TARGET_MONTH}", f"{prev:02d}/{TARGET_MONTH}"]
            patterns = [clean_text(p) for p in raw_patterns]
            grp_col_target = None
            for c in sorted(df_target.columns):
                if c in patterns: grp_col_target = c; break
            
            if grp_col_target:
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
                
                st.success(f"🎯 KẾT QUẢ: {len(final_result)} SỐ")
                st.text_area("Kết quả:", ",".join(final_result))
                
                csv = pd.DataFrame(final_result, columns=["So"]).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải về CSV", csv, "ketqua.csv", "text/csv")
            else: st.error(f"Không tìm thấy cột ngày {prev} trong file ngày {TARGET_DAY}")
        else: st.error(f"Thiếu file ngày {TARGET_DAY}")
else:
    st.info("👈 Vui lòng tải file ở thanh bên trái!")
