import streamlit as st
import pandas as pd
import re
from collections import Counter
import datetime
from datetime import timedelta

# --- CẤU HÌNH ---
st.set_page_config(page_title="Xổ Số V22 (Chuẩn Logic)", page_icon="💎", layout="centered")
st.title("💎 V22: Logic Vote Chuẩn + Fix Ngày")

# --- 1. TẢI FILE ---
uploaded_files = st.file_uploader("Tải file Excel (T12, T1...):", type=['xlsx'], accept_multiple_files=True)

# --- CẤU HÌNH BÊN ---
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ROLLING_WINDOW = st.number_input("Chu kỳ xét (Ngày)", min_value=1, value=10)

# --- HÀM CORE ---
SCORE_MAPPING = {
    'M10': 50, 'M9': 25, 'M8': 15, 'M7': 7, 'M6': 6, 'M5': 5,
    'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, 'M0': 0
}

def get_nums(s):
    if pd.isna(s): return []
    raw_nums = re.findall(r'\d+', str(s))
    return [n.zfill(2) for n in raw_nums if len(n) == 2]

def get_col_score(col_name):
    clean = re.sub(r'[^A-Z0-9]', '', str(col_name).upper())
    if 'M10' in clean: return 50 
    for key, score in SCORE_MAPPING.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: continue
            if key == 'M0' and 'M10' in clean: continue
            return score
    return 0

# --- XỬ LÝ NGÀY THÁNG ---
def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper()
    # 1. Dạng YYYY-MM-DD
    match_iso = re.search(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        # Fix lỗi đảo ngày tháng (nếu có)
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)
    # 2. Dạng DD/MM
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})', s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        curr_y = f_y
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: pass
    return None

def get_file_meta(filename):
    y_match = re.search(r'20\d{2}', filename)
    y = int(y_match.group(0)) if y_match else 2025
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d+)', filename, re.IGNORECASE)
    m = int(m_match.group(1)) if m_match else 1
    return m, y

def get_sheet_date(sheet_name, f_m, f_y):
    s_clean = re.sub(r'[^0-9]', ' ', sheet_name).strip()
    try:
        parts = [int(x) for x in s_clean.split()]
        if not parts: return None
        d = parts[0]
        m, y = f_m, f_y
        if len(parts) >= 3 and parts[2] > 2000: y = parts[2]; m = parts[1]
        return datetime.date(y, m, d)
    except: return None

@st.cache_data(ttl=600)
def load_data_v22(files):
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
                    
                    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
                    h_row = 3
                    for idx, row in preview.iterrows():
                        if "TV TOP" in str(row.values).upper() or "THÀNH VIÊN" in str(row.values).upper():
                            h_row = idx; break
                    df = pd.read_excel(xls, sheet_name=sheet, header=h_row)
                    
                    hist_map = {}
                    for col in df.columns:
                        d_obj = parse_date_smart(col, f_m, f_y)
                        if d_obj: hist_map[d_obj] = col
                    
                    data_map = {}
                    for col in df.columns:
                        c_upper = str(col).strip().upper()
                        if re.match(r'^\d+X$', c_upper):
                            data_map[c_upper.lower()] = col

                    kq_row = None
                    for idx, row in df.iterrows():
                        if str(row.values[0]).strip().upper() == "KQ": kq_row = row; break
                    if kq_row is not None:
                        for d_val, c_name in hist_map.items():
                            val = str(kq_row[c_name])
                            nums = get_nums(val)
                            if nums: kq_db[d_val] = nums[0]

                    cache[t_date] = {'df': df, 'hist_map': hist_map, 'data_map': data_map}
                except: continue
        except: continue
    return cache, kq_db

# --- HÀM TÍNH TOÁN CORE V22 ---
def calculate_v22(target_date, rolling_window, cache, kq_db):
    if target_date not in cache: return [], [], None, "Chưa có Sheet dữ liệu."

    curr_data = cache[target_date]
    df = curr_data['df']
    data_map = curr_data['data_map']
    
    # 1. Tìm cột hôm qua (để phân nhóm)
    prev_date = target_date - timedelta(days=1)
    col_hist_used = curr_data['hist_map'].get(prev_date)
    df_source = df
    
    if not col_hist_used and prev_date in cache:
        col_hist_used = cache[prev_date]['hist_map'].get(prev_date)
        df_source = cache[prev_date]['df']
        
    if not col_hist_used:
        return [], [], None, f"Không tìm thấy cột dữ liệu ngày {prev_date.strftime('%d/%m')}."

    # 2. Xếp hạng Group (Backtest Top 6)
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    score_cols = {c: get_col_score(c) for c in df.columns if get_col_score(c) > 0}

    past_dates = [target_date - timedelta(days=i) for i in range(1, rolling_window + 1)]
    for d in past_dates:
        if d not in kq_db or d not in cache: continue
        # Ưu tiên lấy dữ liệu từ chính sheet ngày d
        d_df = cache[d]['df']
        d_hist_col = cache[d]['hist_map'].get(d)
        if not d_hist_col: continue
        kq = kq_db[d]
        
        for g in groups:
            mask = d_df[d_hist_col].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper())) == g.upper()
            mems = d_df[mask]
            if mems.empty: stats[g]['ranks'].append(999); continue
            
            col_data_name = cache[d]['data_map'].get(g)
            if not col_data_name: continue
            
            # --- LOGIC VOTE CHUẨN V3: MỖI NGƯỜI 1 VÉ ---
            num_stats = {} # {num: {'score':0, 'votes':0}}
            for _, r in mems.iterrows():
                person_votes = set() # Số mà người này đã vote
                for sc_col, pts in score_cols.items():
                    for n in get_nums(r[sc_col]):
                        if n not in num_stats: num_stats[n] = {'score':0, 'votes':0}
                        num_stats[n]['score'] += pts
                        # Chỉ cộng vote nếu người này chưa vote số n
                        if n not in person_votes:
                            num_stats[n]['votes'] += 1
                            person_votes.add(n)
            
            # Sort: Điểm > Vote > Bé
            sorted_nums = sorted(num_stats.keys(), key=lambda n: (-num_stats[n]['score'], -num_stats[n]['votes'], int(n)))
            top80 = sorted_nums[:80]
            
            if kq in top80:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top80.index(kq) + 1)
            else: stats[g]['ranks'].append(999)

    final = []
    for g, inf in stats.items(): final.append((g, -inf['wins'], sum(inf['ranks'])))
    final.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final[:6]]

    # 3. DỰ ĐOÁN
    limits = {
        top6[0]: 80, top6[1]: 80, 
        top6[2]: 65, top6[3]: 65, 
        top6[4]: 60, top6[5]: 60
    }

    def get_alliance_set(alliance_groups):
        alliance_pool = set()
        for g in alliance_groups:
            col_data = data_map.get(g)
            if not col_data: continue

            hist_series = df_source[col_hist_used].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper()))
            L = min(len(df), len(hist_series))
            mask = hist_series.iloc[:L] == g.upper()
            valid_mems = df.iloc[:L][mask.values]
            
            # --- LOGIC VOTE CHUẨN V3 ---
            local_stats = {}
            for _, r in valid_mems.iterrows():
                person_votes = set()
                for sc_col, pts in score_cols.items():
                    for n in get_nums(r[sc_col]):
                        if n not in local_stats: local_stats[n] = {'score':0, 'votes':0}
                        local_stats[n]['score'] += pts
                        if n not in person_votes:
                            local_stats[n]['votes'] += 1
                            person_votes.add(n)
            
            sorted_nums = sorted(local_stats.keys(), key=lambda n: (-local_stats[n]['score'], -local_stats[n]['votes'], int(n)))
            limit = limits.get(g, 60)
            cut_nums = sorted_nums[:limit]
            
            alliance_pool.update(cut_nums)
            
        return alliance_pool

    s1 = get_alliance_set([top6[0], top6[5], top6[3]])
    s2 = get_alliance_set([top6[1], top6[4], top6[2]])
    
    final_result = sorted(list(s1.intersection(s2)))
    return top6, final_result, f"Cột {col_hist_used}", None

# --- UI ---
if uploaded_files:
    data_cache, kq_db = load_data_v22(uploaded_files)
    st.success(f"Đã đọc {len(data_cache)} ngày.")
    
    tab1, tab2 = st.tabs(["DỰ ĐOÁN", "BACKTEST"])
    with tab1:
        last_d = max(data_cache.keys()) if data_cache else datetime.date.today()
        target = st.date_input("Ngày:", value=last_d)
        if st.button("PHÂN TÍCH"):
            top6, res, src, err = calculate_v22(target, ROLLING_WINDOW, data_cache, kq_db)
            if err: st.error(err)
            else:
                st.info(f"Dữ liệu phân nhóm: {src}")
                st.success(f"TOP 6: {', '.join(top6)}")
                st.text_area(f"KẾT QUẢ ({len(res)} số):", ",".join(res))
                if target in kq_db:
                    real = kq_db[target]
                    st.write(f"KQ Thực: {real} - {'WIN 🎉' if real in res else 'MISS ❌'}")

    with tab2:
        c1, c2 = st.columns(2)
        with c1: start = st.date_input("Từ:", value=last_d - timedelta(days=3))
        with c2: end = st.date_input("Đến:", value=last_d)
        if st.button("CHẠY BACKTEST"):
            logs = []
            delta = (end - start).days
            bar = st.progress(0)
            for i in range(delta + 1):
                d = start + timedelta(days=i)
                bar.progress((i+1)/(delta+1))
                try:
                    _, res, _, err = calculate_v22(d, ROLLING_WINDOW, data_cache, kq_db)
                    if err: continue
                    real = kq_db.get(d, "N/A")
                    stt = "WIN" if real in res else "LOSS"
                    logs.append({"Ngày": d.strftime("%d/%m"), "KQ": real, "TT": stt, "Số": len(res)})
                except: pass
            bar.empty()
            st.dataframe(pd.DataFrame(logs))
