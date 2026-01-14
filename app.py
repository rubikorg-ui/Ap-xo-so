import streamlit as st
import pandas as pd
import re
import datetime
import time
import json
import os
from datetime import timedelta
from collections import Counter
from functools import lru_cache
import numpy as np

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="V62 Vote 8x Pro", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.title("🛡️ V24 - CHIẾN THUẬT VOTE 8X (CHUẨN)")
st.caption("🚀 Logic: Nguồn 8X -> Top 6 Nhóm -> 2 Liên Minh Giao Thoa (Fix chuẩn 63 số)")

CONFIG_FILE = 'config.json'

SCORES_PRESETS = {
    "Vote 8x Standard": { 
        "STD": [0]*11, "MOD": [0]*11, # 8x không dùng điểm M
        "LIMITS": {'l12': 80, 'l34': 70, 'l56': 60, 'mod': 80}, # Cấu hình chuẩn ra 63 số
        "ROLLING": 10
    }
}

RE_NUMS = re.compile(r'\d+')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CORE FUNCTIONS (XỬ LÝ DỮ LIỆU)
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    if pd.isna(s): return []
    s_str = str(s).strip()
    if not s_str: return []
    raw_nums = RE_NUMS.findall(s_str)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper().replace('NGAY', '').replace('NGÀY', '').strip()
    match_iso = RE_ISO_DATE.search(s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 != f_m and p2 == f_m: return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)
    match_slash = RE_SLASH_DATE.search(s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        if m < 1 or m > 12 or d < 1 or d > 31: return None
        curr_y = f_y
        if m == 12 and f_m == 1: curr_y -= 1
        elif m == 1 and f_m == 12: curr_y += 1
        try: return datetime.date(curr_y, m, d)
        except: return None
    return None

def extract_meta_from_filename(filename):
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    full_date_match = re.search(r'(\d{1,2})[\.\-](\d{1,2})(?:[\.\-]20\d{2})?', clean_name)
    if full_date_match:
        try:
            d = int(full_date_match.group(1)); m = int(full_date_match.group(2))
            y = int(full_date_match.group(3)) if full_date_match.lastindex >= 3 else y_global
            if m == 12 and m_global == 1: y -= 1 
            elif m == 1 and m_global == 12: y += 1
            return m, y, datetime.date(y, m, d)
        except: pass
    return m_global, y_global, None

def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords): return idx
    return 3

# --- HÀM LẤY SỐ THEO VOTE (Core của 8x) ---
def get_top_nums_by_vote(df_members, col_name, limit):
    if df_members.empty: return []
    all_nums = []
    # Chỉ lấy cột 8x
    vals = df_members[col_name].dropna().astype(str).tolist()
    for val in vals:
        if any(kw in val.upper() for kw in BAD_KEYWORDS): continue
        all_nums.extend(get_nums(val))
    
    counts = Counter(all_nums)
    # Sort: Vote cao -> thấp, Số bé -> lớn
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], int(x[0])))
    return [n for n, c in sorted_items[:int(limit)]]

# --- HÀM XỬ LÝ LOGIC 8X CHUYÊN BIỆT (ĐỂ KHÔNG BỊ SAI) ---
def calculate_vote_8x_strict(target_date, rolling_window, _cache, _kq_db, limits_config):
    if target_date not in _cache: return None, "Không có dữ liệu ngày này"
    curr_data = _cache[target_date]; df = curr_data['df']
    
    # 1. Tìm cột 8X
    col_8x = next((c for c in df.columns if re.match(r'^(8X|80|DÀN|DAN)$', c.strip().upper()) or '8X' in c.strip().upper()), None)
    if not col_8x: return None, "Không tìm thấy cột 8X trong file"

    # 2. Tìm cột Phân Nhóm (Lịch sử hôm qua)
    prev_date = target_date - timedelta(days=1)
    if prev_date not in _cache:
        # Thử lùi thêm nếu dính ngày nghỉ
        for i in range(2, 4):
            if (target_date - timedelta(days=i)) in _cache: prev_date = target_date - timedelta(days=i); break
    
    col_group = curr_data['hist_map'].get(prev_date)
    if not col_group and prev_date in _cache:
        col_group = _cache[prev_date]['hist_map'].get(prev_date)
    
    if not col_group: return None, f"Không tìm thấy cột phân nhóm ngày {prev_date}"

    # 3. BACKTEST: Tìm Top 6 Nhóm Mạnh Nhất
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}
    
    past_dates = []
    check_d = target_date - timedelta(days=1)
    while len(past_dates) < rolling_window:
        if check_d in _cache and check_d in _kq_db: past_dates.append(check_d)
        check_d -= timedelta(days=1)
        if (target_date - check_d).days > 60: break
    
    for d in past_dates:
        d_df = _cache[d]['df']; kq = _kq_db[d]
        # Tìm cột 8x quá khứ
        d_c8 = next((c for c in d_df.columns if '8X' in c.upper()), None)
        # Tìm cột nhóm quá khứ (ngày trước đó của d)
        d_prev = d - timedelta(days=1) # Đơn giản hóa, thực tế cần check map
        sorted_dates = sorted([k for k in _cache[d]['hist_map'].keys() if k < d], reverse=True)
        d_c_grp = _cache[d]['hist_map'].get(sorted_dates[0]) if sorted_dates else None

        if d_c8 and d_c_grp:
            try:
                # Chuẩn hóa cột nhóm
                grp_series = d_df[d_c_grp].astype(str).str.upper().str.replace('S', '6').str.replace(r'[^0-9X]', '', regex=True)
                for g in groups:
                    mems = d_df[grp_series == g.upper()]
                    # Cắt cứng 80 số để so sánh công bằng
                    top80 = get_top_nums_by_vote(mems, d_c8, 80)
                    if kq in top80:
                        stats[g]['wins'] += 1
                        stats[g]['ranks'].append(top80.index(kq))
                    else:
                        stats[g]['ranks'].append(999)
            except: continue

    # Xếp hạng nhóm
    final_rank = []
    for g, inf in stats.items():
        final_rank.append((g, -inf['wins'], sum(inf['ranks'])))
    final_rank.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_rank[:6]]

    # 4. FINAL CUT: Lấy số cho ngày hiện tại (Logic Liên Minh)
    # Chuẩn hóa cột nhóm hiện tại
    hist_series = df[col_group].astype(str).str.upper().str.replace('S', '6').str.replace(r'[^0-9X]', '', regex=True)

    def get_pool(group_names, limit_val):
        pool = []
        for g in group_names:
            mems = df[hist_series == g.upper()]
            nums = get_top_nums_by_vote(mems, col_8x, limit_val)
            pool.extend(nums)
        # Trả về các số xuất hiện >= 2 lần
        counts = Counter(pool)
        return {n for n, c in counts.items() if c >= 2}

    # Cấu hình limit theo yêu cầu:
    # Top 1, 2: L12 (80)
    # Top 3, 4: L34 (70)
    # Top 5, 6: L56 (60)
    
    # LIÊN MINH 1: Top 1, 5, 3
    # Top 1 (idx 0) -> L12
    # Top 5 (idx 4) -> L56
    # Top 3 (idx 2) -> L34
    
    pool1_top1 = get_top_nums_by_vote(df[hist_series == top6[0].upper()], col_8x, limits_config['l12'])
    pool1_top5 = get_top_nums_by_vote(df[hist_series == top6[4].upper()], col_8x, limits_config['l56'])
    pool1_top3 = get_top_nums_by_vote(df[hist_series == top6[2].upper()], col_8x, limits_config['l34'])
    
    raw_pool1 = pool1_top1 + pool1_top5 + pool1_top3
    s1 = {n for n, c in Counter(raw_pool1).items() if c >= 2}

    # LIÊN MINH 2: Top 2, 4, 6
    # Top 2 (idx 1) -> L12
    # Top 4 (idx 3) -> L34
    # Top 6 (idx 5) -> L56
    
    pool2_top2 = get_top_nums_by_vote(df[hist_series == top6[1].upper()], col_8x, limits_config['l12'])
    pool2_top4 = get_top_nums_by_vote(df[hist_series == top6[3].upper()], col_8x, limits_config['l34'])
    pool2_top6 = get_top_nums_by_vote(df[hist_series == top6[5].upper()], col_8x, limits_config['l56'])
    
    raw_pool2 = pool2_top2 + pool2_top4 + pool2_top6
    s2 = {n for n, c in Counter(raw_pool2).items() if c >= 2}

    # GIAO THOA (CHỈ GIỮ SỐ TRÙNG CỦA 2 LIÊN MINH)
    final_dan = sorted(list(s1.intersection(s2)))

    return {
        "top6_std": top6,
        "dan_goc": final_dan,
        "dan_final": final_dan, # Bỏ qua mod, lấy luôn giao thoa
        "source_col": col_group,
        "debug_s1": len(s1),
        "debug_s2": len(s2)
    }, None
# ==============================================================================
# 3. HÀM LOAD DỮ LIỆU (ĐỌC FILE EXCEL/CSV)
# ==============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_data_v24(files):
    cache = {}; kq_db = {}; err_logs = []; file_status = []
    files = sorted(files, key=lambda x: x.name)
    
    for file in files:
        if file.name.upper().startswith('~$') or 'N.CSV' in file.name.upper(): continue
        f_m, f_y, date_from_name = extract_meta_from_filename(file.name)
        
        try:
            dfs_to_process = []
            # Xử lý Excel
            if file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(file, engine='openpyxl')
                for sheet in xls.sheet_names:
                    s_date = None
                    try:
                        clean_s = re.sub(r'[^0-9]', ' ', sheet).strip()
                        parts = [int(x) for x in clean_s.split()]
                        if parts:
                            d_s, m_s, y_s = parts[0], f_m, f_y
                            if len(parts) >= 3 and parts[2] > 2000: y_s = parts[2]; m_s = parts[1]
                            s_date = datetime.date(y_s, m_s, d_s)
                    except: pass
                    if not s_date: s_date = date_from_name
                    
                    if s_date:
                        preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=30, engine='openpyxl')
                        h_row = find_header_row(preview)
                        df = pd.read_excel(xls, sheet_name=sheet, header=h_row, engine='openpyxl')
                        dfs_to_process.append((s_date, df))
                file_status.append(f"✅ Excel: {file.name}")

            # Xử lý CSV
            elif file.name.endswith('.csv'):
                if not date_from_name: continue
                encodings_to_try = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'utf-16']
                df_raw = None; h_row = 0
                for enc in encodings_to_try:
                    try:
                        file.seek(0)
                        preview = pd.read_csv(file, header=None, nrows=30, encoding=enc)
                        h_row = find_header_row(preview)
                        file.seek(0)
                        df_raw = pd.read_csv(file, header=None, encoding=enc); break
                    except: continue
                
                if df_raw is None: err_logs.append(f"❌ Lỗi Encoding: {file.name}"); continue
                
                # Xử lý header
                df = df_raw.iloc[h_row+1:].copy()
                raw_cols = df_raw.iloc[h_row].astype(str).tolist()
                # Fix lỗi trùng tên cột
                seen = {}; final_cols = []
                for c in raw_cols:
                    c = str(c).strip().upper().replace('M 1 0', 'M10')
                    if c in seen: seen[c] += 1; final_cols.append(f"{c}.{seen[c]}")
                    else: seen[c] = 0; final_cols.append(c)
                df.columns = final_cols
                
                dfs_to_process.append((date_from_name, df))
                file_status.append(f"✅ CSV: {file.name}")

            # Xử lý Dataframe sau khi load
            for t_date, df in dfs_to_process:
                # Clean column names
                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]
                
                # Tìm cột chứa KQ (để lưu vào kq_db)
                hist_map = {}
                kq_row = None
                if not df.empty:
                    # Tìm dòng KQ
                    for c_idx in range(min(2, len(df.columns))):
                        col_check = df.columns[c_idx]
                        try:
                            mask_kq = df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')
                            if mask_kq.any(): kq_row = df[mask_kq].iloc[0]; break
                        except: continue
                
                # Map cột lịch sử
                for col in df.columns:
                    if "UNNAMED" in col or col.startswith("M") or col in ["STT", "SCORE_SORT"]: continue
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj: 
                        hist_map[d_obj] = col
                        # Lưu kết quả nếu có
                        if kq_row is not None:
                            try:
                                nums = get_nums(str(kq_row[col]))
                                if nums: kq_db[d_obj] = nums[0]
                            except: pass
                            
                cache[t_date] = {'df': df, 'hist_map': hist_map}
                
        except Exception as e: err_logs.append(f"Lỗi '{file.name}': {str(e)}"); continue
        
    return cache, kq_db, file_status, err_logs

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (MAIN APP)
# ==============================================================================

def main():
    # --- SIDEBAR CẤU HÌNH ---
    with st.sidebar:
        st.header("⚙️ Cấu Hình Vote 8x")
        
        # Load presets
        saved_cfg = None
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f: saved_cfg = json.load(f)
            
        # Default values
        def_l12 = 80; def_l34 = 70; def_l56 = 60
        if saved_cfg:
            def_l12 = saved_cfg.get('L12', 80)
            def_l34 = saved_cfg.get('L34', 70)
            def_l56 = saved_cfg.get('L56', 60)

        with st.expander("✂️ Cắt Số (Limits)", expanded=True):
            L12 = st.number_input("Top 1 & 2 lấy:", value=def_l12, step=1, key="L12")
            L34 = st.number_input("Top 3 & 4 lấy:", value=def_l34, step=1, key="L34")
            L56 = st.number_input("Top 5 & 6 lấy:", value=def_l56, step=1, key="L56")
            st.caption("Gợi ý chuẩn: 80 - 70 - 60")

        ROLLING_WINDOW = st.number_input("Backtest (Ngày)", value=10)
        
        if st.button("💾 LƯU CẤU HÌNH"):
            cfg = {'L12': L12, 'L34': L34, 'L56': L56}
            with open(CONFIG_FILE, 'w') as f: json.dump(cfg, f)
            st.success("Đã lưu!")

        st.divider()
        if st.button("🗑️ XÓA CACHE"): st.cache_data.clear(); st.rerun()

    # --- MAIN SCREEN ---
    uploaded_files = st.file_uploader("📂 Tải file CSV/Excel", type=['xlsx', 'csv'], accept_multiple_files=True)
    
    if uploaded_files:
        data_cache, kq_db, f_status, err_logs = load_data_v24(uploaded_files)
        
        # Show status
        with st.expander("Trạng thái File", expanded=False):
            for s in f_status: st.write(s)
            for e in err_logs: st.error(e)
            
        if data_cache:
            last_d = max(data_cache.keys())
            tab1, tab2 = st.tabs(["📊 SOI CẦU (PREDICT)", "🔙 KIỂM CHỨNG (BACKTEST)"])
            
            # === TAB 1: SOI CẦU ===
            with tab1:
                st.subheader(f"🛡️ Chiến Thuật: V24 Vote 8x (Chuẩn Giao Thoa)")
                col_d, col_b = st.columns([1, 2])
                with col_d:
                    target_date = st.date_input("Ngày soi:", value=last_d)
                
                if st.button("🚀 CHẠY PHÂN TÍCH NGAY", type="primary"):
                    limits = {'l12': L12, 'l34': L34, 'l56': L56, 'mod': 80}
                    res, err = calculate_vote_8x_strict(target_date, ROLLING_WINDOW, data_cache, kq_db, limits)
                    
                    if err:
                        st.error(err)
                    else:
                        st.success(f"✅ Đã phân tích xong ngày {target_date.strftime('%d/%m/%Y')}")
                        
                        # Hiển thị Top 6
                        st.info(f"🏆 Top 6 Nhóm Mạnh Nhất: {', '.join(res['top6_std'])}")
                        
                        # Hiển thị kết quả
                        st.divider()
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Giao Thoa 2 Liên Minh ({len(res['dan_goc'])})**")
                            st.code(",".join(res['dan_goc']), language="text")
                            st.caption(f"Chi tiết: LM1 ({res['debug_s1']} số) ∩ LM2 ({res['debug_s2']} số)")
                            
                        with c2:
                            st.write(f"**🛡️ DÀN FINAL (CHỐT) ({len(res['dan_final'])})**")
                            st.text_area("Copy dàn số:", ",".join(res['dan_final']), height=150)
                        
                        # Check KQ nếu có
                        if target_date in kq_db:
                            kq = kq_db[target_date]
                            st.markdown(f"### 🏁 KẾT QUẢ THỰC TẾ: `{kq}`")
                            if kq in res['dan_final']:
                                st.success(f"🎉 CHÚC MỪNG! Dàn Final ĐÃ ĂN đề {kq}")
                            else:
                                st.error(f"❌ Rất tiếc, Dàn Final xịt đề {kq}")

            # === TAB 2: BACKTEST ===
            with tab2:
                st.write("Kiểm tra hiệu quả chiến thuật trong quá khứ")
                c_b1, c_b2 = st.columns(2)
                with c_b1: d_start = st.date_input("Từ ngày:", value=last_d - timedelta(days=5))
                with c_b2: d_end = st.date_input("Đến ngày:", value=last_d)
                
                if st.button("▶️ CHẠY BACKTEST"):
                    logs = []
                    curr = d_start
                    bar = st.progress(0)
                    days_list = [d_start + timedelta(days=x) for x in range((d_end - d_start).days + 1)]
                    
                    limits = {'l12': L12, 'l34': L34, 'l56': L56, 'mod': 80}
                    
                    for i, d in enumerate(days_list):
                        bar.progress((i + 1) / len(days_list))
                        if d not in kq_db: continue
                        
                        res, err = calculate_vote_8x_strict(d, ROLLING_WINDOW, data_cache, kq_db, limits)
                        real_kq = kq_db[d]
                        
                        if res:
                            is_win = "✅ WIN" if real_kq in res['dan_final'] else "❌ MISS"
                            logs.append({
                                "Ngày": d.strftime("%d/%m"),
                                "KQ": real_kq,
                                "Kết quả": is_win,
                                "Số lượng": len(res['dan_final']),
                                "Top 1": res['top6_std'][0]
                            })
                    
                    if logs:
                        df_log = pd.DataFrame(logs)
                        st.dataframe(df_log, use_container_width=True)
                        wins = df_log[df_log['Kết quả'].str.contains("WIN")].shape[0]
                        st.metric("Tỷ lệ chiến thắng", f"{wins}/{len(df_log)} ({wins/len(df_log)*100:.1f}%)")
                    else:
                        st.warning("Không có dữ liệu kết quả để backtest trong khoảng này.")

if __name__ == "__main__":
    main()
