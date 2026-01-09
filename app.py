
import streamlit as st
import pandas as pd
import re
import datetime
import time
import json
import os
from datetime import timedelta
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & PRESETS (GIỮ NGUYÊN)
# ==============================================================================

st.set_page_config(
    page_title="Quang Pro V56 - Adaptive Engine", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Quang Handsome: V56 Elite Hunter (Adaptive)")
st.caption("🚀 Adaptive Weighting α = 0.6 | Áp cho Dự đoán + Backtest | Giữ nguyên Engine gốc")

CONFIG_FILE = 'config.json'
ALPHA = 0.6   # hệ số Adaptive – ĐÃ CHỐT

# --- CÁC CẤU HÌNH MẪU (PRESETS) ---
SCORES_PRESETS = {
    "Hard Core (Khuyên dùng)": { 
        "STD": [0, 0, 5, 10, 15, 25, 30, 35, 40, 50, 60], 
        "MOD": [0, 5, 10, 20, 25, 45, 50, 40, 30, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88}
    },
    "CH1: Bám Đuôi (An Toàn)": { 
        "STD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70], 
        "MOD": [10, 20, 30, 30, 30, 30, 40, 40, 50, 50, 70],
        "LIMITS": {'l12': 80, 'l34': 75, 'l56': 60, 'mod': 88}
    },
    "Gốc (V24 Standard)": {
        "STD": [0, 1, 2, 3, 4, 5, 6, 7, 15, 25, 50],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40],
        "LIMITS": {'l12': 82, 'l34': 76, 'l56': 70, 'mod': 88}
    },
    "Miền Nam (Experimental)": {
        "STD": [60, 8, 9, 10, 10, 30, 70, 30, 30, 30, 30],
        "MOD": [0, 5, 10, 15, 30, 30, 50, 35, 25, 25, 40],
        "LIMITS": {'l12': 85, 'l34': 80, 'l56': 75, 'mod': 90}
    }
}

# Regex & Sets (GIỮ NGUYÊN)
RE_NUMS = re.compile(r'\d+')
RE_CLEAN_SCORE = re.compile(r'[^A-Z0-9]')
RE_ISO_DATE = re.compile(r'(20\d{2})[\.\-/](\d{1,2})[\.\-/](\d{1,2})')
RE_SLASH_DATE = re.compile(r'(\d{1,2})[\.\-/](\d{1,2})')
BAD_KEYWORDS = frozenset(['N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'NGHỈ', 'LỖI'])

# ==============================================================================
# 2. CORE FUNCTIONS (HELPER) – GIỮ NGUYÊN
# ==============================================================================

@lru_cache(maxsize=10000)
def get_nums(s):
    if pd.isna(s): 
        return []
    s_str = str(s).strip()
    if not s_str: 
        return []
    s_upper = s_str.upper()
    if any(kw in s_upper for kw in BAD_KEYWORDS): 
        return []
    raw_nums = RE_NUMS.findall(s_upper)
    return [n.zfill(2) for n in raw_nums if len(n) <= 2]

@lru_cache(maxsize=1000)
def get_col_score(col_name, mapping_tuple):
    clean = RE_CLEAN_SCORE.sub('', str(col_name).upper().replace(' ', ''))
    mapping = dict(mapping_tuple)
    if 'M10' in clean: 
        return mapping.get('M10', 0)
    for key, score in mapping.items():
        if key in clean:
            if key == 'M1' and 'M10' in clean: 
                continue
            if key == 'M0' and 'M10' in clean: 
                continue
            return score
    return 0
# ==============================================================================
# 3. PARSE DATE – META – HEADER (ENGINE GỐC)
# ==============================================================================

def extract_meta_from_filename(filename):
    name = filename.upper()
    y_match = re.search(r'202[0-9]', name)
    y = int(y_match.group(0)) if y_match else datetime.datetime.now().year

    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', name)
    m = int(m_match.group(1)) if m_match else None

    return m, y


def parse_date_smart(col, f_month, f_year):
    s = str(col).upper().replace('NGÀY', '').replace('NGAY', '').strip()

    m1 = RE_ISO_DATE.search(s)
    if m1:
        y, a, b = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        try:
            return datetime.date(y, a, b)
        except:
            try:
                return datetime.date(y, b, a)
            except:
                return None

    m2 = RE_SLASH_DATE.search(s)
    if m2:
        d, m = int(m2.group(1)), int(m2.group(2))
        if not f_year:
            return None
        try:
            return datetime.date(f_year, m, d)
        except:
            return None

    return None


def find_header_row(df_preview):
    keywords = ['STT', 'MEMBER', 'THÀNH VIÊN', 'TV TOP', 'DANH SÁCH']
    for i in range(min(25, len(df_preview))):
        row = ' '.join([str(x).upper() for x in df_preview.iloc[i].values])
        if any(k in row for k in keywords):
            return i
    return 3


# ==============================================================================
# 4. LOAD DATA ENGINE (FIX st.cache_data → st.cache)
# ==============================================================================

@st.cache
def load_data_v24(files):
    cache = {}
    kq_db = {}
    err_logs = []

    for file in files:
        try:
            f_month, f_year = extract_meta_from_filename(file.name)
            xls = pd.ExcelFile(file, engine='openpyxl')

            for sheet in xls.sheet_names:
                preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=30)
                h_row = find_header_row(preview)
                df = pd.read_excel(xls, sheet_name=sheet, header=h_row)

                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]

                hist_map = {}
                for col in df.columns:
                    d_obj = parse_date_smart(col, f_month, f_year)
                    if d_obj:
                        hist_map[d_obj] = col

                # ---- TÌM DÒNG KẾT QUẢ (KQ) ----
                kq_row = None
                for idx in range(min(2, len(df.columns))):
                    col_check = df.columns[idx]
                    try:
                        mask = df[col_check].astype(str).str.upper().str.contains('KQ|KẾT QUẢ')
                        if mask.any():
                            kq_row = df[mask].iloc[0]
                            break
                    except:
                        continue

                if kq_row is not None:
                    for d_val, c_name in hist_map.items():
                        nums = get_nums(str(kq_row[c_name]))
                        if nums:
                            kq_db[d_val] = nums[0]

                for d_val, c_name in hist_map.items():
                    cache[d_val] = {
                        'df': df,
                        'hist_col': c_name
                    }

        except Exception as e:
            err_logs.append(f"Lỗi file {file.name}: {e}")

    return cache, kq_db, err_logs
# ==============================================================================
# 5. ADAPTIVE WINRATE ENGINE (MỚI – ÁP CHO TOÀN BỘ APP)
# ==============================================================================

def calc_winrate_M(cache, kq_db, rolling_days):
    """
    Tính winrate cho từng cột M dựa trên rolling_days gần nhất
    Dùng cho:
    - Dự đoán hằng ngày
    - Backtest
    """
    stats = defaultdict(lambda: {'hit': 0, 'total': 0})

    all_days = sorted(kq_db.keys())
    recent_days = all_days[-rolling_days:] if rolling_days < len(all_days) else all_days

    for d in recent_days:
        if d not in cache:
            continue

        df = cache[d]['df']
        hist_col = cache[d]['hist_col']
        kq = kq_db.get(d)
        if not kq:
            continue

        for _, row in df.iterrows():
            for col in df.columns:
                if not col.startswith('M'):
                    continue
                nums = get_nums(row[col])
                if not nums:
                    continue
                stats[col]['total'] += 1
                if kq in nums:
                    stats[col]['hit'] += 1

    # tránh chia 0 + tránh triệt tiêu
    winrate = {}
    for m, v in stats.items():
        if v['total'] > 0:
            winrate[m] = v['hit'] / v['total']
        else:
            winrate[m] = 0.05  # floor an toàn

    return winrate


# ==============================================================================
# 6. CORE SCORING ENGINE (GIỮ NGUYÊN LOGIC – CHỈ GẮN ADAPTIVE)
# ==============================================================================

def build_score_maps(df, score_std, score_mod, winrate_M, use_adaptive=True):
    """
    Build p_map_dict và s_map_dict
    Nếu use_adaptive=True → áp Adaptive
    """
    p_map = {}
    s_map = {}

    std_tuple = tuple(score_std.items())
    mod_tuple = tuple(score_mod.items())

    for col in df.columns:
        base_p = get_col_score(col, std_tuple)
        base_s = get_col_score(col, mod_tuple)

        if base_p <= 0 and base_s <= 0:
            continue

        if use_adaptive:
            factor = (winrate_M.get(col, 0.05)) ** ALPHA
        else:
            factor = 1.0

        if base_p > 0:
            p_map[col] = base_p * factor
        if base_s > 0:
            s_map[col] = base_s * factor

    return p_map, s_map


def fast_rank_nums(df, p_map_dict, limit=60):
    counter = Counter()
    for _, row in df.iterrows():
        for col, w in p_map_dict.items():
            for n in get_nums(row[col]):
                counter[n] += w
    return [n for n, _ in counter.most_common(limit)]


def calculate_v24_logic_only(
    target_date,
    cache,
    kq_db,
    score_std,
    score_mod,
    winrate_M,
    use_adaptive=True,
    limit=60
):
    """
    CORE LOGIC – dùng cho:
    - Dự đoán
    - Backtest
    """
    if target_date not in cache:
        return []

    df = cache[target_date]['df']
    hist_col = cache[target_date]['hist_col']

    p_map_dict, s_map_dict = build_score_maps(
        df, score_std, score_mod, winrate_M, use_adaptive
    )

    # lọc member hợp lệ theo cột lịch sử
    hist_series = df[hist_col].astype(str)
    valid_df = df[hist_series.notna()]

    # Gốc
    res_p = fast_rank_nums(valid_df, p_map_dict, limit)
    # Mod
    res_s = fast_rank_nums(valid_df, s_map_dict, limit)

    # Gộp Hybrid (GIỮ NGUYÊN)
    final = list(dict.fromkeys(res_p + res_s))

    return final[:limit]
# ==============================================================================
# 7. UI – SIDEBAR & CẤU HÌNH CHẠY
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Cấu hình chạy")

    preset_name = st.selectbox(
        "🎯 Preset",
        list(SCORES_PRESETS.keys()),
        index=0
    )

    rolling_days = st.number_input(
        "📅 Rolling window (ngày)",
        min_value=3,
        max_value=30,
        value=10
    )

    limit_top = st.number_input(
        "🔢 Số lượng số lấy",
        min_value=30,
        max_value=99,
        value=60
    )

    st.markdown("---")
    st.caption("🧠 Adaptive đang BẬT cho toàn bộ app")
    st.caption("Công thức: score × (winrate ^ 0.6)")

# ==============================================================================
# 8. MAIN UI – LOAD DATA
# ==============================================================================

uploaded_files = st.file_uploader(
    "📂 Upload file Excel (giữ nguyên cấu trúc cũ)",
    type=["xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Đang đọc dữ liệu..."):
        cache, kq_db, err_logs = load_data_v24(uploaded_files)

    if err_logs:
        for e in err_logs:
            st.error(e)

    if not cache or not kq_db:
        st.error("❌ Không đọc được dữ liệu hợp lệ.")
        st.stop()

    all_days = sorted(kq_db.keys())
    last_day = all_days[-1]

    st.success(f"✅ Đã đọc {len(all_days)} ngày dữ liệu.")

    # ==============================================================================
    # 9. DỰ ĐOÁN HẰNG NGÀY (ADAPTIVE)
    # ==============================================================================

    st.subheader("🎯 Dự đoán hằng ngày (Adaptive)")

    selected_day = st.selectbox(
        "📅 Chọn ngày dự đoán",
        options=all_days,
        index=len(all_days) - 1
    )

    preset = SCORES_PRESETS[preset_name]
    score_std = {f"M{i}": v for i, v in enumerate(preset["STD"])}
    score_mod = {f"M{i}": v for i, v in enumerate(preset["MOD"])}

    winrate_M = calc_winrate_M(cache, kq_db, rolling_days)

    result_nums = calculate_v24_logic_only(
        target_date=selected_day,
        cache=cache,
        kq_db=kq_db,
        score_std=score_std,
        score_mod=score_mod,
        winrate_M=winrate_M,
        use_adaptive=True,
        limit=limit_top
    )

    st.write("### ✅ Kết quả đề xuất")
    st.write(", ".join(result_nums))

    st.markdown("---")

    # ==============================================================================
    # 10. HIỂN THỊ KẾT QUẢ THỰC TẾ (NẾU CÓ)
    # ==============================================================================

    real_kq = kq_db.get(selected_day)
    if real_kq:
        st.info(f"🎯 Kết quả thực tế ngày {selected_day.strftime('%d/%m')}: **{real_kq}**")
# ==============================================================================
# 11. BACKTEST – SO SÁNH STATIC vs ADAPTIVE (CÙNG ENGINE)
# ==============================================================================

    st.subheader("🔙 Backtest – So sánh Static vs Adaptive")

    d_start, d_end = st.select_slider(
        "📅 Chọn khoảng ngày backtest",
        options=all_days,
        value=(all_days[0], all_days[-1])
    )

    if st.button("🚀 CHẠY BACKTEST"):
        logs = []

        progress = st.progress(0)
        total = len(all_days)

        for idx, d in enumerate(all_days):
            progress.progress((idx + 1) / total)

            if d < d_start or d > d_end:
                continue

            real = kq_db.get(d)
            if not real:
                continue

            # --- STATIC (không adaptive) ---
            res_static = calculate_v24_logic_only(
                target_date=d,
                cache=cache,
                kq_db=kq_db,
                score_std=score_std,
                score_mod=score_mod,
                winrate_M=winrate_M,
                use_adaptive=False,
                limit=limit_top
            )

            # --- ADAPTIVE ---
            res_adapt = calculate_v24_logic_only(
                target_date=d,
                cache=cache,
                kq_db=kq_db,
                score_std=score_std,
                score_mod=score_mod,
                winrate_M=winrate_M,
                use_adaptive=True,
                limit=limit_top
            )

            logs.append({
                "Ngày": d.strftime("%d/%m"),
                "KQ": real,
                "Static": "WIN" if real in res_static else "MISS",
                "Adaptive": "WIN" if real in res_adapt else "MISS"
            })

        progress.empty()

        if logs:
            df_log = pd.DataFrame(logs)
            st.dataframe(df_log, use_container_width=True, height=600)

            st.markdown("### 📊 Tổng kết")

            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Static",
                    f\"{(df_log['Static']=='WIN').sum()}/{len(df_log)}\",
                    f\"{(df_log['Static']=='WIN').mean()*100:.1f}%\"
                )
            with c2:
                st.metric(
                    "Adaptive",
                    f\"{(df_log['Adaptive']=='WIN').sum()}/{len(df_log)}\",
                    f\"{(df_log['Adaptive']=='WIN').mean()*100:.1f}%\"
                )
# ==============================================================================
# 12. FOOTER – THÔNG TIN & KẾT THÚC
# ==============================================================================

st.markdown("---")
st.caption(
    "🧠 Quang Handsome – V56 Elite Hunter | "
    "Adaptive Weighting α = 0.6 | "
    "Áp cho Dự đoán + Backtest | "
    "Engine gốc + nâng cấp"
)

# ==============================================================================
# 13. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Streamlit tự quản lý vòng đời app
    pass
