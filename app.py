# ==============================================================================
# 4. PARSE NGÀY & LOAD DATA (LOGIC V56 GỐC – GIỮ NGUYÊN)
# ==============================================================================
def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper()
    s = s.replace('NGAY', '').replace('NGÀY', '').strip()

    match_iso = RE_ISO_DATE.search(s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 != f_m and p2 == f_m:
            return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)

    match_slash = RE_SLASH_DATE.search(s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        if m < 1 or m > 12 or d < 1 or d > 31:
            return None
        curr_y = f_y
        try:
            return datetime.date(curr_y, m, d)
        except:
            return None
    return None


def extract_meta_from_filename(filename):
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    return m_global, y_global, None


def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords):
            return idx
    return 3


@st.cache_data(show_spinner=False)
def load_data_v56(files):
    cache = {}
    kq_db = {}
    err_logs = []

    for file in files:
        try:
            f_m, f_y, _ = extract_meta_from_filename(file.name)
            xls = pd.ExcelFile(file, engine='openpyxl')

            for sheet in xls.sheet_names:
                preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=30)
                h_row = find_header_row(preview)
                df = pd.read_excel(xls, sheet_name=sheet, header=h_row)

                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]

                hist_map = {}
                for col in df.columns:
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj:
                        hist_map[d_obj] = col

                # --- TÌM DÒNG KQ ---
                kq_row = None
                for c_idx in range(min(2, len(df.columns))):
                    col_check = df.columns[c_idx]
                    try:
                        mask = df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')
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

                for d_val in hist_map:
                    cache[d_val] = {
                        'df': df,
                        'hist_col': hist_map[d_val]
                    }

        except Exception as e:
            err_logs.append(f"Lỗi file {file.name}: {e}")

    return cache, kq_db, err_logs
# ==============================================================================
# 4. PARSE NGÀY & LOAD DATA (LOGIC V56 GỐC – GIỮ NGUYÊN)
# ==============================================================================
def parse_date_smart(col_str, f_m, f_y):
    s = str(col_str).strip().upper()
    s = s.replace('NGAY', '').replace('NGÀY', '').strip()

    match_iso = RE_ISO_DATE.search(s)
    if match_iso:
        y, p1, p2 = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if p1 != f_m and p2 == f_m:
            return datetime.date(y, p2, p1)
        return datetime.date(y, p1, p2)

    match_slash = RE_SLASH_DATE.search(s)
    if match_slash:
        d, m = int(match_slash.group(1)), int(match_slash.group(2))
        if m < 1 or m > 12 or d < 1 or d > 31:
            return None
        curr_y = f_y
        try:
            return datetime.date(curr_y, m, d)
        except:
            return None
    return None


def extract_meta_from_filename(filename):
    clean_name = filename.upper().replace(".CSV", "").replace(".XLSX", "")
    y_match = re.search(r'202[0-9]', clean_name)
    y_global = int(y_match.group(0)) if y_match else datetime.datetime.now().year
    m_match = re.search(r'(?:THANG|THÁNG|T)[^0-9]*(\d{1,2})', clean_name)
    m_global = int(m_match.group(1)) if m_match else 12
    return m_global, y_global, None


def find_header_row(df_preview):
    keywords = ["STT", "MEMBER", "THÀNH VIÊN", "TV TOP", "DANH SÁCH", "HỌ VÀ TÊN", "NICK"]
    for idx, row in df_preview.head(30).iterrows():
        row_str = str(row.values).upper()
        if any(k in row_str for k in keywords):
            return idx
    return 3


@st.cache_data(show_spinner=False)
def load_data_v56(files):
    cache = {}
    kq_db = {}
    err_logs = []

    for file in files:
        try:
            f_m, f_y, _ = extract_meta_from_filename(file.name)
            xls = pd.ExcelFile(file, engine='openpyxl')

            for sheet in xls.sheet_names:
                preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=30)
                h_row = find_header_row(preview)
                df = pd.read_excel(xls, sheet_name=sheet, header=h_row)

                df.columns = [str(c).strip().upper().replace('\ufeff', '') for c in df.columns]

                hist_map = {}
                for col in df.columns:
                    d_obj = parse_date_smart(col, f_m, f_y)
                    if d_obj:
                        hist_map[d_obj] = col

                # --- TÌM DÒNG KQ ---
                kq_row = None
                for c_idx in range(min(2, len(df.columns))):
                    col_check = df.columns[c_idx]
                    try:
                        mask = df[col_check].astype(str).str.upper().str.contains(r'KQ|KẾT QUẢ')
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

                for d_val in hist_map:
                    cache[d_val] = {
                        'df': df,
                        'hist_col': hist_map[d_val]
                    }

        except Exception as e:
            err_logs.append(f"Lỗi file {file.name}: {e}")

    return cache, kq_db, err_logs
# ==============================================================================
# 5. ADAPTIVE ENGINE – TÍNH WINRATE M (MỚI, KHÔNG PHÁ LOGIC CŨ)
# ==============================================================================

def calc_winrate_M(cache, kq_db, lookback_days=10):
    """
    Tính winrate cho từng cột M dựa trên lookback_days gần nhất
    Dùng CHUNG cho toàn app khi bật Adaptive
    """
    stats = defaultdict(lambda: {'hit': 0, 'total': 0})

    sorted_days = sorted(kq_db.keys())
    recent_days = sorted_days[-lookback_days:]

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

    # tránh chia cho 0 + tránh triệt tiêu M
    winrate = {}
    for m, v in stats.items():
        if v['total'] > 0:
            winrate[m] = v['hit'] / v['total']
        else:
            winrate[m] = 0.05  # floor an toàn

    return winrate


# ==============================================================================
# 6. CORE CALCULATION – GIỮ LOGIC GỐC, CHỈ THAY TRỌNG SỐ
# ==============================================================================

def fast_get_top_nums(df, p_map_dict, top_n=60):
    """
    Rút gọn từ logic cũ:
    - Đếm số xuất hiện
    - Nhân trọng số
    """
    counter = Counter()

    for _, row in df.iterrows():
        for col, w in p_map_dict.items():
            for n in get_nums(row[col]):
                counter[n] += w

    return [n for n, _ in counter.most_common(int(top_n))]


def calculate_one_day(
    target_date,
    cache,
    kq_db,
    score_std,
    weight_mode,
    winrate_M
):
    """
    Chạy 1 ngày – Static hoặc Adaptive
    """
    if target_date not in cache:
        return []

    df = cache[target_date]['df']
    hist_col = cache[target_date]['hist_col']

    score_tuple = tuple(score_std.items())
    p_map_dict = {}

    for col in df.columns:
        base_score = get_col_score(col, score_tuple)
        if base_score <= 0:
            continue

        if weight_mode == 'Adaptive':
            factor = (winrate_M.get(col, 0.05)) ** ALPHA
            p_map_dict[col] = base_score * factor
        else:
            p_map_dict[col] = base_score

    # lọc member theo cột lịch sử (GIỮ LOGIC GỐC)
    hist_series = df[hist_col].astype(str).str.upper()
    valid_df = df[hist_series.notna()]

    return fast_get_top_nums(valid_df, p_map_dict, top_n=60)
# ==============================================================================
# 7. SIDEBAR – CÀI ĐẶT & CHẾ ĐỘ SO SÁNH
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Cài đặt Backtest")

    weight_mode = st.radio(
        "🔁 Weight Mode",
        options=["Static", "Adaptive"],
        index=0,
        help="Static = logic cũ | Adaptive = trọng số động theo winrate"
    )

    lookback_days = st.number_input(
        "📅 Lookback tính winrate (ngày)",
        min_value=3,
        max_value=30,
        value=10
    )

    st.markdown("---")
    st.caption("ℹ️ Adaptive dùng công thức:")
    st.code("score = base_score × (winrate ^ 0.6)")

# ==============================================================================
# 8. MAIN UI – UPLOAD & LOAD DATA
# ==============================================================================

uploaded_files = st.file_uploader(
    "📂 Tải file Excel (giống app cũ)",
    type=['xlsx'],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Đang đọc dữ liệu..."):
        data_cache, kq_db, err_logs = load_data_v56(uploaded_files)

    if err_logs:
        for e in err_logs:
            st.error(e)

    if not data_cache:
        st.error("Không đọc được dữ liệu hợp lệ.")
        st.stop()

    all_dates = sorted(kq_db.keys())
    last_date = max(all_dates)

    st.success(f"Đã đọc {len(all_dates)} ngày dữ liệu.")

    # ==============================================================================
    # 9. BACKTEST – SO SÁNH STATIC vs ADAPTIVE
    # ==============================================================================

    st.subheader("🔙 BACKTEST – SO SÁNH TRỰC TIẾP")

    d_start, d_end = st.select_slider(
        "📅 Chọn khoảng ngày backtest",
        options=all_dates,
        value=(all_dates[0], all_dates[-1])
    )

    if st.button("🚀 CHẠY BACKTEST", type="primary"):
        winrate_M = calc_winrate_M(data_cache, kq_db, lookback_days)

        logs = []

        progress = st.progress(0)
        total_days = len(all_dates)

        for idx, d in enumerate(all_dates):
            progress.progress((idx + 1) / total_days)

            if d < d_start or d > d_end:
                continue

            real_kq = kq_db.get(d)
            if not real_kq:
                continue

            res_static = calculate_one_day(
                d,
                data_cache,
                kq_db,
                score_std={f"M{i}": v for i, v in enumerate([0,1,2,3,4,5,6,7,15,25,50])},
                weight_mode="Static",
                winrate_M=winrate_M
            )

            res_adapt = calculate_one_day(
                d,
                data_cache,
                kq_db,
                score_std={f"M{i}": v for i, v in enumerate([0,1,2,3,4,5,6,7,15,25,50])},
                weight_mode="Adaptive",
                winrate_M=winrate_M
            )

            logs.append({
                "Ngày": d.strftime("%d/%m"),
                "KQ": real_kq,
                "Static": "WIN" if real_kq in res_static else "MISS",
                "Adaptive": "WIN" if real_kq in res_adapt else "MISS"
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
                    f"{(df_log['Static']=='WIN').sum()}/{len(df_log)}",
                    f"{(df_log['Static']=='WIN').mean()*100:.1f}%"
                )
            with c2:
                st.metric(
                    "Adaptive",
                    f"{(df_log['Adaptive']=='WIN').sum()}/{len(df_log)}",
                    f"{(df_log['Adaptive']=='WIN').mean()*100:.1f}%"
                )

else:
    st.info("👆 Upload file Excel để bắt đầu backtest")
# ==============================================================================
# 10. FOOTER & KẾT THÚC APP
# ==============================================================================

st.markdown("---")
st.caption(
    "🧠 Quang Pro V56 Adaptive | "
    "So sánh Static vs Adaptive (α = 0.6) | "
    "Giữ nguyên logic gốc, chỉ thay trọng số"
)

# ==============================================================================
# 11. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Streamlit tự chạy, block này chỉ để rõ cấu trúc
    pass
