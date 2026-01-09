# ==============================================================
# BACKTEST TOOL – STATIC vs ADAPTIVE (α = 0.6)
# DÙNG RIÊNG – KHÔNG PHẢI APP GỐC
# COPY PASTE 1 LẦN – CHẠY ĐƯỢC TRÊN ANDROID
# ==============================================================

import streamlit as st
import pandas as pd
import re
import datetime
from collections import Counter, defaultdict

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Backtest Static vs Adaptive",
    page_icon="📊",
    layout="wide"
)

st.title("📊 BACKTEST: Static vs Adaptive (α = 0.6)")
st.caption("Tool độc lập – dùng data thật – không liên quan app gốc")

ALPHA = 0.6
RE_NUM = re.compile(r"\d{1,2}")

# ===================== HELPERS =====================
def get_nums(x):
    if pd.isna(x):
        return []
    return [n.zfill(2) for n in RE_NUM.findall(str(x))]

def find_header_row(df):
    for i in range(20):
        row = " ".join(df.iloc[i].astype(str))
        if "M0" in row and "M1" in row:
            return i
    return 3

def load_excel(files):
    data = {}
    kq_db = {}

    for file in files:
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            preview = pd.read_excel(xls, sheet_name=sheet, header=None)
            h = find_header_row(preview)
            df = pd.read_excel(xls, sheet_name=sheet, header=h)
            df.columns = [str(c).strip().upper() for c in df.columns]

            # tìm cột ngày
            date_cols = {}
            for c in df.columns:
                m = re.search(r"(\d{1,2})[/-](\d{1,2})", c)
                if m:
                    d, mth = int(m.group(1)), int(m.group(2))
                    try:
                        date_cols[datetime.date(2025, mth, d)] = c
                    except:
                        pass

            # tìm dòng KQ
            kq_row = None
            for c in df.columns[:2]:
                mask = df[c].astype(str).str.contains("KQ", case=False, na=False)
                if mask.any():
                    kq_row = df[mask].iloc[0]
                    break

            if kq_row is None:
                continue

            for d, col in date_cols.items():
                nums = get_nums(kq_row[col])
                if nums:
                    kq_db[d] = nums[0]
                    data[d] = df

    return data, kq_db

def calc_winrate(data, kq_db, lookback=10):
    stats = defaultdict(lambda: {"hit": 0, "total": 0})
    days = sorted(kq_db.keys())[-lookback:]

    for d in days:
        df = data[d]
        kq = kq_db[d]
        for _, row in df.iterrows():
            for c in df.columns:
                if c.startswith("M"):
                    nums = get_nums(row[c])
                    if nums:
                        stats[c]["total"] += 1
                        if kq in nums:
                            stats[c]["hit"] += 1

    return {
        m: (v["hit"] / v["total"] if v["total"] > 0 else 0.05)
        for m, v in stats.items()
    }

def rank_nums(df, weights):
    cnt = Counter()
    for _, row in df.iterrows():
        for c, w in weights.items():
            for n in get_nums(row[c]):
                cnt[n] += w
    return [n for n, _ in cnt.most_common(60)]

# ===================== UI =====================
files = st.file_uploader("📂 Upload file Excel", type=["xlsx"], accept_multiple_files=True)

if files:
    with st.spinner("Đang đọc dữ liệu..."):
        data, kq_db = load_excel(files)

    if not data:
        st.error("Không đọc được dữ liệu")
        st.stop()

    days = sorted(kq_db.keys())
    d1, d2 = st.select_slider(
        "📅 Chọn khoảng ngày",
        options=days,
        value=(days[0], days[-1])
    )

    lookback = st.number_input("Lookback tính winrate", 3, 30, 10)

    if st.button("🚀 CHẠY BACKTEST"):
        winrate = calc_winrate(data, kq_db, lookback)
        logs = []

        for d in days:
            if d < d1 or d > d2:
                continue

            df = data[d]
            kq = kq_db[d]

            static_w = {c: 1 for c in df.columns if c.startswith("M")}
            adapt_w = {
                c: (winrate.get(c, 0.05) ** ALPHA)
                for c in df.columns if c.startswith("M")
            }

            r_static = rank_nums(df, static_w)
            r_adapt = rank_nums(df, adapt_w)

            logs.append({
                "Ngày": d.strftime("%d/%m"),
                "KQ": kq,
                "Static": "WIN" if kq in r_static else "MISS",
                "Adaptive": "WIN" if kq in r_adapt else "MISS"
            })

        res = pd.DataFrame(logs)
        st.dataframe(res, use_container_width=True, height=600)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Static WIN", f"{(res.Static=='WIN').sum()}/{len(res)}")
        with c2:
            st.metric("Adaptive WIN", f"{(res.Adaptive=='WIN').sum()}/{len(res)}")
