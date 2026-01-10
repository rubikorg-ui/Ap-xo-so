
"""
pa2_daysignal_strategy.py
================================
MODULE ĐỘC LẬP – KHÔNG PHÁ LOGIC GỐC

Chức năng:
1. Hiển thị ĐÈN NGÀY 🟩🟨🟥
2. Hiển thị CẢNH BÁO (read-only)
3. KHÔNG tính lại số
4. KHÔNG can thiệp hybrid / prediction

CÁCH DÙNG (CHỈ 2 BƯỚC):
--------------------------------
BƯỚC 1: Ở ĐẦU FILE APP, THÊM:
    import pa2_daysignal_strategy as pa2

BƯỚC 2: SAU KHI ĐÃ CÓ:
    - res_curr
    - res_hc
    - hybrid_goc
    - kq_db
    - target_date

DÁN DÒNG SAU:
    pa2.render_day_signal(
        res_curr=res_curr,
        res_hc=res_hc,
        hybrid=hybrid_goc,
        kq_db=kq_db,
        target_date=target_date
    )
"""

import streamlit as st
from datetime import timedelta


def _safe_len(x):
    try:
        return len(x)
    except:
        return 0


def render_day_signal(
    *,
    res_curr,
    res_hc,
    hybrid,
    kq_db,
    target_date
):
    """
    HÀM DUY NHẤT BẠN CẦN GỌI
    --------------------------------
    Chỉ ĐỌC dữ liệu đã có
    Không thay đổi bất kỳ logic nào
    """

    # ================== SIZE ==================
    size_today = _safe_len(res_curr.get("dan_final"))

    # ================== CONSENSUS ==================
    try:
        set_goc = set(res_curr.get("dan_goc", []))
        set_mod = set(res_curr.get("dan_mod", []))
        set_hc = set(res_hc.get("dan_goc", [])) if res_hc else set()

        union = set_goc | set_mod | set_hc
        inter = set_goc & set_mod & set_hc

        consensus = len(inter) / len(union) if union else 0
    except:
        consensus = 0

    # ================== RECENT HIT ==================
    recent_hits = []
    for i in range(1, 6):
        d = target_date - timedelta(days=i)
        if d in kq_db:
            try:
                recent_hits.append(1 if kq_db[d] in res_curr.get("dan_final", []) else 0)
            except:
                pass
    recent_hit_rate = sum(recent_hits) / len(recent_hits) if recent_hits else 0

    # ================== DAY SCORE ==================
    score = 0
    warnings = []

    if consensus >= 0.35:
        score += 1
    elif consensus < 0.25:
        score -= 1
        warnings.append("Consensus thấp – hệ không đồng thuận")

    if recent_hit_rate >= 0.6:
        score += 1
    elif recent_hit_rate < 0.4:
        score -= 1
        warnings.append("Phong độ 5 ngày gần kém")

    # ================== LABEL ==================
    if score >= 1:
        label = "🟩 NGÀY ĐẸP"
        color = "success"
    elif score <= -1:
        label = "🟥 NGÀY XẤU"
        color = "error"
    else:
        label = "🟨 TRUNG TÍNH"
        color = "warning"

    # ================== RENDER ==================
    st.divider()
    st.subheader("🚦 ĐÁNH GIÁ NGÀY (MODULE)")

    getattr(st, color)(label)
    st.metric("Day Score", score)
    st.metric("Consensus", round(consensus, 2))
    st.metric("Size hôm nay", size_today)

    if warnings:
        st.subheader("🚨 CẢNH BÁO")
        for w in warnings:
            st.warning(w)
    else:
        st.success("Không có cảnh báo nghiêm trọng")

    st.caption("Module chỉ đọc dữ liệu – không can thiệp logic gốc.")
