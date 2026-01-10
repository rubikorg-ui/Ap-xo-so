# ============================================================
# pa2_daysignal_strategy.py
# MODULE ĐÁNH GIÁ NGÀY – ĐÈN TÍN HIỆU & CẢNH BÁO
# ĐỘC LẬP – KHÔNG CAN THIỆP LOGIC GỐC
# ============================================================

import streamlit as st
from datetime import timedelta


# ----------------------------
# Utils an toàn
# ----------------------------
def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0


# ============================================================
# HÀM CHÍNH – CHỈ GỌI HÀM NÀY TỪ app.py
# ============================================================
def render_day_signal(
    *,
    res_curr,
    res_hc,
    hybrid,
    kq_db,
    target_date
):
    """
    Module hiển thị:
    - Đèn đánh giá ngày 🟩🟨🟥
    - Day score
    - Consensus
    - Cảnh báo rủi ro

    LƯU Ý:
    - CHỈ ĐỌC dữ liệu
    - KHÔNG thay đổi số
    - KHÔNG ghi đè biến gốc
    """

    # ========================================================
    # TEST CHẮC CHẮN MODULE ĐANG CHẠY (BẠN CÓ THỂ XÓA SAU)
    # ========================================================
    st.subheader("🚦 ĐÁNH GIÁ NGÀY (MODULE)")

    # ========================================================
    # 1. SIZE DÀN
    # ========================================================
    size_today = _safe_len(res_curr.get("dan_final", []))

    # ========================================================
    # 2. CONSENSUS GIỮA CÁC HỆ
    # ========================================================
    try:
        set_goc = set(res_curr.get("dan_goc", []))
        set_mod = set(res_curr.get("dan_mod", []))
        set_hc = set(res_hc.get("dan_goc", [])) if res_hc else set()

        union = set_goc | set_mod | set_hc
        inter = set_goc & set_mod & set_hc

        consensus = round(len(inter) / len(union), 2) if union else 0.0
    except Exception:
        consensus = 0.0

    # ========================================================
    # 3. PHONG ĐỘ GẦN (5 NGÀY)
    # ========================================================
    recent_hits = []
    for i in range(1, 6):
        d = target_date - timedelta(days=i)
        if d in kq_db:
            try:
                recent_hits.append(
                    1 if kq_db[d] in res_curr.get("dan_final", []) else 0
                )
            except Exception:
                pass

    recent_rate = round(
        sum(recent_hits) / len(recent_hits), 2
    ) if recent_hits else 0.0

    # ========================================================
    # 4. TÍNH ĐIỂM NGÀY
    # ========================================================
    score = 0
    warnings = []

    # Consensus
    if consensus >= 0.35:
        score += 1
    elif consensus < 0.25:
        score -= 1
        warnings.append("Consensus thấp – các hệ không đồng thuận")

    # Phong độ
    if recent_rate >= 0.6:
        score += 1
    elif recent_rate < 0.4:
        score -= 1
        warnings.append("Phong độ 5 ngày gần đây kém")

    # Size
    if size_today > 70:
        warnings.append("Dàn rộng – rủi ro cao")
    elif size_today < 35:
        score += 1

    # ========================================================
    # 5. KẾT LUẬN NGÀY
    # ========================================================
    if score >= 2:
        label = "🟩 NGÀY ĐẸP"
        box = st.success
    elif score <= 0:
        label = "🟥 NGÀY XẤU"
        box = st.error
    else:
        label = "🟨 TRUNG TÍNH"
        box = st.warning

    box(label)

    # ========================================================
    # 6. HIỂN THỊ METRIC
    #њ ========================================================
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Day Score", score)
    with c2:
        st.metric("Consensus", consensus)
    with c3:
        st.metric("Size hôm nay", size_today)

    # ========================================================
    # 7. CẢNH BÁO
    # ========================================================
    if warnings:
        st.markdown("### 🚨 Cảnh báo")
        for w in warnings:
            st.warning(w)
    else:
        st.success("Không có cảnh báo nghiêm trọng")

    st.caption("Module PA2 – chỉ đọc dữ liệu, không can thiệp logic gốc.")
