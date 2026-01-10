# ============================================================
# PA2 – PRE-ANALYSIS (DẠNG CHỮ)
# ĐÁNH GIÁ NGÀY TRƯỚC KHI CÓ KẾT QUẢ
# 3 MỨC: 🟩 / 🟨 / 🟥
# TUYỆT ĐỐI KHÔNG DÙNG KQ, WIN/MISS, HIT-RATE
# ============================================================

import streamlit as st
from collections import Counter


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _safe_set(x):
    try:
        return set(x)
    except Exception:
        return set()


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0


def _density_score(dan):
    """
    Độ tập trung dàn:
    - dàn quá rộng -> phân tán
    - dàn vừa -> tốt
    """
    n = _safe_len(dan)
    if n == 0:
        return -1
    if n < 30:
        return 1
    if 30 <= n <= 60:
        return 2
    if 60 < n <= 80:
        return 0
    return -1


def _hybrid_pressure(hybrid):
    """
    Đánh giá hybrid có 'ép thật' hay không
    """
    if not hybrid:
        return -1

    cnt = Counter(hybrid)
    top = cnt.most_common(1)[0][1]

    if top >= 3:
        return 2
    if top == 2:
        return 1
    return -1


# ============================================================
# HÀM CHÍNH – GỌI TỪ app.py (TRƯỚC MỞ THƯỞNG)
# ============================================================
def render_pa2_preanalysis(
    *,
    res_curr,
    res_hc,
    hybrid_goc,
    res_prev=None   # kết quả hôm trước (CẤU TRÚC, KHÔNG PHẢI KQ)
):
    """
    PA2 – PRE ANALYSIS
    - Chỉ dùng dữ liệu hiện tại
    - Không phụ thuộc kết quả xổ
    """

    st.subheader("🧠 PA2 – ĐÁNH GIÁ TRƯỚC MỞ THƯỞNG")

    reasons_good = []
    reasons_bad = []

    score = 0

    # --------------------------------------------------------
    # 1. CONSENSUS CẤU TRÚC (QUAN TRỌNG NHẤT)
    # --------------------------------------------------------
    goc = _safe_set(res_curr.get("dan_goc"))
    mod = _safe_set(res_curr.get("dan_mod"))
    hc = _safe_set(res_hc.get("dan_goc")) if res_hc else set()

    union = goc | mod | hc
    inter = goc & mod & hc

    consensus = len(inter) / len(union) if union else 0

    if consensus >= 0.35:
        score += 2
        reasons_good.append("Consensus gốc / màn hình / hardcore rõ ràng")
    elif consensus >= 0.25:
        score += 1
        reasons_good.append("Consensus mức trung bình")
    else:
        score -= 2
        reasons_bad.append("Consensus thấp – các hệ không đồng thuận")

    # --------------------------------------------------------
    # 2. ĐỘ TẬP TRUNG DÀN (ENTROPY / DENSITY)
    # --------------------------------------------------------
    dan_final = res_curr.get("dan_final", [])
    dens = _density_score(dan_final)

    if dens == 2:
        score += 1
        reasons_good.append("Dàn tập trung, không phình")
    elif dens == 1:
        reasons_good.append("Dàn hẹp – chọn lọc mạnh")
    elif dens == 0:
        reasons_bad.append("Dàn hơi rộng – nhiễu nhẹ")
    else:
        score -= 1
        reasons_bad.append("Dàn quá rộng – phân tán")

    # --------------------------------------------------------
    # 3. ĐỘ ÉP HYBRID
    # --------------------------------------------------------
    hscore = _hybrid_pressure(hybrid_goc)

    if hscore == 2:
        score += 1
        reasons_good.append("Hybrid ép mạnh vào nhóm rõ ràng")
    elif hscore == 1:
        reasons_good.append("Hybrid có ép nhẹ")
    else:
        score -= 1
        reasons_bad.append("Hybrid ép yếu hoặc chỉ giao hình thức")

    # --------------------------------------------------------
    # 4. ĐỘ ỔN ĐỊNH CẤU TRÚC (SO VỚI HÔM TRƯỚC – NẾU CÓ)
    # --------------------------------------------------------
    if res_prev:
        prev_set = _safe_set(res_prev.get("dan_final"))
        curr_set = _safe_set(dan_final)

        diff = len(curr_set.symmetric_difference(prev_set))

        if diff <= 15:
            score += 1
            reasons_good.append("Cấu trúc ổn định so với hôm trước")
        elif diff >= 30:
            score -= 1
            reasons_bad.append("Cấu trúc thay đổi mạnh so với hôm trước")

    # --------------------------------------------------------
    # 5. KẾT LUẬN CUỐI (3 MỨC)
    # --------------------------------------------------------
    if score >= 3:
        label = "🟩 KẾT LUẬN: ĐÁNG ĐÁNH"
        box = st.success
        action = "👉 Khuyến nghị: Đánh theo plan chính"
    elif score >= 1:
        label = "🟨 KẾT LUẬN: NGUY HIỂM"
        box = st.warning
        action = "👉 Khuyến nghị: Giảm vốn / đánh chọn lọc"
    else:
        label = "🟥 KẾT LUẬN: KHÔNG ĐÁNG ĐÁNH"
        box = st.error
        action = "👉 Khuyến nghị: Nên nghỉ – tránh vào tiền"

    box(label)

    # --------------------------------------------------------
    # 6. LÝ DO
    # --------------------------------------------------------
    st.markdown("### 📌 Lý do")

    for r in reasons_good:
        st.write(f"• {r}")

    for r in reasons_bad:
        st.write(f"• ⚠️ {r}")

    # --------------------------------------------------------
    # 7. HÀNH ĐỘNG
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown(action)

    st.caption("PA2 – Pre-Analysis | Đánh giá trước khi có kết quả | Không dùng WIN/MISS")
