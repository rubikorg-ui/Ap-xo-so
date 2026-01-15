# -*- coding: utf-8 -*-

from collections import Counter
from datetime import timedelta
import re
import pandas as pd

# =============================
# CONFIG (NO VIETNAMESE ACCENT)
# =============================
BAD_KEYWORDS = [
    'N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'LOI'
]

# =====================================================
# HELPER: GET TOP NUMS BY VOTE
# =====================================================
def get_top_nums_by_vote_count(
    df_members,
    col_name,
    limit,
    hc_score_map=None
):
    """
    Lấy danh sách số được Vote nhiều nhất từ cột 8X.
    """
    if df_members.empty:
        return []

    all_nums = []
    # Chuyển đổi dữ liệu cột sang string để xử lý regex
    vals = df_members[col_name].dropna().astype(str).tolist()

    for val in vals:
        val_up = val.upper()
        # Bỏ qua các từ khóa báo nghỉ/xịt
        if not val_up or any(kw in val_up for kw in BAD_KEYWORDS):
            continue
        
        # Regex tìm tất cả các số trong ô (xử lý chuỗi dài 00,01,02...)
        found = re.findall(r'\d+', val_up)
        
        # Chuẩn hóa số về dạng 2 chữ số (01, 02...)
        # Chỉ lấy 2 số cuối nếu chuỗi dài (tránh dính ngày tháng năm)
        valid_nums = []
        for n in found:
            clean_n = n.zfill(2)
            if len(clean_n) > 2: clean_n = clean_n[-2:]
            valid_nums.append(clean_n)
            
        all_nums.extend(valid_nums)

    # Đếm số lần xuất hiện (Vote)
    counts = Counter(all_nums)

    # 1. Sắp xếp theo Vote giảm dần
    sorted_by_vote = sorted(
        counts.items(),
        key=lambda x: -x[1]
    )

    if not sorted_by_vote:
        return []

    # Ép kiểu limit
    try:
        limit = int(limit)
    except:
        limit = 80

    # Nếu tổng số lượng số ít hơn limit, lấy hết
    if len(sorted_by_vote) <= limit:
        return [n for n, _ in sorted_by_vote]

    # 2. Xử lý điểm cắt (Cut Vote)
    try:
        cut_vote = sorted_by_vote[limit - 1][1]
    except IndexError:
        return [n for n, _ in sorted_by_vote]

    higher = [(n, v) for n, v in sorted_by_vote if v > cut_vote]
    equal = [(n, v) for n, v in sorted_by_vote if v == cut_vote]

    slots_left = limit - len(higher)
    if slots_left <= 0:
        return [n for n, _ in higher]

    # 3. Logic Tie-Break (Ưu tiên điểm HC nếu bằng phiếu)
    if len(equal) <= slots_left:
        result = higher + equal
    else:
        def tie_key(item):
            num = item[0]
            hc = hc_score_map.get(num, 0) if hc_score_map else 0
            # Ưu tiên: Điểm HC cao -> Giá trị số nhỏ
            return (-hc, int(num))

        equal_sorted = sorted(equal, key=tie_key)
        result = higher + equal_sorted[:slots_left]

    return [n for n, _ in result]

# =====================================================
# MAIN STRATEGY: VOTE 8X (FIXED DATA COLUMN)
# =====================================================
def calculate_vote_8x_strategy(
    target_date,
    rolling_window,
    data_cache,
    kq_db,
    limits_config,
    top_n=10,
    hc_score_map=None,
    **kwargs
):
    if target_date not in data_cache:
        return None

    curr_data = data_cache[target_date]
    df = curr_data['df']

    # 1. TÌM CỘT 8X CHUẨN (QUAN TRỌNG: TRÁNH CỘT ĐIỂM 'Đ8X')
    col_8x = None
    
    # Cách 1: Tìm cột trùng khớp chính xác "8X" (thường là cột R)
    for c in df.columns:
        if str(c).strip().upper() == '8X':
            col_8x = c
            break
    
    # Cách 2: Nếu không thấy, tìm cột có chứa "8X" nhưng KHÔNG chứa "Đ"
    if not col_8x:
        for c in df.columns:
            name = str(c).strip().upper()
            if '8X' in name and 'Đ' not in name:
                col_8x = c
                break
    
    if not col_8x:
        return None # "NO 8X COLUMN"

    # 2. TÌM CỘT PHÂN NHÓM (GROUP COLUMN)
    col_group = None
    prev_date = target_date - timedelta(days=1)
    
    # Quét ngược 5 ngày để tìm cột Group
    for _ in range(5):
        if prev_date in curr_data['hist_map']:
            col_group = curr_data['hist_map'][prev_date]
            break
        prev_date -= timedelta(days=1)

    # Fallback: Thử tìm trong chính ngày hiện tại hoặc tìm theo tên cột ngày hôm trước
    if not col_group:
        if target_date in curr_data['hist_map']:
            col_group = curr_data['hist_map'][target_date]
        else:
            # Cố gắng tìm cột có tên là ngày hôm trước (vd: "14/01")
            d_prev_explicit = target_date - timedelta(days=1)
            d_str = d_prev_explicit.strftime("%d/%m") # định dạng dd/mm
            for c in df.columns:
                if d_str in str(c):
                    col_group = c
                    break
    
    if not col_group:
        return None # "NO GROUP COLUMN"

    # 3. BACKTEST: TÌM TOP 6 NHÓM MẠNH NHẤT
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}

    past_dates = []
    d = target_date - timedelta(days=1)
    while len(past_dates) < rolling_window:
        if d in data_cache and d in kq_db:
            past_dates.append(d)
        d -= timedelta(days=1)
        if (target_date - d).days > 60:
            break

    for d in past_dates:
        d_df = data_cache[d]['df']
        kq = kq_db[d]

        # Tìm cột 8X cho ngày backtest (cũng phải tránh cột Đ)
        d_c8 = None
        for c in d_df.columns:
            if str(c).strip().upper() == '8X':
                d_c8 = c; break
        if not d_c8:
            for c in d_df.columns:
                if '8X' in str(c).strip().upper() and 'Đ' not in str(c).strip().upper():
                    d_c8 = c; break
        
        # Tìm cột Group cho ngày backtest (Group ngày D dựa trên KQ ngày D-1)
        d_grp_col = None
        d_prev = d - timedelta(days=1)
        if d_prev in data_cache[d]['hist_map']:
            d_grp_col = data_cache[d]['hist_map'][d_prev]
        
        if not d_c8 or not d_grp_col:
            continue

        try:
            hist_series = (
                d_df[d_grp_col]
                .astype(str)
                .str.upper()
                .str.replace('S', '6')
                .str.replace(r'[^0-9X]', '', regex=True)
            )
        except:
            continue

        for g in groups:
            mems = d_df[hist_series == g.upper()]
            # Backtest dùng limit chuẩn 80
            top80 = get_top_nums_by_vote_count(
                mems,
                d_c8,
                80,
                hc_score_map
            )
            if kq in top80:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top80.index(kq))
            else:
                stats[g]['ranks'].append(999)

    # Xếp hạng Group
    final_rank = [
        (g, -v['wins'], sum(v['ranks']))
        for g, v in stats.items()
    ]
    final_rank.sort(key=lambda x: (x[1], x[2]))
    
    # Lấy Top 6 Group
    top6 = [x[0] for x in final_rank[:6]]

    # 4. FINAL CUT & ALLIANCE (LOGIC V24 CHUẨN)
    try:
        hist_series = (
            df[col_group]
            .astype(str)
            .str.upper()
            .str.replace('S', '6')
            .str.replace(r'[^0-9X]', '', regex=True)
        )
    except:
        return None

    limit_map = {
        top6[0]: limits_config.get('l12', 75),
        top6[1]: limits_config.get('l12', 75),
        top6[2]: limits_config.get('l34', 70),
        top6[3]: limits_config.get('l34', 70),
        top6[4]: limits_config.get('l56', 65),
        top6[5]: limits_config.get('l56', 65),
    }

    def get_pool(group_list):
        pool = []
        for g in group_list:
            limit = limit_map.get(g, 80)
            nums = get_top_nums_by_vote_count(
                df[hist_series == g.upper()],
                col_8x,
                limit, 
                hc_score_map
            )
            pool.extend(nums)
        # Logic V24: Giữ số xuất hiện >= 2 lần trong Phe
        return {n for n, c in Counter(pool).items() if c >= 2}

    # PHE 1 (Rank 1, 5, 3)
    s1 = get_pool([top6[0], top6[4], top6[2]])
    # PHE 2 (Rank 2, 4, 6)
    s2 = get_pool([top6[1], top6[3], top6[5]])

    # GIAO THOA (Intersection) - Logic chuẩn V24
    # Do dữ liệu 8X giờ đã lấy đủ, phép giao này sẽ trả về lượng số hợp lý (40-60 số)
    final_dan = sorted(list(s1.intersection(s2)))
    
    # RAW (Hợp Union - dùng để tham khảo bao quát)
    raw_union = sorted(list(s1.union(s2)))

    return {
        "top6_vote": top6,
        "dan_goc": raw_union,
        "dan_final": final_dan,
        "dan_mod": [],
        "source_col": col_group
    }
