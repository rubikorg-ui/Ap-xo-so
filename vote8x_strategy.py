# -*- coding: utf-8 -*-

from collections import Counter
from datetime import timedelta
import re

# =============================
# CONFIG (NO VIETNAMESE ACCENT)
# =============================
BAD_KEYWORDS = [
    'N', 'NGHI', 'SX', 'XIT', 'MISS', 'TRUOT', 'LOI'
]

# =====================================================
# HELPER: GET TOP NUMS BY VOTE
# PRIORITY: VOTE > HC SCORE > VALUE
# =====================================================
def get_top_nums_by_vote_count(
    df_members,
    col_name,
    limit,
    hc_score_map=None
):
    """
    Lấy danh sách số top đầu dựa trên lượt Vote (tần suất xuất hiện).
    Nếu bằng phiếu, dùng điểm HC (nếu có) để phân định.
    """
    if df_members.empty:
        return []

    all_nums = []
    # Lấy dữ liệu từ cột Vote (8X), bỏ giá trị null
    vals = df_members[col_name].dropna().astype(str).tolist()

    for val in vals:
        val_up = val.upper()
        # Bỏ qua các từ khóa xấu (Nghỉ, Xịt...)
        if any(kw in val_up for kw in BAD_KEYWORDS):
            continue
        # Tìm tất cả các số trong chuỗi
        found = re.findall(r'\d+', val_up)
        # Chuẩn hóa về 2 chữ số (01, 02...)
        all_nums.extend([n.zfill(2) for n in found if len(n) <= 2])

    # Đếm số lần xuất hiện (Vote)
    counts = Counter(all_nums)

    # 1. Sắp xếp sơ bộ theo số lượng Vote giảm dần
    sorted_by_vote = sorted(
        counts.items(),
        key=lambda x: -x[1]
    )

    # Đảm bảo limit là số nguyên
    try:
        limit = int(limit)
    except:
        limit = 80

    # Nếu tổng số lượng số ít hơn limit, lấy hết
    if len(sorted_by_vote) <= limit:
        return [n for n, _ in sorted_by_vote]

    # 2. Xử lý điểm cắt (Cut Vote)
    # Lấy số phiếu tại vị trí cắt (ví dụ vị trí 65 có 5 phiếu)
    try:
        cut_vote = sorted_by_vote[limit - 1][1]
    except IndexError:
        return [n for n, _ in sorted_by_vote]

    # Tách thành 2 nhóm: Nhóm chắc chắn đỗ (Higher) và Nhóm bằng điểm cắt (Equal)
    higher = [(n, v) for n, v in sorted_by_vote if v > cut_vote]
    equal = [(n, v) for n, v in sorted_by_vote if v == cut_vote]

    slots_left = limit - len(higher)
    if slots_left <= 0:
        return [n for n, _ in higher]

    # 3. Logic Tie-Break (Phân định khi bằng phiếu)
    if len(equal) <= slots_left:
        # Nếu số lượng bằng phiếu ít hơn chỗ trống, lấy hết
        result = higher + equal
    else:
        # 4. Nếu số lượng bằng phiếu nhiều hơn chỗ trống -> Dùng điểm HC
        def tie_key(item):
            num = item[0]
            # Lấy điểm từ bản đồ HC (nếu có), mặc định 0
            hc = hc_score_map.get(num, 0) if hc_score_map else 0
            # Sắp xếp: Điểm HC cao nhất lên đầu (-hc), sau đó đến giá trị số nhỏ
            return (-hc, int(num))

        equal_sorted = sorted(equal, key=tie_key)
        result = higher + equal_sorted[:slots_left]

    return [n for n, _ in result]

# =====================================================
# MAIN STRATEGY: VOTE 8X (WITH V24 CUT LOGIC)
# =====================================================
def calculate_vote_8x_strategy(
    target_date,
    rolling_window,
    data_cache,
    kq_db,
    limits_config,  # <-- Nhận dict cấu hình cắt {l12, l34, l56}
    top_n=10,
    hc_score_map=None,
    **kwargs
):
    """
    Chiến thuật Vote 8x kết hợp logic cắt thông minh V24.
    """
    # Kiểm tra dữ liệu ngày đích
    if target_date not in data_cache:
        return None

    curr_data = data_cache[target_date]
    df = curr_data['df']

    # 1. TÌM CỘT 8X (Cột chứa dữ liệu Vote)
    col_8x = next(
        (c for c in df.columns if '8X' in c.upper()),
        None
    )
    if not col_8x:
        return None # "NO 8X COLUMN"

    # 2. TÌM CỘT PHÂN NHÓM (GROUP COLUMN)
    # Quét ngược 5 ngày để tìm cột phân nhóm gần nhất
    col_group = None
    prev_date = target_date - timedelta(days=1)

    for _ in range(5):
        if prev_date in curr_data['hist_map']:
            col_group = curr_data['hist_map'][prev_date]
            break
        prev_date -= timedelta(days=1)

    # Fallback: Nếu không tìm thấy trong quá khứ, thử tìm trong chính ngày hiện tại
    if not col_group:
        if target_date in curr_data['hist_map']:
            col_group = curr_data['hist_map'][target_date]
        else:
            return None # "NO GROUP COLUMN"

    # 3. BACKTEST: XẾP HẠNG TOP GROUPS
    # Mục đích: Tìm ra các nhóm (0x, 1x...) phong độ tốt nhất trong chu kỳ
    groups = [f"{i}x" for i in range(10)]
    stats = {g: {'wins': 0, 'ranks': []} for g in groups}

    past_dates = []
    d = target_date - timedelta(days=1)
    
    # Lấy danh sách ngày quá khứ hợp lệ
    while len(past_dates) < rolling_window:
        if d in data_cache and d in kq_db:
            past_dates.append(d)
        d -= timedelta(days=1)
        if (target_date - d).days > 60: # Giới hạn tìm kiếm
            break

    # Chạy loop backtest
    for d in past_dates:
        d_df = data_cache[d]['df']
        kq = kq_db[d]

        d_c8 = next((c for c in d_df.columns if '8X' in c.upper()), None)
        hist_dates = sorted(
            [k for k in data_cache[d]['hist_map'] if k < d],
            reverse=True
        )
        if not d_c8 or not hist_dates:
            continue

        d_grp_col = data_cache[d]['hist_map'][hist_dates[0]]
        
        # Clean dữ liệu cột Group
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
            # Khi Backtest xếp hạng, dùng limit chuẩn 80 để đánh giá năng lực thực
            top_test = get_top_nums_by_vote_count(
                mems,
                d_c8,
                80,
                hc_score_map
            )
            if kq in top_test:
                stats[g]['wins'] += 1
                stats[g]['ranks'].append(top_test.index(kq))
            else:
                stats[g]['ranks'].append(999)

    # Xếp hạng Group: Ưu tiên số lần thắng (wins) -> Tổng hạng (nhỏ tốt hơn)
    final_rank = [
        (g, -v['wins'], sum(v['ranks']))
        for g, v in stats.items()
    ]
    final_rank.sort(key=lambda x: (x[1], x[2]))
    
    # Lấy Top 6 Group tốt nhất
    top6 = [x[0] for x in final_rank[:6]]

    # 4. FINAL CUT: ÁP DỤNG LOGIC V24 (L12, L34, L56)
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

    # Bản đồ Limit dựa trên thứ hạng (Rank)
    # Rank 1-2 dùng l12, Rank 3-4 dùng l34...
    limit_map = {
        top6[0]: limits_config.get('l12', 75),
        top6[1]: limits_config.get('l12', 75),
        top6[2]: limits_config.get('l34', 70),
        top6[3]: limits_config.get('l34', 70),
        top6[4]: limits_config.get('l56', 65),
        top6[5]: limits_config.get('l56', 65),
    }

    def get_pool(group_list):
        """Hàm lấy tập hợp số từ danh sách nhóm, áp dụng limit riêng"""
        pool = []
        for g in group_list:
            # Lấy limit tương ứng với nhóm đó
            limit = limit_map.get(g, 80)
            nums = get_top_nums_by_vote_count(
                df[hist_series == g.upper()],
                col_8x,
                limit, 
                hc_score_map
            )
            pool.extend(nums)
        # Trả về Set các số (Logic: Lấy hợp tất cả số trong nhóm)
        return set(pool)

    # 5. CROSS ALLIANCE (LIÊN MINH CHÉO)
    # Chia Top 6 thành 2 phe để giao thoa kết quả, tăng độ chính xác
    # Phe 1: Rank 1, 5, 3 (Xáo trộn cố ý để cân bằng lực)
    s1 = get_pool([top6[0], top6[4], top6[2]])
    # Phe 2: Rank 2, 4, 6
    s2 = get_pool([top6[1], top6[3], top6[5]])

    # Dàn Final: Giao của 2 phe (Số phải xuất hiện ở cả 2 phe mới được chọn)
    final_dan = sorted(list(s1.intersection(s2)))
    
    # Dàn Gốc (Raw): Hợp của 2 phe (Tất cả các số tiềm năng)
    # Dùng để hiển thị tham khảo hoặc Backup khi Final quá ít số
    raw_union = sorted(list(s1.union(s2)))

    return {
        "top6_vote": top6,
        "dan_goc": raw_union,   # Dàn bao phủ rộng
        "dan_final": final_dan, # Dàn tinh gọn
        "dan_mod": [],          # Không dùng mod trong chiến thuật này
        "source_col": col_group
    }
