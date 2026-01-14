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
# GET TOP NUMS BY VOTE
# PRIORITY: VOTE > HC (ONLY IF TIED AT CUT)
# =====================================================
def get_top_nums_by_vote_count(
    df_members,
    col_name,
    limit,
    hc_score_map=None
):
    if df_members.empty:
        return []

    all_nums = []
    vals = df_members[col_name].dropna().astype(str).tolist()

    for val in vals:
        val_up = val.upper()
        if any(kw in val_up for kw in BAD_KEYWORDS):
            continue
        found = re.findall(r'\d+', val_up)
        all_nums.extend([n.zfill(2) for n in found if len(n) <= 2])

    counts = Counter(all_nums)

    # 1. SORT BY VOTE ONLY
    sorted_by_vote = sorted(
        counts.items(),
        key=lambda x: -x[1]
    )

    if len(sorted_by_vote) <= limit:
        return [n for n, _ in sorted_by_vote]

    # 2. FIND CUT VOTE
    cut_vote = sorted_by_vote[limit - 1][1]

    higher = [(n, v) for n, v in sorted_by_vote if v > cut_vote]
    equal = [(n, v) for n, v in sorted_by_vote if v == cut_vote]

    slots_left = limit - len(higher)

    # 3. NO NEED HC
    if len(equal) <= slots_left:
        result = higher + equal
    else:
        # 4. HC TIE-BREAK ONLY HERE
        def tie_key(item):
            num = item[0]
            hc = hc_score_map.get(num, 0) if hc_score_map else 0
            return (-hc, int(num))

        equal_sorted = sorted(equal, key=tie_key)
        result = higher + equal_sorted[:slots_left]

    return [n for n, _ in result]

# =====================================================
# MAIN STRATEGY: VOTE 8X
# =====================================================
def calculate_vote_8x_strategy(
    target_date,
    rolling_window,
    data_cache,
    kq_db,
    limits_config,
    hc_score_map=None
):
    if target_date not in data_cache:
        return None, "NO DATA FOR THIS DATE"

    curr_data = data_cache[target_date]
    df = curr_data['df']

    # 1. FIND 8X COLUMN
    col_8x = next(
        (c for c in df.columns if '8X' in c.upper()),
        None
    )
    if not col_8x:
        return None, "NO 8X COLUMN"

    # 2. FIND GROUP COLUMN
    col_group = None
    prev_date = target_date - timedelta(days=1)

    for _ in range(5):
        if prev_date in curr_data['hist_map']:
            col_group = curr_data['hist_map'][prev_date]
            break
        prev_date -= timedelta(days=1)

    if not col_group:
        return None, "NO GROUP COLUMN"

    # 3. BACKTEST TO PICK TOP 6 GROUPS
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

        d_c8 = next((c for c in d_df.columns if '8X' in c.upper()), None)
        hist_dates = sorted(
            [k for k in data_cache[d]['hist_map'] if k < d],
            reverse=True
        )
        if not d_c8 or not hist_dates:
            continue

        d_grp_col = data_cache[d]['hist_map'][hist_dates[0]]
        hist_series = (
            d_df[d_grp_col]
            .astype(str)
            .str.upper()
            .str.replace('S', '6')
            .str.replace(r'[^0-9X]', '', regex=True)
        )

        for g in groups:
            mems = d_df[hist_series == g.upper()]
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

    final_rank = [
        (g, -v['wins'], sum(v['ranks']))
        for g, v in stats.items()
    ]
    final_rank.sort(key=lambda x: (x[1], x[2]))
    top6 = [x[0] for x in final_rank[:6]]

    # 4. FINAL CUT
    hist_series = (
        df[col_group]
        .astype(str)
        .str.upper()
        .str.replace('S', '6')
        .str.replace(r'[^0-9X]', '', regex=True)
    )

    limit_map = {
        top6[0]: limits_config['l12'],
        top6[1]: limits_config['l12'],
        top6[2]: limits_config['l34'],
        top6[3]: limits_config['l34'],
        top6[4]: limits_config['l56'],
        top6[5]: limits_config['l56'],
    }

    def get_pool(group_list):
        pool = []
        for g in group_list:
            limit = limit_map.get(g, 80)
            pool.extend(
                get_top_nums_by_vote_count(
                    df[hist_series == g.upper()],
                    col_8x,
                    limit,
                    hc_score_map
                )
            )
        return {n for n, c in Counter(pool).items() if c >= 2}

    # CROSS ALLIANCE (INTENTIONAL DESIGN)
    s1 = get_pool([top6[0], top6[4], top6[2]])
    s2 = get_pool([top6[1], top6[3], top6[5]])

    final_dan = sorted(list(s1.intersection(s2)))

    return {
        "top6_vote": top6,
        "dan_vote": final_dan,
        "source_col": col_group
    }, None
