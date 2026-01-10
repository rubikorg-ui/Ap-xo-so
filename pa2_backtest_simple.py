def pa2_backtest_simple(history):
    total_all = 0
    win_all = 0

    total_green = 0
    win_green = 0

    for row in history:
        kq = row["kq"]
        dan = row["dan"]
        pa2 = row["pa2"]

        # Đánh tất cả ngày
        total_all += 1
        if kq in dan:
            win_all += 1

        # Chỉ đánh ngày xanh
        if pa2 == "GREEN":
            total_green += 1
            if kq in dan:
                win_green += 1

    return {
        "tat_ca_ngay": {
            "so_ngay": total_all,
            "so_trung": win_all,
            "ti_le": round(win_all / total_all, 2) if total_all else 0
        },
        "chi_ngay_xanh": {
            "so_ngay": total_green,
            "so_trung": win_green,
            "ti_le": round(win_green / total_green, 2) if total_green else 0
        }
    }
