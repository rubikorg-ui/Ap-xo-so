# --- HÀM PHÂN TÍCH NHÓM CHUYÊN SÂU (UPDATE: RANKING & PREDICTION) ---
def analyze_group_performance(start_date, end_date, cut_limit, score_map, data_cache, kq_db, min_v, inverse):
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    
    # Cấu trúc lưu trữ thêm 'ranks' để tính điểm xếp hạng chuẩn
    grp_stats = {f"{i}x": {'wins': 0, 'ranks': [], 'history': [], 'last_pred': []} for i in range(10)}
    detailed_rows = [] 
    
    for d in dates:
        day_record = {"Ngày": d.strftime("%d/%m"), "KQ": kq_db.get(d, "N/A")}
        
        # Nếu không có dữ liệu
        if d not in kq_db or d not in data_cache: 
            for g in grp_stats: 
                grp_stats[g]['history'].append(None)
                grp_stats[g]['ranks'].append(999) # 999 biểu thị cho việc không có dữ liệu/Miss
                day_record[g] = "-"
            detailed_rows.append(day_record)
            continue
            
        curr_data = data_cache[d]
        df = curr_data['df']
        
        # Tìm cột lịch sử (Ngày hôm trước)
        prev_date = d - timedelta(days=1)
        if prev_date not in data_cache: 
            for k in range(2, 4):
                 if (d - timedelta(days=k)) in data_cache: 
                     prev_date = d - timedelta(days=k)
                     break
        
        hist_col_name = None
        if prev_date in data_cache:
             hist_col_name = data_cache[d]['hist_map'].get(prev_date)
        
        if not hist_col_name:
             for g in grp_stats: 
                 grp_stats[g]['history'].append(None)
                 grp_stats[g]['ranks'].append(999)
                 day_record[g] = "-"
             detailed_rows.append(day_record)
             continue
             
        hist_series = df[hist_col_name].astype(str).apply(lambda x: re.sub(r'[^0-9X]', '', x.upper().replace('S','6')))
        kq = kq_db[d]

        for g in grp_stats:
            mask = hist_series == g.upper()
            valid_mems = df[mask]
            
            num_stats = {}
            # --- Tính điểm (Scoring) ---
            for _, r in valid_mems.iterrows():
                p_cols = {c: get_col_score(c, score_map) for c in df.columns if get_col_score(c, score_map) > 0}
                processed = set()
                for col, pts in p_cols.items():
                    if col not in valid_mems.columns: continue
                    val = r[col]
                    for n in get_nums(val):
                        if n not in num_stats: num_stats[n] = {'p':0, 'v':0}
                        if n in processed: continue
                        num_stats[n]['p'] += pts
                    processed.update(get_nums(val))
            
            # --- Tính Vote ---
            for n in num_stats: num_stats[n]['v'] = 0
            for _, r in valid_mems.iterrows():
                p_cols = {c: get_col_score(c, score_map) for c in df.columns if get_col_score(c, score_map) > 0}
                found = set()
                for col in p_cols:
                    if col in r:
                         for n in get_nums(r[col]): 
                            if n in num_stats: found.add(n)
                for n in found: num_stats[n]['v'] += 1
            
            # --- Lọc & Xếp hạng ---
            filtered = [n for n, s in num_stats.items() if s['v'] >= min_v]
            
            if inverse:
                 sorted_res = sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['p'], int(n)))
            else:
                 sorted_res = sorted(filtered, key=lambda n: (-num_stats[n]['p'], -num_stats[n]['v'], int(n)))

            # Lấy Top theo cut_limit để xét Win/Miss và tính Rank
            top_list = sorted_res[:cut_limit]
            top_set = set(top_list)
            
            # Lưu dự đoán (chỉ lưu đè liên tục, giá trị cuối cùng sẽ là của ngày cuối)
            grp_stats[g]['last_pred'] = sorted(top_list)
            
            if kq in top_set:
                grp_stats[g]['wins'] += 1
                # Rank là vị trí của KQ trong dàn dự đoán (bắt đầu từ 1)
                rank = top_list.index(kq) + 1
                grp_stats[g]['ranks'].append(rank)
                grp_stats[g]['history'].append("W")
                day_record[g] = f"✅" # Có thể thêm rank vào UI nếu muốn: f"✅({rank})"
            else:
                grp_stats[g]['ranks'].append(999) # Miss = 999 điểm phạt
                grp_stats[g]['history'].append("L")
                day_record[g] = "░"
            
        detailed_rows.append(day_record)
            
    final_report = []
    for g, info in grp_stats.items():
        hist = info['history']
        valid_days = len([x for x in hist if x is not None])
        wins = info['wins']
        ranks = info['ranks']
        sum_ranks = sum(ranks)
        
        # Tính gãy thông
        max_lose = 0
        curr_lose = 0
        temp_lose = 0
        for x in reversed(hist):
            if x == "L": curr_lose += 1
            elif x == "W": break
        for x in hist:
            if x == "L": temp_lose += 1
            else:
                max_lose = max(max_lose, temp_lose)
                temp_lose = 0
        max_lose = max(max_lose, temp_lose)
        
        # Format dự đoán
        pred_str = ",".join(info['last_pred']) if info['last_pred'] else ""
        
        final_report.append({
            "Nhóm": g,
            "Số ngày trúng": wins,
            "Tổng Rank": sum_ranks, # Tiêu chí xếp hạng phụ
            "Tỉ lệ": f"{(wins/valid_days)*100:.1f}%" if valid_days > 0 else "0%",
            "Gãy thông": max_lose,
            "Gãy hiện tại": curr_lose,
            f"Dự đoán (Top {cut_limit})": pred_str,
            "_rank_list": sorted(ranks) # Dùng để sort ẩn, không hiển thị
        })
        
    df_rep = pd.DataFrame(final_report)
    
    # --- SẮP XẾP CHUẨN (GIỐNG CODE LÊN DÀN) ---
    # 1. Số ngày trúng (Giảm dần)
    # 2. Tổng Rank (Tăng dần - càng thấp càng tốt)
    # 3. List Rank chi tiết (Tăng dần - so sánh từng phần tử nếu tổng bằng nhau)
    if not df_rep.empty:
        # Pandas không sort trực tiếp list được dễ dàng, ta dùng sort thủ công
        # Chuyển DataFrame thành list dict để sort
        data_list = df_rep.to_dict('records')
        # Sort key: (-wins, sum_ranks, rank_list)
        data_list.sort(key=lambda x: (-x["Số ngày trúng"], x["Tổng Rank"], x["_rank_list"]))
        
        # Tạo lại DataFrame từ list đã sort, bỏ cột ẩn _rank_list
        df_rep = pd.DataFrame(data_list).drop(columns=["_rank_list"])
        
    return df_rep, pd.DataFrame(detailed_rows)
