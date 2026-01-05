            with tab4:
                st.subheader("🧬 AI GENETIC HUNTER (Săn Tìm Cấu Hình)")
                st.info("Sử dụng giải thuật di truyền để thử hàng ngàn tổ hợp điểm số, tìm ra cấu hình 'độc nhất' phù hợp với dữ liệu hiện tại.")
                
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    target_hunter = st.date_input("Ngày dự đoán:", value=last_d, key="t_hunter")
                    max_nums_hunter = st.slider("Max Số Lượng chấp nhận:", 40, 85, 65, key="mx_hunter")
                    
                    st.write("**Cấu hình AI Scan:**")
                    pop_size = st.select_slider("Kích thước quần thể (Mẫu/Thế hệ):", options=[20, 50, 100], value=50)
                    n_gen = st.select_slider("Số thế hệ (Vòng lặp tiến hóa):", options=[5, 10, 20, 50], value=10)
                    
                    total_scenarios = pop_size * n_gen
                    st.caption(f"⚡ AI sẽ chạy thử nghiệm khoảng **{total_scenarios}** cấu hình.")

                    if st.button("🧬 BẮT ĐẦU SĂN (DEEP SCAN)", type="primary"):
                        # --- KIỂM TRA DỮ LIỆU TRƯỚC KHI CHẠY ---
                        check_past_dates = []
                        check_d = target_hunter - timedelta(days=1)
                        scan_limit = 0
                        while len(check_past_dates) < 7 and scan_limit < 60:
                            if check_d in kq_db and check_d in data_cache:
                                check_past_dates.append(check_d)
                            check_d -= timedelta(days=1)
                            scan_limit += 1
                        
                        if len(check_past_dates) < 5:
                            st.error(f"🔴 KHÔNG CHẠY ĐƯỢC: Thiếu dữ liệu lịch sử!")
                            st.warning(f"AI cần ít nhất 5 ngày có KQ trước ngày {target_hunter.strftime('%d/%m')} để học.")
                            st.write(f"Hiện tại chỉ tìm thấy: {len(check_past_dates)} ngày.")
                            st.info("👉 Gợi ý: Hãy upload thêm file của tháng trước đó.")
                        else:
                            st.toast("🚀 Đủ dữ liệu! AI đang khởi động...", icon="🧬") 
                            prog_bar = st.progress(0)
                            status_txt = st.empty()
                            
                            best_scenarios = run_genetic_search(
                                target_hunter, data_cache, kq_db, limit_cfg, 
                                MIN_VOTES, USE_INVERSE, max_nums_hunter,
                                generations=n_gen, population_size=pop_size,
                                progress_bar=prog_bar, status_text=status_txt
                            )
                            
                            prog_bar.empty()
                            if not best_scenarios:
                                status_txt.warning("⚠️ Đã chạy xong nhưng không tìm được dàn nào dưới số lượng quy định (Max Số Lượng). Hãy tăng Max lên.")
                            else:
                                status_txt.success("✅ Hoàn tất quá trình tiến hóa!")
                                st.session_state['best_scenarios'] = best_scenarios
                
                with c2:
                    if 'best_scenarios' in st.session_state:
                        scenarios = st.session_state['best_scenarios']
                        if not scenarios:
                            st.warning("⚠️ Không tìm thấy cấu hình nào thỏa mãn điều kiện.")
                        else:
                            st.success(f"🎉 Tìm thấy {len(scenarios)} cấu hình ưu tú nhất!")
                            for idx, sc in enumerate(scenarios):
                                with st.expander(f"🏅 #{idx+1} ({sc['Name']}) | Win {sc['WinRate']:.1f}% | TB {sc['AvgNums']:.1f} số", expanded=(idx==0)):
                                    st.write("Cấu hình điểm:")
                                    st.json(sc['Scores'])
                                    if st.button(f"👉 Áp dụng Cấu hình #{idx+1}", key=f"apply_gen_{idx}"):
                                        apply_hunter_callback(sc['Scores'])
