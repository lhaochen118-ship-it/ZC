# 11 — Tài liệu ngoài: Text-to-SQL / Text-to-Code trên bảng (Bước 3 của `docs/13_TASK_AFTER_PLAN1.md`)

*09/11/2026. Chỉ đọc paper / báo cáo (không đọc code repo người khác). Số liệu là số tác giả báo cáo (BIRD = execution accuracy
EX; dev / test ghi rõ); bảng xếp hạng BIRD thay đổi theo thời điểm nên chỉ dùng để so thứ bậc. Nhóm lỗi theo `docs/10` mục 2:
1 ý định, 2 linking, 3 code không chạy / bị cắt, 4 nghiệp vụ, 5 định dạng, 6 câu bẩn, 7 chọn ứng viên.*

## 1. Bức tranh chung

- Hệ thống đứng đầu BIRD đều là **pipeline nhiều bước**: liên kết schema / giá trị → sinh **nhiều ứng viên đa dạng** → sửa
  bằng phản hồi thực thi → **bộ chọn** (selector) học riêng. Tự nhất quán (bỏ phiếu đa số) là mốc; selector học được thêm vài
  điểm. Mô hình mở cỡ 7–32B tới ~67–73 % BIRD nhờ **dữ liệu tổng hợp lớn được kiểm bằng thực thi** và **RL theo phần thưởng
  thực thi (GRPO)**.
- Bài toán của ta khác Text-to-SQL ở 3 điểm: (a) đầu ra là chương trình ngắn trên toolkit nghiệp vụ (`dm`, `industry`) nên
  phần "nghĩa nghiệp vụ" đã nằm trong hàm đã kiểm; (b) câu hỏi tiếng Việt chat (không dấu, viết tắt, phương ngữ); (c) có
  **bỏ trả lời** (sai bị trừ điểm) nên độ tin cậy quan trọng như độ chính xác.

## 2. Từng kỹ thuật

| Kỹ thuật (nguồn) | Ý tưởng | Mức tăng báo cáo | Chi phí | Áp dụng cho ta (nhóm lỗi) |
|---|---|---|---|---|
| **DIN-SQL** (Pourreza & Rafiei, NeurIPS 2023) | Chia bài: schema linking → phân loại độ khó → sinh theo loại → tự sửa | Spider test 85.3 % (GPT-4), BIRD ~56 %; mỗi bước +vài điểm trong ablation | rẻ (prompt) | đã có dạng `<plan>` + `<code>` và gợi ý; bài học: **phân loại câu** để chọn ví dụ / hàm (1, 4) |
| **DAIL-SQL** (Gao et al., VLDB 2024) | Chọn ví dụ theo **khung câu hỏi đã che thực thể** + độ giống của câu truy vấn nháp; trình bày ví dụ dạng (câu, SQL) | Spider test 86.6 % (GPT-4) | rẻ | đã có truy hồi n-gram theo khung (QĐ11); thêm **truy hồi hai bước theo chương trình nháp** (1, 4) |
| **MAC-SQL** (Wang et al., COLING 2025) | 3 agent: Selector (cắt schema), Decomposer, **Refiner** sửa theo lỗi thực thi | BIRD dev ~59 % (GPT-4); Refiner +2–4 | rẻ | lượt sửa lỗi của ta (3) |
| **CHESS** (Talaei et al., 2024) | Truy hồi giá trị (LSH) + mô tả cột, chọn schema, sinh, **unit test** để chọn ứng viên | BIRD test ~66.7 % | trung bình | `hints.entities` = value retriever; **unit test sinh tự động** cho bộ chọn (7) |
| **CHASE-SQL** (Pourreza et al., ICLR 2025) | 3 bộ sinh khác nhau (chia để trị, CoT theo kế hoạch truy vấn, ví dụ tổng hợp trực tuyến) + **bộ chọn so cặp** fine-tune | BIRD test 73.0–75 %; selector hơn tự nhất quán ~+4–5 điểm trên cùng tập ứng viên | trung bình–cao (nhiều mẫu + selector) | biến thể `direct` / `think` / `rephrase` = đa dạng bộ sinh; **selector so cặp** (7) |
| **XiYan-SQL** (Gao et al., 2024–2025) | Nhiều bộ sinh fine-tune với **phong cách khác nhau**, M-Schema, gom và chọn ứng viên bằng mô hình chọn | BIRD test 75.63 % | cao | các thành viên súp như bộ sinh riêng thay vì trộn trọng số; mô hình chọn (7) |
| **OmniSQL / SynSQL-2.5M** (Li et al., 2025) | Tổng hợp 16k DB + 2.5M (câu, SQL, CoT) kiểm bằng thực thi; SFT Qwen2.5-Coder 7/14/32B | 7B: +8.4 điểm trung bình 9 benchmark; 32B BIRD dev 67.0 % (đa số) | cao (sinh dữ liệu) | **nhà máy dữ liệu tổng hợp trên shop giả** sinh chương trình toolkit (1, 3, 4) |
| **Distillery** (Maamari et al., 2024) | Mô hình ngữ cảnh dài không cần cắt schema; tăng cường + sửa | BIRD 67.2 % (GPT-4o) | rẻ | xác nhận QĐ5 (prompt full, không cắt schema) |
| **SQL-R1** (Ma et al., 2025) | RL GRPO trên 7B: thưởng định dạng + thực thi + đúng kết quả + độ dài | BIRD dev ~66–67 %, Spider test ~88 % (7B) | cao (RL) | RL theo thưởng thực thi trên câu train + paraphrase (1, 3, 4) |
| **Reasoning-SQL** (Pourreza et al., 2025) | GRPO với **thưởng từng phần**: schema linking, AI-judge, n-gram, cú pháp, thực thi | 14B RL hơn o3-mini 4 điểm BIRD; trong pipeline CHASE: 72.78 % BIRD test | cao | thưởng từng phần cho chương trình: **kỳ đúng, thực thể đúng, hàm đúng** (1, 4) |
| **Alpha-SQL** (Li et al., ICML 2025) | Zero-shot MCTS trên các bước dựng truy vấn, thưởng tự nhất quán | BIRD dev 69.7 % (32B, không fine-tune); 7B ~66.8 %; nhiều rollout → tăng đều | cao lúc suy luận | tìm kiếm có cấu trúc theo `<plan>` (kỳ / phép đo / bộ lọc) khi phiếu tách (7) |
| **Agentar-Scale-SQL** (2025) | Mở rộng lúc suy luận: sinh đa dạng + suy luận nội tại + sửa lặp + **chọn kiểu đấu loại** | BIRD test 81.67 % (đứng đầu 11/2025) | rất cao | chọn đấu loại giữa các nhóm đáp án khi phiếu tách (7) |
| **Self-consistency** (Wang et al., ICLR 2023), **MBR-exec** (Shi et al., EMNLP 2022) | Gom ứng viên theo **kết quả thực thi**, chọn nhóm lớn | +2–10 điểm tuỳ tác vụ | rẻ–trung bình | đang dùng (vote theo đáp án) (7) |
| **Self-Debugging** (Chen et al., ICLR 2024) | Mô hình giải thích code + đọc lỗi thực thi rồi sửa | Spider +2–3; TransCoder +12 | rẻ | lượt sửa (3); thêm **giải thích kết quả** cho câu nghi ngờ |
| **LEVER** (Ni et al., ICML 2023) | **Verifier học được** chấm (câu, chương trình, kết quả thực thi), kết hợp xác suất sinh | +4.6–10.9 điểm (Spider, WikiTQ, GSM8k) so với mô hình gốc | trung bình (train verifier nhỏ) | **bộ chọn / bộ kiểm** trên log mẫu (7) và điểm tin cậy cho bỏ trả lời |
| **ExCoT** (Zhai et al., 2025) | CoT + **DPO theo phản hồi thực thi** (cặp đúng / sai) | LLaMA-3 70B BIRD dev 57 → 68.5 % | trung bình | DPO trên cặp chương trình đúng / sai của chính mô hình (1, 4) |
| **STaR / RFT** (Zelikman et al., 2022; Yuan et al., 2023) | Lấy mẫu từ chính mô hình, giữ lời giải đúng (kiểm bằng thực thi), SFT lại | +vài điểm mỗi vòng | trung bình | D3 "self" đã làm; lặp lại với gợi ý mới (1, 3) |
| **SENSE** (Yang et al., ACL 2024), **CodeS** (Li et al., SIGMOD 2024), **DTS-SQL** (2024), **SLM-SQL** (2025) | Mô hình nhỏ: dữ liệu tổng hợp từ mô hình mạnh + yếu, tiền huấn luyện SQL, fine-tune tách bước, SFT + GRPO cho 0.5–1.5B | 7B ngang GPT-4 thời điểm đó trên Spider; mô hình rất nhỏ tăng mạnh nhờ GRPO | trung bình | chưng cất từ thầy (Qwen3.6-27B, Gemma 4 31B đã có ở WP5) (1, 4) |
| **TrustSQL** (Lee et al., ICLR 2025) và các bài về bỏ trả lời | Đánh giá cả khả năng **từ chối câu không trả lời được**; phạt trả lời sai | — (benchmark) | — | nhóm "dữ liệu không ghi" (1) + ngưỡng bỏ trả lời |
| **Model soup / WiSE-FT** (Wortsman et al., ICML 2022 / CVPR 2022) | Trung bình trọng số các bản fine-tune, nội suy về gốc | bền hơn khi phân phối lệch | rẻ | đang dùng (QĐ9) |
| **Spider 2.0** (Lei et al., ICLR 2025) | Bài toán doanh nghiệp, agent đọc tài liệu + nhiều bước | agent tốt nhất thời điểm ra mắt ~17–21 % | — | bài học: tài liệu (rulebook) phải được **đưa đúng đoạn** cho mô hình (4) |

## 3. Tóm tắt theo nhóm lỗi của ta

| Nhóm (docs/10) | Kỹ thuật ứng viên, rẻ → đắt |
|---|---|
| 1 Ý định | gợi ý đúng hơn (đã sửa F2, F3) → ví dụ chọn theo chương trình nháp (DAIL) → cảnh báo kỳ thành cờ (F5) → DPO / RL thưởng từng phần (ExCoT, Reasoning-SQL) |
| 2 Linking | value retriever (CHESS; ta: `hints.entities`) → mô tả nhóm / thuộc tính trong card |
| 3 Không chạy / bị cắt | **ngân sách decoding** + ép định dạng (F1) → Self-Debugging → SFT lại cho định dạng gọn (RFT) |
| 4 Nghiệp vụ | đưa đúng định nghĩa rulebook (F3) → hàm toolkit tổng quát cho mẫu câu hay gặp → dữ liệu tổng hợp (OmniSQL) |
| 5 Định dạng | chuẩn hoá theo kiểu được gợi ý (F7, luật) |
| 6 Câu bẩn | làm sạch câu (F4); paraphrase / nhiễu trong huấn luyện (D2N đã có) |
| 7 Chọn ứng viên | bỏ phiếu theo kết quả (đang có) → verifier học được (LEVER) / so cặp (CHASE) / đấu loại (Agentar) → hiệu chỉnh ngưỡng bỏ trả lời |

## 4. Nguồn

- BIRD: Li et al., NeurIPS 2023 D&B; bảng xếp hạng https://bird-bench.github.io (số liệu thay đổi theo thời điểm).
- DIN-SQL: arXiv 2304.11015. DAIL-SQL: arXiv 2308.15363. MAC-SQL: arXiv 2312.11242. CHESS: arXiv 2405.16755.
- CHASE-SQL: arXiv 2410.01943. XiYan-SQL: arXiv 2411.08599 / 2507.04701. OmniSQL: arXiv 2503.02240.
- Distillery: arXiv 2408.07702. SQL-R1: arXiv 2504.08600. Reasoning-SQL: arXiv 2503.23157. Alpha-SQL: arXiv 2502.17248 (ICML 2025).
- Agentar-Scale-SQL: arXiv 2509.24403. Self-consistency: arXiv 2203.11171. MBR-exec: arXiv 2204.11454. Self-Debugging: arXiv 2304.05128.
- LEVER: arXiv 2302.08468. ExCoT: arXiv 2503.19988. STaR: arXiv 2203.14465. RFT: arXiv 2308.01825. SENSE: arXiv 2408.03256.
- CodeS: arXiv 2402.16347. SLM-SQL: arXiv 2507.22478. TrustSQL: arXiv 2403.15879. Model soups: arXiv 2203.05482. WiSE-FT: arXiv 2109.01903.
- Spider 2.0: arXiv 2411.07763.
