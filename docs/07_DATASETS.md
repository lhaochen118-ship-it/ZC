# 07 — Dataset ngoài cho huấn luyện / chưng cất mô hình sinh code truy vấn

*Lập 08/11/2026. Số citation: Semantic Scholar (API batch, tra 08/11/2026; con số thay đổi theo thời gian, chỉ để so
sánh tương đối). Hạng hội nghị theo CORE (A* / A / B / C). Dùng cho §4.8 của `docs/06_HYBRID_FRAMEWORK.md`.*

## 1. Quy tắc dùng dataset (user chốt 08/11 — không hỏi lại)

1. **Dataset công khai (public) thì được dùng**, kể cả giấy phép "chỉ nghiên cứu / giáo dục" (dự án phục vụ mục đích giáo
   dục, nghiên cứu). Không cần hỏi user từng dataset.
2. **Chỉ dùng dataset uy tín:** đã qua bình duyệt ở hội nghị / tạp chí có hạng (CORE A* / A, hoặc Findings / workshop của
   các hội nghị này) **hoặc** được trích dẫn nhiều (mức tham khảo: ≥ ~50 citation trên Semantic Scholar). Dataset chỉ có
   trên arXiv, ít citation → không dùng, trừ khi user đồng ý.
3. Vẫn giữ luật BTC: **không dùng public / private test của cuộc thi để train** (kể cả paraphrase của câu public test);
   mô hình (kể cả mô hình thầy dùng để chưng cất) có trọng số public trước 01/10/2026; tổng tham số lúc inference ≤ 15B.
4. Phần test ẩn của benchmark (gửi tác giả chấm) không dùng được để train; chỉ dùng phần train / dev công khai.
5. Ghi rõ dataset đã dùng (tên, phiên bản, giấy phép) trong `docker/README_SOLUTION.md` khi nộp.

## 2. Text-to-SQL (sinh SQL)

| Dataset | Citation | Nơi công bố (hạng) | Quy mô | Truy cập | Ghi chú cho DataMate |
|---|---|---|---|---|---|
| **Spider** (Yu et al.) | 2414 | EMNLP 2018 (A*) | 10,181 câu / 200 CSDL | Public (CC BY-SA 4.0), cả test (từ 2024) | Chuẩn cross-domain; tiếng Anh |
| **BIRD** | 1387 | NeurIPS 2023 D&B (A*) | 12,751 câu / 95 CSDL lớn, bẩn | Train / dev public; test ẩn | Gần nhất về dữ liệu bẩn + tri thức nghiệp vụ ("evidence") |
| **WikiSQL** (Seq2SQL) | 1747 | arXiv 2017 (không bình duyệt) nhưng trích dẫn rất nhiều | 80,654 câu / 24k bảng | Public (BSD-3) | Một bảng, SQL đơn giản; đạt quy tắc 2 nhờ citation |
| **Spider 2.0** | 391 | ICLR 2025 (A*) | 632 tác vụ doanh nghiệp | Public; một phần cần tài khoản Snowflake / BigQuery | Quy trình nhiều bước, quá khó / khác bối cảnh cho 9B |
| **CoSQL** | 352 | EMNLP 2019 (A*) | ~3k hội thoại, 30k+ lượt | Train / dev public; test ẩn | Hội thoại, ít liên quan |
| **Text2SQL eval (ATIS, GeoQuery, Advising, …)** (Finegan-Dollak) | 329 | ACL 2018 (A*) | 8 bộ cũ (ATIS 5,280, Advising 4,570, …) | Public (GitHub text2sql-data) | Miền hẹp, lâu đời |
| **SParC** | 280 | ACL 2019 (A*) | 4,298 chuỗi câu (~12k câu) | Train / dev public; test ẩn | Câu theo ngữ cảnh |
| **Spider-Realistic** | 246 | NAACL 2021 (A) | 508 câu | Public | Kiểm tra bỏ tên cột khỏi câu |
| **KaggleDBQA** | 213 | ACL 2021 (A*) | 272 câu / 8 CSDL thật | Public | CSDL thật, nhỏ; chỉ để đánh giá |
| **Spider-Syn** | 197 | ACL 2021 (A*) | ~8k câu (đồng nghĩa) | Public | Bền vững với từ đồng nghĩa (giống vấn đề paraphrase của ta) |
| **Spider-DK** | 166 | EMNLP 2021 (A*) | 535 câu | Public | Cần tri thức miền |
| **SynSQL-2.5M** (OmniSQL) | 149 | PVLDB 2025 (A*) | 2.54 triệu mẫu (CSDL + câu + SQL + CoT), 16,583 CSDL | Public (HF, Apache-2.0) | **Lớn nhất, có CoT**; SQLite, tiếng Anh; tốt cho chưng cất kỹ năng truy vấn chung |
| **SQUALL** (WikiTableQuestions + SQL) | 136 | Findings EMNLP 2020 | 11,276 câu | Public | Bảng đơn, có SQL căn chỉnh |
| **Dr.Spider** | 94 | ICLR 2023 (A*) | 17 bộ nhiễu (~15k) | Public | Kiểm tra bền vững (câu / CSDL / SQL) |
| **AmbiQT** | 87 | EMNLP 2023 (A*) | ~3k câu mơ hồ | Public | Câu mơ hồ → nhiều SQL hợp lệ |
| **EHRSQL** | 85 | NeurIPS 2022 D&B (A*) | ~24k câu bệnh án | Câu hỏi public; **CSDL MIMIC / eICU cần xin quyền PhysioNet** | Có câu không trả lời được (abstain); miền y tế |
| **CSpider** | 82 | EMNLP 2019 (A*) | ~10k (Spider tiếng Trung) | Public | Đa ngôn ngữ |
| **SEDE** | 82 | NLP4Prog workshop @ ACL 2021 | 12,023 câu (Stack Exchange) | Public (Apache-2.0) | Câu tự nhiên, SQL thật |
| **DuSQL** | 73 | EMNLP 2020 (A*) | 23,797 câu tiếng Trung | Public qua Baidu LUGE (cần đăng ký tài khoản) | Có câu tính toán (phép toán, ngày) |
| **ViText2SQL** (VinAI) | 68 | Findings EMNLP 2020 | ~10k câu (Spider tiếng Việt) | Public (GitHub / HF), giấy phép nghiên cứu / giáo dục | **Tiếng Việt duy nhất** — dùng được theo quy tắc 1 |
| **CHASE** | 59 | ACL 2021 (A*) | 5,459 chuỗi / 17,940 câu tiếng Trung | Public | Câu theo ngữ cảnh |
| **MultiSpider** | 47 | AAAI 2023 (A*) | Spider × 7 ngôn ngữ (có tiếng Việt) | Public | Gần 50 citation; tiếng Việt máy dịch |
| **Archer** | 32 | EACL 2024 (A) | 1,042 câu (số học, thường thức, giả định) | Train / dev public; test ẩn | Câu tính toán phức tạp; citation thấp nhưng hội nghị hạng A |
| **TrustSQL** | 26 | arXiv 2024 (chưa thấy hội nghị) | 4 CSDL, câu trả lời được / không | Public (phần EHRSQL cần PhysioNet) | Đúng kiểu chấm có phạt như PA; **không đạt quy tắc 2** (ít citation, chưa bình duyệt) |
| **BEAVER** | 83 | arXiv 2024 | CSDL doanh nghiệp | Một phần public | Chưa bình duyệt; dùng nếu user đồng ý |

## 3. Python / pandas / phân tích dữ liệu (gần định dạng "chương trình trên toolkit" của ta)

| Dataset | Citation | Nơi công bố (hạng) | Quy mô | Truy cập | Ghi chú cho DataMate |
|---|---|---|---|---|---|
| **WikiTableQuestions** | 1177 | ACL 2015 (A*) | 22,033 câu / 2,108 bảng | Public (CC BY-SA) | QA trên bảng, đáp án (không có code); dùng tạo dữ liệu với thầy |
| **DS-1000** | 669 | ICML 2023 (A*) | 1,000 bài (pandas 291, numpy, …) | Public (CC BY-SA) | Sinh code pandas có test; chủ yếu để đánh giá |
| **TableBench** (+ TableInstruct) | 197 | AAAI 2025 (A*) | 886 câu test + ~20k mẫu huấn luyện | Public | **Có tập huấn luyện** QA bảng có suy luận số học; có lời giải dạng code |
| **InfiAgent-DABench** | 179 | ICML 2024 (A*) | 257 câu / 52 file CSV | Public | Agent phân tích dữ liệu CSV, đáp án đóng — gần bài toán |
| **DSBench** | 143 | ICLR 2025 (A*) | 466 tác vụ phân tích + 74 mô hình hoá | Public | File Excel / nhiều bảng thật; khó |
| **ARCADE** | 132 | ACL 2023 (A*) | 1,082 bài pandas trong notebook | Public | pandas theo ngữ cảnh notebook |
| **DataSciBench** | 72 | arXiv 2025 | 222 tác vụ | Public | Chưa bình duyệt |
| **DataBench** | 55 | LREC-COLING 2024 (B/C) | 1,300 câu / 65 bảng thật | Public (HF) | Bảng thật, đáp án kiểu số / list / boolean — giống định dạng đáp án DataMate |
| **DABstep** | 54 | arXiv 2025 | 450 tác vụ nhiều bước, nhiều file | Public (dev có nhãn; test chấm qua leaderboard) | **Gần DataMate nhất** (dữ liệu doanh nghiệp, nhiều file, quy tắc nghiệp vụ trong tài liệu); chưa bình duyệt nhưng > 50 citation |
| **Text2Analysis** | 52 | AAAI 2024 (A*) | 2,249 câu | Public | Phân tích nâng cao, câu mơ hồ |

## 4. Gợi ý lựa chọn (để user cân nhắc)

- **Cho chưng cất kỹ năng sinh truy vấn chung:** SynSQL-2.5M (lớn, có CoT, Apache-2.0) + BIRD train (dữ liệu bẩn, có
  tri thức nghiệp vụ) + Spider train. Cần chuyển sang dạng "chương trình trên toolkit / pandas" hoặc giữ SQL làm kỹ
  năng phụ.
- **Cho tiếng Việt:** ViText2SQL (duy nhất có bình duyệt), MultiSpider (tiếng Việt máy dịch) — giúp hiểu câu tiếng Việt,
  không giúp kiến thức nghiệp vụ.
- **Gần bài toán nhất (phân tích dữ liệu doanh nghiệp, đáp án đóng):** InfiAgent-DABench, DABstep (dev), TableBench
  (TableInstruct có tập huấn luyện), DataBench.
- **Cho abstain / câu không trả lời được:** EHRSQL (cần PhysioNet), TrustSQL (không đạt quy tắc 2).
- **Dữ liệu quan trọng nhất vẫn là của ta:** 3000 câu train + câu tổng hợp trên dữ liệu train (đáp án tính bằng toolkit),
  vì định dạng, nghiệp vụ, tiếng Việt và kiểu đáp án khớp đề.

## Nguồn

Semantic Scholar Graph API (`/graph/v1/paper/batch`, 08/11/2026); trang arXiv / ACL Anthology của từng paper; ICLR 2025
DSBench: https://www.iclr.cc/virtual/2025/poster/30458 ; TrustSQL: https://arxiv.org/abs/2403.15879 ;
ViText2SQL: https://huggingface.co/datasets/SEACrowd/vitext2sql ; OmniSQL: https://arxiv.org/abs/2503.02240 .
