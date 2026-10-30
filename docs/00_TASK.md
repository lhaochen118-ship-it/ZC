# 00 — Đề bài: Zalo AI Challenge 2026, track DataMate

*Nguồn: https://challenge.zalo.ai/portal/datamate (đọc ngày 2026-10-30 bằng Chromium headless, gồm các tab Description,
Evaluation, Rule, Submission Guidance, Timeline). Tab **Data** không đọc được trên web; nội dung của nó nằm trong
`data/data.txt` do user cung cấp (§3 sẽ được điền khi có file).*

## 1. Bài toán

Xây một **AI agent** trả lời câu hỏi tiếng Việt đời thường, trực tiếp trên dữ liệu có cấu trúc (CSV, Excel, SQLite) của
các hộ kinh doanh nhỏ ở Việt Nam. Agent trả về đáp án đúng, **hoặc từ chối khi dữ liệu không đủ để trả lời**.

Dữ liệu được mô tả là "messy, real-world": Excel nhiều sheet, tên sản phẩm không thống nhất, cách hỏi khẩu ngữ ("sp bán
dc nhiều nhất tuần rồi là gì"), và logic nghiệp vụ như giá vốn, công nợ quá hạn.

> Correct answers score, wrong answers cost, silence is free: build a system that is right when it speaks.

### Input

- **Gói dữ liệu của một doanh nghiệp:** một hoặc nhiều file (`.csv`, `.xlsx`, `.sqlite`) và một `README.md` ngắn mô tả
  doanh nghiệp.
- **Câu hỏi** bằng tiếng Việt đời thường.
- **as-of date:** "hôm nay", dùng để quy đổi các mốc thời gian tương đối ("tháng này", "tuần rồi", "cùng kỳ năm ngoái").
- **Rulebook** và **lịch âm–dương**, dùng chung cho mọi doanh nghiệp trong cùng split.

Ví dụ một dòng `questions.jsonl`:

```json
{"id": "q_06de33fa21cc", "business_id": "biz_118", "question": "sp bán dc nhiều nhất tuần rồi là gì", "as_of_date": "2026-03-04"}
```

### Output

Một dòng `submission.jsonl` cho mỗi câu hỏi:

```json
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg"}
```

## 2. Kiểu đáp án và luật so khớp (tab Evaluation)

| type | Nội dung | Ví dụ | Luật so khớp |
|---|---|---|---|
| `number` | Số (tiền VND, số đếm, số lượng) | `48250000` | Khớp chính xác. Đáp án chuẩn là số nguyên hoặc làm tròn 2 chữ số thập phân |
| `percent` | Phần trăm, đơn vị % | `-12.45` | Khớp sau khi làm tròn 2 chữ số; **dấu phải đúng** |
| `string` | Tên (sản phẩm, khách, chi nhánh, thứ…) | `"Sting dâu 330ml"` | Chuẩn hoá rồi so bằng: (1) Unicode NFC; (2) lowercase; (3) ký tự không phải chữ/số/khoảng trắng → space; (4) gộp khoảng trắng. **Không bỏ dấu.** Chấp nhận mọi cách viết của tên có trong dữ liệu của doanh nghiệp đó, **và cả mã** (ví dụ `sp0042`) |
| `date` | Ngày ISO | `"2026-08-15"`, `"2025-09"` | Khớp chính xác `yyyy-mm-dd`; câu hỏi về tháng thì `yyyy-mm` |
| `boolean` | `true` / `false` | `true` | Khớp chính xác |
| `list` | Danh sách tên hoặc số, theo thứ tự câu hỏi yêu cầu (hoặc mặc định của rulebook) | `["Cà phê sữa", "Bạc xỉu", "Trà đào"]` | **Luôn có thứ tự**, so theo vị trí; phần tử so bằng luật `string` hoặc `number` |
| `table` | Bảng nhỏ | `{"columns": [...], "rows": [[...], ...]}` | Cột khớp theo tên đã chuẩn hoá, không khớp thì theo vị trí; **hàng so như tập hợp**; ô tên theo luật `string`, ô số theo luật `number` |
| `unanswerable` | Dữ liệu không chứa thông tin cần thiết | — | Đúng khi và chỉ khi đáp án chuẩn là `unanswerable` |
| `abstain` | Agent chọn không trả lời | — | Luôn tính là abstain |

- Câu trả lời đúng khi **type trùng** type chuẩn **và** giá trị khớp; không có điểm từng phần.
- `abstain` hoặc thiếu đáp án → abstain.
- **Câu hỏi xếp hạng:** theo tiêu chí trong câu hỏi, hoặc tiêu chí mặc định của rulebook; giảm dần; **chỉ tính đơn hợp lệ
  (valid orders)**; bằng nhau thì xếp theo **mã tăng dần**. Câu có hoà ở biên k (hoặc ở hạng 1 với câu "… nhất") bị loại
  khỏi test. Có ít hơn k ứng viên thì đáp án là tất cả.

## 3. Dữ liệu (tab Data — chờ `data/data.txt`)

Nguồn: Kaggle dataset public `nhatlong1103/zc-2026` (gồm `data.txt` = tab Data, `train/`, `public_test/`).

```
<split>/data/<business_id>/  *.csv | *.xlsx | *.sqlite + README.md   rulebook.md   calendar.csv   questions.jsonl   answers.jsonl (train)
```

| Split | Doanh nghiệp | Câu hỏi | File dữ liệu | as_of_date |
|---|---|---|---|---|
| train | 30 (`biz_101`–`biz_130`), ~100 câu mỗi DN | 3,000 (có đáp án) | 107 xlsx, 20 csv, 1 sqlite | 2025-01-15 .. 2026-09-20 |
| public_test | 14 (`dev_001`–`dev_014`) | 1,000 | 69 xlsx, 10 csv, 1 sqlite | 2025-02-04 .. 2026-09-30 |

- **Phân bố type (train):** number 1884, percent 578, list 193, date 145, string 88, unanswerable 70, boolean 23, table 19.
- `answers.jsonl`: `{"id", "type", "answer", "alias"}`; `alias` = các cách viết khác + mã (list/table: một list alias cho
  mỗi phần tử/dòng); `unanswerable` có `"answer": null`.
- **Ngành:** tạp hoá, cà phê/trà sữa, nhà hàng, nhà thuốc, vật liệu xây dựng, sửa xe, mẹ và bé, điện thoại… Mỗi DN
  12–30 tháng, vài trăm đến ~300k dòng. **Private test có ngành không có trong train/public.**
- **Rulebook mỗi split một bản, có khác nhau** (public thêm: khách trung thành, giờ cao điểm, nhà cung cấp chính, topping
  ghi chung ô…). Agent phải đọc rulebook của split đang chạy, không hard-code luật train.
- README mỗi DN chất lượng khác nhau: có DN đủ từ điển dữ liệu, có DN gần như trống.
- **Dữ liệu bẩn thật:** dòng tiêu đề trước header, header lặp giữa dữ liệu, dòng "Tổng tháng"/"Ghi chú", số dạng
  `10.000đ`, `105k`, `1tr440`, `531 500`, `1 bao 25kg`; ngày `01/01/2025`, `2025-01-01`, `1 thg 6 2025`, serial Excel
  (cả dạng chuỗi `"46183"`), `1/1` không có năm (lấy năm của sheet); trạng thái viết nhiều kiểu (`Xong`, `ok`,
  `Đã hoàn thành`, `hủy`, `Hủy`…); một danh sách tách nhiều file/sheet theo tháng.
- Docker phải có `predict.sh` đọc `/data`, ghi `/result/submission.jsonl` và log từng câu ở `/result/logs/` (id, đáp
  án, thời gian, cách ra đáp án: code sinh ra, output thực thi).

Phần Rule cho biết: khi chấm, BTC mount test set vào `/data` (read-only) và lấy `/result/submission.jsonl`; image phải
ghi log (chi tiết định dạng log nằm ở mục "Submission format" của tab Data).

## 4. Điểm: Penalized Accuracy

$$\mathrm{PA} = \frac{1}{N}\sum_{i=1}^{N} s_i,\qquad s_i = \begin{cases} 1 & \text{đúng} \\ 0 & \text{abstain} \\ -1 & \text{sai} \end{cases}$$

- Trả lời bất kỳ giá trị nào cho câu `unanswerable` là **sai**.
- Mọi câu có trọng số như nhau.
- **Hoà điểm:** xếp theo **tổng thời gian inference** trong lần BTC chạy lại, rồi theo thời điểm nộp sớm hơn.

**Hệ quả quyết định (dùng xuyên suốt):** với xác suất đúng ước lượng p của đáp án tốt nhất, kỳ vọng điểm khi trả lời là
2p − 1, khi abstain là 0. Vậy **trả lời khi và chỉ khi p > 0.5**. `unanswerable` cũng là một "đáp án" phải có p > 0.5.

## 5. Luật kỹ thuật (tab Rule)

1. Đội và điều kiện theo quy chế chung của Zalo AI Challenge 2026.
2. **Mô hình:** trọng số public **trước 2026-10-01**, giấy phép cho phép dùng; fine-tune thoải mái. **Tổng tham số của
   mọi mô hình dùng lúc inference ≤ 15B** (LLM, embedding, reranker, classifier; tính cả mọi expert của MoE; dù có nạp
   cùng lúc hay không). Quantization tuỳ ý. Phải khai báo mô hình trong README của solution.
3. **Không dịch vụ ngoài:** chạy offline hoàn toàn trong Docker, không mạng.
4. **Không có người trong vòng lặp.** Cấm hard-code đáp án hoặc logic dựa trên id câu hỏi / id doanh nghiệp.
5. Agent được sinh và chạy code trong container. Thư mục dữ liệu read-only; file tạm ghi chỗ khác.
6. **Dữ liệu:** được dùng train data, dữ liệu tự tổng hợp, dataset/mô hình public. **Không** dùng public/private test để
   train, không thu thập đáp án test bằng bất cứ cách nào.
7. **Máy chấm:** 2 × NVIDIA RTX 5090 (32 GB mỗi GPU, 64 GB tổng), 512 GB RAM. **Toàn bộ private test phải xong trong
   6 giờ**, kể cả nạp mô hình. Blackwell cần CUDA ≥ 12.8 và PyTorch/vLLM tương ứng; base image theo Docker guideline.
8. **Điểm cuối = điểm BTC chạy lại Docker** trên private test. Phải đủ tất định: |PA(nộp) − PA(chạy lại)| ≤ 0.03, nếu
   không giải thích được thì bị loại. Quá giờ hoặc dừng sớm: chỉ chấm các dòng đã ghi vào `/result/submission.jsonl`,
   còn lại tính abstain → **ghi đáp án tăng dần (append + flush) trong lúc chạy.**

## 6. Lịch và cách nộp

| Mốc | Việc |
|---|---|
| 26/10 | Mở: dev kit, train + public test, phần mềm chấm |
| 26/10 – 16/11 | Vòng public: nộp `submission.jsonl` trên portal, **tối đa 5 lần/ngày** |
| **16/11 23:59 (GMT+7)** | **Nộp solution hoàn chỉnh** (code train, code inference, dữ liệu) dạng Docker: link Google Drive public + checksum, form "Solution Submission"; tối đa 2 lần, lấy lần cuối |
| 20/11 9:00 | BTC phát private test |
| 20/11 – 23/11 23:59 | Chạy solution đã đóng băng trên private test, nộp jsonl |
| 2/12 | Công bố top 2 mỗi track |
| 12/12 | Bảng xếp hạng cuối |

Giải: nhất 3,500 USD, nhì 2,500 USD, top 5 mỗi track có quà.

## 7. Quan sát từ public leaderboard (30/10)

20 đội đầu đều có **Score = 1** (nộp từ 26–28/10). Có hai khả năng: public test nhỏ/dễ, hoặc điểm hiển thị đã làm tròn.
Nếu nhiều đội đạt PA ≈ 1 thì thứ hạng cuối sẽ do **private test** (khó hơn hoặc khác phân phối) và **thời gian
inference** quyết định. Kế hoạch (`01_PLAN.md`) vì thế tối ưu cả độ bền trên dữ liệu mới lẫn tốc độ.
