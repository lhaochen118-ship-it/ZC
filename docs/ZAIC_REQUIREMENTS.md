# ZAIC.docx — yêu cầu của user (bản text)

*Trích nguyên văn từ `ZAIC.docx` (gốc ở thư mục gốc repo). Token không có trong file và không được ghi vào repo.*


## 1. Mục tiêu chính
Mục tiêu là xây dựng 1 hệ thống có khả năng:
- Lọc và làm sạch dữ liệu
- Sinh được câu truy vấn để trả về đáp án đúng chính xác như yêu cầu của người dùng.
- Trả lời đúng chính xác tất cả câu hỏi trong tập test.
- Khi đã đạt được điều này thì mới tạo file submission cho tôi nộp bài.

## 2. Quy trình nghiên cứu tự động, lặp và cải tiến liên tục
Agent chủ động thực hiện vòng lặp nghiên cứu sau:

### Bước 1 — Phân tích sâu bản chất bài toán
- Nghiên cứu kỹ những hạn chế cốt lõi của proposed method hiện tại.
- Đào sâu các cơ sở lý thuyết liên quan và những hướng tiếp cận có liên quan.
- Tham khảo những giải pháp đã có.
- Xác định nguyên nhân khiến phương pháp hiện tại chưa đạt hiệu quả.

### Bước 2 — Đề xuất giả thuyết và phương pháp mới
- Lựa chọn thuật toán hoặc mô hình phù hợp
- Tìm kiếm giải pháp giải quyết trực tiếp những hạn chế mang tính bản chất.

### Bước 3 — Triển khai và tối ưu mã nguồn
- Tự thiết kế, viết mã và tích hợp phương pháp mới.
- Tối ưu thuật toán, tài nguyên CPU/GPU và khả năng xử lý song song.

### Bước 4 — Đánh giá thực nghiệm
- Đánh giá trên tập test.
- Làm sao cho tối ưu càng nhiều càng tốt trên tập test, nhưng mô hình / thuật toán bạn đề xuất cũng phải đảm bảo xử lý được trong trường hợp tổng quát.
- Tránh fix cứng lên trên tập test này.

### Bước 5 — Phân tích và cải tiến
- Xác định nguyên nhân thành công hoặc thất bại của từng giả thuyết.
- Loại bỏ các hướng không hiệu quả, cải tiến những hướng có triển vọng.
- Khi cần thiết, quay lại nghiên cứu lý thuyết để đề xuất giả thuyết mới.

### Bước 6 — Lặp lại quy trình
Tiếp tục chu trình:
Phân tích bản chất → Đề xuất giả thuyết → Thiết kế phương pháp → Triển khai → Đánh giá → Phân tích → Cải tiến.
Agent tự chủ trong việc lựa chọn và thực hiện các vòng thử nghiệm khả thi, tận dụng tối đa tài nguyên được cấp để tiến tới phương pháp có hiệu quả cao hơn.
Không đánh đổi khả năng tổng quát hóa, tính đúng đắn khoa học hoặc độ chính xác chỉ để tăng score trên một dataset.

## 4. Tài nguyên Kaggle
Danh sách API token của các tài khoản Kaggle được cung cấp dưới đây:
[Chèn danh sách API token Kaggle đã cung cấp vào đây]

### 4.1. Cấu hình và bảo mật
- Sử dụng tất cả tài khoản Kaggle được cung cấp.
- Cấu hình Kaggle CLI thông qua biến môi trường KAGGLE_API_TOKEN hoặc tệp ~/.kaggle/access_token với quyền truy cập 600.
- Xác minh kết nối bằng lệnh kaggle competitions list.
- Bảo mật toàn bộ token gốc: lưu trữ riêng, không đưa trực tiếp vào prompt, mã nguồn, GitHub hoặc báo cáo.

### 4.2. Tự động hóa và quản lý phiên
- Tự động hóa tối đa các thao tác nghiên cứu, triển khai và thử nghiệm thông thường. Chủ động thực hiện mà không cần xin xác nhận ở từng bước nhỏ.
- Nếu phiên không xuất hiện đầu ra trong khoảng 10 phút đối với T4 hoặc 8 phút đối với L4/CPU, hãy kiểm tra trạng thái. Nếu L4 có dấu hiệu bị kẹt trong hàng đợi, ngắt phiên và thử tạo phiên mới, tối đa 5 lần. Nếu vẫn không được, kiểm tra xem có lỗi code, logic gì ko? Nếu có thì sửa rồi chạy lại. Nếu không thì, chuyển sang dùng 2 x GPU T4.
- Nếu vẫn thất bại, kiểm tra log, quota, cấu hình và xác định nguyên nhân.
- Khi phát hiện tiến trình treo, chủ động ngắt và khắc phục. Nếu đã thử các phương án khả thi nhưng không thể ngắt, thông báo để người dùng xử lý thủ công.
- Khi không còn GPU khả dụng, chuyển các tác vụ rất nhẹ sang CPU; các tác vụ tính toán nặng chờ GPU trống.
- Thường xuyên quản lý tài nguyên, kết thúc các tiến trình treo hoặc không còn cần thiết.

### 4.3. Tối ưu GPU và chạy song song
Theo cấu hình và hạn mức được nêu trong yêu cầu gốc:
- Mỗi tài khoản có thể chạy đồng thời tối đa 2 phiên GPU commit.
- Mỗi phiên sử dụng 4 GPU NVIDIA L4, mỗi GPU có 24 GB VRAM, tương đương tổng cộng 96 GB VRAM phân tán.
- Hai phiên độc lập tương đương 8 GPU L4 trên mỗi tài khoản.
Yêu cầu tối ưu:
- Thiết kế mã để khai thác đồng thời cả 4 GPU L4 trong mỗi phiên. Nếu không có L4, chuyển sang 2 GPU T4.
- Lựa chọn chiến lược phân chia tác vụ, batch size, gradient accumulation, mixed precision, Distributed Data Parallel (DDP) và các kỹ thuật tối ưu phù hợp.
- Tối đa hóa throughput, hạn chế GPU nhàn rỗi, tránh lỗi hết VRAM (OOM) và giảm thời gian thực thi mà không làm suy giảm độ chính xác.
- Tận dụng hiệu quả tổng VRAM phân tán của 4 GPU L4, hướng tới mức sử dụng bộ nhớ cao khi có lợi cho hiệu năng. Không mặc định rằng 96 GB VRAM phân tán tương đương một GPU có 96 GB VRAM thống nhất.
- Ưu tiên khai thác tối đa 2 phiên GPU trên mỗi tài khoản khi có đủ tác vụ độc lập, thay vì chỉ sử dụng 1 phiên và để lãng phí tài nguyên.
- Dồn tất cả luồng chạy lên 1 acc. Hết slot rồi mới sang acc khác. Biết rằng 1 acc chạy cùng lúc được: 2 GPU sessions, 5 CPU Sessions. Mỗi GPU sessions dùng được 4 x GPU L4, tổng 96 GB VRAM.
Nguyên tắc xuyên suốt: Luôn chủ động tìm kiếm và áp dụng các phương án thiết kế thuật toán, phân chia công việc, xử lý song song và tối ưu mã nhằm giảm tối đa thời gian chạy, đồng thời bảo đảm độ chính xác và tính tái lập.
Tôi ko quan tâm bảo mật, tôi chỉ cần bạn làm ra được kết quả tốt và nhanh nhất có thể.

### 4.4. Quản lý dữ liệu và tái sử dụng kết quả
- Nếu notebook cần thư viện hoặc tệp phụ trợ để chạy offline, hãy tải trước các thành phần cần thiết, đóng gói thành Kaggle Dataset công khai và sử dụng làm đầu vào cho các tài khoản.
- Đối với mã nguồn hoặc kết quả có thể tái sử dụng, tạo Kaggle Dataset công khai trên tài khoản nhatlong1103, hoặc lưu vào một thư mục được tổ chức rõ ràng trên nhánh GitHub hiện tại.
- Tận dụng lại kết quả đã có, tránh chạy trùng thí nghiệm và lãng phí thời gian tính toán.
- Không đưa token, thông tin nhạy cảm hoặc dữ liệu không được phép công khai vào các Dataset.

### 4.5. Tối ưu quá trình làm việc của Claude
- Thiết lập cơ chế theo dõi và thông báo tự động khi một tài khoản hoặc phiên tính toán hoàn thành công việc, hạn chế việc Agent phải liên tục kiểm tra thủ công và tiêu tốn token.
- Khi tìm kiếm tài liệu tham khảo, ưu tiên đọc nội dung văn bản và paper. Hạn chế đọc toàn bộ repository mã nguồn nếu không thực sự cần thiết.

## Bổ sung của user (2026-11-02, khung chat 3)

- Được chọn mô hình SOTA mạnh nhất hiện có rồi lắp ráp / phát triển trên đó, hoặc tự thiết kế thuật toán / pipeline riêng nếu
  tự tin mạnh hơn, hoặc làm cả hai hướng và chia ra chạy song song. Luôn trong giới hạn BTC: **tổng ≤ 15B tham số**.
- Nếu cần thêm acc Kaggle (và Kaggle không chặn API) thì báo user để gửi thêm.
- Nếu phương án hiện tại bị bỏ: copy nhánh hiện tại sang nhánh chat mới, bỏ mọi file không cần, chỉ giữ đề bài / ngữ cảnh /
  quy trình, rồi tìm giải pháp mới. Bám sát tối ưu code, luồng xử lý, bộ nhớ, GPU, batch size để đạt độ chính xác cao nhất
  trong thời gian ngắn nhất.
