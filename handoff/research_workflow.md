# Research workflow — quy trình làm việc (DataMate)
Cập nhật 14/11/2026 (§9 quy trình kế hoạch 5: S đóng băng, W huấn luyện riêng, đánh giá phán
xử offline); 13/11/2026 (§8 vòng học luật của kế hoạch 3; §5 vòng lặp kế hoạch 2: public gốc +
held-out, sàng lọc → xác nhận; §1b quota / 12 h; §7 bàn giao qua Process/ ). Giữ tinh thần quy
trình gốc (docs/RESEARCH_WORKFLOW.pdf ), đổi đích từ "paper" sang "điểm PA tối đa trên private
test, trong 6 giờ, tái lập được".

## 1. Nguyên tắc chung
- Tự động hoá tối đa. Tự đề xuất → chạy → kiểm chứng → quyết định; không hỏi lại những quyết định đã
có trong tài liệu. Chỉ hỏi user việc chỉ user làm được (token, nộp portal, accept rules, xoá kernel bị chặn,
quyền public dữ liệu).
- Mỗi vòng lặp: Đề xuất → Phản biện → Xây → Chạy trên Kaggle → Chấm ( src/score.py ) → Phân tích lỗi theo
nhóm → Giữ / sửa / bỏ → ghi EXPERIMENT_LOG.md .
- Một thí nghiệm quyết định mỗi lần, nhưng khi đã chắc thì chạy song song tối đa trên mọi GPU.
- Đo lỗi trước khi sửa: luôn lưu đầu ra từng câu (plan, code, kết quả thực thi, đáp án, p) để phân tích
offline; tìm vị trí và điều kiện lỗi rồi mới đề xuất bản sửa.
- Chọn siêu tham số bằng dữ liệu tách riêng (holdout theo doanh nghiệp), một cấu hình chung; mọi
biến thể chọn sau khi nhìn dữ liệu phải được ghi rõ và kiểm lại trên holdout.
- Báo cáo kết quả âm trung thực; ý tưởng không cải thiện thì loại và ghi lý do.
- Kiểm dữ liệu đầu vào trước khi kết luận (header Excel, định dạng số/ngày, mã hoá tiếng Việt).
- Không vi phạm luật BTC: không dùng test để train, không logic theo id, tổng tham số ≤ 15B, offline.

## 1a. Cổng kiểm tra trước khi chạy (user chốt 2026-11-02)
Code xong → tự review diff (logic, công thức, biên, tối ưu) → automation/preflight.py "<jobs>"  in PREFLIGHT OK 
(flag, file kernel, test CPU, chạy thử end-to-end trên CPU với mô hình giả) → mới được đẩy kernel. fanout.py 
tự kiểm và từ chối khởi chạy nếu chưa qua. Chi tiết: CLAUDE.md .

## 1b. Nguyên tắc dùng Kaggle (ZAIC.docx §4, cập nhật 2026-11-02)
- Dùng tất cả acc được cấp. Token chỉ ở ~/.kaggle/tok_<username>  (quyền 600), không đưa vào code,
prompt, GitHub hay báo cáo; kiểm tra kết nối bằng kaggle competitions list . Khung chat mới phải xin lại
token.
- Acc và quota mới nhất: handoff/ENVIRONMENT_SUMMARY.md  §2 và python3 automation/quota.py  (24 acc,
11/11). Quota 30 giờ GPU / tuần / acc (phiên 4×L4 ≈ 1.5–2.2 h quota / giờ tường), làm mới thứ Bảy 00:00
UTC. Acc < 6 h: kiểm đủ giờ trước khi giao task (tự động trong launch_many.py ). Phiên ≤ 12 h: runner
tự đặt hạn chót, huấn luyện tự lưu checkpoint ( CLAUDE.md ). Acc dùng chung với dự án khác: không đụng
kernel không phải zc-* .
- Dồn luồng lên một acc: mỗi acc tối đa 2 phiên GPU + 5 phiên CPU cùng lúc; hết slot mới sang acc khác.
Mỗi phiên GPU = 4×L4 (24 GB/GPU, 96 GB phân tán, không phải một GPU 96 GB); không có L4 → 2×T4.
- Kẹt hàng đợi: không có đầu ra sau ~8 phút (L4 / CPU) hoặc ~10 phút (T4) → kiểm trạng thái; L4 kẹt
hàng đợi → ngắt, tạo phiên mới, tối đa 5 lần; vẫn hỏng → xem lỗi code / logic, sửa rồi chạy lại; không phải lỗi
code → chuyển 2×T4. Vẫn hỏng → xem log, quota, cấu hình.
- Tận dụng hết GPU trong phiên: inference --dp <N>  (N = số GPU) (mỗi GPU một bản mô hình, chia DN cân
bằng theo số câu, prefix caching theo DN); mô hình không vừa 1 GPU thì --tp . Fine-tune: DDP mọi GPU, bf16
(L4) / fp16 (T4), gradient accumulation, gradient checkpointing chống OOM.
- Hết GPU: việc rất nhẹ chạy CPU, việc nặng đợi GPU. Định kỳ dọn tiến trình treo (kernel Kaggle và tiến trình
nền cục bộ); không ngắt được thì báo user.
- Cần thêm acc (và Kaggle không chặn API) → báo user.

## 2. Tối ưu thời gian chạy
- Cache mọi thứ tất định (ingest, profile, entity index) và đăng thành dataset public để acc nào cũng mount.
- Lưu đầu ra LLM theo câu → thay đổi ở phần quyết định / formatter / voting chấm lại offline trong vài
giây, không chạy lại LLM.

- Chia câu hỏi theo doanh nghiệp ra nhiều kernel / GPU; trong kernel dùng vLLM batch + prefix caching.
- Thử nhỏ (vài doanh nghiệp) trước khi chạy toàn bộ.

## 3. Tối ưu token cho Claude
- Không chủ động kiểm tra kernel. Sau khi khởi chạy, bật một watcher nền:
bash python3 automation/watch_done.py <runs> --max 115
chạy bằng Bash run_in_background . Nó thoát (→ Claude được báo ngay) khi bất kỳ kernel nào xong, in tên
kernel và trạng thái; xử lý xong thì bật lại. Hết 115 phút thì in WATCH_TIMEOUT  để bật lại (tác vụ nền có trần
~2 giờ). - Tìm thông tin: chỉ đọc văn bản / paper (alphaXiv, web), không đọc code repo của người
khác. - Dữ liệu lớn: chỉ đọc cấu trúc và vài dòng ( head , python -c  in shape/cột), không in cả file. - Log dài:
chỉ đọc tóm tắt ( score.py  theo nhóm, --errors N ).

## 4. Git
- Branch làm việc: claude/datamate-v3-0dny90  (repo lhaochen118-ship-it/ZC ). Không tạo PR nếu user không
yêu cầu. Chỉ tạo file nộp thử khi kết quả tốt (user chốt).
- Commit kèm trailer theo hướng dẫn của phiên; không ghi model ID vào commit.
- Kết quả mới để file mới ( results/<run>/… ), không ghi đè kết quả cũ.
- Không commit token, dữ liệu sinh ra lớn, cache ( .gitignore ).

## 5. Vòng lặp hiện tại (kế hoạch 2, từ 09/11: đến public gốc 100 %,
held-out không giảm)
Hybrid (user 08/11): mô hình viết chương trình trả lời mọi câu; luật tổng quát chỉ làm ngữ cảnh / kiểm tra.
Vòng lặp ( docs/13_TASK_AFTER_PLAN1.md , docs/12_PLAN2.md ):
1. Thước đo cố định: public gốc ( pub_full , chấm với data/raw/public_gt.jsonl ), held-out pub_ho  (paraphrase
public) + tr_ho  (6 DN train giữ ngoài) → báo cả hai ở mọi mốc; tập con sàng lọc pub_ho400 , tr_ho400 .
2. Phân tích lỗi trên public gốc / train ( tests/list_wrong.py , taxonomy 7 nhóm docs/10 ); held-out chỉ xem
tổng.
3. Sửa bằng cơ chế chung (gợi ý, làm sạch, kiểm tra có tín hiệu sạch trên mọi log, toolkit) hoặc phía mô
hình (dữ liệu fine-tune, thành viên, bộ chọn) + test; không mẫu theo câu / DN.
4. Ước lượng offline trên log có sẵn khi được ( tests/pool_gens.py , replay_vote.py , quét tín hiệu sạch).
5. Sàng lọc (một ứng viên / kernel, song song nhiều acc, tập con, ~40 phút) → xác nhận đủ tập 2 seed →
giữ / bỏ theo §4 docs/12  (public ↑ ≥ 0.3, held-out gộp không ↓ > 0.3).
6. Sau mỗi lần chạy có public: automation/update_final_submission.py <run> ; đổi mô hình nộp:
 automation/set_final_model.py --publish  + chạy đường container; ghi handoff/STATUS.md  (+ Process/scores.csv 
khi bàn giao), commit + push.

## 6. Nhiễu của điểm public
- Luồng hybrid hiện tại: một lần chạy dao động public ±0.4 điểm, held-out gộp ±0.3; chất lượng còn phụ
thuộc seed huấn luyện (cùng dữ liệu: 0.991 vs 0.983) → mọi quyết định ≥ 2 seed; tập con 400 câu nhiễu
±0.6 (chỉ để loại).
- Ưu tiên các sửa giảm phương sai (ổn định ánh xạ, bỏ phiếu biết ánh xạ) trước các sửa tăng trung bình
nhỏ.

## 7. Chuyển khung chat
- Trước khi đóng: commit + push mọi thứ; lưu vào Process/  những gì không nằm trong git (log nén của lần
chạy then chốt, dữ liệu huấn luyện nén, plan khởi chạy, sổ kernel, quota, bản vá worktree;
 Process/README.md ); cập nhật handoff/STATUS.md , ba PDF bàn giao, handoff/environment_summary.pdf ,
 docs/RESEARCH_WORKFLOW.pdf , READING_GUIDE.md ; tắt routine tự chạy của phiên cũ.
- Khung mới: đọc READING_GUIDE.md  → CLAUDE.md  → handoff/STATUS.md ; xin token; tạo routine mỗi giờ; tải
output kernel đang chạy ( Process/kaggle/kernels.csv ).


## 8. Vòng lặp kế hoạch 3 (13/11): luật do mô hình tự học
1. Đo sạch trước: held-out đã lọc nhãn ( *_clean , *400c ), mốc P đo cùng code; báo public + held-out sạch ở
mọi mốc.
2. Sinh thẻ ( src/lrl_gen.py , GPU, chia shard theo GPU): biên dịch rulebook, bí danh giá trị (đầu vào của
split), quy nạp từ ca sai trên câu train ( training/data/lrl_cases.py ).
3. Nhận thẻ tự động ( tests/lrl_accept.py ): thẻ check → quét tín hiệu sạch trên log đã có (CPU); thẻ nội dung
→ A/B hai lần chạy cùng seed trên câu của DN khác. Không ai đọc / sửa tay thẻ.
4. Nội hoá: dữ liệu v3 ( <spec> , phong cách, thẻ trong prompt) → SFT FSDP (chuỗi checkpoint qua acc khi
một acc không đủ giờ).
5. Đánh giá Q / Q2 / P (+ thẻ) song song nhiều acc → quy tắc giữ / bỏ docs/12  §4, đổi bản nộp docs/16  §5.4
→ lặp.

## 6. Bàn giao: automation/archive_runs.py  + 3 PDF + READING_GUIDE.md .

## 9. Quy trình kế hoạch 5 (14/11) — dùng tiếp cho kế hoạch mới
1. Chẩn đoán trước khi đề xuất: phát lại log đã có với đáp án public ( automation/rebuild_public_gt.py ) →
phân nhóm lỗi: có mẫu đúng nhưng thua phiếu / mọi mẫu cùng sai / chưa rõ ( Process/tools/plan5/an3.py ,
 err.py ); trần "có mẫu đúng" khi gộp nhiều log ( an4.py , pool.py ).
2. Tách vai để quy công: mỗi vai một adapter, huấn luyện riêng; đo từng vai ghép cặp trên cùng log S.
3. Mốc zero-shot trước, huấn luyện song song: đẩy zero-shot (1 phiên, mỗi GPU một biến thể) cùng lúc
với các nhánh huấn luyện (mỗi nhánh 1 phiên FSDP 4 GPU, mỗi nhánh một acc).
4. Đánh giá offline tối đa: mọi biến thể phán xử ghi verdicts.jsonl  → lưới chính sách / kết hợp trên CPU;
chọn chính sách ổn định trên ≥ 2–4 log S (trung bình + thấp nhất), không theo một log.
5. Đường container sớm: khi có ứng viên tốt nhất, chạy ngay container × 2 (2 acc song song) để đo điểm
thật + độ tất định; thành phần cải tiến sau được đo lại trên log của chính các lần chạy này.
6. Cổng trước khi đẩy: tests/run_all.sh  + smoke CPU đúng lệnh ( soup_job ... --fake-responder --subset ) + kiểm
độ dài bản ghi / tỉ lệ parse; mỗi acc một lần đẩy tới khi RUNNING (PDF 2 §9).

## 10. Quy trình v2 (14–15/11, chế độ H) — dùng tiếp
1. Một mốc, đo cùng loại file: mốc = cấu hình nộp chạy đường container ( soup_job --runner predict ) ≥ 2
lần; mọi biến thể so ghép cặp từng câu với các lần chạy mốc, cùng loại file (container
 <config>/submission.jsonl  hoặc phán xử judge/ensemble/submission.jsonl , không trộn).
2. Phân tích log trên CPU trước khi tốn GPU ( Process/tools/v2/ ): theo nhóm DN (ngành / không), theo
vòng / biến thể mẫu, theo việc gọi hàm, gộp log (trần của thêm mẫu). Một giả thuyết chỉ được đẩy GPU khi
phân tích cho thấy đòn bẩy (ví dụ: chất lượng mẫu ngành ~0.2 → nhắm vào hàm / dữ liệu, không nhắm vào
số mẫu hay phán xử).
3. Thành phần mới phải được đo đầu-cuối trên phân phối đích (public chế độ H) trước khi xây tiếp; số
đo trên phân phối nguồn (dòng held-out DN train) chỉ là sàng lọc (bài học WT-b).
4. Lưới có hệ thống ( handoff/06 ): yếu tố × biến thể, mỗi biến thể một phiên DP 4; giai đoạn đơn yếu tố →
ghép yếu tố thắng → xác nhận × 2 → Docker. Ghi mọi kết quả vào nhật ký ngay khi về (kể cả âm).
5. Dữ liệu do mô hình sinh (data augmentation, handoff/07 ): sinh bằng Qwen không suy nghĩ + JSON
schema; nhãn qua thực thi; kiểm ngược bằng đường mô hình thứ hai (S giải lại câu sinh ra, giữ khi
trùng); Claude chỉ viết few-shot và kiểm ngẫu nhiên 30 mẫu / nguồn.
6. Dùng lại hàm / link của một lần chạy cho các biến thể S ( units_file , links_file ) để so sạch và nhanh; bản
nộp luôn sinh lại.
7. Chuyển khung chat: như §7 + automation/archive_v2.py  (log v2 → Process/runs/v2/ ),
 Process/scores_v2.csv , Process/kaggle/kernels_v2.csv , tắt routine.


## 10. Quy trình P5 (15–16/11) — thí nghiệm "chỉ thay đúng thành phần" (BẮT BUỘC từ 16/11)

- **Baseline cố định theo loại:** fine-tune = `configs/flow_p5_final.json` (0.952); chế độ H = `flow_p5_final_h` (0.860 TB 4 lượt);
  zero-shot = cùng framework, Qwen3.5-9B gốc (B00-S 0.754). Không dùng 0.952 làm mốc zero-shot.
- **Sinh config bằng code** từ baseline (copy + update khoá), in danh sách khoá khác baseline; điều chỉnh tương thích ghi `_doc`.
- **Một thay đổi / thí nghiệm;** nhiều thay đổi ⇒ ablation riêng. Tổ hợp lai nhiều backbone phải tính tổng tham số (≤ 15B).
- **Kết luận bằng ghép cặp từng câu** (z ≥ 2) với ≥ 2 lượt baseline (nhiễu lượt ±0.003–0.01); không chọn biến thể phán xử theo public.
- **Đánh giá nhanh:** chia cửa hàng (`run_kaggle public_test@i/n`) ra nhiều acc; đổi riêng phán xử = `soup_job --runner judge` trên log có sẵn.
- **Chuỗi tự động** (`Process/tools/p5/after_*.sh`): bước sau chỉ chạy khi bước trước `DONE:complete`; resume-safe sau khi worker khởi động lại.
- **Bản nộp:** user 16/11 09:00 — framework mạnh nhất (0.952) + file submission của nó (`Docker_submission/`); P (0.994) đóng gói riêng (`0.994_Docker_submission/`).
