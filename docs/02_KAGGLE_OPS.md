# 02 — Vận hành Kaggle và container

*Rút từ `handoff/02_ENGINEERING_OPS_NOTES.md` và `handoff/ENVIRONMENT_SUMMARY.md` của `hmm-navigation-2` (bản gốc:
`git show de4360e:handoff/02_ENGINEERING_OPS_NOTES.md`), chỉ giữ phần dùng được cho DataMate. Bài học đã được kiểm chứng qua nhiều tuần chạy thật.*

## 1. Chính sách tính toán (user chỉ định)

1. **Mọi tính toán nặng chạy trên Kaggle.** Container local (4 nhân, không GPU) chỉ cho kiểm tra nhanh dưới khoảng 2 phút
   (ví dụ test `src/score.py`, đọc cấu trúc dữ liệu).
2. **Mỗi acc dùng đủ 2 phiên GPU trước khi sang acc khác** (không rải 1 phiên mỗi acc). `fanout.py` cấp slot theo thứ
   tự (acc1, 0), (acc1, 1), (acc2, 0)…
3. Hết GPU: job **rất nhẹ** (dưới khoảng 30 phút CPU: chấm điểm, phân tích offline) chạy CPU (5 phiên mỗi acc). Job nặng
   (LLM inference, fine-tune) **đợi GPU trống**, không đẩy sang CPU.
4. **Kết quả dùng lại được nhiều lần → dataset PUBLIC của `nhatlong1103`** (`automation/publish_dataset.py`) để acc khác
   mount, không tính lại: dữ liệu BTC, cache ingest/profile, wheels vLLM, trọng số mô hình, dữ liệu tổng hợp, LoRA.
5. **Luôn tìm cách song song hoá** mà không đổi kết quả: chia câu hỏi theo doanh nghiệp ra nhiều kernel, vLLM batch,
   cache kết quả trung gian, đánh giá lại biến thể offline trên cache thay vì chạy lại LLM.
6. Không chạy code bên thứ ba tải về ngay trong container; chạy nó trong kernel Kaggle.

## 2. Tài khoản

Token mỗi acc ở `~/.kaggle/tok_<username>` (chmod 600); `~/.kaggle/access_token` là của `nhatlong1103`. **Repo
public: không commit token, không ghi vào `~/.bashrc`.** Token lấy từ tin nhắn của user, thứ tự trùng bảng dưới. Kiểm
tra: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels list --mine`.

**Acc dùng cho DataMate (token user gửi 2026-10-30):** `yuan1103` (4×L4 + 2×T4; **hết quota GPU tuần 30 h** lúc
30/10), `mxuanquy` (2×T4), `anderson1202` (2×T4). Chưa có token `nhatlong1103` → dataset tái sử dụng tạm đăng public
trên acc khác hoặc dùng output của kernel public (`kernel_sources`).

Bảng acc của dự án trước (tham khảo):

| # | Username | Ghi chú (đến 2026-10-30) |
|---|---|---|
| 1 | `nhatlong1103` | Chủ mọi dataset public |
| 2 | `yuan1103` | **dành cho user** (không chạy job) |
| 3 | `synchan5704` | L4×4 OK |
| 4 | `loinguyentran` | |
| 5 | `yeshenhi` | chỉ T4 (xin L4 thì rơi về CPU, không báo lỗi) |
| 6 | `mucnokieu` | GPU hay bị người khác dùng → chủ yếu CPU; L4×4 OK |
| 7 | `huyenco` | |
| 8 | `nguynduylm` | L4: đã accept rules nhưng lần đầu không có GPU; thử lại |
| 9 | `phuongnehe` | L4×4 OK |
| 10 | `thnliu` | L4×4 OK |
| 11 | `mxuanquy` | chỉ T4; **dành cho user** (không chạy job) |
| 12 | `linhmai24` | L4×4 OK + T4 |

**Trần phiên:** 2 GPU và 5 CPU mỗi acc. Acc chưa xác minh số điện thoại không được cấp GPU.

**GPU L4×4:** kernel-metadata `"machine_shape": "NvidiaL4"`, `"competition_sources": ["gemma-4-developer-agent"]`
(deadline grant 2026-11-25), `"enable_internet": false` → 4× L4 (23 GB mỗi GPU), 48 CPU, 197 GB RAM. Chỉ có tác dụng nếu
acc **đã accept rules** của competition đó trên web (không có API); chưa accept thì kernel lặng lẽ chạy T4. Kiểm bằng
một kernel nhỏ in `torch.cuda.get_device_name(i)`. **L4×4 (96 GB) là môi trường gần nhất với máy chấm 2 × RTX 5090
(64 GB)** → dùng cho đo thời gian và fine-tune.

## 3. Pipeline chạy

```bash
python3 automation/fanout.py <runs>/<run> <prefix> "<script> <args>;..." [n_kernels] [--accounts a,b] [--cpu|--l4] [--ds x/y,z/w]
```

- `make_kernel.py`: upload `src/` thành dataset code `<acc>/zc-code` (mỗi lần một version), viết `run.py` mount code +
  `nhatlong1103/zc-datamate-data` + dataset phụ (`--ds`: trọng số, wheels, cache), chạy các job song song, lưu
  `results/`, `logs/`. Trong job, token `DATA` = thư mục dữ liệu, `INPUT` = `/kaggle/input`, `OUT` = thư mục kết quả.
- `run_kernel.py <kdir> <out> 600 3 14400`: push, poll, kéo output. QUEUED quá 10 phút thì `delete` rồi push lại.
  `--attach` để bám lại kernel đang chạy.
- `fanout.py` chia job vòng tròn ra (acc, slot), có failover sang acc khác khi push lỗi, chờ dataset code `ready` rồi
  mới push, cuối cùng copy `results/*` về `results/` của repo.
- **Ghi tên run vào `<runs>/active_runs.txt` ngay khi khởi chạy** (để `resume.py` và `watch_done.py` biết).
- `<runs>` nên là `<scratchpad>/runs` (không commit).

## 4. Dataset public cần tạo (một lần, dùng chung mọi acc)

| Slug (dự kiến) | Nội dung | Trạng thái |
|---|---|---|
| `nhatlong1103/zc-2026` | Dữ liệu BTC (đã giải nén) + `data.txt` | **có**, public |
| `anderson1202/zc-vllm-t4b` (output kernel public) | wheels vLLM 0.32 + deps (`wheels/`), gắn bằng `--ks` | đang tạo |
| `nhatlong1103/zc-models-*` | Trọng số mô hình ứng viên (nếu không mount được từ Kaggle Models) | chưa |
| `nhatlong1103/zc-cache-*` | Cache ingest/profile, đầu ra LLM theo câu để chấm lại offline | chưa |

> Lưu ý: dữ liệu BTC có thể có điều khoản không cho phát tán. **Hỏi user trước khi để public**; nếu không được thì
> để private trên `nhatlong1103` và chỉ chạy kernel từ acc đó, hoặc share riêng cho các acc khác.

Giới hạn metadata: title 6–50 ký tự, subtitle 20–80. Upload khoảng 18 MB/s; dùng hard link (`cp -al`) để chuẩn bị thư mục.

## 5. Bài học Kaggle

- **Ngắt kernel:** chỉ `kaggle kernels delete -y <owner>/<slug>` có tác dụng; push version mới không ngắt phiên cũ.
  Gọi **từng lệnh một**; vòng `for` có `delete` bị bộ phân loại an toàn chặn. Bị chặn thì gửi user danh sách xoá tay.
- **Log trễ:** API chỉ lộ log gần lúc xong; RUNNING mà chưa có output không có nghĩa là treo.
- **Kernel dùng code tại lúc push.** Hai fanout cùng lúc trên một acc có thể đè version dataset code của nhau →
  **khởi chạy tuần tự** (chờ dòng `launched` trong log) rồi mới gọi fanout tiếp.
- **T4 phải dùng fp16** (bf16 giả lập, chậm khoảng 7×). Với vLLM trên T4: `dtype=half`, mô hình int4 (AWQ/GPTQ) cho 14B.
- Kernel T4 có thể bật internet (pip install); kernel L4 grant thì **không** → cài từ dataset wheels.
- Collector của đợt cũ có thể ghi đè kết quả mới nếu xong sau: dừng collector cũ hoặc chép lại kết quả đợt mới sau cùng.

## 6. Container Claude Code

- **Container có thể khởi động lại bất cứ lúc nào**: mất tiến trình nền (fanout, run_kernel), còn scratchpad, repo, và
  kernel trên Kaggle vẫn chạy. Phục hồi: `python3 automation/resume.py <runs>`.
- Monitor có thể chết vì lỗi proxy nhưng vẫn ghi `STATUS=DONE:error` trong khi kernel vẫn RUNNING → kiểm
  `kaggle kernels status` rồi `reattach.py`.
- Tác vụ nền của Claude có trần khoảng 2 giờ; không dùng `sleep` dài ở foreground.
- `pkill -f <pattern>` có thể giết luôn shell gọi nó → dùng PID cụ thể hoặc `TaskStop`.
- Sửa file bằng Edit hoặc Python replace có `assert`, không dùng `sed` mù.
- Gói: `pip install kaggle markdown xhtml2pdf`. md → pdf: `python3 automation/md2pdf.py <in.md> <out.pdf>`.
