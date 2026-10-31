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

Token: `~/.kaggle/tok_<username>` (chmod 600), `~/.kaggle/access_token` = `nhatlong1103` (chỉ đăng dataset). Không commit
token. Danh sách acc, GPU và trạng thái quota hiện tại: `handoff/02_ENGINEERING_OPS_NOTES.md` §2.1 (bảng acc của dự án
trước đã bỏ vì lỗi thời).

**GPU L4×4:** kernel-metadata `"machine_shape": "NvidiaL4"`, `"competition_sources": ["gemma-4-developer-agent"]`
(deadline grant 2026-11-25), `"enable_internet": false` → 4× L4 (23 GB), 48 CPU. Chỉ có tác dụng nếu acc **đã accept rules**
của competition đó trên web; chưa accept thì push báo lỗi hoặc rơi về CPU. Kiểm bằng kernel probe (handoff 02 §2.1).

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
| `nhatlong1103/zc-assets` (output kernel public, CPU) | `wheels/` (vLLM 0.32 + deps + python-calamine, 3.7 GB) và `models/Qwen3-14B-AWQ/` (9.4 GB); gắn bằng `--ks nhatlong1103/zc-assets`; `run_kaggle.py` tự dùng | **có** |
| `nhatlong1103/zc-m-qwen3-14b-fp8` | `models/Qwen3-14B-FP8/` (16 GB): cho L4 / RTX 5090 | **có** |
| `nhatlong1103/zc-m-qwen35-9b` | `models/Qwen3.5-9B/` (19 GB, bf16): không chạy được trên T4, thử trên L4 | **có** |
| `nhatlong1103/zc-m-gemma4-12b-qat` | `models/gemma-4-12B-it-qat-w4a16-ct/` (9.6 GB) | **có** |
| `nhatlong1103/zc-train-wheels` | `wheels_train/`: bitsandbytes 0.50.2 (+ peft, accelerate) cho fine-tune offline; image Kaggle đã có peft 0.20, accelerate 1.14, transformers 5.16, torch 2.11 | **có** |
| Kaggle Models `qwen-lm/qwen-3/transformers/14b/1` | Qwen3-14B bf16 (28 GB), gắn bằng `--ms`, offline | **có** |
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

## 6. Container và bài học khác

Xem `handoff/02_ENGINEERING_OPS_NOTES.md` (bản cập nhật, thay cho mục 6–7 cũ).
