# Kernel đang chạy (cập nhật 2026-10-30 ~21:00 UTC, khung chat 2)

Kéo kết quả một kernel đã xong: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_<acc>) kaggle kernels output <acc>/<slug> -p <dir>`;
runner cục bộ của khung chat này (`<scratchpad>/runs/<run>/<slug>_out/`) tự kéo khi kernel xong.

| Run | Kernel | GPU | Cấu hình | DN | Ghi chú |
|---|---|---|---|---|---|
| e3h | `linhmai24/zc-e3h-4` | 4×L4 | `--helpers --compile --compile-scope sales`, AABB K=4 | 101,103,112,121,124,127,129 | kernel 1–3 xong (`EXPERIMENT_LOG` entry 3–4) |
| e3b | `thnliu/zc-e3b-2` | 4×L4 | `--helpers`, BBBB K=4 | nửa sau của train | kernel 1 xong |
| e3t | `mxuanquy/zc-e3t-1`, `anderson1202/zc-e3t-2` | 2×T4 | `--helpers --think`, BB K=2, 100 câu holdout / 100 câu dev | | đo tác dụng thinking |
| e5 | `phuongnehe/zc-e5-1` | 4×L4 | e3h + `--norm-q` + auto-header | 105,108,115,118 | |
| e6a | `linhmai24/zc-e6a-1` | 4×L4 | e5 + `--shots` (thư viện v1) | 105,108,115,118 | so với e5-1 |
| e7 | `lelynhan/zc-e7-1` | 4×L4 | mọi sửa đến 20:05 + shots | 114,116,117,125 | so với e3h-2, e4-2 |
| e7b | `lelynhan/zc-e7b-1` | 4×L4 | mọi sửa đến 20:40 + shots | 102,106,107,111,113 | so với e3h-2, e4-1 |

**Cần user ngắt tay** (công cụ tự động bị chặn lệnh xoá): `mxuanquy/zc-e1b-1`, `anderson1202/zc-e1c-1` — baseline của khung
chat cũ, code cũ có lỗi, đã chạy > 9 giờ, không còn giá trị; chỉ tốn quota T4.
