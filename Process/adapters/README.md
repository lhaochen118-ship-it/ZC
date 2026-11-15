# Process/adapters — LoRA adapter W của kế hoạch 5 (Git LFS)

Mỗi thư mục = một adapter (r 32, α 64) học chồng trên **Qwen3.5-9B + V3A × 0.70 đã gộp** (gộp `V3A × 0.70 + <W> × 1.0` = đúng mô hình đã huấn luyện).
`adapter_model.safetensors` lưu bằng **Git LFS** (`git lfs pull` sau khi clone; file > 100 MB không vào git thường).

| Thư mục | Vai | Dùng ở |
|---|---|---|
| `WJR1/` | W3 phán xử (huấn luyện kèm dòng rulebook) — **cấu hình nộp** | `configs/flow_v2_final.json` `judge_lora` |
| `WL1/` | W2 linker | `links_lora` |

Bản trên Kaggle: `nhatlong1103/zc-w-final` (`<NAME>/adapter/`). Chỉ số huấn luyện: `Process/runs/p5_train/<NAME>/`.
Để `src/predict_flow.resolve_adapter` tìm thấy: `ZC_ADAPTER_ROOT=<repo>/Process/adapters` (thư mục `<NAME>/adapter_config.json`).

WJB1 (phán xử bản trước) đã xoá khỏi repo 15/11 (thay bằng WJR1); còn trên Kaggle `nhatlong1103/zc-w-final`. Docker v2 tải W từ `nhatlong1103/zc-w-v2` (WL1, WJR1).
