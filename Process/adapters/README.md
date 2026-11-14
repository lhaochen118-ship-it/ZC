# Process/adapters — LoRA adapter W của kế hoạch 5 (Git LFS)

Mỗi thư mục = một adapter (r 32, α 64) học chồng trên **Qwen3.5-9B + V3A × 0.70 đã gộp** (gộp `V3A × 0.70 + <W> × 1.0` = đúng mô hình đã huấn luyện).
`adapter_model.safetensors` lưu bằng **Git LFS** (`git lfs pull` sau khi clone; file > 100 MB không vào git thường).

| Thư mục | Vai | Dùng ở |
|---|---|---|
| `WJR1/` | W3 phán xử (huấn luyện kèm dòng rulebook) — **cấu hình cuối** | `configs/flow_p5_final.json` `judge_lora` |
| `WL1/` | W2 linker | `links_lora` |
| `WJB1/` | W3 phán xử bản trước (dùng trong 2 lần chạy container F5) | dự phòng |

Bản trên Kaggle: `nhatlong1103/zc-w-final` (`<NAME>/adapter/`). Chỉ số huấn luyện: `Process/runs/p5_train/<NAME>/`.
Để `src/predict_flow.resolve_adapter` tìm thấy: `ZC_ADAPTER_ROOT=<repo>/Process/adapters` (thư mục `<NAME>/adapter_config.json`).
