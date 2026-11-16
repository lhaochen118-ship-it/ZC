#!/usr/bin/env python3
import json

SYSTEM = ("Bạn là chủ một cửa hàng nhỏ ở Việt Nam đang hỏi trợ lý dữ liệu về việc kinh doanh của mình. Bạn viết câu hỏi tiếng Việt "
          "tự nhiên, ngắn gọn, đúng cách người bán hàng hỏi hằng ngày.")


def user_msg(card, as_of, code, answer, fn_help, n=2):
    return (f"Dữ liệu cửa hàng (bảng chuẩn):\n{card}\n\nHôm nay là {as_of}.\n\nChương trình dưới đây đã chạy đúng trên dữ liệu:\n{code}\n"
            f"Kết quả: {json.dumps(answer, ensure_ascii=False)}\n\nÝ nghĩa các hàm:\n{fn_help}\n\n"
            f"Hãy viết {n} câu hỏi KHÁC NHAU mà chương trình này trả lời đúng. Giữ đúng đại lượng, kỳ thời gian (theo ngày hôm nay), "
            "tên mặt hàng / khách / nhân viên, con số và điều kiện lọc; không nhắc tên hàm hay tên cột. Câu 1: văn viết chuẩn có dấu. "
            "Câu 2: văn nói / nhắn tin (có thể viết tắt, thiếu dấu, gọi tắt kỳ thời gian như 'tháng trước', 'quý 3').\n"
            'Chỉ trả về JSON: {"questions": ["...", "..."]}')


def messages(card, as_of, code, answer, fn_help, n=2):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(card, as_of, code, answer, fn_help, n)}]
