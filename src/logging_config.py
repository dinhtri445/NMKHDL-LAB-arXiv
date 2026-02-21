# src/logging_config.py
import logging
import os
import sys

def setup_logger(log_file):
    """
    Cấu hình hệ thống Log vào file cụ thể.
    Mỗi khi gọi hàm này, logger sẽ reset để ghi vào file mới.
    """
    # 1. Lấy Root Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 2. [TỐI ƯU] Xóa sạch và ĐÓNG các handler cũ
    # Giúp giải phóng file, tránh lỗi "file is being used" trên Windows
    if logger.hasHandlers():
        for handler in logger.handlers:
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers.clear()

    # 3. Tạo thư mục chứa log nếu chưa có
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 4. Định dạng log [TỐI ƯU HIỂN THỊ]
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # 5. File Handler (Ghi vào file cụ thể)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter) # Dùng format chi tiết
    logger.addHandler(file_handler)

    # 6. Stream Handler (Ghi ra màn hình)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter) # Dùng format gọn gàng
    logger.addHandler(console_handler)

    # 7. Tắt log rác thư viện ngoài
    logging.getLogger('bibtexparser').setLevel(logging.ERROR)
    logging.getLogger('bibtexparser.bparser').setLevel(logging.ERROR)
    logging.getLogger('nltk').setLevel(logging.ERROR)