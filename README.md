# Lab 2: Parsing and Matching

## Thông tin sinh viên
- **MSSV**: 23120377
- **Họ tên**: Mai Đình Trí
- **Lớp**: Nhập môn khoa học dữ liệu - CQ2023/21

## Mô tả
Trong bài lab này, sinh viên được yêu cầu cụ thể 2 phần:  
	1. Cài đặt một bộ phân tích tự động để xử lý các tệp nguồn LaTeX đã thu thập từ bài lab1.  
	2. Thực hiện tác vụ khớp giữa các tham chiếu đã được trích xuất và chuẩn hóa ở trên thành file refs.bib với dữ liệu references.json đã thu thập. Mục tiêu là xác định mục BibTeX đã trích xuất nào tương ứng với arXiv ID nào trong tệp references.json.

## Cấu trúc thư mục
```
23120377/
├── data/                         # Thư mục chứa 5000 bài báo arXiv đã cào
├── result/
│   ├── data_output/              # Chứa kết quả xử lý
│   ├── stats/                    # Chứa file json thống kê
│   └── logs/                     # Chứa file log
├── src/                          # Thư mục chứa toàn bộ mã nguồn chương trình
│   ├── config.py                 # Cấu hình hệ thống: lưu đường dẫn, regex patterns, và tham số ML
│   ├── logging_config.py         # Thiết lập ghi log: giúp theo dõi quá trình chạy và hiển thị trong video demo
│   ├── main.py                   # File chạy chính, tiếp nhận tham số dòng lệnh (parse/match)
│   ├── hierarchy_parser.py       # Xây dựng cây phân cấp
│   ├── parser_pipeline.py        # Xử lý Mục 2.1: Đệ quy đọc file LaTeX, làm sạch Regex và dựng cây phân cấp
│   ├── matching_pipeline.py      # Xử lý Mục 2.2: Làm sạch dữ liệu, tạo đặc trưng (Features), Train model & dự đoán
│   ├── cleaning.py               # Code xử lý data cleaning trước khi matching
│   ├── feature_engineering.py    # Chứa code để tạo feature
│   ├── model_training.py         # Xây dựng mô hình
│   ├── utils.py                  # Các hàm tiện ích dùng chung (IO, xử lý chuỗi cơ bản)
│   ├── manual_ground_truth.json  # Chứa các bài báo được gán nhãn thủ công
│   └── requirements.txt          # Danh sách các thư viện Python cần thiết (pandas, sklearn, pylatexenc...)
│
├── README.md                     # Hướng dẫn chi tiết: Cách cài đặt môi trường và câu lệnh chạy code
└── Report.pdf                    # Báo cáo tổng kết: Phương pháp luận, phân tích kết quả và link video demo
```
## Hướng dẫn Chạy chương trình
#### Chạy trên terminal (ví dụ: terminal của Visual Studio Code)
Để đảm bảo chương trình chạy ổn định và đồng bộ dữ liệu giữa các máy tính, vui lòng thực hiện theo quy trình các bướcsau:

### Bước 1: Thiết lập Môi trường ảo (Virtual Environment)
Việc sử dụng môi trường ảo giúp tách biệt các thư viện của đồ án này với các dự án khác, tránh xung đột phiên bản.
1. Mở terminal tại thư mục gốc của dự án.
2. Tạo môi trường ảo: 
   ```bash
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   - **Windows:** `.\venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

### Bước 2: Cài đặt Thư viện
Cài đặt chính xác các phiên bản thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```
### Bước 3: Các lệnh chạy chương trình
#### Cấu trúc lệnh chung
```bash
python src/main.py --task [TÊN_TASK] --input_dir [FOLDER_DATA] --result_dir [FOLDER_KET_QUA]
```
Note:
- [TÊN_TASK]: parsing hoặc matching, lựa chọn để chạy cho từng phần
- [FOLDER_DATA]: chứa data của 5000 bài báo được cào nếu task là parsing, còn matching thì chứa data kết quả của phần matching rồi xử lý tiếp
#### Chạy Task 2.1 Parsing
```bash
python src/main.py --task parsing
```
hoặc
```bash
python src/main.py
```
#### Chạy Task 2.2: Matching
```
python src/main.py --task matching --input_dir result
```
