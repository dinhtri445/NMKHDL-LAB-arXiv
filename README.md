# arXiv Scraper - Lab 1 Data Science

## Thông tin sinh viên
- **MSSV**: 23120377
- **Họ tên**: Mai Đình Trí
- **Lớp**: Nhập môn khoa học dữ liệu - CQ2023/21

## Mô tả
Dự án này thực hiện thu thập dữ liệu từ arXiv trong phạm vi ID được giao là từ **2406.4361** đến **2406.9360**, bao gồm:
- Siêu dữ liệu (metadata)
- File nguồn toàn văn (TeX)
- Thông tin trích dẫn (references) từ Semantic Scholar

## Thiết lập Môi trường
### 1. Tải lên Google Drive để chạy Google Colab
Có thể nén folder chạy chương trình thành file .zip rồi upload lên Drive (xử lý trong Google Colab) hoặc upload luôn folder chạy chương trình lên Drive.

Ở Lab này, em dùng upload luôn folder chạy chương trình lên Drive. Cách chạy trên Google Colab như sau:
- **1. Mount Google Drive**
    ```python
        from google.colab import drive
        drive.mount('/content/drive')
    ```
- **2. cd tới thư mục project. Ví dụ như bên dưới**
    ```python
    %cd /content/drive/MyDrive/Colab Notebooks/23120377/src
    ```
- **3. Cài đặt dependencies**
    ```python
    !pip install -r requirements.txt
    ```
    File requirements.txt chứa các thư viện: `arxiv`, `requests`, `psutil`  
Hiện tại phiên bản Python trên Google Colab là `Python 3.12.12`

### 2. Chạy mã với các lệnh cụ thể
**Các tham số tham số truyền vào từ command line khi chạy chương trình:**

- `--start` : chỉ số bắt đầu lấy bài báo từ danh sách arXiv ID
VD: nếu có 1000 paper thì --start 10 sẽ bắt đầu lấy bài báo thứ 11 
- `--batch` : số lượng bài báo cần thu thập trong lần chạy
- `--student_id` : có thể đổi tên thư mục output sang MSSV khác 
- `--resume` : dùng để chạy tiếp từ lần chạy trước nếu có sự cố hoặc chủ động dừng, dựa trên progress đã lưu 
- `--save_interval` : sau bao nhiêu paper thì lưu progress.
- `--workers` : số luồng (threads) song song để xử lý (metadata/source/reference), 

**Các cách chạy trên Google Colab:**
- 1. Chạy cơ bản: cào 10 paper từ đầu (mặc định --start 0)
    ```python
    !python main.py --batch 10
    ```
- 2. Chạy tiếp tục lần trước: Nếu notebook bị ngắt hay chương trình tắt giữa chừng do lỗi mạng,...
    ```python
    !python main.py --resume
    ```
- 3. Chạy song song tăng tốc với 3 worker
    ```python
    !python main.py --batch 20 --workers 3
    ```
- 4. Chạy tiếp tục từ lần trước để cào 500 paper tiếp theo, kết hợp chạy song song với 4 worker
    ```python
    !python main.py --resume --batch 500 --workers 4
    ```