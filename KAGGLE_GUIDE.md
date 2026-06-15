# Hướng Dẫn Chạy CLSS với QLoRA trên Kaggle

Để chạy source code này trên Kaggle, bạn cần thực hiện các bước chuẩn bị mã nguồn, dữ liệu và thiết lập môi trường. Dưới đây là quy trình chi tiết:

## Bước 1: Chuẩn bị Dữ liệu (Kaggle Dataset)

Kaggle không thể truy cập các đường dẫn local của máy tính hoặc server của bạn (như `/media02/...`). Do đó, bạn cần upload dữ liệu lên Kaggle trước:

1. Nén toàn bộ thư mục dữ liệu `sino_nom_punct` và `sino_nom_punct_doc` thành 1 file ZIP (ví dụ: `sino_nom_data.zip`).
2. Vào Kaggle, chọn mục **Datasets** -> **New Dataset** và tải file ZIP này lên.
3. Đặt tên dataset, ví dụ: `sino-nom-clss-data`.

## Bước 2: Chuẩn bị Code (Upload lên Kaggle Notebook)

Có 2 cách để đưa code lên Kaggle:
- **Cách 1**: Đẩy code lên GitHub, sau đó trong Kaggle dùng lệnh `!git clone <link_github>`.
- **Cách 2**: Nén toàn bộ source code thành `clss_source.zip` (nhớ loại bỏ các thư mục rác, log, cache) rồi upload trực tiếp vào Notebook bằng cách chọn **File** -> **Upload data**.

## Bước 3: Thay đổi cấu hình Data Path cho Kaggle

Khi dữ liệu đã ở trên Kaggle, đường dẫn sẽ trông giống như: `/kaggle/input/sino-nom-clss-data/`.
Bạn cần sửa file **`config/sino_nom_doc_cl_qlora.yaml`** như sau (bạn có thể tạo 1 bản copy cho Kaggle):

```yaml
# Sửa ở phần Dataset
punct:
  ColumnCorpus-SinoNomPunct:
    column_format:
      0: text
      1: punct
    # CẬP NHẬT ĐƯỜNG DẪN KAGGLE:
    data_folder: /kaggle/input/sino-nom-clss-data/sino_nom_punct

  ColumnCorpus-SinoNomPunctDoc:
    column_format:
      0: text
      1: punct
    # CẬP NHẬT ĐƯỜNG DẪN KAGGLE:
    data_folder: /kaggle/input/sino-nom-clss-data/sino_nom_punct_doc
```

## Bước 4: Tạo Notebook và Thiết lập Môi trường

1. Tạo một Kaggle Notebook mới.
2. Bật **GPU (P100 hoặc T4 x2)** và **Internet** trong phần Settings (cột bên phải của Notebook).
3. Thêm Dataset dữ liệu của bạn vào Notebook (Nút **Add Input** -> tìm dataset đã tạo ở Bước 1).
4. Tạo một ô code (Cell) mới và dán các lệnh sau để cài đặt môi trường:

```python
# Cài đặt thư viện yêu cầu (di chuyển vào thư mục code nếu bạn dùng git clone)
# !cd CLSS

!pip install -r requirements.txt
# Đảm bảo cài đặt các thư viện QLoRA
!pip install peft bitsandbytes
```

## Bước 5: Chạy Training

Thêm một ô code (Cell) khác để bắt đầu training. Kaggle chỉ cho phép ghi file ở thư mục `/kaggle/working/`, nên ta cần chỉ định `out_dir`:

```python
# (Nếu bạn dùng git clone, hãy chạy !cd CLSS trước khi train)
!python train.py --config config/sino_nom_doc_cl_qlora.yaml --out_dir /kaggle/working/clss_output
```

> [!TIP]
> Bạn có thể lưu model vào thư mục `/kaggle/working/`. Sau khi training xong, Kaggle sẽ tự động nén thư mục này thành `output.tar` hoặc bạn có thể tải tay checkpoint tốt nhất (`best-model.pt`) xuống máy tính.
