# Nghiên cứu các mô hình học máy trong chẩn đoán bệnh từ hình ảnh X-quang
# 📌 Giới thiệu
Dự án này tập trung vào việc nghiên cứu và áp dụng các mô hình học sâu (Deep Learning) như VGG19 và ResNet50 để chẩn đoán các bệnh lý từ hình ảnh X-quang ngực. Các bệnh lý được nghiên cứu bao gồm: xẹp phổi (Atelectasis), tràn khí màng phổi (Pneumothorax), tràn dịch màng phổi (Effusion), viêm phổi (Pneumonia), và dày màng phổi (Pleural Thickening). Dự án sử dụng bộ dữ liệu NIH Chest X-ray, một bộ dữ liệu công khai lớn với hơn 112,000 hình ảnh X-quang.

Mục tiêu của dự án là xây dựng các mô hình học sâu có khả năng phân loại chính xác các bệnh lý từ hình ảnh X-quang, hỗ trợ các bác sĩ trong việc chẩn đoán nhanh chóng và chính xác hơn.

# 📂 Cấu trúc thư mục
Dự án được tổ chức với cấu trúc thư mục như sau:

xray-diagnosis-deeplearning/

│── models/              # Chứa mã nguồn của các mô hình

│   ├── resnet50.py      # Mô hình ResNet50

│   ├── vgg19.py         # Mô hình VGG19

│

│── data/                # Chứa dữ liệu huấn luyện

│   ├── train_data.csv   # Dữ liệu huấn luyện

│   ├── val_data.csv     # Dữ liệu validation

│   ├── test_data.csv    # Dữ liệu kiểm tra

│

│── results/             # Chứa kết quả huấn luyện

│   ├── results_resnet.npz  # Kết quả mô hình ResNet50

│   ├── results_vgg.npz     # Kết quả mô hình VGG19

│

│── README.md            # Hướng dẫn sử dụng

# 📜 Hướng dẫn sử dụng
Cấu hình môi trường
Cài đặt các thư viện cần thiết:
Đảm bảo bạn đã cài đặt Python 3.7 trở lên. Sau đó, cài đặt các thư viện cần thiết bằng cách chạy lệnh sau:

pip install -r requirements.txt
Tải dữ liệu:
Tải bộ dữ liệu NIH Chest X-ray từ Kaggle và đặt trong thư mục data/.

Huấn luyện mô hình
Huấn luyện mô hình ResNet50:
Mở notebook ResNet50.ipynb trong thư mục notebooks/ và chạy các ô lệnh để huấn luyện mô hình ResNet50.

Huấn luyện mô hình VGG19:
Mở notebook VGG19.ipynb và chạy các ô lệnh để huấn luyện mô hình VGG19.

Huấn luyện các mô hình truyền thống:
Mở notebook Traditional_Models.ipynb để huấn luyện các mô hình học máy truyền thống như Logistic Regression (LR), Decision Tree (DT), K-Nearest Neighbors (KNN), và Support Vector Machine (SVM).

Inference
Sau khi huấn luyện, bạn có thể sử dụng các mô hình đã huấn luyện để dự đoán trên dữ liệu mới. Ví dụ, để dự đoán bằng mô hình ResNet50, bạn có thể sử dụng đoạn mã sau:

from src.model_training import load_model, predict

# Tải mô hình đã huấn luyện
model = load_model('models/resnet50_model.pth')

# Dự đoán trên ảnh mới
image_path = 'path_to_your_image.png'
prediction = predict(model, image_path)
print(prediction)
# 📊 Kết quả
Đánh giá mô hình
Các mô hình được đánh giá dựa trên các chỉ số như AUC, Accuracy, Precision, Recall, và F1-Score. Dưới đây là kết quả đánh giá của các mô hình:
![image](https://github.com/user-attachments/assets/dc33041e-f173-4764-8c8c-9267046e717d)
# 🔗 Nguồn tham khảo
NIH Chest X-ray Dataset

CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning

Deep Learning for Chest X-ray Analysis: A Survey
