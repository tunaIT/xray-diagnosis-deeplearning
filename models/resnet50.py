# -*- coding: utf-8 -*-
"""ResNet50.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1W-pbiqTqxH1Xxq5Cd8ZW-FywuU3re5O8

# Chuẩn bị dữ liệu
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/gdrive')
# %cd '/content/gdrive/MyDrive/tieu_luan'

import pandas as pd
import os
from glob import glob

# Đọc file CSV chứa thông tin về các bản chụp X-quang (ChestX-ray14 dataset)
all_xray_df = pd.read_csv('Data_Entry_2017.csv')

# Tìm tất cả các file .png trong các thư mục con bắt đầu bằng "images"
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('.', 'data', 'images*', '*.png'))}

# In ra số lượng ảnh tìm thấy và tổng số dòng dữ liệu trong file CSV
print('Scans found:', len(all_image_paths), ', Total Headers:', all_xray_df.shape[0])

# Ánh xạ từ tên file trong cột 'Image Index' tới đường dẫn đầy đủ tương ứng
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.head(3)
# Bước 4: Giữ lại các nhãn bệnh cần thiết
labels_to_keep = ['Atelectasis',       # xẹp phổi
                  'Pneumothorax',      # tràn khí màng phổi
                  'Effusion',          # tràn dịch
                  'Pneumonia',         # viêm phổi
                  'Pleural_Thickening']# dày màng phổi
# 3. Tách nhãn và giữ lại nhãn cần thiết trong danh sách labels_to_keep
def filter_labels(row, labels_to_keep):
    labels = row['Finding Labels'].split('|')
    filtered_labels = [label for label in labels if label in labels_to_keep]
    return '|'.join(filtered_labels)
all_xray_df['Filtered Labels'] = all_xray_df.apply(lambda row: filter_labels(row, labels_to_keep), axis=1)

# Loại bỏ các dòng không có nhãn nào sau khi lọc
filtered_data = all_xray_df[all_xray_df['Filtered Labels'] != '']
all_xray_df = filtered_data
# Giữ lại các cột cần thiết
columns_to_keep = ['path', 'Filtered Labels']

# Xóa các cột không cần thiết
all_xray_df = all_xray_df[columns_to_keep]
# Xóa row có path là NaN
all_xray_df = all_xray_df.dropna(subset=['path'])

# Kiểm tra lại DataFrame sau khi xóa cột
print(all_xray_df.head())
label_counts = all_xray_df["Filtered Labels"].value_counts()


# Giữ lại các dòng có nhãn nằm trong `labels_to_keep`
filtered_df = all_xray_df[all_xray_df["Filtered Labels"].isin(labels_to_keep)]
label_counts = filtered_df["Filtered Labels"].value_counts()

# Lọc các nhãn có số lượng lớn hơn 500
labels_to_limit = label_counts[label_counts > 100].index

# Giữ lại tối đa 500 dòng cho mỗi nhãn lớn hơn 500
def limit_labels(group):
    if group.name in labels_to_limit:
        return group.sample(n=100, random_state=42)  # Chọn ngẫu nhiên 500 dòng
    return group

all_xray_df = filtered_df.groupby("Filtered Labels", group_keys=False).apply(limit_labels)
# In kết quả
print("Số lượng nhãn sau khi lọc:")
print(all_xray_df["Filtered Labels"].value_counts())
print(all_xray_df)

print(f'Chia thành công dữ liệu: {len(train_df)} mẫu train, {len(val_df)} mẫu val, {len(test_df)} mẫu test.')

# Uninstall the current PyTorch installation
!pip uninstall -y torch torchvision torchaudio

# Reinstall PyTorch with CUDA support. Make sure the CUDA version matches your GPU
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
print(torch.cuda.is_available())  # This should print True

"""# Xây dựng mô hình và đánh giá"""

train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

train_df.head()

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Định nghĩa các hằng số
LABELS = ['Atelectasis', 'Pneumothorax', 'Effusion', 'Pneumonia', 'Pleural_Thickening']
NUM_CLASSES = len(LABELS)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải dữ liệu từ các tệp CSV
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Xử lý cột 'encoded_labels' thành danh sách Python
train_df['encoded_labels'] = train_df['encoded_labels'].apply(eval)
val_df['encoded_labels'] = val_df['encoded_labels'].apply(eval)
test_df['encoded_labels'] = test_df['encoded_labels'].apply(eval)

# Định nghĩa lớp Dataset
class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['path']
        label = torch.tensor(row['encoded_labels'], dtype=torch.float32)  # Sử dụng danh sách nhãn đã xử lý
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Định nghĩa các phép biến đổi dữ liệu
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Tạo datasets và dataloaders
train_dataset = ChestXRayDataset(train_df, transform=transform)
val_dataset = ChestXRayDataset(val_df, transform=transform)
test_dataset = ChestXRayDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Định nghĩa mô hình ResNet50 với Dropout
class MultiLabelResNet50(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelResNet50, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])  # Bỏ phần fully connected
        self.transition_layer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.Dropout(p=0.5)  # Dropout để giảm overfitting
        )
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.prediction_layer = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.transition_layer(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.prediction_layer(x)
        return x

# Tính trọng số dương và âm
all_labels_list = train_df['encoded_labels'].tolist()
all_labels = torch.tensor(np.vstack(all_labels_list), dtype=torch.float32)
pos_weight = all_labels.sum(dim=0) + 1e-7
neg_weight = all_labels.size(0) - pos_weight + 1e-7
beta_p = (pos_weight + neg_weight) / pos_weight
beta_n = (pos_weight + neg_weight) / neg_weight

# Khởi tạo mô hình, criterion và optimizer
model = MultiLabelResNet50(NUM_CLASSES).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=beta_p.to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

# Huấn luyện mô hình với Early Stopping
best_val_loss = float('inf')
patience_counter = 0
patience_threshold = 3

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model_resnet.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience_threshold:
            print("Early stopping triggered.")
            break

    scheduler.step(val_loss)

# Đánh giá trên tập kiểm tra
model.load_state_dict(torch.load('best_model_resnet.pth'))
model.eval()
all_labels, all_predictions = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(outputs.cpu().numpy())

# Lưu kết quả
all_labels = np.vstack(all_labels)
all_predictions = np.vstack(all_predictions)
np.savez('results_resnet.npz', labels=all_labels, predictions=all_predictions)

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from prettytable import PrettyTable

# Load results
results = np.load('results_resnet.npz')
all_labels = results['labels']
all_predictions = results['predictions']

LABELS = ['Atelectasis', 'Pneumothorax', 'Effusion', 'Pneumonia', 'Pleural_Thickening']

# Tối ưu ngưỡng cho từng nhãn dựa trên AUC-ROC
optimal_thresholds_auc = []
for i, label in enumerate(LABELS):
    true_labels = all_labels[:, i]
    best_auc = 0
    best_threshold = 0.0

    # Duyệt qua các ngưỡng từ 0.0 đến 1.0 với bước nhảy 0.01
    for threshold in np.arange(0.0, 1.0, 0.01):
        pred_labels = (all_predictions[:, i] > threshold).astype(int)
        try:
            auc = roc_auc_score(true_labels, pred_labels)
        except ValueError:
            auc = 0  # Nếu không có nhãn dương/âm, AUC không được tính

        if auc > best_auc:
            best_auc = auc
            best_threshold = threshold

    optimal_thresholds_auc.append(best_threshold)

# Bảng hiển thị các chỉ số với ngưỡng tối ưu theo AUC
print("\nMetrics: Accuracy, Precision, Recall, F1-score, AUC-ROC")
metrics_table = PrettyTable()
metrics_table.field_names = ["Disease Label", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]

auc_scores_optimal = []
for i, label in enumerate(LABELS):
    true_labels = all_labels[:, i]
    pred_labels = (all_predictions[:, i] > optimal_thresholds_auc[i]).astype(int)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    # Tính AUC-ROC cho ngưỡng tối ưu
    try:
        auc_roc = roc_auc_score(true_labels, pred_labels)
        auc_scores_optimal.append(auc_roc)
        auc_value = f"{auc_roc:.4f}"
    except ValueError:
        auc_scores_optimal.append(float('nan'))
        auc_value = "N/A"

    metrics_table.add_row([label, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", auc_value])

print(metrics_table)

# Tính Macro Averages
accuracy_macro_optimal = np.mean([
    accuracy_score(all_labels[:, i], (all_predictions[:, i] > optimal_thresholds_auc[i]).astype(int))
    for i in range(len(LABELS))
])
precision_macro_optimal = np.mean([
    precision_score(all_labels[:, i], (all_predictions[:, i] > optimal_thresholds_auc[i]).astype(int), zero_division=0)
    for i in range(len(LABELS))
])
recall_macro_optimal = np.mean([
    recall_score(all_labels[:, i], (all_predictions[:, i] > optimal_thresholds_auc[i]).astype(int), zero_division=0)
    for i in range(len(LABELS))
])
f1_macro_optimal = np.mean([
    f1_score(all_labels[:, i], (all_predictions[:, i] > optimal_thresholds_auc[i]).astype(int), zero_division=0)
    for i in range(len(LABELS))
])
auc_macro_optimal = np.nanmean(auc_scores_optimal)  # Bỏ qua giá trị NaN

# In kết quả Macro Average
metrics_table_macro = PrettyTable()
metrics_table_macro.field_names = ["Metric", "Macro Average"]
metrics_table_macro.add_row(["Accuracy", f"{accuracy_macro_optimal:.4f}"])
metrics_table_macro.add_row(["Precision", f"{precision_macro_optimal:.4f}"])
metrics_table_macro.add_row(["Recall", f"{recall_macro_optimal:.4f}"])
metrics_table_macro.add_row(["F1-Score", f"{f1_macro_optimal:.4f}"])
metrics_table_macro.add_row(["AUC-ROC", f"{auc_macro_optimal:.4f}"])

print("\nMacro Average Metrics")
print(metrics_table_macro)