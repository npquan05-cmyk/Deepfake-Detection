# 🧠 Deepfake Detection using EfficientNetB0

## 🚀 Overview

Dự án này xây dựng mô hình Deep Learning để **phân loại ảnh Deepfake (FAKE) và ảnh thật (REAL)** sử dụng **EfficientNetB0** với kỹ thuật transfer learning.

Pipeline bao gồm:

* Load và augment dữ liệu ảnh
* Fine-tune EfficientNetB0
* Train với EarlyStopping & ReduceLROnPlateau
* Đánh giá bằng ROC Curve, AUC
* Tìm **optimal threshold**
* Test model + Confusion Matrix
* Predict ảnh thực tế với **face detection (OpenCV Haar Cascade)**

---

## 📂 Dataset

Dataset sử dụng cấu trúc thư mục:

```
Dataset/
│
├── Train/
├── Validation/
└── Test/
```

Đường dẫn trong code:

```python
train_dir = "/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset/Train"
val_dir   = "/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset/Validation"
test_dir  = "/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset/Test"
```

---

## ⚙️ Data Preprocessing

Sử dụng `ImageDataGenerator`:

* Train:

  * Rescale (1./255)
  * Horizontal Flip
  * Rotation (10 độ)

* Validation/Test:

  * Chỉ rescale

Kích thước ảnh: **224x224**
Batch size: **32**

---

## 🧠 Model Architecture

### Backbone:

* EfficientNetB0 (pretrained ImageNet)
* `include_top=False`

### Fine-tuning:

* Freeze toàn bộ layer trừ **30 layer cuối**

### Head:

```
GlobalAveragePooling2D
Dense(512, relu) + Dropout(0.5)
Dense(128, relu) + Dropout(0.3)
Dense(1, sigmoid)
```

---

## ⚙️ Training Configuration

* Optimizer: Adam (learning rate = 1e-4)
* Loss: Binary Crossentropy
* Metrics: Accuracy
* Epochs: 10

### Callbacks:

* EarlyStopping:

  * monitor: val_loss
  * patience: 3
* ReduceLROnPlateau:

  * factor: 0.3
  * patience: 2
  * min_lr: 1e-6

---

## 📊 Training Visualization

Sau khi train:

* Vẽ biểu đồ:

  * Accuracy (Train vs Validation)
  * Loss (Train vs Validation)

---

## 📈 Evaluation

### 1. Validation

* Tính:

  * ROC Curve
  * AUC Score
* Tìm **optimal threshold**:

```python
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
```

---

### 2. Test

* Predict trên test set
* Convert sang label bằng threshold

### Metrics:

* Classification Report
* Confusion Matrix (Seaborn Heatmap)
* ROC AUC

---

## 🧪 Inference (Predict ảnh thực tế)

### Pipeline:

1. Đọc ảnh bằng OpenCV
2. Detect face với Haar Cascade
3. Crop + padding (20%)
4. Resize về 224x224
5. Normalize (chia 255)
6. Predict xác suất

### Rule:

```python
if prob > threshold:
    REAL
else:
    FAKE
```

### Output:

```
Image: xxx.jpg
Probability REAL: 0.82
Prediction: REAL
```

---

## 🧠 Face Detection

Sử dụng:

```python
cv2.CascadeClassifier(haarcascade_frontalface_default.xml)
```

* scaleFactor = 1.3
* minNeighbors = 5

---

## ▶️ Run Project

### 1. Train + Evaluate

```bash
python main.py
```

---

### 2. Predict ảnh trong folder

Trong code:

```python
predict_folder(model, "your_folder_path", threshold)
```

---

## 📦 Requirements

```txt
tensorflow
numpy
matplotlib
seaborn
scikit-learn
opencv-python
```

---

## ⚠️ Lưu ý

* Dataset nằm trên Kaggle → không push lên GitHub
* Model `.h5` lớn → nên ignore
* Cần GPU để train nhanh hơn

---

## 👨‍💻 Author

Nguyễn Phú Quân

---

## 📜 License

Dự án phục vụ mục đích học tập và nghiên cứu.

