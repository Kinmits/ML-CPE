# LAB-04: k-Nearest Neighbors (KNN) & K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Task-Classification%20%26%20Clustering-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20NumPy-green)

การศึกษาและประยุกต์ใช้อัลกอริทึม **k-Nearest Neighbors (KNN)** สำหรับงานการจัดคลาส (Classification) และ **K-Means Clustering** สำหรับการจัดกลุ่ม (Clustering) บนชุดข้อมูลสถิตินักเตะ **Fantasy Premier League (FPL)** โดยจำแนกตำแหน่งการเล่นจากสถิติผลงานในสนาม

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB-04/
├── classification/              # โมดูลจำแนกประเภทตำแหน่งนักเตะ (KNN Classification)
│   ├── data_loader.py           # โหลดและ Preprocess ข้อมูล (StandardScaler)
│   ├── knn_tf.py                # อัลกอริทึม KNN (Vectorized Implementation)
│   ├── evaluate.py              # ฟังก์ชันประเมินผล กราฟ k-Curve และ Confusion Matrix
│   ├── main.py                  # สคริปต์หลักสำหรับรันกระบวนการ Classification
│   └── outputs/                 # โฟลเดอร์เก็บผลลัพธ์
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       └── predictions.csv
│
├── clustering/                  # โมดูลจัดกลุ่มพฤติกรรมสถิติ (K-Means Clustering)
│   ├── data_loader.py           # โหลดและ Preprocess ข้อมูลสำหรับ Clustering
│   ├── kmeans_tf.py             # อัลกอริทึม K-Means (Vectorized Implementation)
│   ├── knn_tools.py             # เครื่องมือคำนวณระยะทางและหาเพื่อนบ้านใกล้ที่สุด
│   ├── visualize.py             # ฟังก์ชันพล็อตกราฟ Elbow Curve และ 2D PCA Cluster
│   ├── main.py                  # สคริปต์หลักสำหรับรันกระบวนการ Clustering
│   └── outputs/                 # โฟลเดอร์เก็บผลลัพธ์
│       ├── 01_elbow.png
│       ├── 02_clusters.png
│       ├── cluster_summary.csv
│       └── clustered_animals.csv
│
├── fpl.csv                      # ชุดข้อมูลหลัก (Fantasy Premier League Dataset)
└── README.md                    # เอกสารอธิบายโปรเจกต์