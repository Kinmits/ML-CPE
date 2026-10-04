# LAB-07: 1D CNN Classification on FPL Dataset

## 📌 Project Overview
This project implements a **1D Convolutional Neural Network (1D CNN)** using TensorFlow/Keras to classify English Premier League (FPL) player positions (`GKP`, `DEF`, `MID`, `FWD`) based on 18 numerical statistical features. The project follows a modular pipeline design pattern for clean, maintainable, and scalable deep learning development.

---

## 📂 Project Structure
```text
LAB-07/
├── fpl.csv                    # Dataset file (567 players, 77 columns)
├── classification/
│   ├── __init__.py
│   ├── data_loader.py         # Loads and extracts 18 features from CSV
│   ├── preprocessing.py       # Standardizes features using StandardScaler & reshapes for 1D CNN
│   ├── split_data.py          # Splits dataset into Train, Validation, and Test sets (stratified)
│   ├── cnn_model.py           # Builds 1D CNN architectures (Config 1 & Config 2) and saves models
│   ├── evaluate.py            # Evaluates performance, generates confusion matrices, and plots loss/accuracy
│   ├── test_cnn.py            # Tests saved models with random sample predictions
│   └── main.py                # Main entry point for the training pipeline
└── output/
    ├── eval_Config_1.png      # Training history & confusion matrix for Config 1
    └── eval_Config_2.png      # Training history & confusion matrix for Config 2