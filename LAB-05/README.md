# LAB 05: Support Vector Machine (SVM)

## Objective
- Apply Support Vector Machine (SVM) to classify Premier League player positions based on Fantasy Premier League (FPL) performance statistics.
- Compare performance across three kernels: **Linear**, **Polynomial**, and **RBF**.

## Structure
- `data_load.py`: Load the dataset.
- `preprocess.py`: Extract features and apply `StandardScaler`.
- `split_data.py`: Split dataset into train and test sets (80/20 ratio).
- `svm_model.py`: Train SVM classifiers with different kernels.
- `evaluate.py`: Calculate accuracy scores and save confusion matrix plots.
- `main.py`: Main executable script for the entire machine learning pipeline.

## How to Run
```bash
python main.py