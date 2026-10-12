# Lab 08: Image Classification using VGG16

This project demonstrates how to build, train, and evaluate a deep learning image classification model using a pre-trained **VGG16** architecture with TensorFlow and Keras.

---

## Project Structure

```text
classification/
│
├── data_loader.py       # Handles dataset loading and initial batching
├── preprocessing.py     # Image preprocessing and VGG-specific transformations
├── split_data.py        # Splits dataset into training, validation, and testing sets
├── vgg_model.py         # Defines the VGG16 architecture and fine-tuning layers
├── evaluate.py          # Evaluates model performance on test data
├── test_vgg.py          # Script for testing random images and generating sample plots
├── main.py              # Main pipeline script to run training and evaluation
└── outputs/             # Generated models, logs, and evaluation plots (ignored by Git)