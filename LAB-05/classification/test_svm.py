# สคริปต์สั้นสำหรับทดสอบการรันแยกต่างหาก
from data_load import load_dataset
from preprocess import preprocess_data, scale_features
from split_data import split_data
from svm_model import train_svm_models

df = load_dataset('fpl.csv')
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

models = train_svm_models(X_train_scaled, y_train, ['linear'])
print("Test completed successfully!")