from data_loader import load_dataset
from preprocessing import preprocess_data, scale_features
from split_data import split_data
from nn_model import train_nn_configurations

df = load_dataset('../fpl.csv')
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

test_config = {'Test_Config': (16,)}
models = train_nn_configurations(X_train_scaled, y_train, test_config, [10])
print("[SUCCESS] Test NN module executed successfully!")