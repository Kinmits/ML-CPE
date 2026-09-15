import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_nn_configurations(X_train_scaled, y_train, configs, epoch_list=[50, 100, 200, 500]):
    """
    เทรน Neural Network ตาม Configurations (Hidden Layers) และ Epochs ที่ต่างกัน
    """
    trained_models = {}
    
    for config_name, hidden_layers in configs.items():
        trained_models[config_name] = {}
        for epochs in epoch_list:
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                max_iter=epochs,
                random_state=42,
                activation='relu',
                solver='adam'
            )
            mlp.fit(X_train_scaled, y_train)
            trained_models[config_name][epochs] = mlp
            print(f"[NN TRAIN] Config: {config_name} | Hidden Layers: {hidden_layers} | Epochs: {epochs} Completed")
            
    return trained_models