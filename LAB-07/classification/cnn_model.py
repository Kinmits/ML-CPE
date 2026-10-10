from tensorflow.keras import layers, models
import tensorflow as tf

def build_cnn_model(input_shape, num_classes, config_type=1):
    model = models.Sequential()
    
    if config_type == 1:
        # Config 1: 1 Conv1D Layer (แบบตื้น)
        model.add(layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
    else:
        # Config 2: 2 Conv1D Layers (แบบลึก)
        model.add(layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
        # ไม่ใส่ MaxPooling ซ้ำเพราะฟีเจอร์เรามีแค่ 18 ตัว มิติจะเล็กเกินไป
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(model, X_train, y_train, X_val, y_val, epochs, model_name):
    print(f"\n[INFO] Training model ({model_name}) for {epochs} Epochs...")
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, verbose=1
    )
    model.save(f"{model_name}.h5")
    print(f"[INFO] Model saved as {model_name}.h5")
    return history