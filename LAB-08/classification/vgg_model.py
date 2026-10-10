import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os

def build_vgg_model(input_shape, num_classes):
    print("[INFO] Building VGG16 Transfer Learning Model...")
    
    # Load VGG16 pre-trained on ImageNet without top layers
    base_model = tf.keras.applications.VGG16(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers (Transfer Learning)
    base_model.trainable = False
    print(f"[INFO] VGG16 base model loaded. Frozen layers: {len(base_model.layers)}")
    
    # Build custom classifier head
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="VGG16_Transfer_Learning")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_and_save_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    print(f"\n[INFO] Training VGG16 model for {epochs} epochs...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    os.makedirs("outputs", exist_ok=True)
    model.save("outputs/vgg_model.keras")
    print("[INFO] Model saved to outputs/vgg_model.keras")
    
    # Save history json
    history_dict = {str(k): [float(v) for v in vals] for k, vals in history.history.items()}
    with open("outputs/history.json", "w") as f:
        json.dump(history_dict, f)
    print("[INFO] Training history saved to outputs/history.json")
    
    return history