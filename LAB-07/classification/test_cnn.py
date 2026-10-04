import tensorflow as tf
import numpy as np
import random

def test_random_samples(model_path, X_test, y_test, class_names):
    print(f"\n[INFO] Testing 4 random players with saved model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    indices = random.sample(range(len(X_test)), 4)
    for i, idx in enumerate(indices):
        data = X_test[idx]
        true_label = class_names[y_test[idx]]
        
        # Predict 
        pred_prob = model.predict(np.expand_dims(data, axis=0), verbose=0)
        pred_label = class_names[np.argmax(pred_prob)]
        
        mark = "✅ (Correct)" if true_label == pred_label else "❌ (Wrong)"
        print(f"Player {i+1}: Actual = {true_label:<5} | AI Predicted = {pred_label:<5} {mark}")