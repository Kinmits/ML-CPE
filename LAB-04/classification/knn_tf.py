import numpy as np

class KNNTensorFlow:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype=np.float32)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)
        predictions = []
        for x in X_test:
            dists = np.sum((self.X_train - x) ** 2, axis=1)
            top_k_indices = np.argsort(dists)[:self.k]
            top_k_labels = self.y_train[top_k_indices]
            vals, counts = np.unique(top_k_labels, return_counts=True)
            predictions.append(vals[np.argmax(counts)])
        return np.array(predictions)