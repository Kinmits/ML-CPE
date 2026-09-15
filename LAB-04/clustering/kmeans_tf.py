import numpy as np

class KMeansTensorFlow:
    def __init__(self, k=4, max_iters=100, tol=1e-4, seed=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centroids = None

    def fit(self, X):
        np.random.seed(self.seed)
        X = np.array(X, dtype=np.float32)
        n_samples = X.shape[0]
        
        idx = np.random.choice(n_samples, self.k, replace=False)
        centroids = X[idx]

        for _ in range(self.max_iters):
            dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            cluster_assignments = np.argmin(dists, axis=1)

            new_centroids = np.array([
                X[cluster_assignments == j].mean(axis=0) if np.sum(cluster_assignments == j) > 0 
                else X[np.random.choice(n_samples)]
                for j in range(self.k)
            ])
            
            diff = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            if diff < self.tol:
                break

        self.centroids = centroids
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        dists = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def get_inertia(self, X):
        X = np.array(X, dtype=np.float32)
        dists = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
        min_dists = np.min(dists, axis=1)
        return np.sum(min_dists)