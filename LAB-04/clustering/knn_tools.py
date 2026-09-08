import numpy as np

def find_nearest_neighbors(query_point, dataset, k=5):
    distances = np.linalg.norm(dataset - query_point, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices, distances[nearest_indices]