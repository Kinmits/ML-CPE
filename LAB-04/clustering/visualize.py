import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from kmeans_tf import KMeansTensorFlow

def plot_elbow_curve(X, max_k=10, output_path="outputs/01_elbow.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        km = KMeansTensorFlow(k=k, seed=42)
        km.fit(X)
        inertias.append(km.get_inertia(X))
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o', color='#d62728', linewidth=2)
    plt.title("Elbow Method for Optimal K", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_clusters_2d(X, labels, centroids, output_path="outputs/02_clusters.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)
    
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("Set2", len(np.unique(labels)))
    
    for cluster_id in np.unique(labels):
        points = X_pca[labels == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}", s=60, alpha=0.8, color=palette[cluster_id])
        
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, color='black', label='Centroids', zorder=10)
    plt.title("Player Clusters (2D PCA Projection)", fontsize=14, fontweight='bold')
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def export_clustering_results(df_raw, labels, output_clustered="outputs/clustered_animals.csv", output_summary="outputs/cluster_summary.csv"):
    os.makedirs(os.path.dirname(output_clustered), exist_ok=True)
    
    df_clustered = df_raw.copy()
    df_clustered['cluster'] = labels
    df_clustered.to_csv(output_clustered, index=False)
    
    summary = df_clustered.groupby('cluster').agg(
        count=('cluster', 'count')
    ).reset_index()
    summary.to_csv(output_summary, index=False)