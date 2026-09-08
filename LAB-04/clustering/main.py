from data_loader import load_clustering_data
from kmeans_tf import KMeansTensorFlow
from visualize import plot_elbow_curve, plot_clusters_2d, export_clustering_results

def main():
    print("=== Running Clustering Module ===")
    data = load_clustering_data("../fpl.csv")
    
    plot_elbow_curve(data["X_scaled"], max_k=10, output_path="outputs/01_elbow.png")
    
    optimal_k = 4
    kmeans = KMeansTensorFlow(k=optimal_k, seed=42)
    kmeans.fit(data["X_scaled"])
    cluster_labels = kmeans.predict(data["X_scaled"])
    
    plot_clusters_2d(data["X_scaled"], cluster_labels, kmeans.centroids, output_path="outputs/02_clusters.png")
    export_clustering_results(data["raw_df"], cluster_labels)
    print("Clustering Outputs Saved Successfully!")

if __name__ == "__main__":
    main()