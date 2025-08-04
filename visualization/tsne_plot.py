import argparse
import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.manifold import TSNE

def reduce_dimensions(embeddings, method="umap", n_neighbors=15, min_dist=0.1, random_state=42):
    """Reduce embeddings to 2D using UMAP or t-SNE."""
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
    else:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    
    reduced = reducer.fit_transform(embeddings)
    return reduced

def visualize_clusters(df, reduced_embeddings, method, output_dir):
    """Create static and interactive visualizations of clusters with topic labels."""
    df["x"] = reduced_embeddings[:, 0]
    df["y"] = reduced_embeddings[:, 1]

    os.makedirs(output_dir, exist_ok=True)

    # Static Plot (Seaborn/Matplotlib)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="x", y="y", hue="bertopic_topic", palette="tab10",
        data=df, legend="full", s=50, alpha=0.8
    )
    plt.title(f"{method.upper()} Plot of Academic Paper Clusters")
    plt.savefig(os.path.join(output_dir, f"{method}_plot_static.png"))
    print(f"Saved static plot: {method}_plot_static.png")

    # Interactive Plot (Plotly)
    fig = px.scatter(
        df,
        x="x", y="y",
        color="bertopic_topic",
        hover_data=["title", "bertopic_topic"],
        title=f"{method.upper()} Plot (Interactive)",
        color_continuous_scale="Viridis"
    )
    html_path = os.path.join(output_dir, f"{method}_plot_interactive.html")
    fig.write_html(html_path)
    print(f"Saved interactive plot: {method}_plot_interactive.html")

    return df

def run_dimensionality_reduction(input_csv, embeddings_path, method="umap"):
    df = pd.read_csv(input_csv)
    embeddings = np.load(embeddings_path)

    reduced = reduce_dimensions(embeddings, method=method)

    df_with_coords = visualize_clusters(df, reduced, method, output_dir="data/plots")

    # Save CSV with coordinates
    output_csv = input_csv.replace(".csv", f"_{method}_coords.csv")
    df_with_coords.to_csv(output_csv, index=False)
    print(f" Saved dataframe with coordinates to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize paper clusters using UMAP or t-SNE")
    parser.add_argument("--input", required=True, help="CSV file with clustered papers")
    parser.add_argument("--embeddings", required=True, help="Path to .npy embedding file")
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap", help="Dimensionality reduction method")

    args = parser.parse_args()
    run_dimensionality_reduction(args.input, args.embeddings, args.method)
