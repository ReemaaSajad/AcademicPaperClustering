import pandas as pd
import numpy as np
import argparse
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def run_kmeans(embeddings, n_clusters=10):
    """Cluster embeddings using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f"[KMeans] Silhouette Score: {score:.3f}")
    return labels

def run_bertopic(abstracts, embeddings=None, model_name="all-MiniLM-L6-v2"):
    """Run BERTopic on raw abstracts or embeddings."""
    if embeddings is not None:
        print("[BERTopic] Running BERTopic with precomputed embeddings...")
    else:
        print("[BERTopic] Running BERTopic with transformer model...")
    
    topic_model = BERTopic(embedding_model=model_name, verbose=True)
    topics, probs = topic_model.fit_transform(abstracts, embeddings)

    # Get top topic keywords
    topic_info = topic_model.get_topic_info()
    print(f"[BERTopic] Found {len(topic_info)} topics.")

    return topics, topic_info

def cluster_and_model(input_csv, embeddings_path, output_csv, model_name="all-MiniLM-L6-v2", n_clusters=10):
    """Main driver: Load data, run KMeans + BERTopic, and save labeled output."""
    df = pd.read_csv(input_csv)
    if "cleaned_abstract" not in df.columns:
        raise ValueError("Input CSV must contain 'cleaned_abstract' column.")

    embeddings = np.load(embeddings_path)

    # ---- KMeans Clustering ----
    print("\n🔹 Running KMeans Clustering...")
    kmeans_labels = run_kmeans(embeddings, n_clusters=n_clusters)
    df["kmeans_cluster"] = kmeans_labels

    # ---- BERTopic ----
    print("\n🔹 Running BERTopic Topic Modeling...")
    abstracts = df["cleaned_abstract"].tolist()
    topics, topic_info = run_bertopic(abstracts, embeddings, model_name=model_name)
    df["bertopic_topic"] = topics

    # Save result
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    topic_info.to_csv(output_csv.replace(".csv", "_topics.csv"), index=False)

    print(f"\n✅ Saved clustered data to {output_csv}")
    print(f"✅ Saved BERTopic topics to {output_csv.replace('.csv', '_topics.csv')}")

    return df, topic_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering + topic modeling on academic paper abstracts")
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV file")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings (.npy file)")
    parser.add_argument("--output", required=False, help="Path to save output CSV")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Model name for BERTopic (default: all-MiniLM-L6-v2)")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters for KMeans")

    args = parser.parse_args()

    input_path = args.input
    embeddings_path = args.embeddings
    filename = os.path.basename(input_path).replace(".csv", "_clustered.csv")
    output_path = args.output if args.output else f"data/processed/{filename}"

    cluster_and_model(input_path, embeddings_path, output_path, model_name=args.model, n_clusters=args.clusters)
