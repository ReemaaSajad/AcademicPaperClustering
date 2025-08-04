import argparse
import os
import subprocess

def run_command(command_list):
    print(f"\n🔧 Running: {' '.join(command_list)}")
    # Use the virtual environment's Python executable
    if command_list[0] == "python":
        command_list[0] = os.path.join(".venv", "Scripts", "python.exe")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command_list)}")

def run_pipeline(keyword, source="arxiv", max_results=100, model="all-MiniLM-L6-v2", n_clusters=10, reduction="umap"):
    print(f"\n🚀 Starting NLP Pipeline for keyword: '{keyword}'")

    # Step 1: Fetch papers
    run_command([
        "python", "api/fetch_papers.py",
        keyword,
        "--source", source,
        "--max-results", str(max_results)
    ])
    
    # Set filenames based on source
    source_map = {
        "arxiv": "arxiv_papers.csv",
        "semantic_scholar": "semantic_scholar_papers.csv",
        "semanticscholar": "semantic_scholar_papers.csv",
        "both": "combined_papers.csv"
    }
    base_file = source_map[source.lower()]
    raw_path = os.path.join("data", "raw", base_file)
    cleaned_path = os.path.join("data", "processed", f"cleaned_{base_file}")
    embedding_path = cleaned_path.replace(".csv", ".npy")
    clustered_path = cleaned_path.replace(".csv", "_clustered.csv")
    
    # Step 2: Preprocessing
    run_command([
        "python", "pipeline/preprocessing.py",
        "--input", raw_path,
        "--output", cleaned_path
    ])

    # Step 3: Embedding
    run_command([
        "python", "embeddings/embedder.py",
        "--input", cleaned_path,
        "--output", embedding_path,
        "--model", model
    ])

    # Step 4: Clustering + BERTopic
    run_command([
        "python", "clustering/cluster.py",
        "--input", cleaned_path,
        "--embeddings", embedding_path,
        "--output", clustered_path,
        "--model", model,
        "--clusters", str(n_clusters)
    ])

    # Step 5: Dimensionality Reduction + Visualization
    run_command([
        "python", "visualization/tsne_plot.py",
        "--input", clustered_path,
        "--embeddings", embedding_path,
        "--method", reduction
    ])

    print(f"\n✅ Pipeline complete for keyword: {keyword}")
    print(f"📊 Check outputs in: data/processed/ and data/plots/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full academic paper clustering + visualization pipeline")
    parser.add_argument("--keyword", required=True, help="Keyword to search papers for")
    parser.add_argument("--source", default="arxiv", choices=["arxiv", "semantic_scholar", "both"], help="Data source")
    parser.add_argument("--max-results", type=int, default=100, help="Number of papers to fetch")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters for KMeans")
    parser.add_argument("--reduction", choices=["umap", "tsne"], default="umap", help="Dimensionality reduction method")

    args = parser.parse_args()

    try:
        run_pipeline(
            keyword=args.keyword,
            source=args.source,
            max_results=args.max_results,
            model=args.model,
            n_clusters=args.clusters,
            reduction=args.reduction
        )
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
