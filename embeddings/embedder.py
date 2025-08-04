import pandas as pd
import numpy as np
import argparse
import os
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """Load a SentenceTransformer model (e.g., SciBERT or MiniLM)."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    return model

def embed_abstracts(input_csv, output_npy, model_name="all-MiniLM-L6-v2"):
    """Embed cleaned abstracts and save embedding matrix."""
    df = pd.read_csv(input_csv)

    if "cleaned_abstract" not in df.columns:
        raise ValueError("Input CSV must have a 'cleaned_abstract' column.")

    abstracts = df["cleaned_abstract"].tolist()

    model = load_model(model_name)
    embeddings = model.encode(abstracts, show_progress_bar=True)

    # Save embeddings
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, embeddings)

    print(f"Saved {len(embeddings)} embeddings to {output_npy}")
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentence embeddings for academic abstracts")
    parser.add_argument("--input", required=True, help="Path to preprocessed CSV file (with 'cleaned_abstract')")
    parser.add_argument("--output", required=False, help="Path to save .npy embedding file")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Transformer model name (default: all-MiniLM-L6-v2)")

    args = parser.parse_args()

    input_path = args.input
    filename = os.path.basename(input_path).replace(".csv", ".npy")
    output_path = args.output if args.output else f"data/processed/{filename}"

    embed_abstracts(input_path, output_path, model_name=args.model)
