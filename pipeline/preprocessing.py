import re
import pandas as pd
import unicodedata
import argparse
import os

def clean_abstract(text):
    """
    Clean abstract text for embedding:
    - Normalize unicode
    - Remove LaTeX equations and symbols
    - Remove non-ASCII characters
    - Lowercase
    - Remove multiple spaces
    - Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Remove LaTeX math expressions (e.g., $E = mc^2$)
    text = re.sub(r"\$.*?\$", " ", text)

    # Remove LaTeX commands (e.g., \cite{}, \ref{}, \alpha)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Remove HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()

    # Lowercase
    text = text.lower()

    # Remove unwanted punctuation except intra-word dashes
    text = re.sub(r"[^\w\s\-]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_dataframe(input_csv, output_csv):
    """
    Load raw CSV, clean abstract text, and save preprocessed version.
    """
    df = pd.read_csv(input_csv)

    # Keep only needed columns
    expected_columns = {"title", "abstract", "authors", "link", "source"}
    df = df[[col for col in df.columns if col in expected_columns]]

    # Drop rows with missing abstracts
    df.dropna(subset=["abstract"], inplace=True)

    # Apply preprocessing
    df["cleaned_abstract"] = df["abstract"].apply(clean_abstract)

    # Save preprocessed file
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw academic paper abstracts")
    parser.add_argument("--input", required=True, help="Path to raw CSV file (e.g., data/raw/arxiv_papers.csv)")
    parser.add_argument("--output", required=False, help="Path to save cleaned CSV (default: same name in data/processed/)")

    args = parser.parse_args()

    input_path = args.input
    filename = os.path.basename(input_path)
    output_path = args.output if args.output else f"data/processed/cleaned_{filename}"

    preprocess_dataframe(input_path, output_path)
