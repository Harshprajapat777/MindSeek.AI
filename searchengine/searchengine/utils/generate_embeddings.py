"""
Generate embeddings from AI/ML expert profiles using sentence-transformers.
Uses the all-MiniLM-L6-v2 model for efficient semantic embeddings.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
CSV_PATH = BASE_DIR.parent / 'aiml_expert_profiles.csv'
OUTPUT_DIR = BASE_DIR / 'searchengine' / 'data'
EMBEDDINGS_FILE = OUTPUT_DIR / 'expert_embeddings.pkl'
METADATA_FILE = OUTPUT_DIR / 'expert_metadata.pkl'
BATCH_SIZE = 32


def load_expert_data(csv_path):
    """Load expert data from CSV file."""
    print(f"Loading expert data from: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"Loaded {len(df)} expert profiles")

    return df


def create_searchable_text(row):
    """Combine relevant fields into searchable text for embedding."""
    parts = [
        str(row.get('GenericTitle', '') or ''),
        str(row.get('Biography', '') or ''),
        str(row.get('InternalBiography', '') or ''),
        str(row.get('Skills', '') or ''),
        str(row.get('Products', '') or ''),
        str(row.get('Companies', '') or ''),
        str(row.get('Position', '') or ''),
        str(row.get('Projects', '') or ''),
        str(row.get('EmploymentHistory', '') or ''),
        str(row.get('Education', '') or ''),
        str(row.get('Sector1', '') or ''),
    ]

    # Filter empty strings and join
    text = ' '.join(filter(lambda x: x.strip(), parts))

    # Truncate if too long (model has max token limit)
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


def generate_embeddings(df, model):
    """Generate embeddings for all expert profiles."""
    print("Preparing text for embedding generation...")

    # Create searchable text for each expert
    texts = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing texts"):
        text = create_searchable_text(row)
        if text.strip():
            texts.append(text)
            valid_indices.append(idx)

    print(f"Generating embeddings for {len(texts)} profiles...")

    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )

    return embeddings, valid_indices


def save_embeddings(embeddings, df, valid_indices, output_dir):
    """Save embeddings and metadata to disk."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_data = {
        'embeddings': embeddings,
        'model_name': MODEL_NAME,
        'dimension': embeddings.shape[1]
    }

    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"Saved embeddings to: {EMBEDDINGS_FILE}")

    # Save metadata (expert info for search results)
    metadata = []
    for idx in valid_indices:
        row = df.iloc[idx]
        metadata.append({
            'uid': row.get('UID', ''),
            'first_name': row.get('FirstName', ''),
            'surname': row.get('Surname', ''),
            'email': row.get('EmailAddress', ''),
            'city': row.get('City', ''),
            'country': row.get('Country', ''),
            'title': row.get('GenericTitle', ''),
            'biography': str(row.get('Biography', ''))[:500],  # Truncate for display
            'skills': row.get('Skills', ''),
            'position': row.get('Position', ''),
            'companies': row.get('Companies', ''),
            'sector': row.get('Sector1', ''),
            'external_link': row.get('ExternalProfileLink', ''),
        })

    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to: {METADATA_FILE}")

    return len(metadata)


def main():
    print("=" * 60)
    print("Expert Profile Embedding Generator")
    print("=" * 60)

    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    # Load the model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Load expert data
    df = load_expert_data(CSV_PATH)

    # Generate embeddings
    embeddings, valid_indices = generate_embeddings(df, model)

    # Save to disk
    num_saved = save_embeddings(embeddings, df, valid_indices, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"Successfully generated embeddings for {num_saved} experts")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nNext step: Run build_faiss.py to create the search index")


if __name__ == '__main__':
    main()
