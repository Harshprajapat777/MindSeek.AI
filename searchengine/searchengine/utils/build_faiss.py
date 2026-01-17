"""
Build FAISS index for fast similarity search on expert embeddings.
Creates an optimized index for semantic search queries.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import faiss


# Configuration
DATA_DIR = BASE_DIR / 'searchengine' / 'data'
EMBEDDINGS_FILE = DATA_DIR / 'expert_embeddings.pkl'
METADATA_FILE = DATA_DIR / 'expert_metadata.pkl'
FAISS_INDEX_FILE = DATA_DIR / 'expert_faiss.index'


def load_embeddings():
    """Load embeddings from pickle file."""
    print(f"Loading embeddings from: {EMBEDDINGS_FILE}")

    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: Embeddings file not found at {EMBEDDINGS_FILE}")
        print("Please run generate_embeddings.py first")
        sys.exit(1)

    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    dimension = data['dimension']
    model_name = data['model_name']

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Dimension: {dimension}")
    print(f"Model: {model_name}")

    return embeddings, dimension


def build_faiss_index(embeddings, dimension):
    """Build FAISS index for similarity search."""
    print("\nBuilding FAISS index...")

    num_vectors = len(embeddings)

    # Convert to float32 (required by FAISS)
    embeddings = embeddings.astype(np.float32)

    # Choose index type based on dataset size
    if num_vectors < 1000:
        # For small datasets, use flat index (exact search)
        print("Using Flat index (exact search) for small dataset")
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity (normalized vectors)

    elif num_vectors < 10000:
        # For medium datasets, use IVF with flat quantizer
        print("Using IVF index for medium dataset")
        nlist = min(100, num_vectors // 10)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)

    else:
        # For large datasets, use IVF with PQ compression
        print("Using IVF-PQ index for large dataset")
        nlist = min(1000, num_vectors // 100)
        m = 8  # Number of subquantizers
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
        index.train(embeddings)

    # Add vectors to index
    print(f"Adding {num_vectors} vectors to index...")
    index.add(embeddings)

    print(f"Index built successfully. Total vectors: {index.ntotal}")

    return index


def save_index(index, output_path):
    """Save FAISS index to disk."""
    print(f"\nSaving index to: {output_path}")
    faiss.write_index(index, str(output_path))
    print("Index saved successfully")


def verify_index(index_path, embeddings, dimension):
    """Verify the saved index by running a test query."""
    print("\nVerifying index with test query...")

    # Load the index
    index = faiss.read_index(str(index_path))

    # Use first embedding as test query
    query = embeddings[0:1].astype(np.float32)

    # Search for top 5 similar
    k = min(5, index.ntotal)
    distances, indices = index.search(query, k)

    print(f"Test query results (top {k}):")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  {i+1}. Index: {idx}, Score: {dist:.4f}")

    # First result should be the query itself with score ~1.0
    if indices[0][0] == 0 and distances[0][0] > 0.99:
        print("Index verification PASSED")
        return True
    else:
        print("WARNING: Index verification may have issues")
        return False


def main():
    print("=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)

    # Load embeddings
    embeddings, dimension = load_embeddings()

    # Build index
    index = build_faiss_index(embeddings, dimension)

    # Save index
    save_index(index, FAISS_INDEX_FILE)

    # Verify index
    verify_index(FAISS_INDEX_FILE, embeddings, dimension)

    # Load metadata to show summary
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
        print(f"\nMetadata entries: {len(metadata)}")

    print("\n" + "=" * 60)
    print("FAISS index built successfully!")
    print(f"Index file: {FAISS_INDEX_FILE}")
    print(f"Index size: {FAISS_INDEX_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)
    print("\nYour semantic search is ready to use!")


if __name__ == '__main__':
    main()
