# MindSeek AI - Semantic Search Engine

A semantic search application to find AI/ML experts using vector embeddings and FAISS indexing.

---

## How It Works

### Overview
```
CSV Data → Embeddings (Vectors) → FAISS Index → Fast Semantic Search
```

---

## Core Concepts

### 1. What are Embeddings?

Embeddings convert text into **vectors** (arrays of decimal numbers) that capture semantic meaning.

**Example:**
```
Input Text:
"NLP expert with PyTorch and transformers experience"

Output Vector:
[0.023, -0.156, 0.089, 0.742, -0.334, ...384 decimal numbers]
```

> **Note:** Embeddings are NOT binary (0s and 1s). They are decimal numbers that represent meaning in mathematical space.

---

### 2. Why Use Embeddings?

| Text Comparison | Vector Comparison |
|-----------------|-------------------|
| "NLP expert" vs "Natural Language Processing specialist" | Similar vectors (close in space) |
| Exact keyword matching only | Understands meaning & context |
| Misses synonyms | Captures semantic similarity |

---

### 3. What is FAISS?

**FAISS = Facebook AI Similarity Search**

A library that organizes vectors for **fast similarity search**.

#### Without FAISS (Slow)
```
User Query → Compare with Expert 1
           → Compare with Expert 2
           → Compare with Expert 3
           → ...
           → Compare with Expert 500

Result: Check ALL 500 one by one = SLOW
```

#### With FAISS (Fast)
```
User Query → FAISS jumps directly to similar vectors
           → Skips irrelevant ones

Result: Finds matches in milliseconds = FAST
```

#### Analogy
| Without FAISS | With FAISS |
|---------------|------------|
| Finding a word by reading entire dictionary page by page | Using alphabetical index to jump directly |

---

## System Architecture

### Pre-processing (One-time, Offline)
```
CSV Data (500+ experts)
        ↓
Generate Embeddings (all-MiniLM-L6-v2 model)
        ↓
Build FAISS Index
        ↓
Save to disk (.pkl and .index files)
```

### Query Time (Real-time, Fast)
```
User Query: "NLP expert with PyTorch experience"
        ↓
Convert query to embedding (single text - very fast)
        ↓
FAISS searches pre-built index (milliseconds)
        ↓
Returns top matching experts
```

---

## Scripts & Output Files

### 1. `generate_embeddings.py`

**Purpose:** Converts expert profile text into vector embeddings.

**Process:**
```
Step 1: Load CSV file (aiml_expert_profiles.csv)
Step 2: For each expert, combine text fields:
        → Biography + Skills + Projects + Education + etc.
Step 3: Load AI model (all-MiniLM-L6-v2)
Step 4: Convert each expert's text → 384 numbers (vector)
Step 5: Save vectors → expert_embeddings.pkl
Step 6: Save expert info → expert_metadata.pkl
```

**Output Files:**

| File | Contains |
|------|----------|
| `expert_embeddings.pkl` | Vector numbers (384 dimensions per expert) |
| `expert_metadata.pkl` | Expert info (name, skills, bio, etc.) for displaying results |

---

### 2. `build_faiss.py`

**Purpose:** Builds optimized search index from embeddings.

**Process:**
```
Step 1: Load expert_embeddings.pkl (vectors)
Step 2: Create FAISS index structure
Step 3: Add all vectors to index
Step 4: Optimize for fast search
Step 5: Save → expert_faiss.index
```

**Output File:**

| File | Contains |
|------|----------|
| `expert_faiss.index` | Optimized search index for fast similarity lookup |

---

## File Flow Diagram

```
CSV (raw data)
      ↓
┌─────────────────────────┐
│ generate_embeddings.py  │
└─────────────────────────┘
      ↓
├── expert_embeddings.pkl  (vectors)
├── expert_metadata.pkl    (expert info)
      ↓
┌─────────────────────────┐
│    build_faiss.py       │
└─────────────────────────┘
      ↓
└── expert_faiss.index     (search index)
```

---

## Search Flow

```
Query → Embed → Search faiss.index → Get IDs → Lookup metadata.pkl → Show Results
```

---

## Performance Comparison

| Without Pre-built Index | With Pre-built Index |
|-------------------------|----------------------|
| Generate embeddings for all 500+ experts every query | Embeddings already stored |
| Compare query with each expert one by one | Optimized vector similarity search |
| Slow (seconds to minutes) | Fast (milliseconds) |
| High system load | Low system load |

---

## Summary

| Component | Purpose |
|-----------|---------|
| `aiml_expert_profiles.csv` | Raw expert data |
| `generate_embeddings.py` | Convert text → vectors |
| `expert_embeddings.pkl` | Store meaning as numbers |
| `expert_metadata.pkl` | Store expert info for display |
| `build_faiss.py` | Build search index |
| `expert_faiss.index` | Search vectors FAST |

---

## Quick Start

```bash
# Navigate to project
cd E:\MindSeekAI\MindSeek.AI\searchengine

# Install dependencies
pip install -r requirements.txt

# Generate embeddings from expert data
python searchengine/utils/generate_embeddings.py

# Build FAISS index
python searchengine/utils/build_faiss.py

# Run Django server
python manage.py runserver
```

---

## Tech Stack

- **Backend:** Django 4.2
- **Embedding Model:** all-MiniLM-L6-v2 (sentence-transformers)
- **Vector Search:** FAISS (Facebook AI Similarity Search)
- **Data Processing:** Pandas, NumPy

---

## Project Structure

```
searchengine/
├── manage.py
├── requirements.txt
├── README.md
└── searchengine/
    ├── settings.py
    ├── urls.py
    ├── views.py
    ├── models.py
    ├── templates/
    │   └── Home.html
    ├── data/
    │   ├── expert_embeddings.pkl
    │   ├── expert_metadata.pkl
    │   └── expert_faiss.index
    └── utils/
        ├── generate_embeddings.py
        └── build_faiss.py
```

---

## Model Information

**all-MiniLM-L6-v2**
- Embedding Dimension: 384
- Max Sequence Length: 256 tokens
- Optimized for: Semantic similarity search
- Source: Hugging Face / sentence-transformers

---

*MindSeek AI - Discover Brilliant Minds*
