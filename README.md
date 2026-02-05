# MindSeek AI

**Discover Brilliant Minds** — A semantic search platform that discovers and matches AI/ML experts based on their expertise, research, projects, and skills using advanced AI techniques.

---

## About The Project

MindSeek AI is an intelligent expert-matching search engine built with Django. Instead of relying on simple keyword matching, it uses **vector embeddings**, **FAISS indexing**, and **cross-encoder reranking** to understand the *meaning* behind your search query and find the most relevant experts from a database of 500+ AI/ML professionals.

### Key Features

- **Semantic Search** — Understands meaning, not just keywords ("NLP expert" matches "Natural Language Processing specialist")
- **LLM-Powered Query Analysis** — Uses OpenAI GPT to extract skills, experience level, and requirements from natural language queries
- **Cross-Encoder Reranking** — Re-scores results using a secondary model for higher accuracy
- **Fast Search Mode** — FAISS-only search for instant results without LLM processing
- **Modern Web UI** — Responsive interface with real-time progress tracking, expert cards, and quick search tags
- **Health Monitoring** — Built-in health check endpoint for production monitoring

---

## Architecture & How Components Connect

```
                         +---------------------+
                         |     Web Browser      |
                         |    (Home.html UI)    |
                         +----------+----------+
                                    |  HTTP / JSON
                                    v
                         +---------------------+
                         |   Django Backend     |
                         |    (views.py)        |
                         +--+-------+-------+--+
                            |       |       |
           +----------------+       |       +----------------+
           v                        v                        v
  /api/search/             /api/search/fast/        /api/search/health/
  (Full LLM Search)        (FAISS-Only Search)      (Health Check)
           |                        |
           v                        v
  +------------------+     +------------------+
  | OpenAI GPT-4o    |     |                  |
  | (Query Analysis) |     |                  |
  +--------+---------+     |                  |
           |               |                  |
           v               v                  |
  +------------------------------+            |
  |        FAISS Index           |            |
  |  (Vector Similarity Search)  |            |
  +-------------+----------------+            |
                |                             |
                v                             |
  +------------------------------+            |
  |     Cross-Encoder Reranker   |            |
  |   (ms-marco-MiniLM-L-6-v2)  |            |
  +-------------+----------------+            |
                |                             |
                v                             v
  +----------------------------------------------+
  |          Expert Results (Top 5)               |
  |  name, title, skills, score, match reason     |
  +----------------------------------------------+
```

### Data Pipeline (One-Time Setup)

```
aiml_expert_profiles.csv (500+ expert profiles)
        |
        v
generate_embeddings.py
  - Combines text fields (bio, skills, projects, education)
  - Converts to 384-dim vectors using all-MiniLM-L6-v2
        |
        +--> expert_embeddings.pkl  (16 MB - vector data)
        +--> expert_metadata.pkl    (6.7 MB - expert info)
        |
        v
build_faiss.py
  - Creates optimized search index from embeddings
        |
        +--> expert_faiss.index     (707 KB - search index)
```

### Search Pipeline (Real-Time)

```
User Query: "NLP expert with 5 years PyTorch experience"
    |
    v
Step 1: LLM Query Understanding (OpenAI gpt-4o-mini)
        -> Extracts: skills=[NLP, PyTorch], experience=5 years, etc.
    |
    v
Step 2: FAISS Semantic Search
        -> Encodes query to vector, searches index, retrieves candidates
    |
    v
Step 3: Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
        -> Re-scores top 100 candidates for precision
    |
    v
Step 4: Return Top 5 Results with scores, bio, and match reasons
```

---

## Tech Stack

| Layer            | Technology                                   |
| ---------------- | -------------------------------------------- |
| **Backend**      | Django 4.2, Python                           |
| **LLM**         | OpenAI API (gpt-4o-mini)                     |
| **Embeddings**   | sentence-transformers (all-MiniLM-L6-v2)     |
| **Vector Search**| FAISS (faiss-cpu)                            |
| **Reranking**    | Cross-Encoder (ms-marco-MiniLM-L-6-v2)      |
| **ML Framework** | PyTorch 2.6, HuggingFace Transformers        |
| **Database**     | SQLite3                                      |
| **Frontend**     | HTML5, CSS3, Vanilla JavaScript              |
| **Data**         | Pandas, NumPy                                |
| **Deployment**   | Gunicorn / ASGI compatible                   |

---

## Project Structure

```
MindSeek.AI/
├── LICENSE                          # MIT License
├── README.md                        # This file
├── aiml_expert_profiles.csv         # Source data (500+ AI/ML experts)
└── searchengine/                    # Django project root
    ├── manage.py                    # Django CLI
    ├── db.sqlite3                   # SQLite database
    ├── requirements.txt             # Python dependencies
    ├── .env.example                 # Environment variable template
    ├── .env                         # Your local env config (not committed)
    └── searchengine/                # Django app
        ├── settings.py              # Django configuration
        ├── urls.py                  # URL routing (API endpoints)
        ├── views.py                 # Request handlers
        ├── models.py                # Expert database model
        ├── wsgi.py                  # WSGI entry point
        ├── asgi.py                  # ASGI entry point
        ├── templates/
        │   └── Home.html            # Frontend UI
        ├── data/
        │   ├── expert_embeddings.pkl   # Pre-computed vector embeddings
        │   ├── expert_metadata.pkl     # Expert metadata for results
        │   └── expert_faiss.index      # FAISS search index
        └── utils/
            ├── expert_search.py        # Core search service
            ├── generate_embeddings.py  # Embedding generation script
            └── build_faiss.py          # FAISS index builder
```

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **8 GB+ RAM** (required for loading ML models)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/MindSeek.AI.git
   cd MindSeek.AI/searchengine
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   SEARCH_TOP_K=5
   FAISS_CANDIDATES=-1
   RERANK_CANDIDATES=50
   ```

5. **Generate embeddings** (skip if `data/` files already exist)
   ```bash
   python searchengine/utils/generate_embeddings.py
   ```

6. **Build FAISS index** (skip if `expert_faiss.index` already exists)
   ```bash
   python searchengine/utils/build_faiss.py
   ```

7. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

8. **Start the development server**
   ```bash
   python manage.py runserver
   ```

9. **Open in browser**
   ```
   http://127.0.0.1:8000
   ```

---

## API Endpoints

### `POST /api/search/` — Full Semantic Search (LLM + FAISS + Reranking)

**Request:**
```json
{
  "query": "NLP expert with PyTorch experience",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "query": "NLP expert with PyTorch experience",
  "query_analysis": {
    "interpreted_need": "...",
    "required_skills": ["NLP", "PyTorch"],
    "preferred_experience": "..."
  },
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "name": "Expert Name",
      "title": "Senior ML Engineer",
      "skills": "Python, PyTorch, NLP",
      "location": "San Francisco, USA",
      "scores": {
        "cross_encoder": 8.5,
        "semantic": 85.3
      }
    }
  ]
}
```

### `POST /api/search/fast/` — Fast Search (FAISS Only, No LLM)

Same request format. Returns results without LLM query analysis. Faster response, default `top_k=20`.

### `GET /api/search/health/` — Health Check

```json
{
  "status": "healthy",
  "index_size": 500,
  "metadata_count": 500,
  "openai_configured": true,
  "model": "gpt-4o-mini"
}
```

---

## Environment Variables

| Variable             | Description                                  | Default        |
| -------------------- | -------------------------------------------- | -------------- |
| `OPENAI_API_KEY`     | Your OpenAI API key (required for full search) | —              |
| `OPENAI_MODEL`       | OpenAI model to use for query analysis       | `gpt-4o-mini`  |
| `SEARCH_TOP_K`       | Number of final results to return            | `5`            |
| `FAISS_CANDIDATES`   | FAISS candidates to retrieve (`-1` = all)    | `-1`           |
| `RERANK_CANDIDATES`  | Candidates passed to cross-encoder reranker  | `50`           |

---

## AI Models Used

| Model                        | Purpose                    | Dimensions | Source           |
| ---------------------------- | -------------------------- | ---------- | ---------------- |
| **all-MiniLM-L6-v2**        | Text embedding generation  | 384        | HuggingFace      |
| **ms-marco-MiniLM-L-6-v2**  | Cross-encoder reranking    | —          | HuggingFace      |
| **gpt-4o-mini**             | Query understanding (LLM)  | —          | OpenAI API       |

---

## Deployment

### Production with Gunicorn

```bash
cd searchengine
gunicorn searchengine.wsgi:application --bind 0.0.0.0:8000
```

### Production with ASGI (Uvicorn)

```bash
cd searchengine
uvicorn searchengine.asgi:application --host 0.0.0.0 --port 8000
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2026 Harsh Prajapat**

---

*MindSeek AI — Discover Brilliant Minds*
