# AI Coding Assistant Instructions

## Project Overview
This is a **local RAG (Retrieval Augmented Generation) system** that indexes markdown documentation from GitHub repositories and enables semantic search using LlamaIndex + Qdrant + Ollama. The system clones repos defined in `sources.yaml`, extracts markdown/YAML files, embeds them using HuggingFace models, stores vectors in Qdrant, and answers questions using a local LLM.

## Architecture & Components

### Core Stack
- **Vector Database**: Qdrant (running in Docker, persistent storage in `qdrant_storage/`)
- **LLM**: Ollama (local, model: `llama3.2:latest`)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)
- **Framework**: LlamaIndex for RAG orchestration
- **Interface**: Jupyter Lab notebooks in `workspace/notebooks/`

### Data Flow
1. `sources.yaml` defines GitHub repos to clone → `workspace/markdown_docs/{owner}/`
2. `rag_qa.ipynb` reads markdown/YAML files recursively from `workspace/markdown_docs/`
3. Documents embedded via HuggingFace model → stored in Qdrant collection `markdown_rag`
4. Query engine retrieves top-k similar chunks → Ollama generates contextual answers

## Developer Workflows

### Environment Setup
```bash
mise install              # Install Python 3 + uv via mise
mise run setup            # Install dependencies via uv into .venv
docker compose up -d      # Start Qdrant vector DB
```

### Working with Notebooks
```bash
mise run start            # Launch Jupyter Lab at http://localhost:8888
```
- Main notebook: `workspace/notebooks/rag_qa.ipynb`
- Cells execute sequentially: install deps → clone repos → index docs → query

### Adding New Documentation Sources
Edit `workspace/notebooks/sources.yaml`:
```yaml
dxas90:  # GitHub username/org
  - https://github.com/dxas90/new-repo.git
```
Re-run the "Get the sources" cell in `rag_qa.ipynb` to sync.

### Querying the RAG System
Modify the `question` variable in the final notebook cell:
```python
question = "your question about the docs"
response = query_engine.query(question)
```
Response includes answer + source file paths from metadata.

## Project-Specific Conventions

### Dependency Management
- **mise.toml** handles tooling (Python 3, uv package manager)
- **requeriments.txt** (note: typo in filename) lists Python packages
- Uses `uv` instead of pip for fast installs: `uv pip install -r requeriments.txt`
- Auto-creates `.venv` via mise's `python.uv_venv_auto = true`

### Docker Services
- Only Qdrant runs in Docker (port 6333)
- Ollama + Jupyter are commented out in `docker-compose.yml` (run locally via mise)
- Qdrant data persists in `./qdrant_storage` bind mount

### Collection Schema
- Collection name: `markdown_rag`
- Vector size: **384** (must match embedding model dim)
- Distance metric: Cosine similarity
- File metadata includes `file_path` for source attribution

### File Organization
- Cloned repos: `workspace/markdown_docs/{org}/{repo}/`
- Notebooks: `workspace/notebooks/`
- Qdrant storage: `qdrant_storage/` (gitignored)
- All repos cloned with `--depth=1` for space efficiency

## Critical Patterns

### Embedding Model Initialization
Always set `TOKENIZERS_PARALLELISM=false` to avoid HuggingFace warnings:
```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Ollama Model Management
Use `ensure_model()` function to auto-pull missing models:
```python
ensure_model("llama3.2:latest")  # Checks local registry, pulls if needed
llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")
```

### Document Loading
Explicitly add file paths to metadata for source tracking:
```python
for doc in docs:
    if "file_path" not in doc.metadata:
        doc.metadata["file_path"] = os.path.abspath(doc.metadata.get("file_name", ""))
```

### Collection Creation
Check collection existence before creating to avoid errors:
```python
try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(collection_name=COLLECTION_NAME, ...)
```

## Key Files
- `mise.toml`: Tooling config, defines `setup` and `start` tasks
- `docker-compose.yml`: Qdrant service definition
- `workspace/notebooks/sources.yaml`: Repos to index
- `workspace/notebooks/rag_qa.ipynb`: Main RAG implementation
- `qdrant_storage/collections/markdown_rag/config.json`: Vector DB schema
