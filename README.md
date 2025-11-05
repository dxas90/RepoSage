# Local RAG Documentation Search

A local Retrieval Augmented Generation (RAG) system that indexes markdown documentation from GitHub repositories and enables semantic search using LlamaIndex, Qdrant vector database, and Ollama for local LLM inference.

## 🎯 Overview

This project allows you to:

- Clone and index documentation from multiple GitHub repositories
- Search across all your documentation using semantic search
- Get AI-powered answers with source citations
- Run everything locally (no API keys, no cloud services)

## 🏗️ Architecture

```text
┌─────────────────┐
│  sources.yaml   │ → GitHub repos to clone
└────────┬────────┘
         ↓
┌─────────────────┐
│ markdown_docs/  │ → Cloned repositories
└────────┬────────┘
         ↓
┌─────────────────┐
│ HuggingFace     │ → Generate 384-dim embeddings
│ Embeddings      │   (all-MiniLM-L6-v2)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Qdrant Vector   │ → Store document embeddings
│ Database        │   (Cosine similarity)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Query Engine    │ → Retrieve relevant docs
└────────┬────────┘
         ↓
┌─────────────────┐
│ Ollama LLM      │ → Generate contextual answers
│ (llama3.2)      │
└─────────────────┘
```

## 📋 Prerequisites

- **Docker** (for Qdrant vector database)
- **Ollama** (for local LLM inference)
- **mise** (for Python and dependency management)
- **Git** (for cloning repositories)

### Installing Ollama

```bash
# macOS
brew install ollama
ollama serve  # Start Ollama server

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull the required model
ollama pull llama3.2:latest
```

### Installing mise

```bash
# macOS
brew install mise

# Linux
curl https://mise.run | sh

# Add to your shell config (~/.zshrc or ~/.bashrc)
eval "$(mise activate zsh)"
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd OWN_SEARCH
```

### 2. Setup Environment

```bash
# Install Python 3 and uv package manager via mise
mise install

# Install Python dependencies
mise run setup
```

This will:

- Create a `.venv` virtual environment
- Install all required packages from `requeriments.txt`

### 3. Start Qdrant Vector Database

```bash
docker compose up -d
```

This starts Qdrant on `http://localhost:6333` with persistent storage in `./qdrant_storage/`.

### 4. Configure Documentation Sources

Edit `workspace/notebooks/sources.yaml` to specify which GitHub repos to index:

```yaml
your-github-username:
  - https://github.com/username/repo1.git
  - https://github.com/username/repo2.git

another-org:
  - https://github.com/org/another-repo.git
```

### 5. Start Jupyter Lab

```bash
mise run start
```

This launches Jupyter Lab at `http://localhost:8888`. Open the provided link in your browser.

### 6. Run the RAG Notebook

Open `workspace/notebooks/rag_qa.ipynb` and execute cells sequentially:

1. **Install dependencies** (first cell)
2. **Clone repositories** (pulls repos from `sources.yaml`)
3. **Index documents** (loads markdown files, generates embeddings, stores in Qdrant)
4. **Query the system** (ask questions and get AI-powered answers)

## 📖 Usage

### Adding New Documentation Sources

1. Edit `workspace/notebooks/sources.yaml`:

   ```yaml
   your-org:
     - https://github.com/your-org/new-repo.git
   ```

2. Re-run the "Get the sources" cell in `rag_qa.ipynb` to clone new repos

3. Re-run the "load the markdowns" cell to re-index all documents

### Querying Documentation

In the final cell of `rag_qa.ipynb`, modify the question:

```python
question = """
How do I set up CI/CD for a Python project?
"""

response = query_engine.query(question)

print("🧠 Answer:\n", response.response)
print("\n📄 Sources:")
for src in response.source_nodes:
    print("-", src.metadata.get("file_path", "unknown"))
```

The response includes:

- **Answer**: AI-generated response based on your documentation
- **Sources**: File paths to the documents used to generate the answer

### Customizing Search Parameters

Adjust similarity threshold and number of results:

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Return top 5 most similar chunks (default: 3)
)
```

## 🔧 Configuration

### Environment Variables

Set in notebook cells before initialization:

```python
# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Qdrant Configuration

- **Host**: `localhost`
- **Port**: `6333`
- **Collection**: `markdown_rag`
- **Vector Dimension**: `384` (matches embedding model)
- **Distance Metric**: Cosine similarity

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2`

To change:

```python
embed_model = HuggingFaceEmbedding(model_name="your-model-name")
Settings.embed_model = embed_model
```

### LLM Model

Default: `llama3.2:latest`

To change:

```python
ensure_model("llama3.3:latest")  # Auto-pulls if not available
llm = Ollama(model="llama3.3:latest", base_url="http://localhost:11434")
Settings.llm = llm
```

## 📁 Project Structure

```text
.
├── docker-compose.yml          # Qdrant service definition
├── mise.toml                   # Tooling and task configuration
├── requeriments.txt            # Python dependencies
├── qdrant_storage/             # Persistent vector database storage
│   └── collections/
│       └── markdown_rag/       # Document embeddings
└── workspace/
    ├── markdown_docs/          # Cloned GitHub repositories
    │   └── {org}/{repo}/       # Organized by GitHub user/org
    └── notebooks/
        ├── sources.yaml        # Repository sources configuration
        └── rag_qa.ipynb        # Main RAG notebook
```

## 🛠️ Advanced Usage

### Updating Existing Repositories

The clone script automatically pulls updates for existing repos:

```python
# In the "Get the sources" cell
# Existing repos are updated with: git pull --ff-only
# New repos are cloned with: git clone --depth=1
```

### Clearing the Vector Database

To re-index from scratch:

```bash
# Stop Qdrant
docker compose down

# Remove storage
rm -rf qdrant_storage/

# Restart Qdrant
docker compose up -d
```

Then re-run the indexing cell in the notebook.

### Working with Large Repositories

The system clones repos with `--depth=1` to save space. If you need full history:

Edit the clone command in `rag_qa.ipynb`:

```python
# Remove --depth=1 flag
subprocess.run(["git", "clone", repo], cwd=dir_path, check=True)
```

## 🐛 Troubleshooting

### Ollama Connection Error

```text
⚠️ Ollama API not reachable at http://localhost:11434
```

**Solution**: Ensure Ollama is running:

```bash
ollama serve
```

### Model Not Found

```text
❌ Model 'llama3.2:latest' not found
```

**Solution**: Pull the model manually:

```bash
ollama pull llama3.2:latest
```

### Qdrant Connection Error

```text
ConnectionError: Cannot connect to Qdrant
```

**Solution**: Verify Qdrant is running:

```bash
docker compose ps
docker compose up -d  # If not running
```

### Embedding Dimension Mismatch

```text
Vector dimension mismatch: expected 384, got XXX
```

**Solution**: Delete the collection and re-create:

```bash
docker compose down
rm -rf qdrant_storage/
docker compose up -d
```

Then re-run the indexing cell.

### TOKENIZERS_PARALLELISM Warning

```text
huggingface/tokenizers: The current process just got forked...
```

**Solution**: Already handled via `os.environ["TOKENIZERS_PARALLELISM"] = "false"`

## 🔐 Privacy & Security

- **100% Local**: All processing happens on your machine
- **No API Keys**: No cloud services or external API calls
- **No Data Sharing**: Your documents never leave your system
- **Offline Capable**: Works without internet (after initial setup)

## 📊 Performance Considerations

### Indexing Speed

- **Initial indexing**: ~1-5 minutes for dozens of repos
- **Incremental updates**: Only new/changed files are processed
- **Embedding generation**: CPU-bound (consider GPU for large datasets)

### Query Speed

- **Vector search**: <100ms for most queries
- **LLM generation**: 1-5 seconds depending on hardware

### Storage Requirements

- **Embeddings**: ~1MB per 1000 documents
- **Qdrant storage**: Grows with number of indexed documents
- **Cloned repos**: Depends on repository sizes (shallow clones save space)

## 🤝 Contributing

To add features or fix bugs:

1. Fork the repository
2. Create a feature branch
3. Make your changes in `workspace/notebooks/rag_qa.ipynb`
4. Test thoroughly
5. Submit a pull request

## 📝 License

MIT

## 🙏 Acknowledgments

- **LlamaIndex**: RAG framework
- **Qdrant**: Vector database
- **Ollama**: Local LLM inference
- **HuggingFace**: Embedding models
- **mise**: Development environment management

## 📧 Support

For issues or questions:

- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

---

Built with ❤️ for local, private documentation search
