# Code RAG System

A Retrieval-Augmented Generation (RAG) system for querying your code repository using natural language.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** (for local LLM)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull codellama:7b
   # Or try: ollama pull llama3:8b
   # Or try: ollama pull mistral:7b
   ```

### Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the system on your code repository
python main.py --repo /path/to/your/repo

# Or use current directory
python main.py --repo .
```

### Usage Options

#### Interactive Mode (CLI)
```bash
python main.py --repo /path/to/repo
```

Commands in interactive mode:
- `/help` - Show available commands
- `/sources` - Toggle source display
- `/chunks N` - Change number of chunks retrieved (default: 5)
- `/quit` - Exit

#### Single Question Mode
```bash
python main.py --repo /path/to/repo --question "How does authentication work?"
```

#### Force Reindexing
```bash
python main.py --repo /path/to/repo --reindex
```

#### Web UI (Streamlit)
```bash
streamlit run web_ui.py
```

## 📁 Project Structure

```
rag_script/
├── main.py              # Main orchestrator and CLI
├── code_miner.py        # Extracts code files from repository
├── smart_chunker.py     # Chunks code intelligently
├── vectorizer.py        # Creates embeddings and vector storage
├── query_engine.py      # Query processing and LLM integration
├── web_ui.py           # Streamlit web interface
├── requirements.txt    # Python dependencies
└── run.sh             # Quick start script
```

## 🔧 Configuration

### Change LLM Model

Edit `main.py` line 49-50:

```python
self.engine = CodeQueryEngine(
    vectorizer=self.vectorizer,
    llm_backend="ollama",  # or "openai"
    model="codellama:7b",  # or "llama3:8b", "mistral:7b", etc.
)
```

### Change Chunk Size

Edit `main.py` line 32:

```python
chunker = SmartCodeChunker(chunk_size=800, chunk_overlap=150)
```

### Change Embedding Model

Edit `main.py` line 36:

```python
self.vectorizer = CodeVectorizer(
    model_name="all-MiniLM-L6-v2",  # or other sentence-transformers model
    persist_dir="./chroma_db"
)
```

## 🛠️ Troubleshooting

### "Ollama connection issue"
- Make sure Ollama is running: `ollama serve`
- Check if model is installed: `ollama list`
- Pull the model: `ollama pull codellama:7b`

### "No code files found"
- Check your repo path is correct
- Make sure your file extensions are in `code_miner.py` line 19-48
- Check if directories are being ignored in line 11-18

### "Module not found" errors
- Install dependencies: `pip install -r requirements.txt`
- Consider using a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Linux/Mac
  # or: venv\Scripts\activate  # On Windows
  pip install -r requirements.txt
  ```

## 📊 Features

- ✅ Supports 20+ programming languages
- ✅ Intelligent code chunking at natural boundaries
- ✅ Persistent vector database (ChromaDB)
- ✅ Local LLM support via Ollama
- ✅ Interactive CLI interface
- ✅ Web UI with Streamlit
- ✅ Configurable chunk sizes and retrieval counts
- ✅ Source attribution for answers

## 🔒 Privacy

- All data stays local on your machine
- No external API calls (when using Ollama)
- Vector database stored in `./chroma_db/`

## 📝 Example Questions

- "How does authentication work in this codebase?"
- "Where is the database connection configured?"
- "Show me how to create a new user"
- "What API endpoints are available?"
- "How is error handling implemented?"
- "Where are the tests for the UserService?"

## 🚦 System Requirements

- **RAM**: 4GB minimum, 8GB recommended (for embedding model + LLM)
- **Storage**: ~2GB for models + your codebase embeddings
- **CPU**: Modern multi-core processor recommended
- **GPU**: Optional, but speeds up embeddings significantly

## 📄 License

MIT (or your preferred license)

