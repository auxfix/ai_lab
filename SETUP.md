# Quick Setup Guide

## Option 1: Automated Setup (Recommended)

```bash
chmod +x run.sh
./run.sh
```

This script will:
1. Check for Python 3
2. Install Ollama (if needed)
3. Download CodeLlama model
4. Create virtual environment
5. Install Python dependencies
6. Launch the system

## Option 2: Manual Setup

### Step 1: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama (in a separate terminal)
ollama serve

# Pull a model
ollama pull codellama:7b
```

### Step 2: Install Python Dependencies

```bash
# Optional: Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run the System

```bash
# CLI Mode
python main.py --repo /path/to/your/repo

# Web UI Mode
streamlit run web_ui.py
```

## First Time Usage

1. Point the system to your code repository
2. Wait for initial indexing (this happens once)
3. Start asking questions about your code!

## Tips

- First run will download embedding models (~80MB)
- Indexing time depends on repository size
- Use `--reindex` flag to rebuild the database
- Database is cached in `./chroma_db/` folder

## Common Issues

**"Module not found"**
- Run: `pip install -r requirements.txt`

**"Ollama connection issue"**
- Make sure Ollama is running: `ollama serve`
- Check model is installed: `ollama list`

**"No code files found"**
- Check your repository path
- Make sure it contains supported file types

## System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection (first run only)

