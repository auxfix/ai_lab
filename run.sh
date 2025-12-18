#!/bin/bash
echo "🚀 Code RAG Quick Start"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull the model
echo "📥 Downloading CodeLlama model (this may take a while)..."
ollama pull codellama:7b

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Run the system
echo "🎬 Starting Code RAG Assistant..."
python main.py --repo .

echo "✅ Done! Access the web UI with: streamlit run web_ui.py"
