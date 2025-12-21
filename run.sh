#!/bin/bash
set -e  # Exit on error

echo "🚀 Code RAG Quick Start"
echo "======================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found."
    read -p "Do you want to install Ollama? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📥 Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        echo "⚠️  Skipping Ollama installation. You'll need to install it manually."
    fi
fi

# Check if Ollama is running
if command -v ollama &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "🔄 Starting Ollama service..."
        ollama serve &> /dev/null &
        sleep 2
    fi
    
    # Pull the model if not exists
    if ! ollama list | grep -q "codellama:7b"; then
        echo "📥 Downloading CodeLlama model (this may take a while - ~3.8GB)..."
        ollama pull codellama:7b
    else
        echo "✅ CodeLlama model already installed"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "Choose an option:"
echo "  1) Run interactive CLI"
echo "  2) Launch web UI"
echo "  3) Exit"
echo ""
read -p "Enter choice (1-3): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo "🎬 Starting Code RAG Assistant (CLI)..."
        python main.py --repo .
        ;;
    2)
        echo "🌐 Starting Web UI..."
        streamlit run web_ui.py
        ;;
    3)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option. Run manually:"
        echo "  CLI: python main.py --repo ."
        echo "  Web: streamlit run web_ui.py"
        ;;
esac
