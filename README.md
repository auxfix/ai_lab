# 🧠 AI Lab Repository
_______________________________
Collection of AI projects and experiments.
________________________________________

## 📁 Repository Structure

```
ai_lab/
├── 📂 rag_system/          # Advanced Retrieval-Augmented Generation system
│   ├── 📄 README.md        # RAG system documentation
│   ├── 📄 requirements.txt # Dependencies
│   └── 📂 src/             # Source code
│
├── 📂 voice_assistant/     # 🎙️ GPT-NeoX + Whisper Voice Companion
│   ├── 📄 voice_companion.py  # Main application
│   ├── 📄 requirements.txt     # Python dependencies
│   ├── 📄 config.yaml          # Configuration file
│   ├── 📂 models/              # Downloaded AI models
│   ├── 📂 audio_outputs/       # Generated speech files
│   └── 📄 README.md            # Detailed documentation
│
├── 📄 LICENSE              # MIT License
├── 📄 .gitignore          # Git ignore rules
└── 📄 CONTRIBUTING.md     # Contribution guidelines
```

## 🚀 Quick Navigation

- **[🎙️ Voice Assistant](./voice_assistant/)** – Real-time voice AI companion
- **[🔍 RAG System](./rag_system/)** – Advanced document retrieval system
- **[📋 Installation Guide](#-installation)**
- **[🤝 Contributing](#-contributing)**

## 🎙️ Voice Assistant Project

### 🌟 **What is This?**

A **real-time voice AI companion** that combines three state-of-the-art AI models:
- **🗣️ OpenAI Whisper** – Speech recognition
- **🧠 GPT-NeoX-20B** – Intelligent conversation
- **🎵 Coqui TTS** – Natural speech synthesis

### ✨ **Key Features**

| Feature | Description | Status |
|---------|-------------|--------|
| **🎤 Real-time Speech Recognition** | Converts speech to text with Whisper | ✅ Working |
| **💭 Intelligent Conversations** | GPT-NeoX-20B powered responses | ✅ Working |
| **🔊 Natural Voice Synthesis** | High-quality Coqui TTS voices | ✅ Working |
| **⚡ Multiple Voice Options** | 100+ speaker voices available | ✅ Working |
| **🌍 Multilingual Support** | Whisper supports 99 languages | ✅ Working |
| **💾 Conversation Memory** | Remembers context across turns | ✅ Working |
| **🎚️ Adjustable Parameters** | Control creativity, speed, emotion | ✅ Working |
| **📁 Audio Export** | Save conversations as audio files | ✅ Working |

### 🎯 **Use Cases**

- **🤖 Personal AI Assistant** – Daily task helper
- **🎓 Language Learning** – Practice conversations
- **🧠 Mental Wellness** – Therapeutic conversations
- **🎮 Gaming Companion** – Interactive NPCs
- **🏥 Accessibility Tool** – Voice interface for everyone
- **🔬 Research Platform** – AI conversation studies

### 📊 **Performance Metrics**

| Metric | Value | Target |
|--------|-------|--------|
| 🕒 Response Time | 4-15 seconds | <2 seconds |
| 🔊 Voice Quality | 8/10 human-like | 9.5/10 |
| 🧠 Context Memory | 10 turns | 50+ turns |
| 💬 Languages | 99+ | All major |
| 🎭 Voice Options | 100+ | Unlimited |

### 🛠️ **Technology Stack**

```mermaid
graph LR
    A[Python 3.8+] --> B[PyTorch]
    B --> C[Transformers]
    C --> D[Whisper]
    D --> E[GPT-NeoX-20B]
    E --> F[Coqui TTS]
    F --> G[PyAudio]
    G --> H[SoundDevice]
    
    style A fill:#3572A5
    style B fill:#EE4C2C
    style C fill:#FFD43B
    style D fill:#10A37F
    style E fill:#6F42C1
    style F fill:#FF6B6B
    style G fill:#4CAF50
    style H fill:#2196F3
```

### 🚀 **Quick Start**

```bash
# Clone repository
git clone https://github.com/yourusername/ai_lab.git
cd ai_lab/voice_assistant

# Install dependencies
pip install -r requirements.txt

# Run with microphone
python voice_companion.py --mic

# Run with text input
python voice_companion.py
```

### ⚙️ **Configuration Examples**

**Basic Configuration:**
```yaml
audio:
  sample_rate: 16000
  silence_threshold: 0.01
  
whisper:
  model: "base"
  language: "auto"
  
gpt:
  max_tokens: 150
  temperature: 0.7
  
tts:
  model: "tts_models/en/vctk/vits"
  speaker: "p225"
  speed: 1.0
```

**Advanced Configuration:**
```yaml
system:
  enable_emotion_detection: true
  enable_voice_cloning: false
  save_conversations: true
  cache_responses: true
  
performance:
  use_quantization: true
  cpu_offload: false
  batch_size: 4
  
privacy:
  local_processing_only: true
  encrypt_conversations: false
  auto_delete_audio: true
```

## 🔍 RAG System Project

*(Brief overview - see [RAG System README](./rag_system/README.md) for details)*

A sophisticated **Retrieval-Augmented Generation** system designed for intelligent document processing and question answering. This system combines vector search with large language models to provide accurate, context-aware responses based on document collections.

**Core Capabilities:**
- 📚 Document ingestion and processing
- 🔍 Semantic search and retrieval
- 💬 Context-aware question answering
- 📊 Multi-source information synthesis

## 📋 Installation

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended)
- **16GB+ RAM**
- **50GB+ free disk space**

### Step-by-Step Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/ai_lab.git
   cd ai_lab
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   # Voice Assistant
   cd voice_assistant
   pip install -r requirements.txt
   
   # RAG System
   cd ../rag_system
   pip install -r requirements.txt
   ```

4. **Download Models**
   ```bash
   # Voice Assistant will download models automatically on first run
   python voice_companion.py --download-models
   ```

### 🐳 Docker Installation

```bash
# Build and run Voice Assistant
docker build -t voice-assistant -f voice_assistant/Dockerfile .
docker run --gpus all -p 8000:8000 voice-assistant

# Build and run RAG System
docker build -t rag-system -f rag_system/Dockerfile .
docker run -p 8001:8001 rag-system
```

## 🎮 Usage Examples

### Voice Assistant Demo

```python
from voice_assistant import VoiceCompanion

# Initialize with custom settings
assistant = VoiceCompanion(
    tts_model="tts_models/en/vctk/vits",
    whisper_model="medium",
    gpt_model="EleutherAI/gpt-neox-20b",
    device="cuda"
)

# Have a conversation
response = assistant.chat("What's the weather like today?")
print(f"Assistant: {response}")

# Save conversation
assistant.save_conversation("weather_chat.json")

# Export audio
assistant.export_audio("weather_response.wav")
```

### API Server Mode

```bash
# Start REST API server
python voice_assistant/api_server.py --port 8000

# Make API calls
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "voice": "p225"}'
```

## 📊 Hardware Requirements

### Minimum Requirements

| Component | Voice Assistant | RAG System | Combined |
|-----------|----------------|------------|----------|
| **GPU VRAM** | 8GB | 4GB | 12GB |
| **RAM** | 16GB | 8GB | 24GB |
| **Storage** | 50GB | 20GB | 70GB |
| **CPU** | 4 cores | 2 cores | 6 cores |

### Recommended Setup

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | NVIDIA RTX 3090/4090 (24GB) | Model inference |
| **CPU** | AMD Ryzen 9 / Intel i9 | Data processing |
| **RAM** | 64GB DDR5 | Large model loading |
| **Storage** | 1TB NVMe SSD | Fast model access |
| **OS** | Ubuntu 22.04 LTS | Stable environment |

## 🔧 Troubleshooting

### Common Issues

**Issue:** "Out of memory error"
```bash
# Solution: Use quantization
python voice_companion.py --quantize 4bit

# Or use smaller models
python voice_companion.py --whisper-tiny --gpt-small
```

**Issue:** "Audio device not found"
```bash
# Check audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Set default device
export AUDIO_DEVICE=0  # Linux/Mac
set AUDIO_DEVICE=0     # Windows
```

**Issue:** "Model download failed"
```bash
# Manual download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('EleutherAI/gpt-neox-20b')"

# Or use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Performance Tips

1. **Enable quantization:** Reduce VRAM usage by 75%
2. **Use CPU offloading:** For systems with limited GPU memory
3. **Batch processing:** Process multiple requests together
4. **Model caching:** Cache frequently used models in RAM
5. **Audio compression:** Reduce audio quality for faster processing



### Development Setup

```bash
# Fork repository
git clone https://github.com/yourusername/ai_lab.git
cd ai_lab

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python -m pytest tests/

# Commit changes
git commit -m "Add amazing feature"

# Push to branch
git push origin feature/amazing-feature

# Open Pull Request
```

### Code Standards

- **Python:** Follow PEP 8 guidelines
- **Documentation:** Include docstrings for all functions
- **Testing:** Write unit tests for new features
- **Commits:** Use conventional commit messages
- **Branching:** Follow Git Flow branching model

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
