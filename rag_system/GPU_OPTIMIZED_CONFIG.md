# High-Performance Configuration for RTX 3090 24GB + 126GB RAM

This configuration is optimized for your powerful hardware setup.

## Hardware-Optimized Settings

### Embedding Model
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Why**: More accurate than MiniLM (384 → 768 dimensions)
- **GPU**: Enabled (CUDA acceleration)
- **Speed**: ~10-50x faster with your RTX 3090

### Chunking Strategy
- **Chunk Size**: 1500 tokens (vs 800 default)
- **Overlap**: 200 tokens (vs 150 default)
- **Why**: Better context preservation, your RAM can handle it

### Batch Processing
- **Embedding Batch**: 256 (vs 100 default)
- **Why**: Maximize GPU utilization

### LLM Model Options

#### Recommended: CodeLlama 34B
```bash
ollama pull codellama:34b  # ~19GB - fits in your 24GB VRAM
```
- Best code understanding
- Fits comfortably in your VRAM
- Much better than 7B for complex questions

#### Alternative High-Quality Models
```bash
# Option 1: Mixtral 8x7B (excellent reasoning)
ollama pull mixtral:8x7b  # ~26GB - tight but works

# Option 2: DeepSeek Coder 33B (best for code)
ollama pull deepseek-coder:33b  # ~18GB

# Option 3: Llama 3 70B (quantized)
ollama pull llama3:70b-instruct-q4_0  # ~40GB - requires CPU offloading
```

### Context Window
- **Size**: 8192 tokens (vs 2048 default)
- **Why**: Handle larger code contexts

### Retrieval Settings
- **Default Chunks**: 7-10 (vs 5 default)
- **Why**: More comprehensive context with larger model

## Performance Expectations

### With Your Hardware:

| Operation | Time (Small Repo 100 files) | Time (Large Repo 10K files) |
|-----------|----------------------------|----------------------------|
| Initial Indexing | ~30 seconds | ~10-15 minutes |
| Query (Embedding) | <0.1 seconds | <0.1 seconds |
| LLM Response (34B) | 3-8 seconds | 3-8 seconds |

### Memory Usage:

| Component | RAM Usage | VRAM Usage |
|-----------|-----------|------------|
| Embedding Model | ~2GB | ~1GB |
| ChromaDB | ~100MB per 10K chunks | - |
| CodeLlama 34B | ~2GB | ~19GB |
| **Total** | **~4-8GB** | **~20GB** |

You have plenty of headroom! 🚀

## Optimized Run Commands

### Maximum Quality
```bash
# Uses all optimizations
python main.py --repo /path/to/repo

# In Python code, adjust main.py line 50 to:
model="codellama:34b"
```

### Even More Context (if needed)
```python
# main.py - increase chunks retrieved
result = self.engine.ask(user_input, n_chunks=10)  # vs default 5
```

### Multiple Repositories
With your RAM, you can run multiple instances:
```bash
# Terminal 1
python main.py --repo ~/project1

# Terminal 2  
python main.py --repo ~/project2
```

## GPU Monitoring

Check your GPU usage:
```bash
# Install nvtop for monitoring
sudo apt install nvtop
nvtop

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

Expected during embedding: 80-95% GPU utilization
Expected during LLM inference: 60-80% GPU utilization

## Advanced: Run Even Larger Models

Your 24GB VRAM can handle:

### 1. Llama 3 70B (with CPU offloading)
```bash
ollama pull llama3:70b

# Then edit main.py to use CPU offloading:
# In query_engine.py, add:
options={
    "num_gpu": 35,  # Put 35 layers on GPU, rest on CPU
    "num_ctx": 8192,
}
```

### 2. Multiple Models Simultaneously
With 126GB RAM, run separate models for different tasks:
- One for code questions
- One for documentation
- One for general queries

## Energy Efficiency Tips

Your RTX 3090 is powerful but power-hungry (~350W). To save energy:

```bash
# Power limit to 250W (still very fast, saves ~100W)
sudo nvidia-smi -pl 250

# Reset to default
sudo nvidia-smi -pl 350
```

## Benchmark Your Setup

After installation:
```bash
# Time the embedding process
time python main.py --repo /path/to/small/repo --reindex

# Expected: <30 seconds for 100 files
```

## Configuration Files Updated

✅ `main.py` - Larger chunks, better model
✅ `vectorizer.py` - GPU acceleration, larger batches  
✅ `query_engine.py` - Larger context, more GPU layers
✅ `run.sh` - Downloads 34B model

## Summary

Your hardware can run:
- ✅ Best embedding model (all-mpnet-base-v2)
- ✅ Largest practical model (CodeLlama 34B)
- ✅ Large chunk sizes (1500 tokens)
- ✅ Large batches (256)
- ✅ Extended context (8K tokens)
- ✅ Multiple retrieval chunks (10+)

**You're getting PREMIUM RAG quality!** 🏆

