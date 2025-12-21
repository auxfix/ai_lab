"""
GPU-Optimized Configuration for High-End Hardware
RTX 3090 24GB VRAM + 126GB RAM

This module provides optimized settings for powerful hardware.
Import and use these instead of defaults for maximum performance.
"""

# Embedding Configuration
EMBEDDING_CONFIG = {
    # High-quality model (768-dim vs 384-dim)
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    
    # GPU acceleration
    "use_gpu": True,
    
    # Larger batches for GPU
    "batch_size": 256,
    
    # Database
    "persist_dir": "./chroma_db",
}

# Alternative embedding models (ordered by quality)
EMBEDDING_MODELS = {
    "best_quality": "sentence-transformers/all-mpnet-base-v2",  # 768-dim, slow but accurate
    "balanced": "sentence-transformers/all-MiniLM-L12-v2",  # 384-dim, faster
    "fast": "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, fastest
    "code_specific": "microsoft/codebert-base",  # 768-dim, trained on code
}

# Chunking Configuration
CHUNKING_CONFIG = {
    # Larger chunks for better context
    "chunk_size": 1500,  # vs 800 default
    "chunk_overlap": 200,  # vs 150 default
}

# For very large files or complex codebases
CHUNKING_CONFIG_LARGE = {
    "chunk_size": 2000,
    "chunk_overlap": 300,
}

# LLM Configuration
LLM_CONFIG = {
    "backend": "ollama",
    
    # Recommended models for your VRAM
    "model": "codellama:34b",  # Best for code, fits perfectly
    
    "options": {
        "temperature": 0.7,
        "num_predict": 1000,  # More detailed answers
        "num_ctx": 8192,  # Large context window
        "num_gpu": 99,  # Use all GPU layers possible
    }
}

# Alternative LLM models for your hardware
LLM_MODELS = {
    # Best for code (recommended)
    "codellama_34b": {
        "model": "codellama:34b",
        "vram": "19GB",
        "description": "Best code understanding, perfect fit for RTX 3090"
    },
    
    # Best for general reasoning
    "mixtral_8x7b": {
        "model": "mixtral:8x7b",
        "vram": "26GB", 
        "description": "Excellent reasoning, MoE architecture"
    },
    
    # Best specialized code model
    "deepseek_coder_33b": {
        "model": "deepseek-coder:33b",
        "vram": "18GB",
        "description": "Specialized for code, very accurate"
    },
    
    # Largest that fits with offloading
    "llama3_70b": {
        "model": "llama3:70b-instruct-q4_0",
        "vram": "24GB + CPU offload",
        "description": "Highest quality, needs CPU offloading",
        "options": {
            "num_gpu": 35,  # Partial GPU offloading
            "num_ctx": 8192,
        }
    },
    
    # Smaller for speed
    "codellama_13b": {
        "model": "codellama:13b",
        "vram": "7GB",
        "description": "Fast, leaves VRAM for other tasks"
    },
}

# Query Configuration
QUERY_CONFIG = {
    # Retrieve more chunks with better hardware
    "n_chunks": 8,  # vs 5 default
    
    # Show sources
    "show_sources": True,
}

# For complex queries
QUERY_CONFIG_DEEP = {
    "n_chunks": 15,  # Very comprehensive
    "show_sources": True,
}

# Performance Monitoring
MONITORING = {
    "log_gpu_usage": True,
    "log_timing": True,
    "verbose": True,
}


def get_gpu_info():
    """Get GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
                "cuda_version": torch.version.cuda,
            }
    except ImportError:
        pass
    return {"available": False}


def print_config_summary():
    """Print configuration summary"""
    print("=" * 60)
    print("🚀 GPU-OPTIMIZED CONFIGURATION")
    print("=" * 60)
    
    print("\n📊 Embedding:")
    print(f"  Model: {EMBEDDING_CONFIG['model_name']}")
    print(f"  GPU: {EMBEDDING_CONFIG['use_gpu']}")
    print(f"  Batch: {EMBEDDING_CONFIG['batch_size']}")
    
    print("\n📦 Chunking:")
    print(f"  Size: {CHUNKING_CONFIG['chunk_size']} tokens")
    print(f"  Overlap: {CHUNKING_CONFIG['chunk_overlap']} tokens")
    
    print("\n🤖 LLM:")
    print(f"  Backend: {LLM_CONFIG['backend']}")
    print(f"  Model: {LLM_CONFIG['model']}")
    print(f"  Context: {LLM_CONFIG['options']['num_ctx']} tokens")
    
    print("\n🔍 Retrieval:")
    print(f"  Chunks: {QUERY_CONFIG['n_chunks']}")
    
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"\n⚡ GPU Info:")
        print(f"  Device: {gpu_info['name']}")
        print(f"  Memory: {gpu_info['memory_total']}")
        print(f"  CUDA: {gpu_info['cuda_version']}")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    print_config_summary()
    
    print("\n💡 To use this config:")
    print("1. Import: from config_gpu import EMBEDDING_CONFIG, LLM_CONFIG")
    print("2. Pass to constructors instead of hardcoded values")
    print("\nExample:")
    print("  vectorizer = CodeVectorizer(**EMBEDDING_CONFIG)")
    print("  engine = CodeQueryEngine(vectorizer, **LLM_CONFIG)")

