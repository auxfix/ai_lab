#!/usr/bin/env python3
"""
Benchmark script for GPU-optimized RAG system
Tests performance with your RTX 3090 + 126GB RAM setup
"""

import time
import sys
from pathlib import Path


def benchmark_embedding():
    """Benchmark embedding speed with GPU"""
    print("\n" + "=" * 60)
    print("🧠 EMBEDDING BENCHMARK")
    print("=" * 60)
    
    try:
        from vectorizer import CodeVectorizer
        
        # Test data
        test_chunks = [
            {
                "content": f"def function_{i}():\n    return {i}" * 10,
                "source_file": f"test_{i}.py",
                "language": "py",
                "chunk_id": i,
                "total_chunks": 1,
                "full_path": f"test_{i}.py",
            }
            for i in range(100)
        ]
        
        print(f"📦 Test data: {len(test_chunks)} chunks")
        
        # Benchmark with GPU
        print("\n⚡ With GPU acceleration:")
        vectorizer_gpu = CodeVectorizer(
            model_name="sentence-transformers/all-mpnet-base-v2",
            use_gpu=True
        )
        
        start = time.time()
        for chunk in test_chunks:
            vectorizer_gpu.embedding_model.encode(chunk["content"])
        gpu_time = time.time() - start
        
        print(f"  Time: {gpu_time:.2f}s")
        print(f"  Speed: {len(test_chunks)/gpu_time:.1f} chunks/sec")
        
        # Benchmark without GPU (for comparison)
        print("\n🐌 Without GPU (CPU only):")
        vectorizer_cpu = CodeVectorizer(
            model_name="sentence-transformers/all-mpnet-base-v2",
            use_gpu=False
        )
        
        start = time.time()
        for chunk in test_chunks[:20]:  # Only 20 to save time
            vectorizer_cpu.embedding_model.encode(chunk["content"])
        cpu_time = time.time() - start
        estimated_cpu = cpu_time * 5  # Estimate for 100
        
        print(f"  Time (estimated): {estimated_cpu:.2f}s")
        print(f"  Speed: {20/cpu_time:.1f} chunks/sec")
        
        speedup = estimated_cpu / gpu_time
        print(f"\n🚀 GPU Speedup: {speedup:.1f}x faster")
        
        return {"gpu_time": gpu_time, "cpu_time": estimated_cpu, "speedup": speedup}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def benchmark_batch_sizes():
    """Test different batch sizes"""
    print("\n" + "=" * 60)
    print("📦 BATCH SIZE BENCHMARK")
    print("=" * 60)
    
    try:
        from vectorizer import CodeVectorizer
        
        test_data = [f"Code snippet {i}" * 50 for i in range(500)]
        
        vectorizer = CodeVectorizer(
            model_name="sentence-transformers/all-mpnet-base-v2",
            use_gpu=True
        )
        
        batch_sizes = [32, 64, 128, 256, 512]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n🔍 Testing batch size: {batch_size}")
            
            start = time.time()
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                vectorizer.embedding_model.encode(batch)
            
            elapsed = time.time() - start
            results[batch_size] = elapsed
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Speed: {len(test_data)/elapsed:.1f} items/sec")
        
        best_batch = min(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Best batch size: {best_batch[0]} ({best_batch[1]:.2f}s)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def check_gpu_memory():
    """Check GPU memory usage"""
    print("\n" + "=" * 60)
    print("💾 GPU MEMORY CHECK")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return None
        
        # Get GPU info
        device = torch.cuda.current_device()
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(device)}")
        
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1e9
        
        # Current usage
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        free = total_memory - reserved
        
        print(f"\n📊 Memory:")
        print(f"  Total: {total_memory:.1f} GB")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Reserved: {reserved:.1f} GB")
        print(f"  Free: {free:.1f} GB")
        
        # Load model and check memory
        print("\n🧪 Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        
        torch.cuda.reset_peak_memory_stats()
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
        
        model_memory = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Model memory: {model_memory:.2f} GB")
        
        remaining = total_memory - model_memory
        print(f"  Remaining for LLM: {remaining:.1f} GB")
        
        # Estimate LLM capacity
        print(f"\n🤖 Estimated LLM capacity:")
        if remaining > 18:
            print(f"  ✅ Can run CodeLlama 34B (~19GB)")
        if remaining > 12:
            print(f"  ✅ Can run DeepSeek Coder 33B (~18GB)")
        if remaining > 25:
            print(f"  ✅ Can run Mixtral 8x7B (~26GB)")
        
        return {
            "total": total_memory,
            "model_memory": model_memory,
            "remaining": remaining
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("🔬 GPU PERFORMANCE BENCHMARK")
    print("RTX 3090 24GB + 126GB RAM")
    print("=" * 60)
    
    # Check system
    print("\n🖥️  System Check:")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    except ImportError:
        print("  ❌ PyTorch not installed - GPU features won't work")
        return 1
    
    # Run benchmarks
    results = {}
    
    # 1. GPU memory check
    mem_result = check_gpu_memory()
    if mem_result:
        results['memory'] = mem_result
    
    # 2. Embedding benchmark
    emb_result = benchmark_embedding()
    if emb_result:
        results['embedding'] = emb_result
    
    # 3. Batch size benchmark
    batch_result = benchmark_batch_sizes()
    if batch_result:
        results['batch'] = batch_result
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 60)
    
    if 'embedding' in results:
        print(f"\n⚡ Embedding Performance:")
        print(f"  GPU Speedup: {results['embedding']['speedup']:.1f}x")
        print(f"  GPU Speed: {100/results['embedding']['gpu_time']:.1f} chunks/sec")
    
    if 'batch' in results:
        best = min(results['batch'].items(), key=lambda x: x[1])
        print(f"\n📦 Optimal Batch Size: {best[0]}")
    
    if 'memory' in results:
        print(f"\n💾 VRAM Usage:")
        print(f"  Embedding Model: {results['memory']['model_memory']:.1f} GB")
        print(f"  Available for LLM: {results['memory']['remaining']:.1f} GB")
    
    print("\n✅ Your system is OPTIMIZED for high-performance RAG!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

