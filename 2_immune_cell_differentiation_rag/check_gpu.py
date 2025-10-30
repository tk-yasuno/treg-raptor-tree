"""
GPU動作確認スクリプト
Immune Cell Differentiation RAPTOR Tree RAG System

GPU環境の詳細情報を確認し、最適な設定を提案します。
"""

import torch
import sys
from pathlib import Path

def check_gpu_environment():
    """GPU環境の詳細チェック"""
    
    print("🔧 GPU Environment Check")
    print("=" * 50)
    
    # Python環境
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA確認
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        # GPU詳細情報
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            
            # メモリ使用状況
            if torch.cuda.is_available():
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Cached: {cached:.2f} GB")
        
        # CUDA版本
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # 推奨設定
        print("\n🎯 Recommended Settings:")
        if gpu_memory >= 12:
            print("✅ High-end GPU detected (12GB+)")
            print("  - Use faiss-gpu for fast indexing")
            print("  - Enable large batch processing")
            print("  - Use full PubMed literature (50+ articles per query)")
        elif gpu_memory >= 8:
            print("✅ Mid-range GPU detected (8-12GB)")
            print("  - Use faiss-gpu with moderate indexing")
            print("  - Medium batch size recommended")
            print("  - Limit PubMed articles to 30 per query")
        else:
            print("⚠️ Low-memory GPU detected (<8GB)")
            print("  - Consider using CPU for large operations")
            print("  - Use small batch sizes")
            print("  - Limit PubMed articles to 20 per query")
            
    else:
        print("\n❌ No CUDA-capable GPU detected")
        print("\n🎯 CPU-only Recommendations:")
        print("  - Use faiss-cpu for vector indexing")
        print("  - Reduce batch sizes")
        print("  - Expect longer processing times")
        print("  - Consider cloud GPU services for heavy workloads")
    
    # パッケージ推奨
    print("\n📦 Package Recommendations:")
    if cuda_available and gpu_memory >= 8:
        print("  pip install faiss-gpu  # For GPU acceleration")
    else:
        print("  pip install faiss-cpu  # For CPU-only operation")
    
    # メモリテスト
    print("\n🧪 Memory Test:")
    try:
        if cuda_available:
            device = torch.device("cuda")
            # 小さなテンソルでテスト
            test_tensor = torch.randn(1000, 1000).to(device)
            print("✅ GPU memory allocation successful")
            
            # メモリクリア
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU memory cleanup successful")
        else:
            device = torch.device("cpu")
            test_tensor = torch.randn(1000, 1000).to(device)
            print("✅ CPU memory allocation successful")
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
    
    print("\n" + "=" * 50)
    print("GPU environment check completed!")
    
    return cuda_available, gpu_count if cuda_available else 0

def get_optimal_config(cuda_available: bool, gpu_memory: float = 0):
    """最適な設定を生成"""
    
    config = {
        "device": "cuda" if cuda_available else "cpu",
        "use_gpu": cuda_available,
        "batch_size": 32 if cuda_available and gpu_memory >= 8 else 16,
        "max_articles_per_query": 50 if gpu_memory >= 12 else (30 if gpu_memory >= 8 else 20),
        "faiss_backend": "faiss-gpu" if cuda_available and gpu_memory >= 8 else "faiss-cpu",
        "embedding_batch_size": 64 if cuda_available else 32
    }
    
    return config

def save_config(config: dict, output_file: str = "gpu_config.json"):
    """設定をファイルに保存"""
    
    import json
    
    config_with_metadata = {
        "gpu_config": config,
        "generated_at": str(torch.cuda.get_device_name(0)) if config["use_gpu"] else "CPU-only",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }
    
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_with_metadata, f, indent=2)
    
    print(f"🎯 Optimal configuration saved to: {output_path}")

if __name__ == "__main__":
    cuda_available, gpu_count = check_gpu_environment()
    
    # GPU memory calculation
    gpu_memory = 0
    if cuda_available and gpu_count > 0:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # 最適設定生成
    config = get_optimal_config(cuda_available, gpu_memory)
    save_config(config)
    
    print("\n🚀 Ready to run Immune Cell RAPTOR Tree RAG System!")
    
    if cuda_available:
        print("💡 Use GPU-accelerated mode for best performance")
    else:
        print("💡 Running in CPU mode - consider cloud GPU for large datasets")