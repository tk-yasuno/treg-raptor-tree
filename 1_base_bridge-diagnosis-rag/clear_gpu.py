import torch
import gc

print("🧹 GPUメモリをクリア中...")

# GPUキャッシュをクリア
if torch.cuda.is_available():
    print(f"クリア前: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    print(f"         {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    # 全てのGPUキャッシュをクリア
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    
    print(f"クリア後: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    print(f"         {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    print("✅ GPUメモリクリア完了")
else:
    print("❌ CUDAが利用できません")
