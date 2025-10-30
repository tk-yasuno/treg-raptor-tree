import torch

print("🚀 GPU Test for 16x Scale RAPTOR")
print("=" * 40)
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"Device object: {torch.device('cuda')}")
    
    # 簡単なGPUテスト
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.mm(x, y)
    print(f"GPU tensor operation test: {z.shape}")
    print("✅ GPU is working correctly!")
else:
    print("❌ No GPU available - using CPU")