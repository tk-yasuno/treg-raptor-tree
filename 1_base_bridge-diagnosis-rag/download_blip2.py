"""
BLIP2-OPT-2.7B モデルを hf_transfer で高速ダウンロード
"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import snapshot_download
import torch

print("🚀 hf_transfer による高速ダウンロードを開始...")
print(f"   CUDA available: {torch.cuda.is_available()}")

model_id = "Salesforce/blip2-opt-2.7b"

try:
    cache_dir = snapshot_download(
        repo_id=model_id,
        resume_download=True,
        local_files_only=False,
    )
    print(f"\n✅ ダウンロード完了!")
    print(f"   キャッシュ先: {cache_dir}")
except Exception as e:
    print(f"\n❌ エラー: {e}")
    raise
