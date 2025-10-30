# 🚀 Quick Start Guide

## 5分でRAPTORを始める

### ステップ1: 環境確認

```bash
# NVIDIA GPUの確認
nvidia-smi

# Python バージョン確認 (3.11+ 推奨)
python --version
```

### ステップ2: セットアップ

```bash
# リポジトリクローン
git clone https://github.com/langchain-ai/learning-langchain.git
cd learning-langchain/treg-raptor-tree

# 仮想環境作成
python -m venv raptor_env
source raptor_env/bin/activate  # Linux/Mac
# または
raptor_env\Scripts\activate     # Windows

# 依存関係インストール
pip install -r requirements.txt
```

### ステップ3: GPU対応PyTorchインストール

```bash
# CUDA対応版インストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### ステップ4: 実行

```bash
# RAPTORツリー構築
python true_raptor_builder.py

# 結果確認
python analyze_clustered_tree.py

# 可視化生成
python visualize_raptor_tree.py

# 可視化表示
python show_raptor_viz.py
```

## 期待される結果

### コンソール出力例
```
🚀 GPU detected: NVIDIA GeForce RTX 4060 Ti (16.0GB)
🔥 Using OPT-2.7B for GPU with 16GB+ memory
📊 Processing level 0: 35 nodes
📊 Processing level 1: 10 nodes
📊 Processing level 2: 2 nodes
🌟 Root node created

📊 Results Summary:
   Generated nodes: 14
   Tree levels: 4
   Improvement: +180%
```

### 生成ファイル
- `raptor_tree_visualization_*.png` - ツリー構造図
- `raptor_statistics_*.png` - 統計分析
- `data/immune_cell_differentiation/raptor_trees/*.json` - ツリーデータ

## トラブルシューティング

### 🔧 GPU認識されない
```bash
# PyTorchのCUDA確認
python -c "import torch; print(torch.cuda.is_available())"

# 再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 🔧 メモリ不足
- GPUメモリが不足する場合、より小さなモデルが自動選択されます
- 必要に応じて `config.yaml` でバッチサイズを調整

### 🔧 依存関係エラー
```bash
# 全て再インストール
pip install --upgrade --force-reinstall -r requirements.txt
```

## 🎯 成功の確認

✅ **14ノード以上生成**  
✅ **4階層達成**  
✅ **180%以上改善**  
✅ **可視化ファイル生成**  

## 次のステップ

1. **カスタムデータ**: 独自の文書セットで実験
2. **パラメータ調整**: `config.yaml` で設定カスタマイズ
3. **拡張機能**: `DEVELOPER_GUIDE.md` で高度な機能を学習

## サポート

- 📖 詳細: `README.md`
- 🔬 開発者向け: `DEVELOPER_GUIDE.md`
- ❓ 問題: GitHub Issues で質問