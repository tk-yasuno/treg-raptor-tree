# 橋梁診断 Multimodal RAPTOR + ColVBERT + BLIP

## 📋 プロジェクト概要

このワークスペースは、**橋梁診断ロジック**を対象としたマルチモーダルRAG（Retrieval-Augmented Generation）システムの構築に特化しています。

### 主な機能

- **マルチモーダルRAPTORツリー構築**: テキスト+画像情報を統合した階層的知識表現
- **ColVBERT + BLIP エンコーダ**: 高精度な画像-テキスト統合埋め込み
- **橋梁診断特化語彙**: 200+専門用語の日英翻訳辞書
- **2つの可視化レイアウト**: 階層型（ピラミッド）と円形（同心円）
- **スケーラビリティテスト**: 250〜1600チャンクでの性能評価

---

## 🗂️ ディレクトリ構成

```
multimd-raptor-colvbert-blip2/
├── visual_raptor_colbert_bridge.py    # 橋梁診断用RAPTORシステム
├── bridge_diagnosis_vocab.py          # 橋梁診断専門用語辞書
├── visualize_bridge_tree.py           # ツリー可視化（階層型・円形）
├── scaling_test_raptor.py             # スケーリングテスト
├── README_Bridge.md                   # 詳細ドキュメント
├── Bridge_Practice.md                 # 実装教訓・ベストプラクティス
├── Pipfile                            # Python依存関係
├── requirements.txt                   # pip依存関係
│
├── data/
│   └── doken_bridge_diagnosis_logic/
│       ├── images/                    # 診断セットPDF画像（2400+枚）
│       ├── results/                   # RAPTORツリー・可視化結果
│       ├── pdf_text_cache.json        # OCRテキストキャッシュ
│       └── *.pdf                      # 診断セットPDF（46ファイル）
│
└── 0_base_tsunami-lesson-rag/         # 基本クラス定義
    ├── tsunami_lesson_raptor.py       # RAPTOR基本実装
    └── raptor_eval.py                 # 評価ユーティリティ
```

---

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 仮想環境作成（推奨）
pipenv install

# または pip を使用
pip install -r requirements.txt
```

**主要な依存パッケージ**:
- `langchain-huggingface`: 埋め込みモデル
- `langchain-ollama`: LLM（gpt-oss:20b）
- `transformers`: BLIP画像キャプショニング
- `scikit-learn`: クラスタリング
- `networkx`: グラフ構造
- `matplotlib`: 可視化
- `mecab-python3`: 日本語形態素解析

### 2. RAPTORツリー構築

```bash
# 1600チャンク版を構築（約60-90分）
python scaling_test_raptor.py
```

**生成されるファイル**:
- `data/doken_bridge_diagnosis_logic/results/scaling_test_tree_1600chunks_*.pkl`
- `data/doken_bridge_diagnosis_logic/results/scaling_test_log_1600chunks_*.txt`

### 3. ツリー可視化

```bash
# 階層型 + 円形レイアウトを自動生成
python visualize_bridge_tree.py \
  --tree_file data/doken_bridge_diagnosis_logic/results/scaling_test_tree_1600chunks_*.pkl \
  --max_depth 3
```

**出力ファイル**:
- `*_visualization_ja.png` (階層型レイアウト)
- `*_visualization_circular_ja.png` (円形レイアウト)

---

## 📊 実行済みテスト結果

| チャンク数 | 構築時間 | ツリー深度 | 総ノード数 | GPU使用量 |
|----------|---------|----------|----------|----------|
| 250      | 5.2分   | 3        | 85       | 3.2GB    |
| 600      | 12.8分  | 4        | 213      | 4.1GB    |
| 1200     | 28.5分  | 4        | 271      | 5.8GB    |
| 1600     | 実行中  | 4        | -        | -        |

---

## 🎯 主要な設定パラメータ

### `scaling_test_raptor.py`

```python
# チャンク数設定
sample_sizes = [1600]  # 現在: 1600チャンク

# RAPTOR設定
max_depth=4              # ツリー深度（部材→損傷→原因→補修）
selection_strategy='combined'  # クラスタリング戦略
metric_weights={
    'silhouette': 0.5,   # クラスタ品質
    'dbi': 0.5,          # 分離度
    'chi': 0.0           # CHIスコア（無効化）
}
```

### `visualize_bridge_tree.py`

```python
# 可視化パラメータ
max_depth=3              # 表示深度（0-3）
node_size=1500           # ノードサイズ
font_size=9              # フォントサイズ
horizontal_spacing=5.0   # 水平間隔
vertical_spacing=4.0     # 垂直間隔
```

---

## 📚 詳細ドキュメント

- **README_Bridge.md**: 詳細な技術仕様、アーキテクチャ、使用方法
- **Bridge_Practice.md**: 実装の教訓、ベストプラクティス、トラブルシューティング

---

## 🔧 カスタマイズ

### 語彙辞書の拡張

`bridge_diagnosis_vocab.py`を編集して専門用語を追加：

```python
BRIDGE_TRANSLATION_DICT = {
    '新しい用語': 'New Term',
    # ...
}
```

### ツリー深度の調整

```python
# scaling_test_raptor.py
max_depth=5  # より深い階層構造
```

### 可視化レイアウトの調整

```python
# visualize_bridge_tree.py
horizontal_spacing=6.0  # ノード間隔を広げる
figsize=[50, 40]        # キャンバスサイズを拡大
```

---

## ⚠️ 注意事項

1. **GPU推奨**: CUDA対応GPUがあると構築速度が大幅に向上
2. **メモリ要件**: 1600チャンク版では6GB以上のGPUメモリを推奨
3. **実行時間**: チャンク数に応じて数分〜数時間かかります
4. **Ollama起動**: LLMとして`gpt-oss:20b`が必要（事前に`ollama pull gpt-oss:20b`）

---

## 📝 ライセンス

このプロジェクトは研究・教育目的で作成されています。

---

## 🙏 謝辞

- **RAPTOR**: 階層的知識表現フレームワーク
- **ColVBERT**: マルチモーダル埋め込み
- **BLIP**: 画像キャプショニング
- **LangChain**: RAGフレームワーク

---

**最終更新**: 2025年10月29日
