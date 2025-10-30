# セットアップガイド

## 📋 前提条件

- Python 3.10以上
- CUDA対応GPU（推奨：8GB以上のVRAM）
- Ollama（LLM用）

---

## 🔧 ステップ1: Python環境のセットアップ

### Option A: Pipenv（推奨）

```bash
cd C:\Users\yasun\LangChain\learning-langchain\multimd-raptor-colvbert-blip2

# 仮想環境作成とパッケージインストール
pipenv install

# 仮想環境をアクティベート
pipenv shell
```

### Option B: venv + pip

```bash
cd C:\Users\yasun\LangChain\learning-langchain\multimd-raptor-colvbert-blip2

# 仮想環境作成
python -m venv venv

# アクティベート（Windows）
.\venv\Scripts\Activate.ps1

# パッケージインストール
pip install -r requirements.txt
```

---

## 🤖 ステップ2: Ollamaのセットアップ

### 2.1 Ollamaのインストール

```bash
# Ollamaをダウンロード・インストール
# https://ollama.ai/

# インストール確認
ollama --version
```

### 2.2 LLMモデルのダウンロード

```bash
# gpt-oss:20b モデルをダウンロード（約12GB）
ollama pull gpt-oss:20b

# モデル確認
ollama list
```

### 2.3 Ollamaサーバーの起動

```bash
# バックグラウンドで起動
ollama serve

# または、別のターミナルウィンドウで起動
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve"
```

### 2.4 動作確認

```bash
# テストリクエスト
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Hello",
  "stream": false
}'
```

---

## 🎨 ステップ3: MeCabのインストール（日本語形態素解析）

### Windows

```bash
# MeCabインストーラーをダウンロード
# https://github.com/ikegami-yukino/mecab/releases

# インストール後、環境変数を設定
$env:MECAB_PATH = "C:\Program Files (x86)\MeCab\bin"
```

### 確認

```python
import MeCab
tagger = MeCab.Tagger()
print(tagger.parse("橋梁診断"))
# 出力: 橋梁	名詞,一般,*,*,*,*,橋梁,キョウリョウ,キョーリョー
```

---

## 🔍 ステップ4: 動作確認

### 4.1 データの確認

```bash
# 画像ファイルの確認
ls data\doken_bridge_diagnosis_logic\images | Measure-Object
# 期待値: 2400+ファイル

# PDFファイルの確認
ls data\doken_bridge_diagnosis_logic\*.pdf | Measure-Object
# 期待値: 46ファイル

# テキストキャッシュの確認
Test-Path data\doken_bridge_diagnosis_logic\pdf_text_cache.json
# 期待値: True
```

### 4.2 スクリプトの実行テスト

```bash
# 語彙辞書のテスト
python -c "from bridge_diagnosis_vocab import BRIDGE_TRANSLATION_DICT; print(len(BRIDGE_TRANSLATION_DICT), '語彙登録済み')"

# インポートテスト
python -c "from visual_raptor_colbert_bridge import BridgeRAPTORColBERT; print('✅ Import successful')"
```

---

## 🚀 ステップ5: 初回実行

### 5.1 小規模テスト（250チャンク）

```python
# scaling_test_raptor.py を編集
sample_sizes = [250]  # 250チャンクでテスト
```

```bash
# 実行（約5-10分）
python scaling_test_raptor.py
```

### 5.2 結果の確認

```bash
# 生成ファイルの確認
ls data\doken_bridge_diagnosis_logic\results\

# 期待される出力:
# - scaling_test_tree_250chunks_*.pkl
# - scaling_test_log_250chunks_*.txt
# - scaling_test_*.json
# - scaling_test_graph_*.png
```

### 5.3 可視化テスト

```bash
# ツリーファイルのパスを確認
$treefile = (Get-ChildItem data\doken_bridge_diagnosis_logic\results\scaling_test_tree_250chunks_*.pkl | Select-Object -First 1).FullName

# 可視化実行
python visualize_bridge_tree.py --tree_file $treefile --max_depth 3
```

---

## 🎯 ステップ6: 本番実行（1600チャンク）

```python
# scaling_test_raptor.py を編集
sample_sizes = [1600]  # 1600チャンク
```

```bash
# バックグラウンドで実行（60-90分）
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; python scaling_test_raptor.py 2>&1 | Tee-Object -FilePath 'build_1600chunks.log'" `
  -WindowStyle Minimized

# 進捗確認（別ターミナル）
Get-Content build_1600chunks.log -Tail 30 -Wait
```

---

## ⚠️ トラブルシューティング

### GPU メモリ不足

```python
# scaling_test_raptor.py でバッチサイズを削減
encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}  # 64 → 32
```

### Ollama接続エラー

```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/ps

# 再起動
taskkill /F /IM ollama.exe
ollama serve
```

### MeCab エラー

```bash
# 環境変数を確認
echo $env:MECAB_PATH

# パスを設定
$env:MECAB_PATH = "C:\Program Files (x86)\MeCab\bin"
```

### 日本語出力の文字化け

```python
# スクリプト冒頭に追加
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

---

## 📞 サポート

問題が解決しない場合は、以下を確認してください：

1. **README_Bridge.md**: 詳細な技術仕様
2. **Bridge_Practice.md**: 実装の教訓とベストプラクティス
3. ログファイル: `data/doken_bridge_diagnosis_logic/results/scaling_test_log_*.txt`

---

**セットアップ完了！🎉**
