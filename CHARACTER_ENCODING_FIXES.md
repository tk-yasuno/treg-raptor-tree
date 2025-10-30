# 🔧 Character Encoding Fixes for RAPTOR Tree Visualization

## 📋 問題の特定

1. **文字化け症状**: ノードラベルに「CLP□□□□□□□□□□(2)」や「CD4□□T□□(2)」のような placeholder文字（□）が表示
2. **フォント問題**: 一部の特殊文字やUnicode文字がWindowsのフォントシステムで正しく表示されない
3. **エンコーディング混在**: 日本語、英語、特殊記号が混在し、レンダリング時に問題発生

## 🎯 実装された解決策

### 1. ASCII優先ラベル生成 (`immune_cell_vocab.py`)

```python
# 完全ASCII文字のみでシンプルなラベル生成
def generate_immune_label(content, level, cluster_id, cluster_size):
    """免疫学的に意味のあるラベルを生成（文字化け対策強化版）"""
    
    if level == 1:  # CLP
        clp_terms = ['CLP', 'IL-7', 'lymphoid', 'progenitor']
        # 英語キーワードを優先使用
        
    elif level == 2:  # CD4+T
        cd4_terms = ['CD4', 'TCR', 'helper', 'T cell']
        # CD4関連の安全な英語用語のみ
        
    elif level == 3:  # Treg
        treg_terms = ['Foxp3', 'TGF-b', 'IL-10', 'CTLA-4', 'Treg']
        # 特殊文字を安全な文字に置換（β → b）
```

### 2. 強化された文字清浄化 (`visualize_raptor_tree.py`)

```python
def clean_label_text(text):
    """ラベルテキストの安全な清浄化"""
    
    # 特定の問題文字を除去・置換
    replacements = {
        '□': '',        # placeholder文字を完全除去
        '？': '?',      # 全角疑問符を半角に
        '　': ' ',      # 全角スペースを半角に
        'β': 'b',       # ベータを安全な文字に
        '－': '-',      # 全角ハイフンを半角に
    }
    
    # 最終的にASCII範囲のみに制限
    for char in text:
        if ord(char) < 128:  # ASCII範囲
            cleaned += char
        elif char == '\n':
            cleaned += char
        else:
            cleaned += '_'  # 非ASCII文字は_に置換
```

### 3. モノスペースフォント使用

```python
# 複数のフォールバックオプション
font_options = [
    {'family': 'monospace', 'size': 8, 'weight': 'normal'},
    {'family': 'DejaVu Sans Mono', 'size': 8, 'weight': 'normal'},
    {'family': 'Courier New', 'size': 8, 'weight': 'normal'},
    {'family': 'Arial', 'size': 8, 'weight': 'normal'},
]
```

## ✅ 修正効果

### Before (修正前)
```
CLP□□□□□□□□□□(2)
CD4□□T□□(2)
Treg□□□□□□(1)
```

### After (修正後)
```
CLP
IL-7
(2)

CD4+T
TCR
(3)

Treg
Foxp3
(1)
```

## 🔍 技術詳細

### 1. 文字コード制限
- **ASCII範囲**: 0-127の文字のみ許可
- **改行文字**: \nは例外的に許可
- **フォールバック**: 非ASCII文字は「_」に置換

### 2. 用語標準化
- **英語優先**: 国際的な免疫学用語を使用
- **略語使用**: 長い用語は標準的な略語に変換
- **特殊文字回避**: β、－等の特殊文字を英語表記に変換

### 3. レンダリング最適化
- **モノスペースフォント**: 文字幅の一定性を確保
- **フォントフォールバック**: 複数のフォントオプションで互換性向上
- **エラーハンドリング**: フォント読み込み失敗時の安全な代替処理

## 📊 結果検証

- **文字化け解消**: □文字の完全除去
- **表示安定性**: 全ノードで一貫した文字表示
- **免疫学的正確性**: 専門用語の適切な英語表記
- **レイアウト改善**: モノスペースフォントによる整列改善

## 🚀 今後の改善点

1. **より多くのフォント対応**: システムフォントの自動検出
2. **動的サイズ調整**: ラベル長に応じたフォントサイズ最適化
3. **カラー最適化**: 背景色とのコントラスト改善
4. **多言語対応**: 安全な多言語表示システムの構築

---
*修正完了日: 2025年10月31日 02:16*
*対象ファイル: immune_cell_vocab.py, visualize_raptor_tree.py*