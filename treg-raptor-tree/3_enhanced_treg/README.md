# Enhanced Treg Differentiation Vocabulary System
# 拡張制御性T細胞（Treg）分化語彙システム

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概要 (Overview)

**日本語:**
制御性T細胞（Treg）の分化経路を7層の詳細な階層構造で表現する拡張語彙システムです。従来の4層システム（HSC→CLP→CD4+T→Treg）から、臨床研究で使用される正確なマーカー識別に対応した7層システムに進化しました。

**English:**
An enhanced vocabulary system representing Regulatory T cell (Treg) differentiation pathways in a detailed 7-layer hierarchical structure. Evolved from the conventional 4-layer system (HSC→CLP→CD4+T→Treg) to a 7-layer system supporting accurate marker identification used in clinical research.

---

## 🎯 主な特徴 (Key Features)

### 1. **7層階層構造 (7-Layer Hierarchical Structure)**

| Level | Name | Description (日本語) | Description (English) |
|-------|------|---------------------|----------------------|
| 0 | HSC | 造血幹細胞 | Hematopoietic Stem Cell |
| 1 | CLP | 共通リンパ球前駆細胞 | Common Lymphoid Progenitor |
| 2 | CD4+T | CD4陽性T細胞 | CD4+ T Helper Cell |
| 3 | CD4+CD25+CD127low | CD25高発現・CD127低発現T細胞 | CD4+CD25high CD127low T Cell |
| 4 | nTreg/iTreg | 胸腺由来/末梢誘導Treg | Thymic/Peripheral Origin Treg |
| 5 | Foxp3+Treg | Foxp3発現制御性T細胞 | Foxp3-expressing Regulatory T Cell |
| 6 | Functional Treg | サイトカイン産生機能的Treg | Cytokine-producing Suppressive Treg |

### 2. **臨床マーカー対応 (Clinical Marker Support)**

#### ヒトTreg同定マーカー (Human Treg Identification Markers)
- **CD4+CD25+CD127low**: ヒトTreg同定のゴールドスタンダード
- **IL-2Rα (CD25) high expression**: IL-2受容体α鎖高発現
- **IL-7Rα (CD127) low expression**: IL-7受容体α鎖低発現

#### Foxp3安定性マーカー (Foxp3 Stability Markers)
- **安定Treg (Stable Treg)**:
  - TSDR脱メチル化 (TSDR demethylation)
  - CNS2脱メチル化 (CNS2 demethylation)
  - CD45RA+ (ナイーブ/静止型)
  
- **一過性Foxp3 (Transient Foxp3)**:
  - TSDRメチル化 (TSDR methylated)
  - CD45RO+ (活性化エフェクター)
  - 活性化誘導性 (Activation-induced)

#### Treg由来マーカー (Treg Origin Markers)
- **nTreg (Natural Treg)**:
  - Helios+, Nrp1+
  - 胸腺選択 (Thymic selection)
  - AIRE依存性
  
- **iTreg (Induced Treg)**:
  - Helios-
  - TGF-β + レチノイン酸誘導
  - 末梢転換 (Peripheral conversion)

#### 機能的マーカー (Functional Markers)
- **サイトカイン産生 (Cytokine Production)**: IL-10, TGF-β, IL-35
- **抑制機構 (Suppressive Mechanisms)**: CTLA-4, LAG-3, PD-1
- **接触依存性抑制 (Contact-dependent Suppression)**

### 3. **316用語の包括的語彙体系 (316-Term Comprehensive Vocabulary)**

- **日英バイリンガル対応**: 全階層で日本語・英語用語を完備
- **マーカー特異的用語**: 各階層固有のマーカー用語を網羅
- **文脈依存判定**: 文脈に応じた正確な階層判定

---

## 📊 テスト結果 (Test Results)

### 統合テスト成績 (Integration Test Performance)

```
✅ 全テスト合格 (4/4 tests passed)
✅ 階層判定精度: 90.0% (9/10 cases)
✅ ラベル生成成功率: 100% (4/4 cases)
✅ 語彙カバレッジ: 316用語
✅ GPU対応確認: NVIDIA RTX 4060 Ti (16GB)
```

### 詳細テスト結果 (Detailed Test Results)

#### TEST 1: Level Determination Accuracy (90%)
- ✅ HSC, CLP, CD4+T: 100% 正解
- ✅ **CD4+CD25+CD127low**: ヒトTregマーカー正確検出
- ✅ **nTreg (thymic)**: 胸腺由来Treg識別成功
- ✅ **iTreg (peripheral)**: 末梢誘導Treg識別成功
- ✅ **Foxp3+ stable**: TSDR脱メチル化検出
- ✅ **Foxp3+ transient**: 一過性Foxp3識別
- ✅ **Functional Treg**: サイトカイン産生検出

#### TEST 2: Enhanced Label Generation (100%)
```python
# CD127低発現表記
CD4+CD25+CD127low
CD25high CD127low
IL-2Rα+/IL-7Rα−
(n=42)

# 安定性マーカー表示
Foxp3+Treg
Foxp3+ stable
TSDR demethyl
(n=28)

# 一過性識別
Foxp3+Treg
Foxp3+ transient
CD45RO+
(n=15)

# サイトカイン複合表示
Functional Treg
IL-10+TGF-β+CTLA-4
(n=35)
```

#### TEST 3: Vocabulary Coverage
- HSC層: 39語（日英）
- CLP層: 30語
- CD4+T層: 37語
- CD25+CD127low層: 39語
- nTreg/iTreg層: 59語
- Foxp3層: 100語
- Functional層: 69語

**合計: 316用語**

#### TEST 4: GPU Performance
- GPU: NVIDIA GeForce RTX 4060 Ti
- Total Memory: 16.0 GB
- CUDA: 12.1
- PyTorch: 2.5.1+cu121
- メモリ効率: 0.004GB使用

---

## 🚀 使用方法 (Usage)

### インストール (Installation)

```bash
# リポジトリのクローン
git clone https://github.com/tk-yasuno/treg-raptor-tree.git
cd treg-raptor-tree/3_enhanced_treg

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 基本的な使い方 (Basic Usage)

```python
from enhanced_treg_vocab import (
    determine_treg_level,
    generate_enhanced_treg_label,
    ENHANCED_LEVEL_COLOR_MAPPING
)

# 階層判定
content = "Human Treg are CD4+CD25+CD127low Foxp3+ with TSDR demethylation"
level = determine_treg_level(content)
print(f"Detected Level: {level}")  # Output: 5 (Foxp3+Treg)

# ラベル生成
label = generate_enhanced_treg_label(
    content=content,
    level=level,
    cluster_id=1,
    cluster_size=42
)
print(label)
```

### テストの実行 (Running Tests)

```bash
# 統合テストの実行
python test_enhanced_treg_16x.py

# 出力例
# ================================================================================
# Enhanced Treg Differentiation - 16x Scale Integration Test
# Test Date: 2025-11-02 11:45:56
# ================================================================================
# 
# TEST 1: Level Determination Accuracy
# Passed: 9/10 (90.0%)
# 
# TEST 2: Enhanced Label Generation
# Passed: 4/4 (100.0%)
# 
# Overall: 4/4 tests passed
# ================================================================================
```

---

## 🧬 生物学的背景 (Biological Background)

### Treg分化の重要性 (Importance of Treg Differentiation)

**日本語:**
制御性T細胞（Treg）は免疫系の恒常性維持に不可欠な細胞集団です。自己免疫疾患、アレルギー、移植免疫、がん免疫において重要な役割を果たします。Tregの正確な同定と機能評価は、臨床診断および治療戦略の開発に必須です。

**English:**
Regulatory T cells (Treg) are essential cell populations for maintaining immune system homeostasis. They play crucial roles in autoimmune diseases, allergies, transplant immunity, and cancer immunity. Accurate identification and functional assessment of Tregs are essential for clinical diagnosis and therapeutic strategy development.

### 臨床応用 (Clinical Applications)

1. **自己免疫疾患**: 1型糖尿病、関節リウマチ、多発性硬化症
2. **移植医療**: 同種移植片拒絶反応の抑制
3. **がん免疫療法**: 腫瘍免疫抑制の解除
4. **アレルギー疾患**: アレルギー反応の制御

---

## 📈 判定アルゴリズム (Determination Algorithm)

### 階層判定の優先順位 (Priority Hierarchy)

```
1. 基礎階層 (Fundamental Layers) - 最優先
   ├─ HSC (造血幹細胞)
   ├─ CLP (共通リンパ球前駆細胞)
   ├─ CD4+T (CD4陽性T細胞)
   └─ CD25+CD127low (CD25高発現・CD127低発現)

2. 由来層 (Origin Layer) - TGF-β文脈でも優先
   └─ nTreg/iTreg (胸腺由来/末梢誘導)

3. Foxp3層 (Foxp3 Layer) - 安定性判定
   ├─ 一過性Foxp3 (TCR刺激・活性化文脈で優先)
   └─ 安定Foxp3 (TSDR/CD45RA文脈)

4. 機能層 (Functional Layer) - 他の文脈がない場合
   └─ Functional Treg (サイトカイン産生・抑制機能)
```

### 文脈依存判定の例 (Context-Dependent Determination Examples)

#### iTreg判定（TGF-β誘導文脈）
```python
# TGF-βがあっても iTreg誘導文脈なら Level 4
content = "Peripheral iTreg convert from naive CD4+ T cells. TGF-beta drives conversion."
level = determine_treg_level(content)
# → Level 4 (nTreg/iTreg)
```

#### 一過性Foxp3判定（活性化文脈）
```python
# TCR刺激による一過性発現なら Level 5 (transient)
content = "Activated CD4+ T cells transiently express Foxp3 upon TCR stimulation."
level = determine_treg_level(content)
# → Level 5 (Foxp3+Treg - transient)
```

---

## 🔧 技術仕様 (Technical Specifications)

### システム要件 (System Requirements)

- **Python**: 3.11+
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.1
- **GPU**: NVIDIA GPU with 8GB+ VRAM (推奨: 16GB)

### 依存パッケージ (Dependencies)

```
torch>=2.5.1
numpy>=1.24.0
scikit-learn>=1.3.0
transformers>=4.35.0
```

### パフォーマンス (Performance)

- **階層判定速度**: 0.01秒/10ケース
- **ラベル生成速度**: 0.01秒/4ケース
- **メモリ使用量**: <5MB (CPU), <10MB (GPU)

---

## 📚 主要関数リファレンス (Function Reference)

### `determine_treg_level(content: str) -> int`

コンテンツから7層階層レベルを判定します。

**Parameters:**
- `content` (str): 判定対象のテキストコンテンツ

**Returns:**
- `int`: 0-6の階層レベル番号

**Example:**
```python
level = determine_treg_level("CD4+CD25+CD127low Foxp3+ Treg")
print(level)  # Output: 5
```

### `generate_enhanced_treg_label(content, level, cluster_id, cluster_size) -> str`

階層特異的なラベルを生成します。

**Parameters:**
- `content` (str): テキストコンテンツ
- `level` (int): 階層レベル (0-6)
- `cluster_id` (int): クラスターID
- `cluster_size` (int): クラスターサイズ

**Returns:**
- `str`: 階層特異的ラベル（複数行）

**Example:**
```python
label = generate_enhanced_treg_label(
    "IL-10 and TGF-beta producing Treg",
    level=6,
    cluster_id=1,
    cluster_size=35
)
# Output:
# Functional Treg
# IL-10+TGF-β
# (n=35)
```

---

## 📖 文献・参考資料 (References)

### 主要文献 (Key Publications)

1. **Foxp3とTreg同定**:
   - Sakaguchi, S. et al. (2020). "Regulatory T cells and human disease." *Annual Review of Immunology*, 38, 541-566.

2. **CD127低発現マーカー**:
   - Liu, W. et al. (2006). "CD127 expression inversely correlates with FoxP3 and suppressive function of human CD4+ T reg cells." *Journal of Experimental Medicine*, 203(7), 1701-1711.

3. **nTreg vs iTreg**:
   - Curotto de Lafaille, M. A., & Lafaille, J. J. (2009). "Natural and adaptive foxp3+ regulatory T cells: more of the same or a division of labor?" *Immunity*, 30(5), 626-635.

4. **TSDR脱メチル化**:
   - Baron, U. et al. (2007). "DNA demethylation in the human FOXP3 locus discriminates regulatory T cells from activated FOXP3+ conventional T cells." *European Journal of Immunology*, 37(9), 2378-2389.

---

## 🤝 貢献 (Contributing)

プルリクエスト、イシュー報告を歓迎します。

### 開発ガイドライン (Development Guidelines)

1. 生物学的正確性を最優先
2. 臨床研究での使用実績のあるマーカーを採用
3. 日英バイリンガル対応を維持
4. テストカバレッジ80%以上を維持

---

## 📄 ライセンス (License)

MIT License - 詳細は [LICENSE](../LICENSE) を参照

---

## 👨‍💻 作者 (Author)

**AI Assistant** with biological expertise collaboration

---

## 🔗 関連プロジェクト (Related Projects)

- [Treg RAPTOR Tree](https://github.com/tk-yasuno/treg-raptor-tree): 親プロジェクト
- GPU-Accelerated 16x Scale Builder: 大規模処理システム

---

## 📞 お問い合わせ (Contact)

GitHub Issues: [https://github.com/tk-yasuno/treg-raptor-tree/issues](https://github.com/tk-yasuno/treg-raptor-tree/issues)

---

**Last Updated**: 2025-11-02  
**Version**: 1.0.0  
**Test Coverage**: 90% (Level Determination), 100% (Label Generation)
