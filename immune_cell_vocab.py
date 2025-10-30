"""
免疫細胞分化ドメイン専門語彙
Immune Cell Differentiation Domain Vocabulary

免疫細胞分化経路（HSC → CLP → CD4+ T → Treg）に特化した専門用語辞書
4階層構造に対応したラベリングシステム
"""

# ストップワード（ノードラベルから除外する語）
IMMUNE_STOP_WORDS = {
    # 一般的なストップワード
    '本研究', '本論文', '我々', '研究', '実験', '結果', '考察', '解析',
    '報告', '検討', '調査', '観察', '測定', '評価', '分析',
    '図', '表', '写真', 'データ', 'グラフ', 'チャート',
    'ページ', '参照', '以下', '上記', '前述', '後述',
    '場合', '状態', '程度', '部位', '領域', '全体',
    'こと', 'もの', 'ため', 'など', 'による', 'について', 'において',
    '細胞', '免疫', '分化',  # 頻出するため除外
    
    # 英語ストップワード
    'cells', 'cell', 'immune', 'immunology', 'differentiation',
    'study', 'research', 'analysis', 'result', 'conclusion',
    'figure', 'table', 'data', 'experiment', 'method',
}

# 4階層免疫細胞分化語彙体系
IMMUNE_HIERARCHY_VOCAB = {
    
    # Level 0: HSC (造血幹細胞) レベル
    "hsc_level": {
        "japanese": {
            # 幹細胞関連
            '造血幹細胞', 'HSC', '幹細胞', '多能性幹細胞', '全能性幹細胞',
            '長期再構築型HSC', '短期再構築型HSC', 'LT-HSC', 'ST-HSC',
            '多分化能', '自己複製', '再生能', '造血系統',
            
            # 骨髄微小環境
            '骨髄', '骨髄微小環境', 'ニッチ', '骨髄ニッチ', '造血ニッチ',
            '間質細胞', '骨芽細胞', '血管内皮細胞', 'CXCL12', 'SCF',
            
            # マーカー
            'Lin-', 'Sca-1+', 'c-Kit+', 'LSK', 'CD34-', 'CD150+', 'CD48-',
            'SLAM', 'Flk2-', 'CD135-', 'Thy1.1low',
            
            # 増殖・分化制御
            'Notch', 'Wnt', 'BMP', 'TGF-β', 'TPO', 'FL', 'IL-3', 'IL-6',
            '転写因子', 'RUNX1', 'GATA2', 'SCL', 'PU.1',
        },
        "english": {
            'hematopoietic stem cell', 'HSC', 'stem cell', 'multipotent',
            'long-term HSC', 'short-term HSC', 'LT-HSC', 'ST-HSC',
            'self-renewal', 'bone marrow', 'niche', 'hematopoietic niche',
            'stromal cell', 'osteoblast', 'endothelial cell',
            'CXCL12', 'SCF', 'stem cell factor', 'lineage negative',
            'Sca-1 positive', 'c-Kit positive', 'LSK',
        }
    },
    
    # Level 1: CLP (共通リンパ球前駆細胞) レベル  
    "clp_level": {
        "japanese": {
            # リンパ球前駆細胞
            '共通リンパ球前駆細胞', 'CLP', 'リンパ球前駆細胞', '前駆細胞',
            'リンパ系統', 'リンパ球系統決定', 'リンパ球分化',
            
            # マーカー
            'Lin-', 'Sca-1low', 'c-Kitlow', 'IL-7Rα+', 'CD127+', 'Flk2+', 'CD135+',
            'Flt3+', 'CD25-', 'CD44+', 'CD93+',
            
            # 転写因子・分化制御
            'Ikaros', 'E2A', 'EBF1', 'PAX5', 'Notch1', 'GATA3',
            'IL-7', 'IL-7シグナル', 'STAT5', 'JAK3',
            'Bcl-2', 'Bcl-xL', 'アポトーシス制御',
            
            # 分化経路
            'B細胞前駆細胞', 'T細胞前駆細胞', 'NK細胞前駆細胞',
            'ILC前駆細胞', 'プロB細胞', 'プロT細胞',
        },
        "english": {
            'common lymphoid progenitor', 'CLP', 'lymphoid progenitor',
            'lymphoid lineage', 'lymphoid commitment', 'lymphocyte differentiation',
            'IL-7 receptor alpha', 'CD127', 'Flt3', 'CD135', 'Flk2',
            'Ikaros', 'E2A', 'EBF1', 'PAX5', 'Notch1', 'GATA3',
            'interleukin-7', 'STAT5', 'JAK3', 'B cell progenitor',
            'T cell progenitor', 'NK cell progenitor', 'ILC progenitor',
        }
    },
    
    # Level 2: CD4+ T細胞レベル
    "cd4_t_level": {
        "japanese": {
            # CD4+ T細胞基本
            'CD4陽性T細胞', 'CD4+ T細胞', 'ヘルパーT細胞', 'Th細胞',
            'ナイーブT細胞', 'ナイーブCD4+ T細胞', '初期T細胞',
            
            # T細胞受容体・シグナル
            'TCR', 'T細胞受容体', 'TCRαβ', 'CD3', 'CD28', 'CD40L',
            'MHCクラスII', 'HLA-DR', 'ペプチド-MHC複合体',
            'シナプス', '免疫シナプス', 'TCRシグナル',
            
            # 胸腺での分化
            '胸腺', 'DP', 'CD4+CD8+', 'SP', 'CD4+CD8-',
            '正の選択', '負の選択', '胸腺上皮細胞', 'Aire',
            
            # 活性化・増殖
            'T細胞活性化', '活性化T細胞', 'CD25', 'CD69', 'CD44',
            'IL-2', 'IL-2受容体', 'CD25', 'IL-2R', 'STAT5',
            'mTOR', 'PI3K', 'Akt', 'NFATc1', 'AP-1', 'NF-κB',
            
            # Th細胞サブセット
            'Th1', 'Th2', 'Th17', 'Tfh', 'Th9', 'Th22',
            'IFN-γ', 'IL-4', 'IL-17', 'IL-21', 'IL-9', 'IL-22',
            'T-bet', 'GATA3', 'RORγt', 'Bcl6', 'PU.1', 'AHR',
        },
        "english": {
            'CD4 positive T cell', 'CD4+ T cell', 'helper T cell', 'Th cell',
            'naive T cell', 'naive CD4+ T cell', 'T cell receptor', 'TCR',
            'MHC class II', 'HLA-DR', 'peptide-MHC complex',
            'immunological synapse', 'thymus', 'double positive', 'DP',
            'single positive', 'SP', 'positive selection', 'negative selection',
            'T cell activation', 'activated T cell', 'interleukin-2', 'IL-2',
            'IL-2 receptor', 'CD25', 'STAT5', 'mTOR', 'PI3K', 'Akt',
            'Th1', 'Th2', 'Th17', 'Tfh', 'T-bet', 'GATA3', 'RORγt', 'Bcl6',
            'interferon-gamma', 'interleukin-4', 'interleukin-17',
        }
    },
    
    # Level 3: Treg (制御性T細胞) レベル
    "treg_level": {
        "japanese": {
            # 制御性T細胞基本
            '制御性T細胞', 'Treg', '調節性T細胞', 'レギュラトリーT細胞',
            '免疫抑制', '免疫制御', '免疫寛容', '自己免疫抑制',
            
            # マーカー
            'Foxp3', 'CD25', 'CTLA-4', 'CD152', 'GITR', 'CD103', 'αEβ7',
            'LAG-3', 'TIM-3', 'PD-1', 'CD39', 'CD73', 'Helios', 'Neuropilin-1',
            
            # 分化・発生
            'Foxp3誘導', 'iTreg', '誘導性Treg', 'tTreg', '胸腺由来Treg',
            'nTreg', '自然発生Treg', 'eTreg', '末梢誘導Treg',
            
            # 転写制御
            'Foxp3', 'STAT5', 'Smad3', 'NFAT', 'Runx1', 'Eos', 'IRF4',
            'Blimp-1', 'Bcl-xL', 'c-Rel', 'Satb1',
            
            # サイトカイン・増殖因子
            'TGF-β', 'IL-10', 'IL-35', 'IL-2', 'レチノイン酸', 'RA',
            'ビタミンD3', 'VDR', 'AhR', 'リガンド',
            
            # 抑制機構
            '接触依存性抑制', 'グランザイム', 'パーフォリン', 'Galectin-1',
            'cAMP', 'CD39/CD73', 'アデノシン', 'IDO', 'トリプトファン',
            '代謝制御', 'IL-2消費', 'IL-2競合',
            
            # 病態・疾患
            '自己免疫疾患', 'アレルギー', '移植免疫', '腫瘍免疫',
            'GVHD', '移植片対宿主病', '炎症性腸疾患', 'IBD',
            '関節リウマチ', 'RA', '多発性硬化症', 'MS',
            
            # 機能
            '免疫応答抑制', 'エフェクター機能抑制', '増殖抑制',
            'サイトカイン産生抑制', 'APC機能阻害', '樹状細胞制御',
        },
        "english": {
            'regulatory T cell', 'Treg', 'immune suppression', 'immune regulation',
            'immune tolerance', 'autoimmune suppression', 'Foxp3',
            'forkhead box P3', 'CD25', 'CTLA-4', 'CD152', 'GITR', 'CD103',
            'LAG-3', 'TIM-3', 'PD-1', 'CD39', 'CD73', 'Helios', 'Neuropilin-1',
            'induced Treg', 'iTreg', 'thymic Treg', 'tTreg', 'natural Treg', 'nTreg',
            'peripherally induced Treg', 'eTreg', 'TGF-beta', 'transforming growth factor beta',
            'interleukin-10', 'IL-10', 'interleukin-35', 'IL-35',
            'retinoic acid', 'vitamin D3', 'contact-dependent suppression',
            'granzyme', 'perforin', 'Galectin-1', 'cyclic AMP', 'adenosine',
            'indoleamine 2,3-dioxygenase', 'IDO', 'tryptophan', 'metabolic control',
            'autoimmune disease', 'allergy', 'transplant immunity', 'tumor immunity',
            'graft-versus-host disease', 'GVHD', 'inflammatory bowel disease', 'IBD',
        }
    }
}

# 階層別色分けマッピング
LEVEL_COLOR_MAPPING = {
    0: {"color": "#FF6B6B", "name": "HSC", "description": "造血幹細胞"},
    1: {"color": "#4ECDC4", "name": "CLP", "description": "共通リンパ球前駆細胞"},
    2: {"color": "#45B7D1", "name": "CD4+T", "description": "CD4陽性T細胞"},
    3: {"color": "#96CEB4", "name": "Treg", "description": "制御性T細胞"},
    4: {"color": "#FFEAA7", "name": "ROOT", "description": "ルートノード"}
}

# レベル別専門用語抽出
def extract_level_keywords(content, level):
    """指定レベルに特化したキーワード抽出"""
    if level == 0:
        vocab_set = IMMUNE_HIERARCHY_VOCAB["hsc_level"]
    elif level == 1:
        vocab_set = IMMUNE_HIERARCHY_VOCAB["clp_level"]
    elif level == 2:
        vocab_set = IMMUNE_HIERARCHY_VOCAB["cd4_t_level"]
    elif level == 3:
        vocab_set = IMMUNE_HIERARCHY_VOCAB["treg_level"]
    else:
        return []
    
    # 日本語・英語キーワードを統合
    all_keywords = set()
    all_keywords.update(vocab_set["japanese"])
    all_keywords.update(vocab_set["english"])
    
    # コンテンツ内のキーワード検出
    found_keywords = []
    content_lower = content.lower()
    
    for keyword in all_keywords:
        if keyword.lower() in content_lower:
            found_keywords.append(keyword)
    
    return found_keywords

# 免疫学的に適切なラベル生成
def generate_immune_label(content, level, cluster_id, cluster_size):
    """免疫学的に意味のあるラベルを生成（文字化け対策強化版）"""
    
    # レベル別キーワード抽出
    keywords = extract_level_keywords(content, level)
    
    if level == 4:  # ルートノード
        return f"ROOT\n({cluster_size})"
    
    # 完全ASCII文字のみでシンプルなラベル生成
    if level == 1:  # CLP
        # CLPレベルでよく見つかるキーワード
        clp_terms = ['CLP', 'IL-7', 'lymphoid', 'progenitor']
        found_term = None
        for term in clp_terms:
            if any(term.lower() in k.lower() for k in keywords):
                found_term = term
                break
        
        if found_term:
            label = f"CLP\n{found_term}\n({cluster_size})"
        else:
            label = f"CLP\nLevel1\n({cluster_size})"
            
    elif level == 2:  # CD4+T
        # CD4 Tレベルでよく見つかるキーワード
        cd4_terms = ['CD4', 'TCR', 'helper', 'T cell']
        found_term = None
        for term in cd4_terms:
            if any(term.lower() in k.lower() for k in keywords):
                found_term = term
                break
                
        if found_term:
            label = f"CD4+T\n{found_term}\n({cluster_size})"
        else:
            label = f"CD4+T\nLevel2\n({cluster_size})"
            
    elif level == 3:  # Treg
        # Tregレベルでよく見つかるキーワード
        treg_terms = ['Foxp3', 'TGF-b', 'IL-10', 'CTLA-4', 'Treg']
        found_term = None
        for term in treg_terms:
            if any(term.lower() in k.lower() for k in keywords):
                found_term = term
                break
                
        if found_term:
            # 特殊文字を安全な文字に置換
            safe_term = found_term.replace('β', 'b').replace('-', '_')
            label = f"Treg\n{safe_term}\n({cluster_size})"
        else:
            label = f"Treg\nLevel3\n({cluster_size})"
            
    else:
        # その他のレベル
        label = f"L{level}\nC{cluster_id}\n({cluster_size})"
    
    return label

# 階層別専門用語辞書
IMMUNE_TERM_HIERARCHY = {
    "level_0_hsc": {
        "primary": ["造血幹細胞", "HSC", "骨髄", "幹細胞"],
        "secondary": ["多分化能", "自己複製", "ニッチ", "LSK"],
        "markers": ["Lin-", "Sca-1+", "c-Kit+", "CD150+", "CD48-"],
        "factors": ["SCF", "TPO", "CXCL12", "Notch", "Wnt"]
    },
    "level_1_clp": {
        "primary": ["共通リンパ球前駆細胞", "CLP", "リンパ球前駆細胞"],
        "secondary": ["リンパ系統決定", "リンパ球分化"],
        "markers": ["IL-7Rα+", "CD127+", "Flk2+", "CD135+"],
        "factors": ["IL-7", "Ikaros", "E2A", "EBF1", "PAX5"]
    },
    "level_2_cd4": {
        "primary": ["CD4+ T細胞", "ヘルパーT細胞", "Th細胞"],
        "secondary": ["T細胞活性化", "MHCクラスII", "TCR"],
        "markers": ["CD4+", "CD25+", "CD44+", "CD69+"],
        "factors": ["IL-2", "TCR", "CD28", "STAT5", "mTOR"]
    },
    "level_3_treg": {
        "primary": ["制御性T細胞", "Treg", "免疫制御"],
        "secondary": ["免疫抑制", "免疫寛容", "自己免疫"],
        "markers": ["Foxp3+", "CD25+", "CTLA-4+", "CD103+"],
        "factors": ["TGF-β", "IL-10", "Foxp3", "STAT5"]
    }
}

# 免疫学専門用語検証
def validate_immune_terminology(label_text):
    """免疫学用語の妥当性を検証"""
    
    # 不適切な用語の組み合わせチェック
    invalid_combinations = [
        ("HSC", "Foxp3"),  # 造血幹細胞にFoxp3は不適切
        ("CLP", "CD8+"),   # CLPはCD8+ではない
        ("Treg", "IL-17"), # TregはIL-17を産生しない（通常）
        ("B細胞", "CD4+")   # B細胞はCD4+ではない
    ]
    
    for term1, term2 in invalid_combinations:
        if term1 in label_text and term2 in label_text:
            return False, f"不適切な組み合わせ: {term1} + {term2}"
    
    return True, "適切"

# 使用例とテスト
if __name__ == "__main__":
    # テスト用のコンテンツ
    test_contents = [
        "造血幹細胞（HSC）は骨髄に存在し、多分化能と自己複製能を持つ",
        "共通リンパ球前駆細胞（CLP）はIL-7Rαを発現し、リンパ系統に分化する",
        "CD4+ T細胞はMHCクラスIIを認識し、TCRシグナルにより活性化される",
        "制御性T細胞（Treg）はFoxp3を発現し、TGF-βとIL-10により免疫を抑制する"
    ]
    
    # 各レベルのラベル生成テスト
    for level, content in enumerate(test_contents):
        label = generate_immune_label(content, level, level+1, (level+1)*5)
        is_valid, message = validate_immune_terminology(label)
        
        print(f"Level {level}: {label}")
        print(f"検証結果: {message}")
        print("-" * 50)