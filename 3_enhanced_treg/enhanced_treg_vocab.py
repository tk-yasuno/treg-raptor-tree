"""
Enhanced Treg Differentiation Vocabulary System
Treg分化経路の詳細階層構造

新階層構造（7層）:
Level 0: HSC (造血幹細胞)
Level 1: CLP (共通リンパ球前駆細胞) 
Level 2: CD4+ T (CD4陽性T細胞)
Level 3: CD4+CD25+ T (CD25高発現T細胞)
Level 4: nTreg/iTreg Origin (胸腺由来/末梢誘導)
Level 5: Foxp3+ Treg (Foxp3発現制御性T細胞)
Level 6: Functional Treg (サイトカイン産生機能的Treg)
"""

# 拡張された階層別色分けマッピング（7層構造）
ENHANCED_LEVEL_COLOR_MAPPING = {
    0: {
        "color": "#E74C3C", 
        "name": "HSC", 
        "description": "Hematopoietic Stem Cell",
        "markers": ["Lin-", "Sca-1+", "c-Kit+", "CD34-", "CD150+"],
        "keywords": ["stem cell", "multipotent", "self-renewal", "bone marrow"]
    },
    1: {
        "color": "#3498DB", 
        "name": "CLP", 
        "description": "Common Lymphoid Progenitor",
        "markers": ["IL-7R+", "Flt3+", "Lin-", "Sca-1+"],
        "keywords": ["lymphoid", "IL-7", "progenitor", "commitment"]
    },
    2: {
        "color": "#9B59B6", 
        "name": "CD4+T", 
        "description": "CD4+ T Helper Cell",
        "markers": ["CD4+", "TCR+", "CD3+", "CD8-"],
        "keywords": ["helper T", "MHC-II", "TCR", "thymic selection"]
    },
    3: {
        "color": "#1ABC9C", 
        "name": "CD4+CD25+CD127low", 
        "description": "CD4+CD25high CD127low T Cell",
        "markers": ["CD4+", "CD25high", "CD127low", "IL-7Rαlow", "GITR+"],
        "human_treg_markers": ["CD4+", "CD25+", "CD127low/−", "IL-7Rα low"],
        "keywords": ["CD25 high expression", "CD127 low", "IL-7R alpha low", "IL-2R", "activated T cell", "Treg precursor", "human Treg identification"]
    },
    4: {
        "color": "#F39C12", 
        "name": "nTreg/iTreg", 
        "description": "Thymic/Peripheral Origin Treg",
        "markers": {
            "nTreg": ["thymic", "Helios+", "Nrp1+", "natural"],
            "iTreg": ["peripheral", "Helios-", "induced", "antigen-driven"]
        },
        "keywords": ["thymic selection", "peripheral conversion", "TGF-beta induction", "antigen recognition"]
    },
    5: {
        "color": "#16A085", 
        "name": "Foxp3+Treg", 
        "description": "Foxp3-expressing Regulatory T Cell",
        "markers": ["Foxp3+", "CD4+", "CD25high", "CD127low", "CTLA-4+"],
        "stability_markers": {
            "stable": ["TSDR demethylation", "CNS2 demethylation", "CD45RA+", "resting Treg"],
            "transient": ["TSDR methylated", "activated effector", "CD45RO+", "temporary Foxp3"]
        },
        "keywords": ["Foxp3", "transcription factor", "epigenetic stability", "Treg identity", "TSDR", "demethylation", "CD45RA", "stable vs transient"]
    },
    6: {
        "color": "#27AE60", 
        "name": "Functional Treg", 
        "description": "Cytokine-producing Suppressive Treg",
        "markers": ["Foxp3+", "IL-10+", "TGF-beta+", "CTLA-4+"],
        "cytokines": ["IL-10", "TGF-beta", "IL-35"],
        "mechanisms": ["CTLA-4", "LAG-3", "PD-1", "contact-dependent"],
        "keywords": ["immunosuppression", "tolerance", "cytokine production", "suppressive function"]
    },
    7: {
        "color": "#95A5A6", 
        "name": "ROOT", 
        "description": "Root Node",
        "markers": [],
        "keywords": ["hierarchy root", "top level"]
    }
}

# 詳細なTreg分化語彙体系
TREG_DIFFERENTIATION_VOCAB = {
    # Level 0: HSC (造血幹細胞)
    "hsc_level": {
        "japanese": {
            '造血幹細胞', 'HSC', '幹細胞', '多能性', '自己複製',
            '骨髄', '骨髄ニッチ', 'Lin-', 'Sca-1+', 'c-Kit+',
            'CD34-', 'CD150+', 'CD48-', 'SLAM', 'LSK',
            'SCF', 'TPO', 'Flt3L', 'IL-3', 'Notch', 'Wnt'
        },
        "english": {
            'hematopoietic stem cell', 'HSC', 'stem cell', 'multipotency', 
            'self-renewal', 'bone marrow', 'niche', 'quiescence',
            'Lin-', 'Sca-1+', 'c-Kit+', 'CD34-', 'CD150+', 'CD48-',
            'SCF', 'TPO', 'Flt3L', 'cytokine signaling'
        }
    },
    
    # Level 1: CLP (共通リンパ球前駆細胞)
    "clp_level": {
        "japanese": {
            'CLP', '共通リンパ球前駆細胞', 'リンパ球前駆細胞',
            'IL-7R', 'IL-7受容体', 'Flt3', 'Lin-', 'Sca-1+',
            'リンパ球系列', 'B細胞', 'T細胞', 'NK細胞',
            'IL-7', 'Flt3L', 'CXCL12', 'ストローマ細胞'
        },
        "english": {
            'common lymphoid progenitor', 'CLP', 'lymphoid progenitor',
            'IL-7R+', 'IL-7 receptor', 'Flt3+', 'lymphoid lineage',
            'B cell potential', 'T cell potential', 'NK potential',
            'IL-7 signaling', 'Flt3L', 'CXCL12', 'stromal support'
        }
    },
    
    # Level 2: CD4+ T細胞
    "cd4_t_level": {
        "japanese": {
            'CD4陽性T細胞', 'CD4+T細胞', 'ヘルパーT細胞',
            'CD4', 'CD3', 'TCR', 'T細胞受容体',
            'MHC-II', 'クラスII', '抗原提示', '胸腺選択',
            '正の選択', '負の選択', 'ダブルポジティブ',
            'Th1', 'Th2', 'Th17', 'CD28', 'CD80', 'CD86'
        },
        "english": {
            'CD4+ T cell', 'helper T cell', 'CD4 positive',
            'TCR', 'T cell receptor', 'MHC class II', 'antigen recognition',
            'thymic selection', 'positive selection', 'negative selection',
            'double positive', 'CD4+CD8+', 'single positive',
            'costimulation', 'CD28', 'B7', 'activation'
        }
    },
    
    # Level 3: CD4+CD25+ T細胞（CD25高発現 + CD127低発現）
    "cd25_high_cd127_low_level": {
        "japanese": {
            'CD25高発現', 'CD25high', 'CD25+', 'IL-2受容体',
            'IL-2Rα', 'CD127低発現', 'CD127low', 'CD127-', 'IL-7Rα低発現',
            'GITR', 'GITR陽性', '活性化T細胞', 'Treg前駆細胞', 
            'IL-2応答性', '高親和性IL-2受容体', 'CD25発現上昇',
            'ヒトTreg同定', 'CD4+CD25+CD127low', 'Tregマーカー'
        },
        "english": {
            'CD25 high expression', 'CD25high', 'CD25+', 'IL-2 receptor',
            'IL-2R alpha', 'CD127low', 'CD127 low', 'CD127-', 'IL-7Rα low',
            'IL-7R alpha low', 'GITR+', 'GITR positive',
            'activated T cell', 'Treg precursor', 'IL-2 responsiveness',
            'high-affinity IL-2R', 'upregulated CD25',
            'human Treg identification', 'CD4+CD25+CD127low', 'Treg marker combination'
        }
    },
    
    # Level 4: nTreg/iTreg（由来による分類）
    "treg_origin_level": {
        "ntreg": {
            "japanese": {
                '胸腺由来Treg', 'nTreg', '自然発生Treg', 'ナチュラルTreg',
                'Helios陽性', 'Nrp1陽性', '胸腺選択', '自己抗原認識',
                '中枢性免疫寛容', '胸腺髄質', 'AIRE', 
                '組織特異的抗原', 'TCR親和性', '高親和性TCR'
            },
            "english": {
                'thymic Treg', 'natural Treg', 'nTreg', 'tTreg',
                'Helios+', 'Nrp1+', 'thymic selection', 'self-antigen',
                'central tolerance', 'thymic medulla', 'AIRE',
                'tissue-specific antigen', 'high-affinity TCR', 'natural selection'
            }
        },
        "itreg": {
            "japanese": {
                '末梢誘導Treg', 'iTreg', '誘導性Treg', 'インデュースドTreg',
                'Helios陰性', '末梢転換', 'TGF-β誘導', '抗原認識',
                '末梢性免疫寛容', 'レチノイン酸', 'RA', 'TGF-β',
                '腸管免疫', '粘膜免疫', '食餌抗原', '腸内細菌'
            },
            "english": {
                'peripheral Treg', 'induced Treg', 'iTreg', 'pTreg',
                'Helios-', 'peripheral conversion', 'TGF-beta induced',
                'antigen-driven', 'peripheral tolerance', 'retinoic acid',
                'gut immunity', 'mucosal tolerance', 'dietary antigen',
                'microbiota', 'conversion from effector T'
            }
        }
    },
    
    # Level 5: Foxp3+ Treg（Foxp3発現 - 安定性による分類）
    "foxp3_treg_level": {
        "stable_treg": {
            "japanese": {
                'Foxp3陽性', 'Foxp3発現', 'Foxp3+Treg', '安定Treg',
                'FOXP3転写因子', 'Treg分化', 'Treg同一性',
                'エピジェネティック安定性', 'DNA脱メチル化', 'TSDR',
                'Treg特異的脱メチル化領域', 'TSDR脱メチル化', 'CNS1', 'CNS2', 'CNS3',
                'CNS2脱メチル化', 'Foxp3安定性', 'Treg系譜', '系譜決定',
                'CD45RA陽性', 'ナイーブTreg', '静止型Treg', 'rTreg',
                '真のTreg', '安定発現', '恒常的発現'
            },
            "english": {
                'Foxp3 positive', 'Foxp3+', 'Foxp3 expression', 'stable Treg',
                'FOXP3 transcription factor', 'Treg identity', 'lineage commitment',
                'epigenetic stability', 'DNA demethylation', 'TSDR',
                'Treg-specific demethylated region', 'TSDR demethylation',
                'CNS1', 'CNS2', 'CNS3', 'CNS2 demethylation',
                'Foxp3 stability', 'lineage determination', 'master regulator',
                'CD45RA+', 'naive Treg', 'resting Treg', 'rTreg',
                'bona fide Treg', 'stable expression', 'constitutive Foxp3'
            }
        },
        "transient_foxp3": {
            "japanese": {
                '一過性Foxp3発現', '一時的Foxp3', '活性化誘導Foxp3',
                'TSDRメチル化', 'エピジェネティック不安定',
                'CD45RO陽性', 'エフェクターT細胞', '活性化T細胞',
                '非Treg', '偽Treg', 'Foxp3+非Treg', 
                '一過性発現', '不安定発現', 'TCR刺激誘導'
            },
            "english": {
                'transient Foxp3 expression', 'temporary Foxp3', 'activation-induced Foxp3',
                'TSDR methylated', 'epigenetically unstable',
                'CD45RO+', 'effector T cell', 'activated T cell',
                'non-Treg', 'pseudo-Treg', 'Foxp3+ non-Treg',
                'transient expression', 'unstable expression', 'TCR-induced',
                'activation-dependent', 'non-suppressive Foxp3+'
            }
        },
        "discrimination": {
            "japanese": {
                'CD45RA/CD45RO', 'TSDR解析', 'エピジェネティック解析',
                '真のTreg識別', 'Treg純度', '機能的Treg',
                'メチル化解析', 'バイサルファイト法', 'PCR法'
            },
            "english": {
                'CD45RA/CD45RO discrimination', 'TSDR analysis', 'epigenetic profiling',
                'bona fide Treg identification', 'Treg purity', 'functional Treg',
                'methylation analysis', 'bisulfite sequencing', 'Treg stability assay'
            }
        }
    },
    
    # Level 6: Functional Treg（機能的Treg - サイトカイン産生）
    "functional_treg_level": {
        "cytokines": {
            "japanese": {
                'IL-10産生', 'TGF-β産生', 'IL-35産生',
                'インターロイキン10', 'トランスフォーミング増殖因子β',
                '抑制性サイトカイン', 'サイトカイン分泌',
                '免疫抑制サイトカイン', '抗炎症サイトカイン'
            },
            "english": {
                'IL-10 production', 'TGF-beta production', 'IL-35 production',
                'interleukin-10', 'transforming growth factor beta',
                'suppressive cytokine', 'anti-inflammatory cytokine',
                'cytokine secretion', 'immunosuppressive cytokine'
            }
        },
        "mechanisms": {
            "japanese": {
                'CTLA-4', 'LAG-3', 'PD-1', 'PD-L1', 'TIGIT',
                '接触依存性抑制', '細胞間接触', '共刺激阻害',
                'CD80/CD86競合', 'IDO誘導', 'トリプトファン代謝',
                'cAMP移行', 'グランザイムB', 'パーフォリン'
            },
            "english": {
                'CTLA-4', 'LAG-3', 'PD-1', 'PD-L1', 'TIGIT',
                'contact-dependent suppression', 'cell-cell contact',
                'costimulation blockade', 'CD80/CD86 competition',
                'IDO induction', 'tryptophan metabolism', 'cAMP transfer',
                'granzyme B', 'perforin', 'cytolysis'
            }
        },
        "functions": {
            "japanese": {
                '免疫抑制機能', '抑制機能', '免疫寛容',
                'エフェクターT細胞抑制', '炎症抑制', '組織修復',
                '自己免疫抑制', '同種免疫抑制', '腫瘍免疫',
                '移植免疫寛容', 'アレルギー抑制'
            },
            "english": {
                'immunosuppression', 'suppressive function', 'immune tolerance',
                'effector T cell suppression', 'inflammation control',
                'tissue repair', 'autoimmunity prevention', 'allograft tolerance',
                'tumor immunity', 'transplant tolerance', 'allergy suppression'
            }
        }
    }
}

# 階層判定関数の拡張（優先度調整版）
def determine_treg_level(content):
    """
    コンテンツから詳細なTreg分化階層レベルを判定
    
    判定優先順位（文脈依存）:
    1. HSC/CLP/CD4+T/CD25+ (基礎階層) - 常に最優先
    2. nTreg/iTreg (由来) - TGF-β文脈でも優先
    3. Foxp3 (一過性) - TCR刺激・活性化文脈で優先
    4. Foxp3 (安定) - TSDR/CD45RA文脈
    5. Functional Treg (サイトカイン) - 他の文脈がない場合
    
    Returns:
        int: 0-7のレベル番号
    """
    content_lower = content.lower()
    
    # Level 0-3: 基礎階層（最優先）
    # Level 0: HSC
    hsc_markers = ['hematopoietic stem', 'hsc', 'lin-', 'sca-1+', 'c-kit+', 'bone marrow niche']
    if any(marker in content_lower for marker in hsc_markers):
        return 0
    
    # Level 1: CLP
    clp_markers = ['common lymphoid progenitor', 'clp', 'il-7r+', 'flt3+', 'lymphoid lineage']
    if any(marker in content_lower for marker in clp_markers):
        return 1
    
    # Level 2: CD4+ T cell
    cd4_markers = ['cd4+ t cell', 'cd4 positive', 'helper t', 'th1', 'th2', 'th17', 
                   'mhc class ii', 'thymic selection']
    # CD25やTregキーワードがない場合のみCD4+T判定
    if any(marker in content_lower for marker in cd4_markers):
        if not any(treg in content_lower for treg in ['cd25', 'treg', 'regulatory', 'foxp3']):
            return 2
    
    # Level 3: CD25high + CD127low
    cd25_markers = ['cd25high', 'cd25 high', 'cd25+', 'il-2r', 'cd127low', 'cd127 low', 
                    'il-7rα low', 'il-7r alpha low']
    if any(marker in content_lower for marker in cd25_markers):
        # Foxp3やnTreg/iTregキーワードがない場合のみ
        if not any(advanced in content_lower for advanced in ['foxp3', 'thymic treg', 'peripheral treg', 
                                                               'ntreg', 'itreg', 'il-10', 'suppressive']):
            return 3
    
    # Level 4: nTreg/iTreg (由来) - TGF-β文脈でも優先
    # iTreg特異的マーカー（TGF-βを含む文脈）
    itreg_context = ['peripheral', 'induced', 'conversion', 'gut mucosa', 'mucosal', 
                     'retinoic acid', 'iTreg', 'ptreg']
    ntreg_context = ['thymic', 'natural', 'ntreg', 'ttreg', 'helios+', 'nrp1+', 'aire']
    
    # iTreg判定（TGF-βがあっても iTreg文脈なら Level 4）
    if any(itreg in content_lower for itreg in itreg_context):
        # TGF-βが「誘導因子」として言及されている場合はiTreg
        if 'tgf' in content_lower and any(induction in content_lower for induction in 
                                          ['induc', 'convert', 'driven', 'peripheral']):
            return 4
        # その他のiTreg文脈
        if any(itreg in content_lower for itreg in ['itreg', 'induced treg', 'peripheral treg']):
            return 4
    
    # nTreg判定
    if any(ntreg in content_lower for ntreg in ntreg_context):
        return 4
    
    # Level 5: Foxp3+ Treg（一過性を優先検出）
    # 一過性Foxp3の文脈キーワード
    transient_context = ['transient', 'temporary', 'activation-induced', 'tcr stimulation',
                        'activated cd4', 'effector t', 'cd45ro+', 'non-suppressive',
                        'pseudo-treg', 'unstable']
    
    # 安定Tregの文脈キーワード
    stable_context = ['tsdr demethyl', 'cd45ra', 'stable', 'resting', 'naive treg',
                     'bona fide', 'cns2 demethyl', 'epigenetic stability']
    
    foxp3_markers = ['foxp3', 'foxp3+', 'foxp3 positive', 'foxp3 expression']
    
    if any(marker in content_lower for marker in foxp3_markers):
        # 一過性文脈があればLevel 5（一過性）
        if any(trans in content_lower for trans in transient_context):
            return 5
        # 安定性文脈があればLevel 5（安定）
        elif any(stable in content_lower for stable in stable_context):
            return 5
        # 機能的マーカー（IL-10, suppressive）がなければLevel 5
        elif not any(func in content_lower for func in ['il-10', 'suppress', 'cytokine production']):
            return 5
    
    # Level 6: Functional Treg (サイトカイン産生・抑制機能)
    # 他の文脈がない場合のみ
    functional_markers = ['il-10', 'il-35', 'suppressive function', 'immunosuppression', 
                         'cytokine production', 'contact-dependent']
    # CTLA-4/LAG-3/TGF-βは機能的文脈でのみカウント
    if any(marker in content_lower for marker in functional_markers):
        return 6
    
    # CTLA-4, LAG-3, TGF-βが抑制機能文脈で言及されている場合
    if any(mech in content_lower for mech in ['ctla-4', 'lag-3', 'tgf-beta', 'tgf-β']):
        if any(func in content_lower for func in ['suppress', 'mediate', 'mechanism', 'function']):
            return 6
    
    # どの特異的マーカーにも該当しない場合はLevel 0（デフォルト）
    return 0


# 階層特異的ラベル生成
def generate_enhanced_treg_label(content, level, cluster_id, cluster_size):
    """
    拡張Treg階層に基づくラベル生成
    
    Args:
        content: テキストコンテンツ
        level: 階層レベル (0-7)
        cluster_id: クラスターID
        cluster_size: クラスターサイズ
        
    Returns:
        str: 階層特異的ラベル
    """
    if level not in ENHANCED_LEVEL_COLOR_MAPPING:
        level = 0
    
    level_info = ENHANCED_LEVEL_COLOR_MAPPING[level]
    base_name = level_info["name"]
    
    # レベル特異的キーワード抽出
    content_lower = content.lower()
    detected_markers = []
    
    if level == 6:  # Functional Treg
        cytokines = []
        if 'il-10' in content_lower:
            cytokines.append('IL-10')
        if 'tgf-beta' in content_lower or 'tgf-β' in content_lower:
            cytokines.append('TGF-β')
        if 'ctla-4' in content_lower:
            cytokines.append('CTLA-4')
        if cytokines:
            return f"{base_name}\n{'+'.join(cytokines)}\n(n={cluster_size})"
    
    elif level == 5:  # Foxp3+ Treg
        stability_type = ""
        if 'tsdr demethyl' in content_lower or 'cd45ra' in content_lower:
            stability_type = "stable"
        elif 'transient' in content_lower or 'cd45ro' in content_lower:
            stability_type = "transient"
        
        if 'foxp3' in content_lower:
            if stability_type == "stable":
                return f"{base_name}\nFoxp3+ stable\nTSDR demethyl\n(n={cluster_size})"
            elif stability_type == "transient":
                return f"{base_name}\nFoxp3+ transient\nCD45RO+\n(n={cluster_size})"
            else:
                return f"{base_name}\nFoxp3+\n(n={cluster_size})"
    
    elif level == 4:  # nTreg/iTreg
        if 'thymic' in content_lower or 'helios+' in content_lower:
            return f"{base_name}\nnTreg-thymic\n(n={cluster_size})"
        elif 'peripheral' in content_lower or 'induced' in content_lower:
            return f"{base_name}\niTreg-peripheral\n(n={cluster_size})"
    
    elif level == 3:  # CD25high + CD127low
        cd127_status = "CD127low" if 'cd127' in content_lower or 'il-7r' in content_lower else ""
        if cd127_status:
            return f"{base_name}\nCD25high CD127low\nIL-2Rα+/IL-7Rα−\n(n={cluster_size})"
        else:
            return f"{base_name}\nCD25high\nIL-2Rα high\n(n={cluster_size})"
    
    # デフォルトラベル
    return f"{base_name}\nCluster {cluster_id}\n(n={cluster_size})"

# 後方互換性のための関数マッピング
LEVEL_COLOR_MAPPING = ENHANCED_LEVEL_COLOR_MAPPING
generate_immune_label = generate_enhanced_treg_label

def validate_immune_terminology(label):
    """免疫学用語の妥当性検証"""
    if not label or len(label) < 2:
        return False, "Label too short"
    
    # ASCIIチェック
    try:
        label.encode('ascii')
        return True, "Valid ASCII label"
    except UnicodeEncodeError:
        return False, "Contains non-ASCII characters"

def extract_level_keywords(content, level):
    """レベル特異的キーワード抽出"""
    content_lower = content.lower()
    keywords = set()
    
    if level in ENHANCED_LEVEL_COLOR_MAPPING:
        level_info = ENHANCED_LEVEL_COLOR_MAPPING[level]
        if 'markers' in level_info and isinstance(level_info['markers'], list):
            keywords.update([m.lower() for m in level_info['markers']])
        if 'keywords' in level_info:
            keywords.update([k.lower() for k in level_info['keywords']])
    
    # コンテンツ内のキーワードマッチング
    found_keywords = [kw for kw in keywords if kw in content_lower]
    return found_keywords[:5]  # 上位5個を返す
