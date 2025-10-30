"""
橋梁診断RAPTOR Tree可視化スクリプト
NetworkXとMatplotlibを使用してツリー構造を可視化

保存されたscaling_test_tree_*.pklファイルからRAPTOR Treeを読み込み、
階層構造を視覚化してPNG/PDFとして保存します。
"""

import os
import sys
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# 形態素解析
try:
    from fugashi import Tagger, GenericTagger
    FUGASHI_AVAILABLE = True
except ImportError:
    FUGASHI_AVAILABLE = False
    print("⚠️ fugashiがインストールされていません。形態素解析ラベルはTF-IDFのみ使用されます")

# 橋梁ドメイン語彙
from bridge_diagnosis_vocab import (
    filter_bridge_keywords, 
    is_bridge_keyword, 
    is_stop_word,
    STOP_WORDS,
    BRIDGE_DOMAIN_KEYWORDS, 
    BRIDGE_TRANSLATION_DICT,
    get_priority_keywords_by_depth,
)

# pykakasiのグローバル初期化（1回のみ）
_KAKASI_INSTANCE = None

def get_kakasi():
    """pykakasiインスタンスを取得（シングルトン）"""
    global _KAKASI_INSTANCE
    if _KAKASI_INSTANCE is None:
        try:
            import pykakasi
            _KAKASI_INSTANCE = pykakasi.kakasi()
        except ImportError:
            _KAKASI_INSTANCE = False  # インポート失敗をマーク
    return _KAKASI_INSTANCE

def translate_keyword(keyword: str) -> str:
    """
    日本語キーワードを英語に翻訳（辞書ベース + ローマ字化）
    
    1. 既にASCIIならそのまま返す
    2. 翻訳辞書から検索
    3. pykakasiでローマ字化
    4. 失敗時はフォールバック
    """
    # 既に英数字のみの場合はそのまま返す
    if keyword.isascii():
        return keyword
    
    # 橋梁診断辞書から翻訳を取得
    translated = BRIDGE_TRANSLATION_DICT.get(keyword)
    if translated:
        return translated
    
    # pykakasiでローマ字化を試行
    kks = get_kakasi()
    if kks and kks is not False:
        try:
            result = kks.convert(keyword)
            # hepburnキーでローマ字を取得し、単語ごとに先頭大文字化
            romaji_parts = []
            for item in result:
                if 'hepburn' in item and item['hepburn']:
                    romaji_parts.append(item['hepburn'].capitalize())
            
            if romaji_parts:
                return ' '.join(romaji_parts)
        except Exception as e:
            print(f"⚠️ pykakasi変換エラー ({keyword}): {e}")
    
    # フォールバック: ひらがな・カタカナ・漢字を"[JA]"に置換
    return f"[JA:{keyword[:3]}]"


def extract_keywords_from_text(text: str, top_n: int = 5, depth: int = 0) -> List[str]:
    """
    テキストから重要キーワードを抽出（橋梁診断ドメイン特化）
    
    深さに応じて優先キーワードカテゴリを変更：
    - depth=0: ROOT（全カテゴリ）
    - depth=1: 部材（床版、桁、支承など）
    - depth=2: 損傷（腐食、ひび割れ、疲労など）
    - depth=3: 原因（塩害、凍害、荷重など）
    - depth>=4: 補修工法（補修、補強、表面被覆など）
    
    Args:
        text: 抽出対象のテキスト
        top_n: 抽出するキーワード数
        depth: ノードの深さ（階層レベル）
    """
    keywords = []
    
    # 深さに応じた優先キーワードセットを取得
    priority_keywords = get_priority_keywords_by_depth(depth)
    
    # 1. 優先キーワードカテゴリから抽出
    if priority_keywords:
        priority_found = []
        for keyword in priority_keywords:
            if keyword in text and not is_stop_word(keyword):
                priority_found.append(keyword)
        
        # 優先キーワードを先頭に追加
        keywords.extend(priority_found[:top_n])
    
    # 2. まだ不足している場合、一般的な橋梁ドメインキーワードを追加
    if len(keywords) < top_n:
        domain_keywords = filter_bridge_keywords(text)
        for kw in domain_keywords:
            if kw not in keywords:
                keywords.append(kw)
            if len(keywords) >= top_n:
                break
    
    # 2. 形態素解析で名詞を抽出（fugashi利用可能時）
    if FUGASHI_AVAILABLE and len(keywords) < top_n:
        try:
            # MeCabの設定ファイルパスを環境変数に設定
            mecabrc_paths = [
                r'C:\Program Files\MeCab\etc\mecabrc',
                r'C:\Program Files (x86)\MeCab\etc\mecabrc',
                r'C:\mecab\etc\mecabrc'
            ]
            
            for mecabrc_path in mecabrc_paths:
                if os.path.exists(mecabrc_path):
                    os.environ['MECABRC'] = mecabrc_path
                    break
            
            # MeCabの辞書パス設定
            dicdir = None
            
            # 1. unidic-liteを試行
            try:
                import unidic_lite
                dicdir = unidic_lite.DICDIR
            except ImportError:
                pass
            
            # 2. システムMeCabを試行
            if dicdir is None:
                mecab_dic_paths = [
                    r'C:\Program Files\MeCab\dic\ipadic',
                    r'C:\Program Files (x86)\MeCab\dic\ipadic',
                    r'C:\mecab\dic\ipadic'
                ]
                for path in mecab_dic_paths:
                    if os.path.exists(path):
                        dicdir = path
                        break
            
            # Tagger初期化
            if dicdir:
                if 'ipadic' in dicdir.lower():
                    tagger = GenericTagger(f'-d "{dicdir}"')
                else:
                    tagger = Tagger(f'-d "{dicdir}"')
            else:
                try:
                    tagger = Tagger()
                except:
                    tagger = GenericTagger()
            
            words = []
            for word in tagger(text):
                # 名詞のみ抽出（ストップワードを除外）
                # GenericTaggerはfeatureがタプル、Taggerはオブジェクト
                try:
                    # Tagger形式 (UniDic)
                    pos = word.feature.pos1
                except AttributeError:
                    # GenericTagger形式 (IPADIC) - タプル[0]が品詞
                    pos = word.feature[0] if isinstance(word.feature, tuple) else None
                
                if pos == '名詞' and len(word.surface) > 1:
                    if not is_stop_word(word.surface):
                        words.append(word.surface)
            
            # 頻度カウント
            word_counts = Counter(words)
            most_common = [w for w, c in word_counts.most_common(top_n * 2)]
            
            # まだ含まれていないキーワードを追加
            for word in most_common:
                if word not in keywords:
                    keywords.append(word)
                if len(keywords) >= top_n:
                    break
        except Exception as e:
            print(f"⚠️ 形態素解析エラー: {e}")
    
    # 3. TF-IDFでフォールバック（まだ不足している場合）
    if len(keywords) < top_n:
        # 簡易的な単語分割（スペース・句読点で分割）
        words = re.findall(r'\w+', text)
        # 2文字以上の単語をカウント（ストップワードを除外）
        word_counts = Counter([w for w in words if len(w) >= 2 and not is_stop_word(w)])
        most_common = [w for w, c in word_counts.most_common(top_n * 2)]
        
        for word in most_common:
            if word not in keywords:
                keywords.append(word)
            if len(keywords) >= top_n:
                break
    
    return keywords[:top_n]


def load_raptor_tree(pkl_file: str) -> Dict[str, Any]:
    """
    .pklファイルからRAPTOR Treeを読み込み
    
    戻り値:
        tree_data: {
            'tree': {
                'summaries': List[Dict],  # サマリーノード情報
                'clusters': Dict,         # クラスター情報
                'documents': int,         # ドキュメント数
                'depth': int,             # ツリーの深さ
            },
            'stats': Dict,  # 統計情報
            ...
        }
    """
    print(f"\n📂 Loading RAPTOR Tree from: {pkl_file}")
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Tree file not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        tree_data = pickle.load(f)
    
    # 基本統計
    tree = tree_data.get('tree', {})
    summaries = tree.get('summaries', [])
    stats = tree_data.get('stats', {})
    
    print(f"✅ Loaded tree data")
    print(f"   Depth: {tree.get('depth', 0)}")
    print(f"   Documents: {tree.get('documents', 0)}")
    print(f"   Summaries: {len(summaries)}")
    print(f"   Stats: {stats}")
    
    return tree_data


def build_graph_from_tree(tree_data: Dict[str, Any], max_depth: int = None) -> Tuple[nx.DiGraph, Dict]:
    """
    RAPTOR TreeからNetworkXグラフを構築
    
    Args:
        tree_data: ツリーデータ
        max_depth: 可視化する最大深度（Noneの場合は全て）
    
    visualize_raptor_tree.pyの実装を参考に、
    clustersを再帰的にイテレートしてグラフを構築
    """
    stats = tree_data.get('stats', {})
    num_leaf_nodes = stats.get('num_leaf_nodes', 0)
    num_internal_nodes = stats.get('num_internal_nodes', 0)
    tree_max_depth = stats.get('max_depth', 3)
    
    # max_depthが指定されていない場合はツリーの最大深度を使用
    if max_depth is None:
        max_depth = tree_max_depth
    
    G = nx.DiGraph()
    node_info = {}
    node_counter = 0
    
    print("\n🔨 Building NetworkX graph...")
    print(f"   Target: {num_internal_nodes} internal + {num_leaf_nodes} leaf = {num_internal_nodes + num_leaf_nodes} nodes")
    print(f"   Tree max depth: {tree_max_depth}")
    print(f"   Visualizing up to depth: {max_depth}")
    
    def get_node_id():
        """ユニークなノードIDを生成"""
        nonlocal node_counter
        node_counter += 1
        return f"node_{node_counter}"
    
    def process_clusters(tree_dict, depth=0, parent_id=None):
        """
        再帰的にclustersを処理してノードとエッジを作成
        
        重要: すべてのdepthで、clustersをイテレートしてノードを作成
        """
        # 深度制限チェック
        if max_depth is not None and depth > max_depth:
            return
        
        if not isinstance(tree_dict, dict):
            return
        
        # clustersを処理
        if 'clusters' not in tree_dict or not tree_dict['clusters']:
            return
        
        clusters = tree_dict['clusters']
        
        for cluster_id, cluster_data in clusters.items():
            # サマリーテキスト取得
            summary_text = ""
            if 'summary' in cluster_data:
                summary = cluster_data['summary']
                if hasattr(summary, 'page_content'):
                    summary_text = summary.page_content
                elif isinstance(summary, dict):
                    summary_text = summary.get('page_content', summary.get('text', ''))
            
            # 子の有無判定 (visualize_raptor_tree.pyと同じパターン)
            has_children = ('children' in cluster_data and 
                          cluster_data['children'] and 
                          'clusters' in cluster_data['children'] and
                          cluster_data['children']['clusters'])
            
            # ノードID生成
            node_id = get_node_id()
            layer = tree_max_depth - depth
            
            # キーワード抽出（深さに応じた優先キーワードを使用）
            keywords = extract_keywords_from_text(summary_text, top_n=3, depth=depth)
            translated = [translate_keyword(kw) for kw in keywords]
            # 2つのキーワードを表示
            if len(translated) >= 2:
                label_en = f"{translated[0]}\n{translated[1]}"
            else:
                label_en = translated[0] if translated else f"C{cluster_id}"
            
            if len(keywords) >= 2:
                label_ja = f"{keywords[0]}\n{keywords[1]}"
            else:
                label_ja = keywords[0] if keywords else f"C{cluster_id}"
            
            # ノードタイプ
            node_type = 'internal' if has_children else 'leaf'
            
            # ノード情報を保存
            node_info[node_id] = {
                'label': label_en if label_en else f"C{cluster_id}",
                'label_ja': label_ja if label_ja else f"C{cluster_id}",
                'layer': layer,
                'depth': depth,
                'cluster_id': cluster_id,
                'text': summary_text[:100],
                'keywords': keywords,
                'node_type': node_type,
            }
            
            # グラフにノード追加
            G.add_node(node_id, **node_info[node_id])
            
            # 親ノードとの接続
            if parent_id:
                G.add_edge(parent_id, node_id)
            
            # 子ツリーを再帰的に処理
            if has_children:
                process_clusters(
                    cluster_data['children'],
                    depth + 1,
                    node_id
                )
    
    # ルートツリーから処理開始
    root_tree = tree_data.get('tree', {})
    if not isinstance(root_tree, dict):
        print("❌ エラー: ツリーが辞書形式ではありません")
        return G, node_info
    
    # 仮想ルートノードを作成 (statsで1個としてカウントされているため)
    root_node_id = get_node_id()
    layer = tree_max_depth
    
    node_info[root_node_id] = {
        'label': 'ROOT',
        'layer': layer,
        'depth': 0,
        'cluster_id': 'root',
        'text': 'Root of tree',
        'keywords': [],
        'node_type': 'internal',
    }
    
    G.add_node(root_node_id, **node_info[root_node_id])
    
    # ルートツリーのclustersを処理 (rootノードの子として)
    if 'clusters' in root_tree and root_tree['clusters']:
        clusters = root_tree['clusters']
        
        for cluster_id, cluster_data in clusters.items():
            # サマリーテキスト取得
            summary_text = ""
            if 'summary' in cluster_data:
                summary = cluster_data['summary']
                if hasattr(summary, 'page_content'):
                    summary_text = summary.page_content
                elif isinstance(summary, dict):
                    summary_text = summary.get('page_content', summary.get('text', ''))
            
            # 子の有無判定
            has_children = ('children' in cluster_data and 
                          cluster_data['children'] and 
                          'clusters' in cluster_data['children'] and
                          cluster_data['children']['clusters'])
            
            # ノードID生成
            node_id = get_node_id()
            layer = tree_max_depth - 1  # depth=1相当
            
            # キーワード抽出（深さ=1なので部材カテゴリ優先）
            keywords = extract_keywords_from_text(summary_text, top_n=3, depth=1)
            translated = [translate_keyword(kw) for kw in keywords]
            # 2つのキーワードを表示
            if len(translated) >= 2:
                label_en = f"{translated[0]}\n{translated[1]}"
            else:
                label_en = translated[0] if translated else f"C{cluster_id}"
            
            if len(keywords) >= 2:
                label_ja = f"{keywords[0]}\n{keywords[1]}"
            else:
                label_ja = keywords[0] if keywords else f"C{cluster_id}"
            
            # ノードタイプ
            node_type = 'internal' if has_children else 'leaf'
            
            # ノード情報を保存
            node_info[node_id] = {
                'label': label_en if label_en else f"C{cluster_id}",
                'label_ja': label_ja if label_ja else f"C{cluster_id}",
                'layer': layer,
                'depth': 1,
                'cluster_id': cluster_id,
                'text': summary_text[:100],
                'keywords': keywords,
                'node_type': node_type,
            }
            
            # グラフにノード追加
            G.add_node(node_id, **node_info[node_id])
            
            # ルートノードとの接続
            G.add_edge(root_node_id, node_id)
            
            # 子ツリーを再帰的に処理
            if has_children:
                process_clusters(
                    cluster_data['children'],
                    depth=2,  # 次はdepth=2
                    parent_id=node_id
                )
    
    # 統計情報
    leaf_count = sum(1 for info in node_info.values() if info['node_type'] == 'leaf')
    internal_count = sum(1 for info in node_info.values() if info['node_type'] == 'internal')
    
    # Depth別ノード数
    from collections import Counter
    depth_counts = Counter(info['depth'] for info in node_info.values())
    
    print(f"✅ Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")
    print(f"   Internal nodes: {internal_count}")
    print(f"   Leaf nodes: {leaf_count}")
    print(f"   Layers: {sorted(set(info['layer'] for info in node_info.values()))}")
    print(f"   Nodes by depth: {dict(sorted(depth_counts.items()))}")
    
    return G, node_info


def compute_hierarchical_layout(G: nx.DiGraph, node_info: Dict) -> Dict[int, Tuple[float, float]]:
    """
    階層レイアウトを計算（レイヤーごとに配置）
    
    戻り値:
        pos: ノードID → (x, y) 座標のマッピング
    """
    # レイヤーごとにノードをグループ化
    layers = {}
    for node_id, info in node_info.items():
        layer = info['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node_id)
    
    pos = {}
    max_layer = max(layers.keys())
    
    for layer, nodes in layers.items():
        # Y座標: レイヤーが高いほど上（rootが上になるように反転）
        y = layer * 4.0  # レイヤー間の縦間隔を拡大（3.0 → 4.0）
        
        # X座標: ノードを等間隔に配置
        num_nodes = len(nodes)
        for i, node_id in enumerate(sorted(nodes)):
            x = (i - num_nodes / 2) * 5.0  # 横間隔を拡大（3.5 → 5.0）
            pos[node_id] = (x, y)
    
    return pos


def compute_circular_layout(G: nx.DiGraph, node_info: Dict) -> Dict[int, Tuple[float, float]]:
    """
    円形レイアウトを計算（ルートを中心に同心円状に配置）
    
    戻り値:
        pos: ノードID → (x, y) 座標のマッピング
    """
    import math
    
    # レイヤーごとにノードをグループ化
    layers = {}
    for node_id, info in node_info.items():
        layer = info['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node_id)
    
    pos = {}
    max_layer = max(layers.keys())
    
    for layer, nodes in layers.items():
        num_nodes = len(nodes)
        
        if layer == max_layer:
            # ルートノードは中心に配置
            pos[nodes[0]] = (0, 0)
        else:
            # 同心円の半径（レイヤーが低いほど外側）
            radius = (max_layer - layer) * 8.0  # 円の間隔
            
            # ノードを円周上に等間隔に配置
            for i, node_id in enumerate(sorted(nodes)):
                angle = 2 * math.pi * i / num_nodes
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos[node_id] = (x, y)
    
    return pos


def visualize_tree(
    G: nx.DiGraph,
    node_info: Dict,
    output_file: str = "raptor_tree_bridge.png",
    title: str = "Bridge Diagnosis RAPTOR Tree",
    figsize: Tuple[int, int] = (20, 12),
    dpi: int = 150
):
    """
    RAPTOR Treeを可視化して保存（日本語版と英語版の2つを生成）
    """
    print(f"\n🎨 Visualizing tree (Japanese and English versions)...")
    
    # 日本語フォント設定
    matplotlib.rcParams['font.family'] = 'Yu Gothic'
    matplotlib.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # レイアウト計算（共通）
    pos = compute_hierarchical_layout(G, node_info)
    
    # レイヤーごとに色分け（共通）
    layers = set(info['layer'] for info in node_info.values())
    max_layer = max(layers)
    colors = plt.cm.viridis(np.linspace(0, 1, max_layer + 1))
    
    # ラベル準備
    labels_en = {nid: info['label'] for nid, info in node_info.items()}  # 英語ラベル
    labels_ja = {nid: info.get('label_ja', info['label']) for nid, info in node_info.items()}  # 日本語ラベル
    
    # 統計情報
    stats_text = f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)} | Depth: {max_layer + 1}"
    
    # レイヤー凡例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=colors[layer], markersize=10,
                  label=f'Layer {layer}')
        for layer in sorted(layers)
    ]
    
    # ファイル名のベース部分を取得
    if output_file.endswith('.png'):
        base_name = output_file[:-4]
    else:
        base_name = output_file
    
    # ========== 日本語版生成 ==========
    output_file_ja = f"{base_name}_ja.png"
    print(f"💾 Saving Japanese version to: {output_file_ja}")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    for layer in sorted(layers):
        layer_nodes = [nid for nid, info in node_info.items() if info['layer'] == layer]
        nx.draw_networkx_nodes(G, pos, nodelist=layer_nodes,
                              node_color=[colors[layer]], node_size=1500, alpha=0.8, ax=ax)  # 2000 → 1500（ノードを少し小さく）
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                          arrowsize=15, arrowstyle='->', alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels_ja, font_size=9, font_weight='bold', ax=ax)  # 10 → 9（フォントも少し小さく）
    
    ax.set_title(f"{title} (Japanese)", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, ha='center', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file_ja, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ Japanese PNG saved: {output_file_ja}")
    
    print(f"\n✅ Visualization complete! Generated file:")
    print(f"   - {output_file_ja}")


def visualize_tree_circular(
    G: nx.DiGraph,
    node_info: Dict,
    output_file: str = "raptor_tree_bridge_circular.png",
    title: str = "Bridge Diagnosis RAPTOR Tree (Circular Layout)",
    figsize: Tuple[int, int] = (24, 24),
    dpi: int = 150
):
    """
    RAPTOR Treeを円形レイアウトで可視化して保存（日本語版）
    """
    print(f"\n🎨 Visualizing tree with circular layout...")
    
    # 日本語フォント設定
    matplotlib.rcParams['font.family'] = 'Yu Gothic'
    matplotlib.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 円形レイアウト計算
    pos = compute_circular_layout(G, node_info)
    
    # レイヤーごとに色分け
    layers = set(info['layer'] for info in node_info.values())
    max_layer = max(layers)
    colors = plt.cm.viridis(np.linspace(0, 1, max_layer + 1))
    
    # 日本語ラベル準備
    labels_ja = {nid: info.get('label_ja', info['label']) for nid, info in node_info.items()}
    
    # 統計情報
    stats_text = f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)} | Depth: {max_layer + 1}"
    
    # レイヤー凡例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=colors[layer], markersize=10,
                  label=f'Layer {layer}')
        for layer in sorted(layers)
    ]
    
    # ファイル名のベース部分を取得
    if output_file.endswith('.png'):
        base_name = output_file[:-4]
    else:
        base_name = output_file
    
    # 円形レイアウト版生成
    output_file_circular = f"{base_name}_circular_ja.png"
    print(f"💾 Saving circular layout to: {output_file_circular}")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # ノード描画（レイヤー別に色分け）
    for layer in sorted(layers):
        layer_nodes = [nid for nid, info in node_info.items() if info['layer'] == layer]
        node_size = 2500 if layer == max_layer else 1800  # ルートノードは大きく
        nx.draw_networkx_nodes(G, pos, nodelist=layer_nodes,
                              node_color=[colors[layer]], node_size=node_size, alpha=0.8, ax=ax)
    
    # エッジ描画
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                          arrowsize=15, arrowstyle='->', alpha=0.3, ax=ax, width=1.5)
    
    # ラベル描画
    nx.draw_networkx_labels(G, pos, labels=labels_ja, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title(f"{title}", fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
           fontsize=11, ha='center', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file_circular, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ Circular layout PNG saved: {output_file_circular}")
    
    print(f"\n✅ Circular visualization complete! Generated file:")
    print(f"   - {output_file_circular}")



def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Bridge Diagnosis RAPTOR Tree')
    parser.add_argument('--tree_file', type=str, default=None,
                       help='Path to tree .pkl file (default: latest in results/)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--title', type=str, default='Bridge Diagnosis RAPTOR Tree',
                       help='Chart title')
    parser.add_argument('--figsize', type=int, nargs=2, default=[40, 30],
                       help='Figure size (width height)')  # 30x20 → 40x30に拡大
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output DPI')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Maximum depth to visualize (default: all depths)')
    
    args = parser.parse_args()
    
    # ツリーファイルを検索
    if args.tree_file is None:
        # 橋梁診断用のresultsディレクトリから最新を検索
        results_dir = Path("data/doken_bridge_diagnosis_logic/results")
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return
        
        tree_files = sorted(results_dir.glob("scaling_test_tree_*.pkl"))
        if not tree_files:
            print(f"❌ No tree files found in {results_dir}")
            return
        
        args.tree_file = str(tree_files[-1])  # 最新
        print(f"Using latest tree file: {args.tree_file}")
    
    # 出力ファイル名
    if args.output is None:
        # ツリーファイル名から出力名を生成
        tree_name = Path(args.tree_file).stem  # "scaling_test_tree_250chunks_20251028_172826"
        results_dir = Path("data/doken_bridge_diagnosis_logic/results")
        args.output = str(results_dir / f"{tree_name}_visualization.png")
    
    # 実行
    try:
        # 1. ツリー読み込み
        tree_data = load_raptor_tree(args.tree_file)
        
        # 2. グラフ構築
        G, node_info = build_graph_from_tree(tree_data, max_depth=args.max_depth)
        
        # 3. 階層型レイアウトで可視化
        print("\n" + "="*60)
        print("📊 Generating Hierarchical Layout...")
        print("="*60)
        visualize_tree(
            G, node_info,
            output_file=args.output,
            title=args.title,
            figsize=tuple(args.figsize),
            dpi=args.dpi
        )
        
        # 4. 円形レイアウトで可視化
        print("\n" + "="*60)
        print("🎯 Generating Circular Layout...")
        print("="*60)
        visualize_tree_circular(
            G, node_info,
            output_file=args.output,
            title=args.title + " (Circular)",
            figsize=(24, 24),  # 円形は正方形
            dpi=args.dpi
        )
        
        print("\n" + "="*60)
        print("✅ All visualizations complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
