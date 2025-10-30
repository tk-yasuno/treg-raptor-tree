#!/usr/bin/env python3
"""
RAPTOR Tree Visualization Tool
Creates detailed visual representations of the RAPTOR tree structure
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from pathlib import Path
import numpy as np
from datetime import datetime
import seaborn as sns
from immune_cell_vocab import (
    generate_immune_label, 
    validate_immune_terminology,
    LEVEL_COLOR_MAPPING,
    extract_level_keywords
)

# Windows対応フォント設定（フォント警告を完全回避）
import matplotlib
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10
# 文字エンコーディング問題を回避
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.monospace'] = ['Consolas', 'Arial']

# 全ての警告を抑制
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=UnicodeWarning)
import sys
import os
# 標準出力のエンコーディングを設定
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'encoding'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

class RAPTORTreeVisualizer:
    """Advanced RAPTOR Tree visualization"""
    
    def __init__(self):
        self.colors = {
            'level_1': '#FF6B6B',  # 赤系
            'level_2': '#4ECDC4',  # 青緑系
            'level_3': '#45B7D1',  # 青系
            'level_4': '#96CEB4',  # 緑系
            'root': '#FFEAA7'      # 黄系
        }
        
    def load_latest_tree(self):
        """最新のRAPTORツリーファイルを読み込み"""
        data_dir = Path("data/immune_cell_differentiation/raptor_trees")
        if not data_dir.exists():
            data_dir = Path(".")
            
        # 最新のファイルを検索
        json_files = list(data_dir.glob("*raptor*tree*.json"))
        if not json_files:
            raise FileNotFoundError("RAPTORツリーファイルが見つかりません")
            
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Loading: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f), latest_file.name
    
    def create_tree_graph(self, tree_data):
        """ネットワークグラフとしてツリーを構築"""
        G = nx.DiGraph()
        node_levels = {}
        node_sizes = {}
        node_contents = {}
        
        # ノード追加（ROOTノードを除外）
        for node_id, node_data in tree_data['nodes'].items():
            # ROOTノードをスキップ
            if 'root' in node_id.lower():
                continue
                
            G.add_node(node_id)
            
            # レベル判定（ROOTノードは除外）
            if 'root' in node_id.lower():
                continue
            elif '_L3_' in node_id:
                level = 3
            elif '_L2_' in node_id:
                level = 2
            elif '_L1_' in node_id:
                level = 1
            else:
                level = 0  # 原文書
                
            node_levels[node_id] = level
            node_sizes[node_id] = node_data.get('cluster_size', 1)
            
            # 内容の要約を取得
            content = node_data.get('content', '')
            if len(content) > 100:
                content = content[:100] + "..."
            node_contents[node_id] = content
            
        # エッジ追加（階層関係、ROOTノード除外）
        for node_id, node_data in tree_data['nodes'].items():
            # ROOTノードをスキップ
            if 'root' in node_id.lower():
                continue
                
            if 'source_documents' in node_data:
                for source_doc in node_data['source_documents']:
                    if source_doc in tree_data['nodes'] and 'root' not in source_doc.lower():
                        G.add_edge(source_doc, node_id)
        
        return G, node_levels, node_sizes, node_contents
    
    def plot_hierarchical_tree(self, tree_data, filename):
        """階層構造の可視化"""
        G, node_levels, node_sizes, node_contents = self.create_tree_graph(tree_data)
        
        # 図のサイズを大きく設定
        plt.figure(figsize=(20, 16))
        
        # レベル別の位置設定（1-3レベルのみ）
        pos = {}
        level_counts = {level: 0 for level in range(1, 4)}
        level_positions = {level: [] for level in range(1, 4)}
        
        # 各レベルのノード数をカウント
        for node_id, level in node_levels.items():
            level_counts[level] += 1
            
        # 位置を計算
        for node_id, level in node_levels.items():
            level_positions[level].append(node_id)
            
        # Y座標をレベルに基づいて設定（左側スペースを削減）
        for level in range(1, 4):
            nodes_at_level = level_positions[level]
            if nodes_at_level:
                width = max(len(nodes_at_level) * 3, 10)
                # X座標を右寄りに調整（左側の余白を削減）
                x_positions = np.linspace(-width/2 + 2, width/2 + 2, len(nodes_at_level))
                for i, node_id in enumerate(nodes_at_level):
                    pos[node_id] = (x_positions[i], level * 4)
        
        # ノードの色とサイズを設定
        node_colors = []
        node_size_list = []
        
        for node_id in G.nodes():
            level = node_levels[node_id]
            
            # 免疫学階層に基づく色分け（ROOTレベル除外）
            if level == 3:
                node_colors.append(LEVEL_COLOR_MAPPING[3]['color'])  # Treg
            elif level == 2:
                node_colors.append(LEVEL_COLOR_MAPPING[2]['color'])  # CD4+T
            elif level == 1:
                node_colors.append(LEVEL_COLOR_MAPPING[1]['color'])  # CLP
            else:
                node_colors.append(LEVEL_COLOR_MAPPING[0]['color'])  # HSC
                
            # ノードサイズをクラスターサイズに比例
            size = min(node_sizes.get(node_id, 1) * 100, 2000)
            node_size_list.append(max(size, 300))
        
        # グラフ描画
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=node_size_list,
                with_labels=False,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                alpha=0.8,
                width=2)
        
        # ノードラベルを追加（免疫学用語に基づく、ROOTノード除外）
        for node_id, (x, y) in pos.items():
            level = node_levels[node_id]
            
            # ROOTノードをスキップ
            if 'root' in node_id.lower():
                continue
                
            if level > 0:  # レベル1以上のみラベル表示
                cluster_size = node_sizes.get(node_id, 1)
                
                # 免疫学的に正しいラベル生成
                node_content = tree_data['nodes'].get(node_id, {}).get('content', '')
                cluster_id = node_id.split('_C')[1].split('_')[0] if '_C' in node_id else '0'
                
                label = generate_immune_label(node_content, level, cluster_id, cluster_size)
                
                # 文字化け対策: 安全な文字列処理
                safe_label = ""
                for char in label:
                    if ord(char) < 128 or char in ['ー', '・', '+', '-', '(', ')', '\n']:
                        safe_label += char
                    else:
                        safe_label += "_"  # 不明文字を_に置換
                
                # 不要な文字を除去
                safe_label = safe_label.replace('□', '').replace('?', '')
                safe_label = safe_label.replace('___', '_').replace('__', '_')
                safe_label = safe_label.strip('_').strip()
                
                # 用語の妥当性検証
                is_valid, validation_msg = validate_immune_terminology(safe_label)
                
                # ラベルが空または無効な場合のフォールバック
                if not safe_label or safe_label == "" or safe_label == "_":
                    level_info = LEVEL_COLOR_MAPPING.get(level, {"name": f"L{level}", "description": "Unknown"})
                    # 英語のみのフォールバックラベル
                    safe_label = f"{level_info['name']}\n(Size:{cluster_size})"
                
                plt.text(x, y, safe_label, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=9, fontweight='bold',
                        fontfamily='Arial',  # Arialフォント明示
                        bbox=dict(boxstyle="round,pad=0.4", 
                                facecolor='white', alpha=0.95, 
                                edgecolor='gray', linewidth=1.2))
        
        # 右上に凡例を追加（免疫学的階層説明、ROOTレベル除外）
        legend_y_start = 0.95
        legend_spacing = 0.08
        
        # 英語のみの安全な凡例ラベル
        safe_legend_labels = {
            1: "Level 1: CLP - Common Lymphoid Progenitor",
            2: "Level 2: CD4+T - CD4 Positive T Cell", 
            3: "Level 3: Treg - Regulatory T Cell"
        }
        
        for i, level in enumerate([1, 2, 3]):  # ROOTレベル（4）を除外
            if level in LEVEL_COLOR_MAPPING:
                level_info = LEVEL_COLOR_MAPPING[level]
                # ASCII文字のみを使用してラベルを構築
                legend_item = safe_legend_labels[level]
                
                plt.text(0.75, legend_y_start - (i * legend_spacing), legend_item, 
                        transform=plt.gca().transAxes,
                        fontsize=10, fontweight='bold',
                        fontfamily='Arial',
                        verticalalignment='top',
                        color=level_info['color'],
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', alpha=0.8, edgecolor=level_info['color']))
        
        plt.title("Immune Cell Differentiation Tree (HSC-CLP-CD4T-Treg) - Hierarchical Analysis", 
                 fontsize=16, fontweight='bold', pad=20, family='Arial')
        
        # 統計情報を左上に移動
        total_nodes = len(G.nodes())
        max_level = max(node_levels.values())
        max_cluster = max(node_sizes.values())
        
        stats_lines = [
            f"Total Nodes: {total_nodes}",
            f"Hierarchy Levels: {max_level}",
            f"Max Cluster Size: {max_cluster}"
        ]
        stats_text = "\n".join(stats_lines)
        
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes,
                fontsize=10,
                fontfamily='Arial',
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 階層ツリー可視化を保存: {filename}")
    
    def plot_cluster_statistics(self, tree_data, filename):
        """クラスター統計の可視化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # レベル別ノード数
        level_counts = {}
        cluster_sizes = []
        levels = []
        
        for node_id, node_data in tree_data['nodes'].items():
            # ROOTノードをスキップ
            if 'root' in node_id.lower():
                continue
            elif '_L3_' in node_id:
                level = 3
            elif '_L2_' in node_id:
                level = 2
            elif '_L1_' in node_id:
                level = 1
            else:
                continue
                
            level_counts[level] = level_counts.get(level, 0) + 1
            cluster_sizes.append(node_data.get('cluster_size', 1))
            levels.append(level)
        
        # 1. レベル別ノード数
        ax1.bar(level_counts.keys(), level_counts.values(), 
               color=[self.colors[f'level_{k}'] if k != 4 else self.colors['root'] 
                     for k in level_counts.keys()])
        ax1.set_xlabel('Tree Level', fontfamily='Arial')
        ax1.set_ylabel('Number of Nodes', fontfamily='Arial')
        ax1.set_title('Nodes per Level', fontfamily='Arial')
        ax1.grid(True, alpha=0.3)
        
        # 2. クラスターサイズ分布
        ax2.hist(cluster_sizes, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Cluster Size', fontfamily='Arial')
        ax2.set_ylabel('Frequency', fontfamily='Arial')
        ax2.set_title('Cluster Size Distribution', fontfamily='Arial')
        ax2.grid(True, alpha=0.3)
        
        # 3. レベル別クラスターサイズ
        level_cluster_data = {}
        for level, size in zip(levels, cluster_sizes):
            if level not in level_cluster_data:
                level_cluster_data[level] = []
            level_cluster_data[level].append(size)
        
        box_data = [level_cluster_data.get(level, []) for level in sorted(level_cluster_data.keys())]
        ax3.boxplot(box_data, labels=sorted(level_cluster_data.keys()))
        ax3.set_xlabel('Tree Level', fontfamily='Arial')
        ax3.set_ylabel('Cluster Size', fontfamily='Arial')
        ax3.set_title('Cluster Size by Level', fontfamily='Arial')
        ax3.grid(True, alpha=0.3)
        
        # 4. 改善比較（ROOTノード除外）
        comparison_data = {
            'Previous System': 5,
            'New RAPTOR': len([n for n in tree_data['nodes'].keys() 
                             if any(x in n for x in ['_L']) and 'root' not in n.lower()])
        }
        
        bars = ax4.bar(comparison_data.keys(), comparison_data.values(), 
                      color=['#FF6B6B', '#4ECDC4'])
        ax4.set_ylabel('Number of Nodes', fontfamily='Arial')
        ax4.set_title('System Comparison', fontfamily='Arial')
        
        # 改善率を表示
        improvement = (comparison_data['New RAPTOR'] / comparison_data['Previous System'] - 1) * 100
        improvement_text = f'+{improvement:.0f}% improvement'
        ax4.text(0.5, max(comparison_data.values()) * 0.8, 
                improvement_text, 
                ha='center', fontsize=12, fontweight='bold', fontfamily='Arial',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        for bar, value in zip(bars, comparison_data.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold', fontfamily='Arial')
        
        plt.suptitle('RAPTOR Tree Analysis Dashboard', fontsize=16, fontweight='bold', fontfamily='Arial')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 統計分析を保存: {filename}")

def main():
    print("🎨 RAPTOR Tree Visualization Tool")
    print("=" * 50)
    
    visualizer = RAPTORTreeVisualizer()
    
    try:
        # ツリーデータを読み込み
        tree_data, filename = visualizer.load_latest_tree()
        
        print(f"📊 Tree loaded: {len(tree_data['nodes'])} nodes")
        
        # 可視化ファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tree_viz_file = f"raptor_tree_visualization_{timestamp}.png"
        stats_viz_file = f"raptor_statistics_{timestamp}.png"
        
        # 1. 階層ツリー可視化
        print("\n🌳 Creating hierarchical tree visualization...")
        visualizer.plot_hierarchical_tree(tree_data, tree_viz_file)
        
        # 2. 統計分析可視化
        print("\n📊 Creating statistical analysis...")
        visualizer.plot_cluster_statistics(tree_data, stats_viz_file)
        
        print(f"\n✅ Visualization completed!")
        print(f"📁 Files saved:")
        print(f"   🌳 Tree structure: {tree_viz_file}")
        print(f"   📊 Statistics: {stats_viz_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()