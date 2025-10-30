#!/usr/bin/env python3
"""
Transformer RAPTOR Tree Structure Analysis
新しく生成されたクラスタリングツリーの分析
"""

import json
import sys
from pathlib import Path

def analyze_clustered_tree_structure():
    """新しく生成されたクラスタリングツリーの構造を分析"""
    
    # 最新のクラスタリングツリーファイルを探す
    raptor_dir = Path(r"C:\Users\yasun\LangChain\learning-langchain\treg-raptor-tree\data\immune_cell_differentiation\raptor_trees")
    
    # transformer_raptor_clustered_tree で始まるファイルを探す
    clustered_files = list(raptor_dir.glob("transformer_raptor_clustered_tree_*.json"))
    
    if not clustered_files:
        print("❌ No clustered tree files found")
        return
    
    # 最新のファイルを選択
    latest_file = max(clustered_files, key=lambda x: x.stat().st_mtime)
    
    print("🔍 TRANSFORMER RAPTOR CLUSTERED TREE ANALYSIS")
    print("=" * 60)
    print(f"📁 Analyzing file: {latest_file.name}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📊 Tree Metadata:")
        metadata = data.get('metadata', {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\n📊 Total nodes in clustered tree: {len(data['nodes'])}")
        
        # レベル別の統計
        level_stats = {}
        cluster_info = {}
        
        for node_id, node in data['nodes'].items():
            level = node.get('level', 0)
            cluster_id = node.get('cluster_id')
            
            if level not in level_stats:
                level_stats[level] = []
            level_stats[level].append(node_id)
            
            if cluster_id is not None and level not in cluster_info:
                cluster_info[level] = {}
            if cluster_id is not None:
                if cluster_id not in cluster_info[level]:
                    cluster_info[level][cluster_id] = []
                cluster_info[level][cluster_id].append(node_id)
        
        print(f"\n🌳 Tree Structure by Level:")
        print("-" * 40)
        for level in sorted(level_stats.keys()):
            nodes = level_stats[level]
            print(f"  Level {level}: {len(nodes)} nodes")
            
            # サンプルノードの詳細表示
            for i, node_id in enumerate(nodes[:3]):  # 最大3個まで表示
                node = data['nodes'][node_id]
                content_preview = node.get('content', '')[:100] + "..." if len(node.get('content', '')) > 100 else node.get('content', '')
                cluster_size = node.get('cluster_size', 0)
                print(f"    {node_id}:")
                print(f"      Cluster size: {cluster_size}")
                print(f"      Content: {content_preview}")
                print()
            
            if len(nodes) > 3:
                print(f"    ... and {len(nodes) - 3} more nodes\n")
        
        print(f"\n🔗 Clustering Summary:")
        print("-" * 30)
        for level in sorted(cluster_info.keys()):
            clusters = cluster_info[level]
            print(f"  Level {level}: {len(clusters)} clusters")
            for cluster_id, cluster_nodes in clusters.items():
                print(f"    Cluster {cluster_id}: {len(cluster_nodes)} nodes")
        
        # 比較結果
        print(f"\n✅ CLUSTERING SUCCESS COMPARISON:")
        print("=" * 50)
        print(f"  🔴 Previous system: 5 nodes (fixed hierarchy only)")
        print(f"  🟢 New system: {len(data['nodes'])} nodes ({len(data['nodes']) - 5} additional clustered nodes!)")
        print(f"  📈 Improvement: {((len(data['nodes']) - 5) / 5 * 100):.0f}% more nodes due to clustering")
        print(f"  🌳 Tree depth: {max(level_stats.keys())} levels")
        print(f"  🧮 Algorithm: {metadata.get('algorithm', 'Unknown')}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error analyzing clustered tree: {e}")
        return None

if __name__ == "__main__":
    analyze_clustered_tree_structure()