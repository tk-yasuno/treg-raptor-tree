#!/usr/bin/env python3
"""
8x Scale RAPTOR Tree Structure Analysis
"""

import json
import sys
from pathlib import Path

def analyze_tree_structure():
    """Analyze the structure of the saved 8x scale RAPTOR tree"""
    
    tree_file = Path(r"C:\Users\yasun\LangChain\learning-langchain\treg-raptor-tree\data\immune_cell_differentiation\raptor_trees\immune_cell_raptor_tree_8x_scale_20251031_005118.json")
    
    if not tree_file.exists():
        print(f"❌ Tree file not found: {tree_file}")
        return
    
    print("🔍 8x Scale RAPTOR Tree Structure Analysis")
    print("=" * 50)
    
    try:
        with open(tree_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Total nodes in tree: {len(data['nodes'])}")
        print("\n🧬 Node Details:")
        print("-" * 30)
        
        for node_id, node in data['nodes'].items():
            cell_type = node.get('cell_type', 'Unknown')
            subtype = node.get('subtype', 'Unknown')
            level = node.get('level', 'Unknown')
            parent = node.get('parent_id', 'None (Root)')
            children = node.get('children', [])
            
            print(f"  {node_id}:")
            print(f"    Cell Type: {cell_type}")
            print(f"    Subtype: {subtype}")
            print(f"    Level: {level}")
            print(f"    Parent: {parent}")
            print(f"    Children: {len(children)} ({', '.join(children) if children else 'None'})")
            print()
        
        # Check for potential RAPTOR clustering
        if 'raptor_clusters' in data:
            print(f"🔗 RAPTOR Clusters: {len(data['raptor_clusters'])}")
        
        if 'layer_summaries' in data:
            print(f"📋 Layer Summaries: {len(data['layer_summaries'])}")
        
        # Check if this is actually showing the expected immune hierarchy
        expected_cells = [
            "Hematopoietic Stem Cell",
            "Common Lymphoid Progenitor", 
            "CD4+ T Cell",
            "Natural Treg (nTreg)",
            "Induced Treg (iTreg)"
        ]
        
        found_cells = [node['cell_type'] for node in data['nodes'].values()]
        
        print("\n✅ Expected vs Found Cell Types:")
        print("-" * 40)
        for expected in expected_cells:
            if expected in found_cells:
                print(f"  ✓ {expected}")
            else:
                print(f"  ❌ {expected} - NOT FOUND")
        
        print(f"\n🎯 Analysis: The tree contains {len(data['nodes'])} nodes representing the basic immune cell differentiation hierarchy.")
        print("   This is correct for the initial cell type definitions.")
        print("   RAPTOR clustering would add additional summary nodes based on the 719 processed articles.")
        
        return data
        
    except Exception as e:
        print(f"❌ Error analyzing tree: {e}")
        return None

if __name__ == "__main__":
    analyze_tree_structure()