"""
統合実行スクリプト
Immune Cell Differentiation RAPTOR Tree RAG System

免疫細胞分化系譜のRAPTOR Tree構築からMIRAGE評価まで一括実行
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

def main():
    """メイン実行関数"""
    
    print("🧬 Immune Cell Differentiation RAPTOR Tree RAG System")
    print("=" * 70)
    print("Starting full pipeline execution...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # GPU環境確認
    print("Step 1: GPU Environment Check")
    print("-" * 30)
    try:
        from check_gpu import check_gpu_environment, get_optimal_config, save_config
        cuda_available, gpu_count = check_gpu_environment()
        
        # 最適設定生成
        import torch
        gpu_memory = 0
        if cuda_available and gpu_count > 0:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        config = get_optimal_config(cuda_available, gpu_memory)
        save_config(config)
        print("✅ GPU environment check completed")
        
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")
        print("Continuing with default CPU configuration...")
    
    print("\n" + "=" * 70)
    
    # RAPTOR Tree構築
    print("Step 2: RAPTOR Tree Construction")
    print("-" * 30)
    try:
        from immune_raptor_tree import main as build_raptor_tree
        build_raptor_tree()
        print("✅ RAPTOR Tree construction completed")
        
    except Exception as e:
        print(f"❌ RAPTOR Tree construction failed: {e}")
        print("Please check dependencies and data files")
        return False
    
    print("\n" + "=" * 70)
    
    # MIRAGE評価実行
    print("Step 3: MIRAGE Evaluation")
    print("-" * 30)
    try:
        from mirage_evaluator import main as run_mirage_evaluation
        run_mirage_evaluation()
        print("✅ MIRAGE evaluation completed")
        
    except Exception as e:
        print(f"❌ MIRAGE evaluation failed: {e}")
        print("RAPTOR Tree construction may not have completed successfully")
        return False
    
    print("\n" + "=" * 70)
    
    # 完了メッセージ
    print("🎯 All pipeline steps completed successfully!")
    print("")
    print("Generated files:")
    
    base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
    cache_dir = base_dir / "data/immune_cell_differentiation"
    
    # 出力ファイル確認
    output_files = [
        cache_dir / "raptor_trees/immune_cell_raptor_tree.json",
        cache_dir / "raptor_trees/immune_hierarchy_visualization.png",
        cache_dir / "evaluation_results" / "mirage_report_*.md"
    ]
    
    for file_pattern in output_files:
        if '*' in str(file_pattern):
            # グロブパターンでファイル検索
            parent_dir = file_pattern.parent
            pattern = file_pattern.name
            if parent_dir.exists():
                matching_files = list(parent_dir.glob(pattern))
                for file_path in matching_files[-3:]:  # 最新3ファイル
                    print(f"  📄 {file_path}")
        else:
            if file_pattern.exists():
                print(f"  📄 {file_pattern}")
    
    print("")
    print("🚀 System is ready for immune cell differentiation analysis!")
    print("")
    print("Next steps:")
    print("  1. Review evaluation reports in evaluation_results/")
    print("  2. Check hierarchy visualization")
    print("  3. Use the system for your research queries")
    print("")
    print("For custom queries, use:")
    print("  python immune_raptor_tree.py  # Interactive mode")
    print("  python mirage_evaluator.py    # Custom evaluation")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Pipeline execution completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline execution failed!")
        sys.exit(1)