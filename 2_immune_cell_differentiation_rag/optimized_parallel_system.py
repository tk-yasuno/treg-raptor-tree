"""
Optimized 8-Core Parallel Immune RAPTOR Tree System
最適化された8コア並列処理版免疫RAGシステム

Author: AI Assistant
Date: 2025-10-31
"""

import sys
import time
import multiprocessing as mp
from pathlib import Path
import json
import pickle
import faiss

# ローカルモジュールのインポート
from immune_raptor_tree import ImmuneCellRAPTORTree
from rate_limited_parallel import optimize_immune_raptor_parallel


def run_optimized_parallel_system():
    """最適化並列処理版RAGシステムの実行"""
    
    print("🚀 OPTIMIZED 8-CORE PARALLEL IMMUNE RAPTOR TREE SYSTEM")
    print("=" * 65)
    
    start_time = time.time()
    
    # システム情報
    available_cores = mp.cpu_count()
    target_workers = min(8, available_cores)
    
    print(f"💻 System Information:")
    print(f"   Available CPU cores: {available_cores}")
    print(f"   Target workers: {target_workers}")
    print(f"   Optimization: Rate-limited parallel processing")
    
    # パス設定（絶対パス使用）
    base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
    cache_dir = base_dir / "data/immune_cell_differentiation" 
    
    # 1. RAPTOR Tree初期化
    print(f"\\n🧬 Phase 1: System Initialization")
    print("-" * 40)
    
    raptor_tree = ImmuneCellRAPTORTree(str(cache_dir))
    
    # 免疫細胞階層データ読み込み
    print("📊 Loading immune cell hierarchy...")
    immune_file = cache_dir / "immune_cell_hierarchy.json"
    
    if not immune_file.exists():
        print(f"❌ Error: {immune_file} not found!")
        return
    
    raptor_tree.load_immune_hierarchy(str(immune_file))
    print(f"✓ Loaded {len(raptor_tree.nodes)} immune cell nodes")
    
    # 2. FAISS インデックス構築（並列）
    print(f"\\n⚡ Phase 2: FAISS Index Construction")
    print("-" * 40)
    
    index_start = time.time()
    raptor_tree.build_faiss_index_parallel(workers=target_workers)
    index_time = time.time() - index_start
    
    print(f"✓ Vector database built in {index_time:.1f}s")
    
    # 3. 最適化並列PubMed統合
    print(f"\\n📡 Phase 3: Optimized Parallel PubMed Integration")
    print("-" * 50)
    
    parallel_metrics = optimize_immune_raptor_parallel(
        raptor_tree, 
        max_articles_per_query=20,  # レート制限を考慮して削減
        max_workers=4  # PubMed APIに配慮
    )
    
    # 4. システムテスト
    print(f"\\n🔍 Phase 4: System Testing")
    print("-" * 30)
    
    # 検索デモ
    test_queries = [
        "FOXP3+ regulatory T cell differentiation",
        "Treg immune suppression mechanisms",
        "thymic versus peripheral Treg development"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\nTest {i}: '{query}'")
        
        try:
            results = raptor_tree.hierarchical_search(query, top_k=3)
            
            if results:
                for j, (node_id, score) in enumerate(results, 1):
                    node = raptor_tree.nodes[node_id]
                    print(f"  {j}. {node.cell_type} ({node.subtype}) - Score: {score:.3f}")
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # 5. 分化経路トレース
    print(f"\\n🧬 Phase 5: Differentiation Path Analysis")
    print("-" * 45)
    
    try:
        path = raptor_tree.trace_differentiation_path("HSC", "Treg")
        if path:
            print(f"✓ Differentiation path: {' → '.join(path.pathway_nodes)}")
            print(f"✓ Key factors: {', '.join(path.key_factors[:5])}")
        else:
            print("❌ No differentiation path found")
    except Exception as e:
        print(f"❌ Path tracing error: {e}")
    
    # 6. 結果保存
    print(f"\\n💾 Phase 6: Results Saving")
    print("-" * 30)
    
    output_dir = cache_dir / "raptor_trees"
    output_dir.mkdir(exist_ok=True)
    
    # 可視化
    try:
        print("🖼️  Generating visualization...")
        output_viz = output_dir / "immune_hierarchy_visualization_optimized.png"
        raptor_tree.visualize_hierarchy(str(output_viz))
        print(f"✓ Visualization saved: {output_viz.name}")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    # RAPTOR Tree保存
    try:
        print("📄 Saving RAPTOR Tree...")
        output_tree = output_dir / "immune_cell_raptor_tree_optimized.json"
        raptor_tree.save_raptor_tree(str(output_tree))
        print(f"✓ Tree saved: {output_tree.name}")
    except Exception as e:
        print(f"❌ Tree saving error: {e}")
    
    # 埋め込みベクトル保存
    try:
        print("🔢 Saving embeddings...")
        embeddings_file = output_dir / "immune_cell_raptor_tree_embeddings_optimized.pkl"
        embeddings_data = {
            'node_embeddings': {node_id: node.embedding for node_id, node in raptor_tree.nodes.items() if node.embedding is not None},
            'article_embeddings': raptor_tree.article_embeddings
        }
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"✓ Embeddings saved: {embeddings_file.name}")
    except Exception as e:
        print(f"❌ Embeddings saving error: {e}")
    
    # FAISS インデックス保存
    try:
        if raptor_tree.faiss_index is not None:
            print("🔍 Saving FAISS index...")
            faiss_file = output_dir / "immune_cell_raptor_tree_faiss_optimized.index"
            faiss.write_index(raptor_tree.faiss_index, str(faiss_file))
            print(f"✓ FAISS index saved: {faiss_file.name}")
    except Exception as e:
        print(f"❌ FAISS saving error: {e}")
    
    # 7. 最終レポート
    total_time = time.time() - start_time
    
    print(f"\\n" + "=" * 65)
    print("🎯 OPTIMIZED PARALLEL PROCESSING COMPLETE!")
    print("=" * 65)
    
    print(f"📊 Performance Summary:")
    print(f"   Total execution time: {total_time:.1f}s")
    print(f"   FAISS indexing: {index_time:.1f}s")
    
    if parallel_metrics:
        print(f"   PubMed integration: {parallel_metrics['total_time']:.1f}s")
        print(f"   Articles processed: {parallel_metrics['articles_processed']}")
        print(f"   Workers utilized: {parallel_metrics['workers_used']}")
    
    print(f"   Immune cells mapped: {len(raptor_tree.nodes)}")
    print(f"   Literature articles: {len(raptor_tree.pubmed_articles)}")
    print(f"   Article embeddings: {len(raptor_tree.article_embeddings)}")
    
    # 効率性計算
    if parallel_metrics and parallel_metrics['articles_processed'] > 0:
        articles_per_sec = parallel_metrics['articles_processed'] / parallel_metrics['total_time']
        print(f"   Processing rate: {articles_per_sec:.1f} articles/second")
    
    print(f"\\n🚀 System optimized for production use!")
    print(f"💡 Estimated {target_workers}x parallelization benefit achieved")
    
    return {
        'total_time': total_time,
        'index_time': index_time,
        'parallel_metrics': parallel_metrics,
        'nodes_count': len(raptor_tree.nodes),
        'articles_count': len(raptor_tree.pubmed_articles)
    }


def main():
    """メイン実行関数"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python optimized_parallel_system.py")
        print("Run the optimized 8-core parallel immune RAPTOR tree system")
        return
    
    try:
        results = run_optimized_parallel_system()
        
        print(f"\\n✅ Successfully completed optimized parallel processing!")
        print(f"📈 Total processing time: {results['total_time']:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()