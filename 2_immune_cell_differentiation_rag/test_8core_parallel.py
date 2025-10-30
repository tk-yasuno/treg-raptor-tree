"""
8-Core Parallel Processing Test for Immune Cell RAPTOR Tree
8コア並列処理による性能テストとベンチマーク

Author: AI Assistant
Date: 2025-10-31
"""

import time
import json
import multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from immune_raptor_tree import ImmuneCellRAPTORTree


def run_8core_parallel_test():
    """8コア並列処理テスト実行"""
    
    print("🚀 8-Core Parallel Processing Test")
    print("=" * 50)
    print(f"Available CPU cores: {mp.cpu_count()}")
    print(f"Target workers: 8")
    
    # システム初期化
    start_time = time.time()
    
    # 絶対パスでRAPTOR Tree初期化
    base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
    cache_dir = base_dir / "data/immune_cell_differentiation"
    raptor_tree = ImmuneCellRAPTORTree(str(cache_dir))
    
    print("\n📁 Loading immune cell hierarchy...")
    immune_file = cache_dir / "immune_cell_hierarchy.json"
    raptor_tree.load_immune_hierarchy(str(immune_file))
    print(f"✓ Loaded {len(raptor_tree.nodes)} immune cell nodes")
    
    print("\n🔧 Building FAISS index with 8-core parallel processing...")
    index_start = time.time()
    raptor_tree.build_faiss_index_parallel(workers=8)
    index_time = time.time() - index_start
    print(f"✓ FAISS index built in {index_time:.2f} seconds")
    
    print("\n⚡ Running optimized 8-core PubMed integration...")
    pubmed_start = time.time()
    
    # 8コア最適化並列処理実行
    parallel_metrics = raptor_tree.integrate_pubmed_knowledge_parallel_optimized(
        max_articles_per_query=30,
        workers=8
    )
    
    pubmed_time = time.time() - pubmed_start
    
    print(f"\n🎯 8-Core Test Results:")
    print("-" * 30)
    print(f"FAISS index time: {index_time:.2f}s")
    print(f"PubMed parallel time: {pubmed_time:.2f}s")
    print(f"Total articles: {parallel_metrics['articles_count']}")
    print(f"Speedup achieved: {parallel_metrics['speedup']:.1f}x")
    print(f"Parallel efficiency: {parallel_metrics['speedup']/8*100:.0f}%")
    
    # 階層検索デモ
    print("\n🔍 Testing hierarchical search...")
    search_queries = [
        "FOXP3+ regulatory T cell differentiation",
        "Treg immune suppression mechanisms",
        "nTreg vs iTreg comparison"
    ]
    
    search_results = {}
    for query in search_queries:
        results = raptor_tree.hierarchical_search(query, max_depth=4)
        search_results[query] = results
        print(f"\\nQuery: '{query}'")
        for i, (node, score) in enumerate(results, 1):
            print(f"  {i}. {node.cell_type} ({node.subtype}) - Score: {score:.3f}")
    
    # 可視化生成
    print("\n🎨 Generating hierarchy visualization...")
    output_viz = cache_dir / "raptor_trees/immune_hierarchy_8core_parallel.png"
    raptor_tree.visualize_hierarchy(str(output_viz))
    print(f"✓ Visualization saved: {output_viz.name}")
    
    # ファイル保存
    print("\n💾 Saving optimized RAPTOR Tree...")
    output_tree = cache_dir / "raptor_trees/immune_cell_raptor_tree_8core.json"
    raptor_tree.save_raptor_tree(str(output_tree))
    
    # 埋め込みベクトル保存
    import pickle
    embeddings_file = cache_dir / "raptor_trees/immune_cell_raptor_tree_8core_embeddings.pkl"
    embeddings_data = {
        'node_embeddings': {node_id: node.embedding for node_id, node in raptor_tree.nodes.items()},
        'article_embeddings': raptor_tree.article_embeddings,
        'parallel_metrics': parallel_metrics
    }
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"✓ Embeddings saved: {embeddings_file.name}")
    
    # FAISS インデックス保存
    import faiss
    if raptor_tree.faiss_index is not None:
        faiss_file = cache_dir / "raptor_trees/immune_cell_raptor_tree_8core_faiss.index"
        faiss.write_index(raptor_tree.faiss_index, str(faiss_file))
        print(f"✓ FAISS index saved: {faiss_file.name}")
    
    total_time = time.time() - start_time
    
    # 性能比較データ作成
    results = {
        "test_type": "8-core_parallel_optimized",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "system_info": {
            "cpu_cores": mp.cpu_count(),
            "workers_used": 8,
            "python_version": "3.11"
        },
        "performance": {
            "total_time": total_time,
            "index_time": index_time,
            "pubmed_time": pubmed_time,
            "articles_count": parallel_metrics['articles_count'],
            "speedup": parallel_metrics['speedup'],
            "efficiency": parallel_metrics['speedup']/8*100
        },
        "comparison_baseline": {
            "sequential_1x": {"time": 50.8, "articles": 133},
            "sequential_2x": {"time": 66.2, "articles": 248}
        }
    }
    
    # 結果保存
    results_file = f"8core_parallel_test_results_{results['timestamp']}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # 性能比較可視化
    create_performance_comparison(results)
    
    print("\n" + "=" * 60)
    print("🎉 8-Core Parallel Processing Test Completed!")
    print(f"📊 Total execution time: {total_time:.2f} seconds")
    print(f"⚡ Speedup vs sequential: {parallel_metrics['speedup']:.1f}x")
    print(f"🎯 Parallel efficiency: {parallel_metrics['speedup']/8*100:.0f}%")
    print(f"📁 Files saved in: {cache_dir}/raptor_trees/")
    print("=" * 60)
    
    return results


def create_performance_comparison(results):
    """性能比較可視化作成"""
    
    # データ準備
    labels = ['Sequential\\n1x', 'Sequential\\n2x', '8-Core\\nParallel']
    times = [
        results['comparison_baseline']['sequential_1x']['time'],
        results['comparison_baseline']['sequential_2x']['time'],
        results['performance']['total_time']
    ]
    articles = [
        results['comparison_baseline']['sequential_1x']['articles'],
        results['comparison_baseline']['sequential_2x']['articles'],
        results['performance']['articles_count']
    ]
    
    # 可視化作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 実行時間比較
    colors = ['skyblue', 'lightgreen', 'gold']
    bars1 = ax1.bar(labels, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 時間ラベル
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time:.1f}s', ha='center', va='bottom')
    
    # 2. 記事数比較
    bars2 = ax2.bar(labels, articles, color=colors, alpha=0.8)
    ax2.set_ylabel('Articles Retrieved')
    ax2.set_title('Articles Count Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 記事数ラベル
    for bar, count in zip(bars2, articles):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom')
    
    # 3. スループット比較
    throughputs = [a/t for a, t in zip(articles, times)]
    bars3 = ax3.bar(labels, throughputs, color=colors, alpha=0.8)
    ax3.set_ylabel('Throughput (articles/second)')
    ax3.set_title('Processing Throughput')
    ax3.grid(True, alpha=0.3)
    
    # スループットラベル
    for bar, throughput in zip(bars3, throughputs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{throughput:.1f}', ha='center', va='bottom')
    
    # 4. 効率性指標
    efficiency_labels = ['Speedup', 'Efficiency (%)', 'Workers Used']
    efficiency_values = [
        results['performance']['speedup'],
        results['performance']['efficiency'],
        8
    ]
    
    bars4 = ax4.bar(efficiency_labels, efficiency_values, 
                   color=['orange', 'red', 'purple'], alpha=0.8)
    ax4.set_ylabel('Value')
    ax4.set_title('8-Core Parallel Efficiency')
    ax4.grid(True, alpha=0.3)
    
    # 効率性ラベル
    for bar, value in zip(bars4, efficiency_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    viz_file = f"8core_parallel_performance_{results['timestamp']}.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Performance visualization saved: {viz_file}")


def compare_with_baseline():
    """ベースライン性能との比較分析"""
    
    print("\\n📈 PERFORMANCE COMPARISON ANALYSIS:")
    print("=" * 50)
    
    # ベースラインデータ
    baseline_1x = {"time": 50.8, "articles": 133, "scale": 1}
    baseline_2x = {"time": 66.2, "articles": 248, "scale": 2}
    
    # 予想される8コア並列性能（推定）
    estimated_parallel_time = baseline_2x["time"] / 3  # 3x speedup期待
    
    print(f"Baseline Sequential Performance:")
    print(f"  1x scale: {baseline_1x['time']:.1f}s, {baseline_1x['articles']} articles")
    print(f"  2x scale: {baseline_2x['time']:.1f}s, {baseline_2x['articles']} articles")
    print(f"")
    print(f"Expected 8-Core Parallel Performance:")
    print(f"  2x scale: ~{estimated_parallel_time:.1f}s, {baseline_2x['articles']} articles")
    print(f"  Expected speedup: ~3.0x")
    print(f"  Expected efficiency: ~37.5%")


if __name__ == "__main__":
    # ベースライン比較表示
    compare_with_baseline()
    
    # 8コア並列テスト実行
    results = run_8core_parallel_test()