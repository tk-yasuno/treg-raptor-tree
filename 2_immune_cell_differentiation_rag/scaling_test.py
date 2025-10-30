"""
Scaling Test for Immune Cell RAPTOR Tree System
免疫細胞RAPTOR Tree システムのスケーリング性能テスト

Author: AI Assistant
Date: 2025-10-31
"""

import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

from immune_raptor_tree import ImmuneCellRAPTORTree, ImmuneCellEmbedder
from pubmed_retriever import ImmuneCellPubMedRetriever


class ScalingTestRunner:
    """スケーリングテスト実行クラス"""
    
    def __init__(self, test_name: str = "scaling_test"):
        self.test_name = test_name
        self.results = {}
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
    def run_single_scale_test(self, scale_factor: int, max_articles: int = 30) -> Dict:
        """単一スケールでのテスト実行"""
        
        print(f"\n🧪 Running test with scale factor: {scale_factor}x")
        print(f"   Articles per query: {max_articles * scale_factor}")
        
        start_time = time.time()
        
        # RAPTOR Tree構築
        raptor_tree = ImmuneCellRAPTORTree()
        
        # データ読み込み
        load_start = time.time()
        raptor_tree.load_immune_hierarchy("../data/immune_cell_differentiation/immune_cell_hierarchy.json")
        load_time = time.time() - load_start
        
        # FAISSインデックス構築
        index_start = time.time()
        raptor_tree.build_embeddings()
        index_time = time.time() - index_start
        
        # PubMed統合（スケール調整）
        pubmed_start = time.time()
        raptor_tree.integrate_pubmed_knowledge(max_articles * scale_factor)
        pubmed_time = time.time() - pubmed_start
        
        # 可視化
        viz_start = time.time()
        raptor_tree.visualize_hierarchy(f"hierarchy_viz_scale_{scale_factor}x.png")
        viz_time = time.time() - viz_start
        
        total_time = time.time() - start_time
        
        result = {
            "scale_factor": scale_factor,
            "articles_per_query": max_articles * scale_factor,
            "total_articles": len(raptor_tree.pubmed_articles),
            "times": {
                "load": load_time,
                "index": index_time,
                "pubmed": pubmed_time,
                "visualization": viz_time,
                "total": total_time
            },
            "memory_usage": self._get_memory_usage(),
            "cpu_count": mp.cpu_count()
        }
        
        print(f"✅ Scale {scale_factor}x completed in {total_time:.2f} seconds")
        print(f"   Articles retrieved: {result['total_articles']}")
        
        return result
    
    def run_parallel_pubmed_test(self, scale_factor: int, max_articles: int = 30, workers: int = 8) -> Dict:
        """並列処理でのPubMed取得テスト"""
        
        print(f"\n🚀 Running PARALLEL test with {workers} workers")
        print(f"   Scale factor: {scale_factor}x, Articles per query: {max_articles * scale_factor}")
        
        start_time = time.time()
        
        # RAPTOR Tree構築
        raptor_tree = ImmuneCellRAPTORTree()
        raptor_tree.load_immune_hierarchy("../data/immune_cell_differentiation/immune_cell_hierarchy.json")
        raptor_tree.build_embeddings()
        
        # 並列PubMed取得
        pubmed_start = time.time()
        retriever = ImmuneCellPubMedRetriever(raptor_tree.cache_dir)
        
        # クエリを並列実行
        queries = list(retriever.immune_queries.items())
        
        def fetch_query_parallel(query_info):
            query, description = query_info
            return retriever.search_pubmed(query, max_articles * scale_factor)
        
        # 並列実行
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(fetch_query_parallel, queries))
        
        # 結果統合
        all_articles = {}
        for articles in results:
            for article in articles:
                if article.pmid not in all_articles:
                    all_articles[article.pmid] = article
        
        raptor_tree.pubmed_articles = all_articles
        
        # 並列ベクトル化
        def encode_article_parallel(item):
            pmid, article = item
            embedding = raptor_tree.embedder.encode_pubmed_article(article)
            return pmid, embedding
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            embedding_results = list(executor.map(encode_article_parallel, all_articles.items()))
        
        raptor_tree.article_embeddings = dict(embedding_results)
        
        pubmed_time = time.time() - pubmed_start
        total_time = time.time() - start_time
        
        result = {
            "scale_factor": scale_factor,
            "articles_per_query": max_articles * scale_factor,
            "total_articles": len(raptor_tree.pubmed_articles),
            "workers": workers,
            "times": {
                "pubmed_parallel": pubmed_time,
                "total": total_time
            },
            "memory_usage": self._get_memory_usage(),
            "cpu_count": mp.cpu_count()
        }
        
        print(f"✅ Parallel test completed in {total_time:.2f} seconds")
        print(f"   PubMed parallel time: {pubmed_time:.2f} seconds")
        print(f"   Articles retrieved: {result['total_articles']}")
        
        return result
    
    def run_scaling_comparison(self, scale_factors: List[int] = [1, 2], max_articles: int = 30) -> Dict:
        """スケーリング比較テスト"""
        
        print(f"\n🔬 Starting scaling comparison test")
        print(f"Scale factors: {scale_factors}")
        print(f"Base articles per query: {max_articles}")
        
        results = {
            "test_info": {
                "name": self.test_name,
                "timestamp": self.timestamp,
                "scale_factors": scale_factors,
                "base_articles": max_articles,
                "cpu_cores": mp.cpu_count()
            },
            "sequential_results": [],
            "parallel_results": []
        }
        
        # シーケンシャル実行
        print("\\n📊 Sequential execution tests:")
        for scale in scale_factors:
            result = self.run_single_scale_test(scale, max_articles)
            results["sequential_results"].append(result)
        
        # 並列実行（スケール2xで比較）
        print("\\n⚡ Parallel execution tests:")
        for scale in scale_factors:
            if scale >= 2:  # 並列処理は負荷が高い場合のみ
                result = self.run_parallel_pubmed_test(scale, max_articles, workers=8)
                results["parallel_results"].append(result)
        
        # 結果分析
        self._analyze_results(results)
        
        # 結果保存
        self._save_results(results)
        
        return results
    
    def _analyze_results(self, results: Dict):
        """結果分析"""
        
        print("\\n📈 SCALING ANALYSIS:")
        print("=" * 50)
        
        seq_results = results["sequential_results"]
        
        if len(seq_results) >= 2:
            base_time = seq_results[0]["times"]["total"]
            scale_2x_time = seq_results[1]["times"]["total"]
            
            scaling_ratio = scale_2x_time / base_time
            print(f"Base (1x): {base_time:.2f} seconds")
            print(f"Scale (2x): {scale_2x_time:.2f} seconds")
            print(f"Scaling ratio: {scaling_ratio:.2f}x")
            
            if scaling_ratio > 1.8:
                print("⚠️  Nearly linear scaling detected!")
                print("💡 Recommendation: Use parallel processing for better performance")
            else:
                print("✅ Sub-linear scaling - current implementation is efficient")
        
        # 並列処理比較
        if results["parallel_results"]:
            par_result = results["parallel_results"][0]
            seq_result = next(r for r in seq_results if r["scale_factor"] == par_result["scale_factor"])
            
            speedup = seq_result["times"]["total"] / par_result["times"]["total"]
            print(f"\\n⚡ PARALLEL PROCESSING:")
            print(f"Sequential (2x): {seq_result['times']['total']:.2f} seconds")
            print(f"Parallel (2x): {par_result['times']['total']:.2f} seconds")
            print(f"Speedup: {speedup:.2f}x")
    
    def _get_memory_usage(self) -> Dict:
        """メモリ使用量取得"""
        try:
            import psutil
            process = psutil.Process()
            return {
                "rss": process.memory_info().rss / 1024 / 1024,  # MB
                "vms": process.memory_info().vms / 1024 / 1024   # MB
            }
        except ImportError:
            return {"rss": 0, "vms": 0}
    
    def _save_results(self, results: Dict):
        """結果保存"""
        
        results_file = f"scaling_test_results_{self.timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\\n💾 Results saved to: {results_file}")
        
        # 可視化
        self._create_visualization(results)
    
    def _create_visualization(self, results: Dict):
        """結果可視化"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 総実行時間比較
        seq_results = results["sequential_results"]
        if len(seq_results) >= 2:
            scales = [r["scale_factor"] for r in seq_results]
            times = [r["times"]["total"] for r in seq_results]
            
            ax1.plot(scales, times, 'o-', label='Sequential', linewidth=2, markersize=8)
            ax1.set_xlabel('Scale Factor')
            ax1.set_ylabel('Total Time (seconds)')
            ax1.set_title('Scaling Performance')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. 処理段階別時間
        if seq_results:
            stages = ['load', 'index', 'pubmed', 'visualization']
            for i, result in enumerate(seq_results):
                times = [result["times"].get(stage, 0) for stage in stages]
                ax2.bar([f"{stage}\\n({result['scale_factor']}x)" for stage in stages], 
                       times, alpha=0.7, label=f"Scale {result['scale_factor']}x")
            
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title('Processing Stage Times')
            ax2.legend()
        
        # 3. 記事数とスケール
        articles = [r["total_articles"] for r in seq_results]
        scales = [r["scale_factor"] for r in seq_results]
        
        ax3.bar(scales, articles, alpha=0.7, color='skyblue')
        ax3.set_xlabel('Scale Factor')
        ax3.set_ylabel('Total Articles Retrieved')
        ax3.set_title('Articles vs Scale Factor')
        ax3.grid(True, alpha=0.3)
        
        # 4. 並列処理比較
        if results["parallel_results"]:
            par_times = [r["times"]["total"] for r in results["parallel_results"]]
            seq_times = [r["times"]["total"] for r in seq_results if r["scale_factor"] >= 2]
            
            x = np.arange(len(par_times))
            width = 0.35
            
            ax4.bar(x - width/2, seq_times, width, label='Sequential', alpha=0.7)
            ax4.bar(x + width/2, par_times, width, label='Parallel (8 workers)', alpha=0.7)
            ax4.set_xlabel('Scale Factor')
            ax4.set_ylabel('Total Time (seconds)')
            ax4.set_title('Sequential vs Parallel')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No parallel results', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Sequential vs Parallel')
        
        plt.tight_layout()
        viz_file = f"scaling_test_visualization_{self.timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {viz_file}")


def main():
    """メイン実行"""
    
    print("🧬 Immune Cell RAPTOR Tree Scaling Test")
    print("=" * 50)
    
    # テスト実行
    runner = ScalingTestRunner("immune_raptor_scaling")
    
    # 1x と 2x でスケーリングテスト
    results = runner.run_scaling_comparison(
        scale_factors=[1, 2],
        max_articles=15  # 基本記事数を少し減らしてテスト時間短縮
    )
    
    print("\\n🎯 Test completed!")
    print("Check the generated files for detailed results and visualizations.")


if __name__ == "__main__":
    main()