"""
Large-Scale Scaling Test for Optimized 8-Core Parallel Immune RAPTOR Tree System
最適化8コア並列処理システムの大規模スケーリングテスト（4x, 8x, 12x, 16x）

Author: AI Assistant
Date: 2025-10-31
"""

import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

# ローカルモジュールのインポート
from immune_raptor_tree import ImmuneCellRAPTORTree
from rate_limited_parallel import optimize_immune_raptor_parallel


class LargeScaleScalingTester:
    """大規模スケーリングテスト実行クラス"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or "C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
        self.cache_dir = self.base_dir / "data/immune_cell_differentiation"
        self.results_dir = self.cache_dir / "scaling_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # テストスケールとターゲット文献数
        self.scaling_tests = {
            "1x": {"articles_per_query": 20, "description": "ベースライン"},
            "4x": {"articles_per_query": 80, "description": "4倍スケール"},
            "8x": {"articles_per_query": 160, "description": "8倍スケール"},
            "12x": {"articles_per_query": 240, "description": "12倍スケール"},
            "16x": {"articles_per_query": 320, "description": "16倍スケール"}
        }
        
        self.results = {}
        
    def run_single_scale_test(self, scale: str, articles_per_query: int, workers: int = 4) -> dict:
        """単一スケールでのテスト実行"""
        
        print(f"\\n{'='*70}")
        print(f"🔬 SCALING TEST: {scale} ({articles_per_query} articles/query)")
        print(f"{'='*70}")
        
        test_start = time.time()
        
        # 1. システム初期化
        print("🧬 Phase 1: System Initialization")
        raptor_tree = ImmuneCellRAPTORTree(str(self.cache_dir))
        
        # 免疫細胞階層データ読み込み
        immune_file = self.cache_dir / "immune_cell_hierarchy.json"
        raptor_tree.load_immune_hierarchy(str(immune_file))
        print(f"✓ Loaded {len(raptor_tree.nodes)} immune cell nodes")
        
        # 2. FAISS インデックス構築
        index_start = time.time()
        raptor_tree.build_faiss_index_parallel(workers=workers)
        index_time = time.time() - index_start
        print(f"✓ FAISS index built in {index_time:.1f}s")
        
        # 3. 最適化並列PubMed統合（スケール調整）
        print(f"📡 Phase 2: Scaled PubMed Integration ({articles_per_query} articles/query)")
        
        integration_start = time.time()
        parallel_metrics = optimize_immune_raptor_parallel(
            raptor_tree,
            max_articles_per_query=articles_per_query,
            max_workers=workers
        )
        integration_time = time.time() - integration_start
        
        # 4. システムテスト
        print("🔍 Phase 3: System Testing")
        search_start = time.time()
        
        test_queries = [
            "FOXP3+ regulatory T cell differentiation",
            "Treg immune suppression mechanisms",
            "thymic versus peripheral Treg development"
        ]
        
        search_results = []
        for query in test_queries:
            try:
                results = raptor_tree.hierarchical_search(query, top_k=5)
                search_results.append(len(results))
            except Exception as e:
                print(f"Search error: {e}")
                search_results.append(0)
        
        search_time = time.time() - search_start
        
        # 5. 結果収集
        total_time = time.time() - test_start
        
        test_result = {
            'scale': scale,
            'articles_per_query': articles_per_query,
            'total_articles': len(raptor_tree.pubmed_articles),
            'total_time': total_time,
            'index_time': index_time,
            'integration_time': integration_time,
            'search_time': search_time,
            'workers_used': workers,
            'articles_per_second': len(raptor_tree.pubmed_articles) / integration_time if integration_time > 0 else 0,
            'search_results_count': sum(search_results),
            'parallel_metrics': parallel_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # パフォーマンス分析
        if parallel_metrics:
            test_result['retrieval_time'] = parallel_metrics.get('retrieval_time', 0)
            test_result['encoding_time'] = parallel_metrics.get('encoding_time', 0)
            test_result['parallel_efficiency'] = parallel_metrics.get('parallel_efficiency', 0)
        
        print(f"\\n📊 {scale} SCALE TEST RESULTS:")
        print(f"   Total execution time: {total_time:.1f}s")
        print(f"   Articles processed: {test_result['total_articles']}")
        print(f"   Processing rate: {test_result['articles_per_second']:.1f} articles/second")
        print(f"   Search results: {test_result['search_results_count']}")
        
        return test_result
    
    def run_comprehensive_scaling_test(self, max_workers: int = 4):
        """包括的スケーリングテスト実行"""
        
        print("🚀 COMPREHENSIVE LARGE-SCALE SCALING TEST")
        print("=" * 80)
        print(f"💻 System: {mp.cpu_count()} cores available, using {max_workers} workers")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 全スケールでテスト実行
        for scale, config in self.scaling_tests.items():
            try:
                result = self.run_single_scale_test(
                    scale=scale,
                    articles_per_query=config["articles_per_query"],
                    workers=max_workers
                )
                self.results[scale] = result
                
                # 中間結果保存
                self.save_intermediate_results()
                
            except Exception as e:
                print(f"❌ Error in {scale} scale test: {e}")
                self.results[scale] = {
                    'scale': scale,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # 最終分析
        self.analyze_scaling_performance()
        self.generate_scaling_report()
        self.create_visualizations()
        
        return self.results
    
    def save_intermediate_results(self):
        """中間結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"large_scale_test_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Intermediate results saved: {results_file.name}")
    
    def analyze_scaling_performance(self):
        """スケーリング性能の分析"""
        
        print("\\n" + "=" * 80)
        print("📈 COMPREHENSIVE SCALING PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # 成功したテストのみ分析
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            print("❌ Insufficient valid results for analysis")
            return
        
        # スケーリング効率計算
        baseline_key = "1x"
        if baseline_key not in valid_results:
            baseline_key = min(valid_results.keys())
        
        baseline = valid_results[baseline_key]
        baseline_time = baseline['total_time']
        baseline_articles = baseline['total_articles']
        
        print(f"📊 Baseline ({baseline_key}): {baseline_time:.1f}s for {baseline_articles} articles")
        print("\\n📋 Scaling Analysis:")
        print("-" * 60)
        
        scaling_analysis = {}
        
        for scale, result in valid_results.items():
            if scale == baseline_key:
                continue
                
            scale_factor = int(scale.replace('x', ''))
            expected_articles = baseline_articles * scale_factor
            actual_articles = result['total_articles']
            
            time_ratio = result['total_time'] / baseline_time
            articles_ratio = actual_articles / baseline_articles
            
            # スケーリング効率
            time_efficiency = scale_factor / time_ratio  # 理想は1.0
            articles_efficiency = articles_ratio / scale_factor  # 理想は1.0
            
            # スループット
            throughput = result['articles_per_second']
            baseline_throughput = baseline['articles_per_second']
            throughput_improvement = throughput / baseline_throughput if baseline_throughput > 0 else 1
            
            analysis = {
                'scale_factor': scale_factor,
                'expected_articles': expected_articles,
                'actual_articles': actual_articles,
                'time_ratio': time_ratio,
                'articles_ratio': articles_ratio,
                'time_efficiency': time_efficiency,
                'articles_efficiency': articles_efficiency,
                'throughput': throughput,
                'throughput_improvement': throughput_improvement
            }
            
            scaling_analysis[scale] = analysis
            
            print(f"{scale:>3} Scale: {result['total_time']:>6.1f}s | "
                  f"{actual_articles:>4} articles | "
                  f"Time ratio: {time_ratio:>4.1f}x | "
                  f"Efficiency: {time_efficiency*100:>5.1f}% | "
                  f"Throughput: {throughput:>5.1f} art/s")
        
        # 最高性能の特定
        best_efficiency = max(scaling_analysis.values(), key=lambda x: x['time_efficiency'])
        best_throughput = max(scaling_analysis.values(), key=lambda x: x['throughput'])
        
        print("\\n🏆 PERFORMANCE HIGHLIGHTS:")
        print(f"   Best time efficiency: {best_efficiency['time_efficiency']*100:.1f}% at {[k for k, v in scaling_analysis.items() if v == best_efficiency][0]} scale")
        print(f"   Best throughput: {best_throughput['throughput']:.1f} articles/second at {[k for k, v in scaling_analysis.items() if v == best_throughput][0]} scale")
        
        # 線形性分析
        scale_factors = [v['scale_factor'] for v in scaling_analysis.values()]
        time_ratios = [v['time_ratio'] for v in scaling_analysis.values()]
        
        if len(scale_factors) >= 2:
            # 線形回帰で傾向分析
            correlation = np.corrcoef(scale_factors, time_ratios)[0, 1]
            
            print(f"\\n📊 Linearity Analysis:")
            print(f"   Scale vs Time correlation: {correlation:.3f}")
            
            if correlation < 0.8:
                print("   ✅ Excellent sub-linear scaling (高効率)")
            elif correlation < 0.95:
                print("   ✅ Good sub-linear scaling (良好)")
            else:
                print("   ⚠️  Near-linear scaling (標準)")
        
        self.scaling_analysis = scaling_analysis
    
    def generate_scaling_report(self):
        """詳細スケーリングレポート生成"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"large_scale_report_{timestamp}.md"
        
        report_lines = []
        report_lines.append("# Large-Scale Scaling Test Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("## Test Configuration")
        report_lines.append(f"- System: {mp.cpu_count()} CPU cores")
        report_lines.append(f"- Workers: 4 (optimized for PubMed API)")
        report_lines.append(f"- Scales tested: {', '.join(self.scaling_tests.keys())}")
        report_lines.append("")
        
        # 結果サマリー
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        report_lines.append("## Results Summary")
        report_lines.append("| Scale | Articles/Query | Total Articles | Time (s) | Rate (art/s) | Efficiency |")
        report_lines.append("|-------|---------------|---------------|----------|--------------|------------|")
        
        for scale, result in valid_results.items():
            if 'total_time' in result:
                efficiency = ""
                if hasattr(self, 'scaling_analysis') and scale in self.scaling_analysis:
                    efficiency = f"{self.scaling_analysis[scale]['time_efficiency']*100:.1f}%"
                
                report_lines.append(f"| {scale} | {result['articles_per_query']} | {result['total_articles']} | {result['total_time']:.1f} | {result['articles_per_second']:.1f} | {efficiency} |")
        
        # 詳細分析
        if hasattr(self, 'scaling_analysis'):
            report_lines.append("")
            report_lines.append("## Detailed Analysis")
            
            for scale, analysis in self.scaling_analysis.items():
                report_lines.append(f"### {scale} Scale Analysis")
                report_lines.append(f"- Expected articles: {analysis['expected_articles']}")
                report_lines.append(f"- Actual articles: {analysis['actual_articles']}")
                report_lines.append(f"- Time efficiency: {analysis['time_efficiency']*100:.1f}%")
                report_lines.append(f"- Throughput improvement: {analysis['throughput_improvement']:.1f}x")
                report_lines.append("")
        
        # レポート保存
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\\n".join(report_lines))
        
        print(f"📄 Scaling report generated: {report_file.name}")
    
    def create_visualizations(self):
        """スケーリング結果の可視化"""
        
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v and 'total_time' in v}
        
        if len(valid_results) < 2:
            print("❌ Insufficient data for visualization")
            return
        
        # データ準備
        scales = list(valid_results.keys())
        scale_factors = [int(s.replace('x', '')) for s in scales]
        times = [valid_results[s]['total_time'] for s in scales]
        articles = [valid_results[s]['total_articles'] for s in scales]
        throughputs = [valid_results[s]['articles_per_second'] for s in scales]
        
        # 可視化作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 実行時間 vs スケール
        ax1.plot(scale_factors, times, 'bo-', linewidth=2, markersize=8)
        ax1.plot(scale_factors, [times[0] * sf for sf in scale_factors], 'r--', alpha=0.5, label='Linear expectation')
        ax1.set_xlabel('Scale Factor')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Scale Factor')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. スループット vs スケール
        ax2.plot(scale_factors, throughputs, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Scale Factor')
        ax2.set_ylabel('Throughput (articles/second)')
        ax2.set_title('Processing Throughput vs Scale Factor')
        ax2.grid(True, alpha=0.3)
        
        # 3. 効率性分析
        if hasattr(self, 'scaling_analysis'):
            efficiencies = [self.scaling_analysis[s]['time_efficiency']*100 for s in scales if s in self.scaling_analysis]
            scale_factors_eff = [int(s.replace('x', '')) for s in scales if s in self.scaling_analysis]
            
            ax3.bar(range(len(efficiencies)), efficiencies, color='orange', alpha=0.7)
            ax3.set_xlabel('Scale')
            ax3.set_ylabel('Time Efficiency (%)')
            ax3.set_title('Scaling Efficiency')
            ax3.set_xticks(range(len(efficiencies)))
            ax3.set_xticklabels([f"{sf}x" for sf in scale_factors_eff])
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Ideal efficiency')
            ax3.legend()
        
        # 4. 文献数 vs 時間の散布図
        ax4.scatter(articles, times, c=scale_factors, s=100, cmap='viridis', alpha=0.7)
        ax4.set_xlabel('Total Articles Processed')
        ax4.set_ylabel('Execution Time (seconds)')
        ax4.set_title('Time vs Articles (colored by scale)')
        ax4.grid(True, alpha=0.3)
        
        # カラーバー追加
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Scale Factor')
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = self.results_dir / f"large_scale_visualization_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Scaling visualization saved: {viz_file.name}")


def main():
    """メイン実行関数"""
    
    print("🚀 LARGE-SCALE SCALING TEST FOR OPTIMIZED PARALLEL SYSTEM")
    print("=" * 80)
    
    # テスター初期化
    tester = LargeScaleScalingTester()
    
    # 大規模スケーリングテスト実行
    try:
        print("🔬 Starting comprehensive scaling tests...")
        results = tester.run_comprehensive_scaling_test(max_workers=4)
        
        print("\\n" + "="*80)
        print("🎯 LARGE-SCALE SCALING TEST COMPLETED!")
        print("="*80)
        
        successful_tests = sum(1 for r in results.values() if 'error' not in r)
        print(f"✅ Successful tests: {successful_tests}/{len(results)}")
        
        # 結果ディレクトリの表示
        print(f"📁 Results saved in: {tester.results_dir}")
        
    except KeyboardInterrupt:
        print("\\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()