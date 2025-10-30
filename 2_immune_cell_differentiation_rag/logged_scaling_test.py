"""
Large-Scale Scaling Test with Comprehensive Logging
大規模スケーリングテスト（4x, 8x, 12x, 16x）- ログ出力対応版

Author: AI Assistant
Date: 2025-10-31
"""

import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import logging

# ローカルモジュールのインポート
from immune_raptor_tree import ImmuneCellRAPTORTree
from rate_limited_parallel import optimize_immune_raptor_parallel


class LoggedScalingTester:
    """ログ対応大規模スケーリングテスト実行クラス"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or "C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
        self.cache_dir = self.base_dir / "data/immune_cell_differentiation"
        self.results_dir = self.cache_dir / "scaling_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # ログ設定
        self.setup_logging()
        
        # テストスケール設定
        self.scaling_tests = {
            "1x": {"articles_per_query": 20, "description": "ベースライン"},
            "4x": {"articles_per_query": 80, "description": "4倍スケール"},
            "8x": {"articles_per_query": 160, "description": "8倍スケール"},
            "12x": {"articles_per_query": 240, "description": "12倍スケール"},
            "16x": {"articles_per_query": 320, "description": "16倍スケール"}
        }
        
        self.results = {}
        
    def setup_logging(self):
        """ログシステムの設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.results_dir / f"large_scale_test_log_{timestamp}.txt"
        
        # ロガー設定
        self.logger = logging.getLogger('ScalingTest')
        self.logger.setLevel(logging.INFO)
        
        # ファイルハンドラー
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマット設定
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラー追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
        self.logger.info(f"ログファイル初期化: {log_file}")
        
    def log_and_print(self, message: str, level: str = "INFO"):
        """ログ出力とコンソール表示"""
        if level == "INFO":
            self.logger.info(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def run_single_scale_test(self, scale: str, articles_per_query: int, workers: int = 4) -> dict:
        """単一スケールでのテスト実行（ログ対応）"""
        
        self.log_and_print(f"{'='*70}")
        self.log_and_print(f"🔬 SCALING TEST: {scale} ({articles_per_query} articles/query)")
        self.log_and_print(f"{'='*70}")
        
        test_start = time.time()
        
        try:
            # 1. システム初期化
            self.log_and_print("🧬 Phase 1: System Initialization")
            raptor_tree = ImmuneCellRAPTORTree(str(self.cache_dir))
            
            # 免疫細胞階層データ読み込み
            immune_file = self.cache_dir / "immune_cell_hierarchy.json"
            raptor_tree.load_immune_hierarchy(str(immune_file))
            self.log_and_print(f"✓ Loaded {len(raptor_tree.nodes)} immune cell nodes")
            
            # 2. FAISS インデックス構築
            self.log_and_print("⚡ Phase 2: FAISS Index Construction")
            index_start = time.time()
            raptor_tree.build_faiss_index_parallel(workers=workers)
            index_time = time.time() - index_start
            self.log_and_print(f"✓ FAISS index built in {index_time:.1f}s")
            
            # 3. 最適化並列PubMed統合
            self.log_and_print(f"📡 Phase 3: Scaled PubMed Integration ({articles_per_query} articles/query)")
            integration_start = time.time()
            
            # PubMed統合実行（ログ記録）
            parallel_metrics = optimize_immune_raptor_parallel(
                raptor_tree,
                max_articles_per_query=articles_per_query,
                max_workers=workers
            )
            
            integration_time = time.time() - integration_start
            
            # 統合結果ログ
            total_articles = len(raptor_tree.pubmed_articles)
            self.log_and_print(f"✓ PubMed integration completed: {total_articles} articles in {integration_time:.1f}s")
            
            if parallel_metrics:
                self.log_and_print(f"   Retrieval time: {parallel_metrics.get('retrieval_time', 0):.1f}s")
                self.log_and_print(f"   Encoding time: {parallel_metrics.get('encoding_time', 0):.1f}s")
                self.log_and_print(f"   Processing rate: {parallel_metrics.get('articles_processed', 0) / integration_time:.1f} articles/second")
            
            # 4. システムテスト
            self.log_and_print("🔍 Phase 4: System Testing")
            search_start = time.time()
            
            test_queries = [
                "FOXP3+ regulatory T cell differentiation",
                "Treg immune suppression mechanisms",
                "thymic versus peripheral Treg development"
            ]
            
            search_results = []
            for i, query in enumerate(test_queries, 1):
                try:
                    results = raptor_tree.hierarchical_search(query, top_k=5)
                    search_results.append(len(results))
                    self.log_and_print(f"   Test {i}: '{query[:50]}...' - {len(results)} results")
                except Exception as e:
                    self.log_and_print(f"   Test {i} error: {e}", "ERROR")
                    search_results.append(0)
            
            search_time = time.time() - search_start
            total_time = time.time() - test_start
            
            # 5. 結果収集とログ
            test_result = {
                'scale': scale,
                'articles_per_query': articles_per_query,
                'total_articles': total_articles,
                'total_time': total_time,
                'index_time': index_time,
                'integration_time': integration_time,
                'search_time': search_time,
                'workers_used': workers,
                'articles_per_second': total_articles / integration_time if integration_time > 0 else 0,
                'search_results_count': sum(search_results),
                'parallel_metrics': parallel_metrics,
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS'
            }
            
            # 成功ログ
            self.log_and_print(f"✅ {scale} SCALE TEST COMPLETED SUCCESSFULLY")
            self.log_and_print(f"   Total execution time: {total_time:.1f}s")
            self.log_and_print(f"   Articles processed: {total_articles}")
            self.log_and_print(f"   Processing rate: {test_result['articles_per_second']:.1f} articles/second")
            self.log_and_print(f"   Search test results: {sum(search_results)}")
            
            return test_result
            
        except Exception as e:
            error_time = time.time() - test_start
            self.log_and_print(f"❌ {scale} SCALE TEST FAILED: {str(e)}", "ERROR")
            
            # エラー結果
            return {
                'scale': scale,
                'articles_per_query': articles_per_query,
                'error': str(e),
                'execution_time': error_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'FAILED'
            }
    
    def run_comprehensive_scaling_test(self, max_workers: int = 4):
        """包括的スケーリングテスト実行（ログ記録）"""
        
        self.log_and_print("🚀 COMPREHENSIVE LARGE-SCALE SCALING TEST STARTED")
        self.log_and_print("=" * 80)
        self.log_and_print(f"💻 System: {mp.cpu_count()} cores available, using {max_workers} workers")
        self.log_and_print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_and_print(f"📝 Log file: {self.log_file}")
        
        test_start_time = time.time()
        
        # 全スケールでテスト実行
        for i, (scale, config) in enumerate(self.scaling_tests.items(), 1):
            self.log_and_print(f"\\n📊 Progress: Test {i}/{len(self.scaling_tests)} - {scale} scale")
            
            try:
                result = self.run_single_scale_test(
                    scale=scale,
                    articles_per_query=config["articles_per_query"],
                    workers=max_workers
                )
                self.results[scale] = result
                
                # 中間結果保存
                self.save_intermediate_results()
                
                # 進捗ログ
                if result['status'] == 'SUCCESS':
                    self.log_and_print(f"✅ {scale} test completed successfully")
                else:
                    self.log_and_print(f"❌ {scale} test failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                self.log_and_print(f"❌ Critical error in {scale} scale test: {e}", "ERROR")
                self.results[scale] = {
                    'scale': scale,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'CRITICAL_FAILED'
                }
        
        # 最終分析とログ
        total_test_time = time.time() - test_start_time
        self.log_and_print(f"\\n⏱️ Total testing time: {total_test_time:.1f}s")
        
        self.analyze_scaling_performance()
        self.generate_scaling_report()
        
        return self.results
    
    def save_intermediate_results(self):
        """中間結果の保存（ログ記録）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"large_scale_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            self.log_and_print(f"💾 Intermediate results saved: {results_file.name}")
            
        except Exception as e:
            self.log_and_print(f"❌ Failed to save results: {e}", "ERROR")
    
    def analyze_scaling_performance(self):
        """スケーリング性能の分析（ログ記録）"""
        
        self.log_and_print("\\n" + "=" * 80)
        self.log_and_print("📈 COMPREHENSIVE SCALING PERFORMANCE ANALYSIS")
        self.log_and_print("=" * 80)
        
        # 成功したテストのみ分析
        valid_results = {k: v for k, v in self.results.items() if v.get('status') == 'SUCCESS'}
        failed_tests = {k: v for k, v in self.results.items() if v.get('status') != 'SUCCESS'}
        
        self.log_and_print(f"📊 Test summary: {len(valid_results)} successful, {len(failed_tests)} failed")
        
        if failed_tests:
            self.log_and_print("❌ Failed tests:")
            for scale, result in failed_tests.items():
                self.log_and_print(f"   {scale}: {result.get('error', 'Unknown error')}")
        
        if len(valid_results) < 2:
            self.log_and_print("❌ Insufficient valid results for analysis")
            return
        
        # スケーリング効率計算
        baseline_key = "1x"
        if baseline_key not in valid_results:
            baseline_key = min(valid_results.keys())
        
        baseline = valid_results[baseline_key]
        baseline_time = baseline['total_time']
        baseline_articles = baseline['total_articles']
        
        self.log_and_print(f"📊 Baseline ({baseline_key}): {baseline_time:.1f}s for {baseline_articles} articles")
        self.log_and_print("\\n📋 Detailed Scaling Analysis:")
        self.log_and_print("-" * 80)
        
        scaling_analysis = {}
        
        for scale, result in valid_results.items():
            if scale == baseline_key:
                continue
                
            scale_factor = int(scale.replace('x', ''))
            time_ratio = result['total_time'] / baseline_time
            articles_ratio = result['total_articles'] / baseline_articles
            
            # スケーリング効率
            time_efficiency = scale_factor / time_ratio
            throughput = result['articles_per_second']
            baseline_throughput = baseline['articles_per_second']
            throughput_improvement = throughput / baseline_throughput if baseline_throughput > 0 else 1
            
            analysis = {
                'scale_factor': scale_factor,
                'time_ratio': time_ratio,
                'articles_ratio': articles_ratio,
                'time_efficiency': time_efficiency,
                'throughput': throughput,
                'throughput_improvement': throughput_improvement
            }
            
            scaling_analysis[scale] = analysis
            
            # 詳細ログ
            self.log_and_print(f"{scale:>3} Scale:")
            self.log_and_print(f"   Time: {result['total_time']:>6.1f}s (ratio: {time_ratio:>4.1f}x)")
            self.log_and_print(f"   Articles: {result['total_articles']:>4} (ratio: {articles_ratio:>4.1f}x)")
            self.log_and_print(f"   Efficiency: {time_efficiency*100:>5.1f}%")
            self.log_and_print(f"   Throughput: {throughput:>5.1f} art/s (improvement: {throughput_improvement:>4.1f}x)")
        
        # パフォーマンスハイライト
        if scaling_analysis:
            best_efficiency = max(scaling_analysis.values(), key=lambda x: x['time_efficiency'])
            best_throughput = max(scaling_analysis.values(), key=lambda x: x['throughput'])
            
            self.log_and_print("\\n🏆 PERFORMANCE HIGHLIGHTS:")
            
            best_eff_scale = [k for k, v in scaling_analysis.items() if v == best_efficiency][0]
            best_thr_scale = [k for k, v in scaling_analysis.items() if v == best_throughput][0]
            
            self.log_and_print(f"   🥇 Best time efficiency: {best_efficiency['time_efficiency']*100:.1f}% at {best_eff_scale} scale")
            self.log_and_print(f"   🚀 Best throughput: {best_throughput['throughput']:.1f} articles/second at {best_thr_scale} scale")
            
            # サブリニア性能の評価
            avg_efficiency = sum(a['time_efficiency'] for a in scaling_analysis.values()) / len(scaling_analysis)
            
            if avg_efficiency > 0.8:
                performance_rating = "🌟 EXCELLENT"
            elif avg_efficiency > 0.6:
                performance_rating = "✅ GOOD"
            elif avg_efficiency > 0.4:
                performance_rating = "⚠️ FAIR"
            else:
                performance_rating = "❌ POOR"
            
            self.log_and_print(f"   📊 Average efficiency: {avg_efficiency*100:.1f}% - {performance_rating}")
        
        self.scaling_analysis = scaling_analysis
    
    def generate_scaling_report(self):
        """詳細スケーリングレポート生成（ログ記録）"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"large_scale_report_{timestamp}.md"
        
        try:
            report_lines = []
            report_lines.append("# Large-Scale Scaling Test Report")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # テスト設定
            report_lines.append("## Test Configuration")
            report_lines.append(f"- System: {mp.cpu_count()} CPU cores")
            report_lines.append(f"- Workers: 4 (optimized for PubMed API)")
            report_lines.append(f"- Scales tested: {', '.join(self.scaling_tests.keys())}")
            report_lines.append(f"- Log file: {self.log_file.name}")
            report_lines.append("")
            
            # 結果サマリー
            valid_results = {k: v for k, v in self.results.items() if v.get('status') == 'SUCCESS'}
            failed_results = {k: v for k, v in self.results.items() if v.get('status') != 'SUCCESS'}
            
            report_lines.append("## Results Summary")
            report_lines.append(f"- Successful tests: {len(valid_results)}")
            report_lines.append(f"- Failed tests: {len(failed_results)}")
            report_lines.append("")
            
            if valid_results:
                report_lines.append("### Successful Tests")
                report_lines.append("| Scale | Articles/Query | Total Articles | Time (s) | Rate (art/s) | Efficiency |")
                report_lines.append("|-------|---------------|---------------|----------|--------------|------------|")
                
                for scale, result in valid_results.items():
                    efficiency = ""
                    if hasattr(self, 'scaling_analysis') and scale in self.scaling_analysis:
                        efficiency = f"{self.scaling_analysis[scale]['time_efficiency']*100:.1f}%"
                    
                    report_lines.append(f"| {scale} | {result['articles_per_query']} | {result['total_articles']} | {result['total_time']:.1f} | {result['articles_per_second']:.1f} | {efficiency} |")
            
            if failed_results:
                report_lines.append("")
                report_lines.append("### Failed Tests")
                for scale, result in failed_results.items():
                    report_lines.append(f"- {scale}: {result.get('error', 'Unknown error')}")
            
            # 詳細分析
            if hasattr(self, 'scaling_analysis') and self.scaling_analysis:
                report_lines.append("")
                report_lines.append("## Detailed Performance Analysis")
                
                for scale, analysis in self.scaling_analysis.items():
                    report_lines.append(f"### {scale} Scale")
                    report_lines.append(f"- Scale factor: {analysis['scale_factor']}x")
                    report_lines.append(f"- Time ratio: {analysis['time_ratio']:.1f}x")
                    report_lines.append(f"- Articles ratio: {analysis['articles_ratio']:.1f}x")
                    report_lines.append(f"- Time efficiency: {analysis['time_efficiency']*100:.1f}%")
                    report_lines.append(f"- Throughput improvement: {analysis['throughput_improvement']:.1f}x")
                    report_lines.append("")
            
            # レポート保存
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("\\n".join(report_lines))
            
            self.log_and_print(f"📄 Comprehensive scaling report generated: {report_file.name}")
            
        except Exception as e:
            self.log_and_print(f"❌ Failed to generate report: {e}", "ERROR")


def main():
    """メイン実行関数"""
    
    print("🚀 LARGE-SCALE SCALING TEST WITH COMPREHENSIVE LOGGING")
    print("=" * 80)
    
    # テスター初期化
    tester = LoggedScalingTester()
    
    # 大規模スケーリングテスト実行
    try:
        print("🔬 Starting comprehensive scaling tests...")
        results = tester.run_comprehensive_scaling_test(max_workers=4)
        
        tester.log_and_print("\\n" + "="*80)
        tester.log_and_print("🎯 LARGE-SCALE SCALING TEST COMPLETED!")
        tester.log_and_print("="*80)
        
        successful_tests = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
        tester.log_and_print(f"✅ Final summary: {successful_tests}/{len(results)} tests successful")
        tester.log_and_print(f"📁 Results and logs saved in: {tester.results_dir}")
        tester.log_and_print(f"📝 Main log file: {tester.log_file}")
        
    except KeyboardInterrupt:
        print("\\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()