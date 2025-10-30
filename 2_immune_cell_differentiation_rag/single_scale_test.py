"""
Single Scale Test for Logging Verification
ログ機能確認用の単一スケールテスト

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


class SingleScaleLogger:
    """単一スケールテスト用ログシステム"""
    
    def __init__(self, scale: str = "4x", articles_per_query: int = 80):
        self.base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
        self.cache_dir = self.base_dir / "data/immune_cell_differentiation"
        self.results_dir = self.cache_dir / "scaling_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.scale = scale
        self.articles_per_query = articles_per_query
        
        # ログ設定
        self.setup_logging()
        
    def setup_logging(self):
        """ログシステムの設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.results_dir / f"single_scale_test_{self.scale}_{timestamp}.log"
        
        # ロガー設定
        self.logger = logging.getLogger('SingleScaleTest')
        self.logger.setLevel(logging.INFO)
        
        # 既存ハンドラーをクリア
        self.logger.handlers.clear()
        
        # ファイルハンドラー
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
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
        
        self.logger.info(f"Single Scale Test Log initialized: {self.log_file}")
        
    def log_info(self, message: str):
        """情報ログ"""
        self.logger.info(message)
        
    def log_error(self, message: str):
        """エラーログ"""
        self.logger.error(message)
        
    def run_test(self):
        """単一スケールテスト実行"""
        
        self.log_info("🚀 SINGLE SCALE TEST STARTED")
        self.log_info("=" * 60)
        self.log_info(f"Scale: {self.scale}")
        self.log_info(f"Articles per query: {self.articles_per_query}")
        self.log_info(f"CPU cores available: {mp.cpu_count()}")
        self.log_info(f"Workers to use: 4")
        
        test_start = time.time()
        
        try:
            # Phase 1: システム初期化
            self.log_info("🧬 Phase 1: System Initialization")
            
            raptor_tree = ImmuneCellRAPTORTree(str(self.cache_dir))
            
            # 免疫細胞階層データ読み込み
            immune_file = self.cache_dir / "immune_cell_hierarchy.json"
            if not immune_file.exists():
                self.log_error(f"Required file not found: {immune_file}")
                return False
                
            raptor_tree.load_immune_hierarchy(str(immune_file))
            self.log_info(f"✓ Loaded {len(raptor_tree.nodes)} immune cell nodes")
            
            # Phase 2: FAISS インデックス構築
            self.log_info("⚡ Phase 2: FAISS Index Construction")
            index_start = time.time()
            
            raptor_tree.build_faiss_index_parallel(workers=4)
            
            index_time = time.time() - index_start
            self.log_info(f"✓ FAISS index built in {index_time:.1f}s")
            
            # Phase 3: PubMed統合
            self.log_info(f"📡 Phase 3: PubMed Integration ({self.articles_per_query} articles/query)")
            integration_start = time.time()
            
            parallel_metrics = optimize_immune_raptor_parallel(
                raptor_tree,
                max_articles_per_query=self.articles_per_query,
                max_workers=4
            )
            
            integration_time = time.time() - integration_start
            total_articles = len(raptor_tree.pubmed_articles)
            
            self.log_info(f"✓ PubMed integration completed")
            self.log_info(f"   Total articles: {total_articles}")
            self.log_info(f"   Integration time: {integration_time:.1f}s")
            self.log_info(f"   Processing rate: {total_articles / integration_time:.1f} articles/second")
            
            if parallel_metrics:
                self.log_info(f"   Parallel metrics:")
                self.log_info(f"     Retrieval time: {parallel_metrics.get('retrieval_time', 0):.1f}s")
                self.log_info(f"     Encoding time: {parallel_metrics.get('encoding_time', 0):.1f}s")
                self.log_info(f"     Workers used: {parallel_metrics.get('workers_used', 4)}")
            
            # Phase 4: システムテスト
            self.log_info("🔍 Phase 4: System Testing")
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
                    self.log_info(f"   Test {i}: '{query[:50]}...' → {len(results)} results")
                except Exception as e:
                    self.log_error(f"   Test {i} failed: {e}")
                    search_results.append(0)
            
            search_time = time.time() - search_start
            total_time = time.time() - test_start
            
            # 結果まとめ
            result = {
                'scale': self.scale,
                'articles_per_query': self.articles_per_query,
                'total_articles': total_articles,
                'total_time': total_time,
                'index_time': index_time,
                'integration_time': integration_time,
                'search_time': search_time,
                'articles_per_second': total_articles / integration_time if integration_time > 0 else 0,
                'search_results_count': sum(search_results),
                'parallel_metrics': parallel_metrics,
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS'
            }
            
            # 結果保存
            result_file = self.results_dir / f"single_scale_result_{self.scale}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # 成功ログ
            self.log_info("✅ SINGLE SCALE TEST COMPLETED SUCCESSFULLY")
            self.log_info("=" * 60)
            self.log_info(f"📊 Final Results:")
            self.log_info(f"   Scale: {self.scale}")
            self.log_info(f"   Total execution time: {total_time:.1f}s")
            self.log_info(f"   Articles processed: {total_articles}")
            self.log_info(f"   Processing rate: {result['articles_per_second']:.1f} articles/second")
            self.log_info(f"   Search tests passed: {sum(search_results)}")
            self.log_info(f"   Result saved: {result_file.name}")
            self.log_info(f"   Log saved: {self.log_file.name}")
            
            return True
            
        except Exception as e:
            error_time = time.time() - test_start
            self.log_error(f"❌ SINGLE SCALE TEST FAILED: {str(e)}")
            self.log_error(f"   Execution time before failure: {error_time:.1f}s")
            
            # エラー結果保存
            error_result = {
                'scale': self.scale,
                'articles_per_query': self.articles_per_query,
                'error': str(e),
                'execution_time': error_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'FAILED'
            }
            
            error_file = self.results_dir / f"single_scale_error_{self.scale}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            
            self.log_error(f"   Error details saved: {error_file.name}")
            
            return False


def main():
    """メイン実行関数"""
    
    print("🧪 SINGLE SCALE TEST FOR LOGGING VERIFICATION")
    print("=" * 60)
    
    # 4倍スケールでテスト
    tester = SingleScaleLogger(scale="4x", articles_per_query=80)
    
    try:
        success = tester.run_test()
        
        if success:
            print(f"\\n✅ Test completed successfully!")
            print(f"📝 Log file: {tester.log_file}")
        else:
            print(f"\\n❌ Test failed. Check log file: {tester.log_file}")
            
    except KeyboardInterrupt:
        print("\\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Critical error: {e}")


if __name__ == "__main__":
    main()