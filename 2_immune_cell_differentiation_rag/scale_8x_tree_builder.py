"""
8x Scale Parallel Processing with Tree Construction and Visualization
8倍スケール並列処理によるツリー構築と可視化

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


class Scale8xTreeBuilder:
    """8倍スケール用ツリー構築・可視化システム"""
    
    def __init__(self):
        self.base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
        self.cache_dir = self.base_dir / "data/immune_cell_differentiation"
        self.results_dir = self.cache_dir / "scaling_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 8倍スケール設定
        self.scale = "8x"
        self.articles_per_query = 160  # 8倍スケール
        self.workers = 4
        
        # ログ設定
        self.setup_logging()
        
    def setup_logging(self):
        """ログシステムの設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.results_dir / f"scale_8x_tree_build_{timestamp}.log"
        
        # ロガー設定
        self.logger = logging.getLogger('Scale8xTreeBuilder')
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
        
        self.logger.info(f"8x Scale Tree Builder Log initialized: {self.log_file}")
        
    def log_info(self, message: str):
        """情報ログ"""
        self.logger.info(message)
        
    def log_error(self, message: str):
        """エラーログ"""
        self.logger.error(message)
        
    def build_8x_scale_tree(self):
        """8倍スケールでのツリー構築実行"""
        
        self.log_info("🚀 8X SCALE PARALLEL TREE CONSTRUCTION STARTED")
        self.log_info("=" * 70)
        self.log_info(f"Scale: {self.scale}")
        self.log_info(f"Articles per query: {self.articles_per_query}")
        self.log_info(f"CPU cores available: {mp.cpu_count()}")
        self.log_info(f"Workers to use: {self.workers}")
        
        total_start = time.time()
        
        try:
            # Phase 1: システム初期化
            self.log_info("\\n🧬 Phase 1: System Initialization")
            
            raptor_tree = ImmuneCellRAPTORTree(str(self.cache_dir))
            
            # 免疫細胞階層データ読み込み
            immune_file = self.cache_dir / "immune_cell_hierarchy.json"
            if not immune_file.exists():
                self.log_error(f"Required file not found: {immune_file}")
                return False
                
            raptor_tree.load_immune_hierarchy(str(immune_file))
            self.log_info(f"✓ Loaded {len(raptor_tree.nodes)} immune cell nodes")
            
            # Phase 2: FAISS インデックス構築
            self.log_info("\\n⚡ Phase 2: FAISS Index Construction")
            index_start = time.time()
            
            raptor_tree.build_faiss_index_parallel(workers=self.workers)
            
            index_time = time.time() - index_start
            self.log_info(f"✓ FAISS index built in {index_time:.1f}s")
            
            # Phase 3: 8倍スケールPubMed統合
            self.log_info(f"\\n📡 Phase 3: 8x Scale PubMed Integration ({self.articles_per_query} articles/query)")
            integration_start = time.time()
            
            parallel_metrics = optimize_immune_raptor_parallel(
                raptor_tree,
                max_articles_per_query=self.articles_per_query,
                max_workers=self.workers
            )
            
            integration_time = time.time() - integration_start
            total_articles = len(raptor_tree.pubmed_articles)
            
            self.log_info(f"✓ 8x Scale PubMed integration completed")
            self.log_info(f"   Total articles: {total_articles}")
            self.log_info(f"   Integration time: {integration_time:.1f}s")
            self.log_info(f"   Processing rate: {total_articles / integration_time:.1f} articles/second")
            
            if parallel_metrics:
                self.log_info(f"   Parallel metrics:")
                self.log_info(f"     Retrieval time: {parallel_metrics.get('retrieval_time', 0):.1f}s")
                self.log_info(f"     Encoding time: {parallel_metrics.get('encoding_time', 0):.1f}s")
                self.log_info(f"     Workers used: {parallel_metrics.get('workers_used', 4)}")
                self.log_info(f"     Parallel efficiency: {parallel_metrics.get('parallel_efficiency', 0):.0f}%")
            
            # Phase 4: システムテスト
            self.log_info("\\n🔍 Phase 4: System Testing")
            search_start = time.time()
            
            test_queries = [
                "FOXP3+ regulatory T cell differentiation",
                "Treg immune suppression mechanisms", 
                "thymic versus peripheral Treg development",
                "IL-10 production by regulatory T cells",
                "CTLA-4 checkpoint inhibition mechanism"
            ]
            
            search_results = []
            for i, query in enumerate(test_queries, 1):
                try:
                    results = raptor_tree.hierarchical_search(query, top_k=5)
                    search_results.append(len(results))
                    self.log_info(f"   Test {i}: '{query[:50]}...' → {len(results)} results")
                    
                    # トップ結果の詳細ログ
                    if results:
                        top_node_id, top_score = results[0]
                        top_node = raptor_tree.nodes[top_node_id]
                        self.log_info(f"     Top result: {top_node.cell_type} ({top_node.subtype}) - Score: {top_score:.3f}")
                        
                except Exception as e:
                    self.log_error(f"   Test {i} failed: {e}")
                    search_results.append(0)
            
            search_time = time.time() - search_start
            
            # Phase 5: ツリー可視化
            self.log_info("\\n🖼️  Phase 5: Tree Visualization")
            viz_start = time.time()
            
            # 8倍スケール用可視化ファイル
            output_dir = self.cache_dir / "raptor_trees"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            try:
                # 階層可視化
                viz_file = output_dir / f"immune_hierarchy_8x_scale_{timestamp}.png"
                raptor_tree.visualize_hierarchy(str(viz_file))
                self.log_info(f"✓ Hierarchy visualization saved: {viz_file.name}")
                
                # 分化経路可視化（利用可能な場合）
                try:
                    path = raptor_tree.trace_differentiation_path("HSC", "Treg")
                    if path:
                        self.log_info(f"✓ Differentiation path traced: {' → '.join(path.pathway_nodes)}")
                        self.log_info(f"   Key factors: {', '.join(path.key_factors[:5])}")
                    else:
                        self.log_info("ℹ️  No differentiation path found")
                except Exception as e:
                    self.log_info(f"ℹ️  Differentiation path tracing not available: {e}")
                
            except Exception as e:
                self.log_error(f"Visualization error: {e}")
            
            viz_time = time.time() - viz_start
            
            # Phase 6: 8倍スケールツリー保存
            self.log_info("\\n💾 Phase 6: 8x Scale Tree Saving")
            save_start = time.time()
            
            try:
                # RAPTOR Tree保存
                tree_file = output_dir / f"immune_cell_raptor_tree_8x_scale_{timestamp}.json"
                raptor_tree.save_raptor_tree(str(tree_file))
                self.log_info(f"✓ 8x Scale Tree saved: {tree_file.name}")
                
                # 埋め込みベクトル保存
                embeddings_file = output_dir / f"immune_cell_raptor_tree_embeddings_8x_{timestamp}.pkl"
                embeddings_data = {
                    'node_embeddings': {node_id: node.embedding for node_id, node in raptor_tree.nodes.items() if node.embedding is not None},
                    'article_embeddings': raptor_tree.article_embeddings
                }
                
                import pickle
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(embeddings_data, f)
                self.log_info(f"✓ 8x Scale Embeddings saved: {embeddings_file.name}")
                
                # FAISS インデックス保存
                if raptor_tree.faiss_index is not None:
                    import faiss
                    faiss_file = output_dir / f"immune_cell_raptor_tree_faiss_8x_{timestamp}.index"
                    faiss.write_index(raptor_tree.faiss_index, str(faiss_file))
                    self.log_info(f"✓ 8x Scale FAISS index saved: {faiss_file.name}")
                
            except Exception as e:
                self.log_error(f"Saving error: {e}")
            
            save_time = time.time() - save_start
            total_time = time.time() - total_start
            
            # 結果まとめ
            result = {
                'scale': self.scale,
                'articles_per_query': self.articles_per_query,
                'total_articles': total_articles,
                'total_time': total_time,
                'index_time': index_time,
                'integration_time': integration_time,
                'search_time': search_time,
                'visualization_time': viz_time,
                'save_time': save_time,
                'articles_per_second': total_articles / integration_time if integration_time > 0 else 0,
                'search_results_count': sum(search_results),
                'parallel_metrics': parallel_metrics,
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS',
                'output_files': {
                    'visualization': viz_file.name if 'viz_file' in locals() else None,
                    'tree_json': tree_file.name if 'tree_file' in locals() else None,
                    'embeddings': embeddings_file.name if 'embeddings_file' in locals() else None,
                    'faiss_index': faiss_file.name if 'faiss_file' in locals() else None
                }
            }
            
            # 結果保存
            result_file = self.results_dir / f"scale_8x_result_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # 成功ログ
            self.log_info("\\n✅ 8X SCALE TREE CONSTRUCTION COMPLETED SUCCESSFULLY")
            self.log_info("=" * 70)
            self.log_info(f"📊 Final Results Summary:")
            self.log_info(f"   Scale: {self.scale}")
            self.log_info(f"   Total execution time: {total_time:.1f}s")
            self.log_info(f"   Articles processed: {total_articles}")
            self.log_info(f"   Processing rate: {result['articles_per_second']:.1f} articles/second")
            self.log_info(f"   Search tests passed: {sum(search_results)}/{len(test_queries)}")
            self.log_info(f"   Visualization time: {viz_time:.1f}s")
            self.log_info(f"   Save time: {save_time:.1f}s")
            self.log_info(f"\\n📁 Output Files:")
            for file_type, filename in result['output_files'].items():
                if filename:
                    self.log_info(f"   {file_type}: {filename}")
            self.log_info(f"\\n📝 Complete log: {self.log_file.name}")
            self.log_info(f"📊 Result data: {result_file.name}")
            
            return True
            
        except Exception as e:
            error_time = time.time() - total_start
            self.log_error(f"❌ 8X SCALE TREE CONSTRUCTION FAILED: {str(e)}")
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
            
            error_file = self.results_dir / f"scale_8x_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            
            self.log_error(f"   Error details saved: {error_file.name}")
            
            return False


def main():
    """メイン実行関数"""
    
    print("🚀 8X SCALE PARALLEL TREE CONSTRUCTION & VISUALIZATION")
    print("=" * 70)
    
    # 8倍スケールツリービルダー初期化
    builder = Scale8xTreeBuilder()
    
    try:
        success = builder.build_8x_scale_tree()
        
        if success:
            print(f"\\n✅ 8x Scale Tree Construction completed successfully!")
            print(f"📝 Detailed log: {builder.log_file}")
            print(f"📁 Output files saved in: {builder.cache_dir}/raptor_trees/")
            print(f"📊 Results saved in: {builder.results_dir}")
        else:
            print(f"\\n❌ 8x Scale Tree Construction failed.")
            print(f"📝 Check log file: {builder.log_file}")
            
    except KeyboardInterrupt:
        print("\\n⚠️  Construction interrupted by user")
    except Exception as e:
        print(f"\\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()