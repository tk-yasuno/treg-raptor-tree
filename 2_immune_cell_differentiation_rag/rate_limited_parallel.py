"""
Rate-Limited Parallel Processing for PubMed API
PubMedのレート制限に対応した並列処理システム

Author: AI Assistant
Date: 2025-10-31
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import queue

class RateLimitedPubMedProcessor:
    """レート制限対応PubMed並列処理クラス"""
    
    def __init__(self, requests_per_second: float = 3.0, max_workers: int = 4):
        """
        Args:
            requests_per_second: 秒あたりのリクエスト数制限
            max_workers: 最大ワーカー数
        """
        self.requests_per_second = requests_per_second
        self.max_workers = min(max_workers, 4)  # PubMedのため保守的に制限
        self.request_times = queue.Queue()
        self.lock = threading.Lock()
        
    def _wait_for_rate_limit(self):
        """レート制限を遵守するための待機"""
        current_time = time.time()
        
        with self.lock:
            # 古いリクエスト時刻を削除（1秒より古いもの）
            temp_times = []
            while not self.request_times.empty():
                req_time = self.request_times.get()
                if current_time - req_time < 1.0:
                    temp_times.append(req_time)
            
            # 戻す
            for t in temp_times:
                self.request_times.put(t)
            
            # レート制限チェック
            if self.request_times.qsize() >= self.requests_per_second:
                sleep_time = 1.0 / self.requests_per_second
                time.sleep(sleep_time)
            
            # 現在の時刻を記録
            self.request_times.put(current_time)
    
    def parallel_pubmed_search(self, queries: List[str], search_func, max_articles: int = 30) -> Dict[str, List[Any]]:
        """レート制限対応の並列PubMed検索"""
        
        print(f"🔄 Rate-limited parallel search with {self.max_workers} workers ({self.requests_per_second} req/s)")
        
        def search_with_rate_limit(query_data):
            query, index = query_data
            
            # レート制限待機
            self._wait_for_rate_limit()
            
            thread_start = time.time()
            try:
                result = search_func(query, max_articles)
                thread_time = time.time() - thread_start
                print(f"✓ Worker {index+1}: '{query[:50]}...' - {len(result) if result else 0}本 ({thread_time:.1f}s)")
                return query, result, thread_time
            except Exception as e:
                print(f"✗ Worker {index+1} error: {e}")
                return query, [], 0
        
        # クエリにインデックスを付加
        query_data = [(query, i) for i, query in enumerate(queries)]
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {executor.submit(search_with_rate_limit, qd): qd for qd in query_data}
            
            for future in as_completed(future_to_query):
                query, result, thread_time = future.result()
                results[query] = result
        
        return results


def optimize_immune_raptor_parallel(raptor_tree, max_articles_per_query: int = 20, max_workers: int = 4):
    """
    免疫RAGシステム用の最適化された並列処理
    レート制限とエラーハンドリングを強化
    """
    
    print("\\n🚀 OPTIMIZED RATE-LIMITED PARALLEL PROCESSING")
    print("=" * 55)
    
    start_time = time.time()
    
    # レート制限プロセッサー初期化
    processor = RateLimitedPubMedProcessor(
        requests_per_second=2.5,  # 保守的なレート
        max_workers=max_workers
    )
    
    # 1. 並列PubMed検索（レート制限対応）
    print("📡 Phase 1: PubMed Literature Retrieval")
    print("-" * 40)
    
    queries = raptor_tree.pubmed_retriever.immune_queries
    
    def safe_search_articles(query, max_articles):
        """安全なPubMed検索（キャッシュ優先）"""
        try:
            # まずキャッシュをチェック
            cached_articles = raptor_tree.pubmed_retriever._get_cached_articles(query)
            if cached_articles:
                return cached_articles[:max_articles]
            
            # APIから検索（レート制限対応）
            pmids = raptor_tree.pubmed_retriever.pubmed_api.search_articles(query, max_articles)
            if pmids:
                articles = raptor_tree.pubmed_retriever.pubmed_api.fetch_article_details(pmids)
                
                # 関連度スコア計算
                for article in articles:
                    article.relevance_score = raptor_tree.pubmed_retriever.pubmed_api.calculate_relevance_score(
                        article, raptor_tree.pubmed_retriever.target_terms
                    )
                
                return articles
            else:
                return []
        except Exception as e:
            print(f"Search error for '{query}': {e}")
            return []
    
    # 並列実行
    search_results = processor.parallel_pubmed_search(
        queries, safe_search_articles, max_articles_per_query
    )
    
    # 結果統合
    all_articles = {}
    total_retrieved = 0
    for query, articles in search_results.items():
        total_retrieved += len(articles)
        for article in articles:
            if article.pmid not in all_articles:
                all_articles[article.pmid] = article
    
    raptor_tree.pubmed_articles = all_articles
    retrieval_time = time.time() - start_time
    
    print(f"📊 Retrieval completed: {len(all_articles)} unique articles")
    print(f"   Total retrieved: {total_retrieved}, Time: {retrieval_time:.1f}s")
    
    # 2. 並列エンコーディング（チャンク処理）
    print("\\n⚡ Phase 2: Parallel Text Encoding")
    print("-" * 40)
    
    encoding_start = time.time()
    
    if all_articles:
        articles_list = list(all_articles.items())
        chunk_size = max(1, len(articles_list) // max_workers)
        
        def encode_chunk_safe(chunk_data):
            """安全なチャンクエンコーディング"""
            chunk, chunk_index = chunk_data
            chunk_start = time.time()
            results = []
            
            try:
                for pmid, article in chunk:
                    embedding = raptor_tree.embedder.encode_pubmed_article(article)
                    results.append((pmid, embedding))
                
                chunk_time = time.time() - chunk_start
                print(f"✓ Encoding chunk {chunk_index+1}: {len(chunk)} articles ({chunk_time:.1f}s)")
                return results
            except Exception as e:
                print(f"✗ Encoding error in chunk {chunk_index+1}: {e}")
                return []
        
        # チャンク分割
        chunks = []
        for i in range(0, len(articles_list), chunk_size):
            chunk = articles_list[i:i + chunk_size]
            chunks.append((chunk, len(chunks)))
        
        # 並列エンコーディング実行
        all_embeddings = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_futures = [executor.submit(encode_chunk_safe, chunk_data) for chunk_data in chunks]
            
            for future in as_completed(chunk_futures):
                chunk_results = future.result()
                for pmid, embedding in chunk_results:
                    all_embeddings[pmid] = embedding
        
        raptor_tree.article_embeddings = all_embeddings
        encoding_time = time.time() - encoding_start
        
        print(f"📊 Encoding completed: {len(all_embeddings)} embeddings in {encoding_time:.1f}s")
    else:
        encoding_time = 0
        print("📊 No articles to encode")
    
    # 3. パフォーマンス分析
    total_time = time.time() - start_time
    
    print("\\n🎯 PERFORMANCE SUMMARY")
    print("=" * 30)
    print(f"Total execution time: {total_time:.1f}s")
    print(f"  Literature retrieval: {retrieval_time:.1f}s ({retrieval_time/total_time*100:.0f}%)")
    print(f"  Text encoding: {encoding_time:.1f}s ({encoding_time/total_time*100:.0f}%)")
    print(f"Articles processed: {len(all_articles)}")
    print(f"Workers used: {max_workers}")
    print(f"Rate limit: {processor.requests_per_second} req/s")
    
    # 効率性計算
    if len(all_articles) > 0:
        articles_per_second = len(all_articles) / total_time
        print(f"Processing rate: {articles_per_second:.1f} articles/second")
        
        # 並列効率推定
        estimated_sequential = len(queries) * 2.0 + len(all_articles) * 0.1  # 保守的見積もり
        speedup = estimated_sequential / total_time if total_time > 0 else 1
        efficiency = speedup / max_workers * 100
        
        print(f"Estimated speedup: {speedup:.1f}x")
        print(f"Parallel efficiency: {efficiency:.0f}%")
    
    return {
        'total_time': total_time,
        'retrieval_time': retrieval_time,
        'encoding_time': encoding_time,
        'articles_processed': len(all_articles),
        'workers_used': max_workers,
        'rate_limit': processor.requests_per_second
    }