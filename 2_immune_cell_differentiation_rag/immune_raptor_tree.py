"""
Immune Cell Differentiation RAPTOR Tree System
免疫細胞分化系譜（HSC → CLP → CD4+ T → Treg）専用RAPTOR Tree構築

Author: AI Assistant  
Date: 2024-10-30
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import faiss
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# PubMed検索システムをインポート
from pubmed_retriever import ImmuneCellPubMedRetriever, PubMedArticle

@dataclass
class ImmuneCellNode:
    """免疫細胞ノードデータクラス"""
    id: str
    parent_id: Optional[str]
    level: int
    cell_type: str
    subtype: str
    markers: List[str]
    functions: List[str]
    location: str
    differentiation_factors: List[str] = None
    suppression_mechanisms: List[str] = None
    pubmed_refs: List[str] = None
    summary: str = ""
    embedding: Optional[np.ndarray] = None
    children: List[str] = None
    
    def __post_init__(self):
        if self.differentiation_factors is None:
            self.differentiation_factors = []
        if self.suppression_mechanisms is None:
            self.suppression_mechanisms = []
        if self.pubmed_refs is None:
            self.pubmed_refs = []
        if self.children is None:
            self.children = []

@dataclass 
class ImmuneDifferentiationPath:
    """免疫分化経路データクラス"""
    path_id: str
    source_cell: str
    target_cell: str
    pathway_nodes: List[str]
    key_factors: List[str]
    regulatory_mechanisms: List[str]
    clinical_relevance: List[str]
    confidence_score: float = 0.0

class ImmuneCellEmbedder:
    """免疫細胞知識のベクトル化"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def encode_cell_node(self, node: ImmuneCellNode) -> np.ndarray:
        """細胞ノードをベクトル化"""
        
        # 細胞情報を文字列に統合
        cell_description = f"""
        Cell Type: {node.cell_type}
        Subtype: {node.subtype}
        Markers: {', '.join(node.markers)}
        Functions: {', '.join(node.functions)}
        Location: {node.location}
        Differentiation Factors: {', '.join(node.differentiation_factors)}
        Summary: {node.summary}
        """
        
        if node.suppression_mechanisms:
            cell_description += f"\nSuppression Mechanisms: {', '.join(node.suppression_mechanisms)}"
        
        return self._encode_text(cell_description)
    
    def encode_pubmed_article(self, article: PubMedArticle) -> np.ndarray:
        """PubMed論文をベクトル化"""
        
        article_text = f"""
        Title: {article.title}
        Abstract: {article.abstract}
        Keywords: {', '.join(article.keywords)}
        MeSH Terms: {', '.join(article.mesh_terms)}
        """
        
        return self._encode_text(article_text)
    
    def _encode_text(self, text: str) -> np.ndarray:
        """テキストをベクトル化"""
        
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 平均プーリング
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy().flatten()

class ImmuneCellRAPTORTree:
    """免疫細胞分化系譜RAPTOR Tree"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or "./immune_raptor_cache"
        Path(self.cache_dir).mkdir(exist_ok=True)
        
        self.embedder = ImmuneCellEmbedder()
        self.pubmed_retriever = ImmuneCellPubMedRetriever(cache_dir)
        
        # ノードとインデックス
        self.nodes: Dict[str, ImmuneCellNode] = {}
        self.hierarchy_graph = nx.DiGraph()
        self.faiss_index = None
        self.id_to_node_map: Dict[int, str] = {}
        
        # PubMed知識ベース
        self.pubmed_articles: Dict[str, PubMedArticle] = {}
        self.article_embeddings: Dict[str, np.ndarray] = {}
        
    def load_immune_hierarchy(self, hierarchy_file: str):
        """免疫細胞階層データを読み込み"""
        
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            hierarchy_data = json.load(f)
        
        immune_data = hierarchy_data['immune_cell_hierarchy']
        
        # ルートノードを処理
        root_data = immune_data['root']
        root_node = ImmuneCellNode(
            id=root_data['id'],
            parent_id=root_data['parent'],
            level=1,
            cell_type=root_data['cell_type'],
            subtype=root_data['subtype'],
            markers=root_data['markers'],
            functions=root_data['functions'],
            location=root_data['location'],
            pubmed_refs=root_data['pubmed_refs'],
            summary=root_data['summary'],
            children=root_data['children']
        )
        
        self.nodes[root_node.id] = root_node
        self.hierarchy_graph.add_node(root_node.id, **asdict(root_node))
        
        # 子ノードを処理
        for node_data in immune_data['nodes']:
            node = ImmuneCellNode(
                id=node_data['id'],
                parent_id=node_data['parent'],
                level=node_data['level'],
                cell_type=node_data['cell_type'],
                subtype=node_data['subtype'],
                markers=node_data['markers'],
                functions=node_data['functions'],
                location=node_data['location'],
                differentiation_factors=node_data.get('differentiation_factors', []),
                suppression_mechanisms=node_data.get('suppression_mechanisms', []),
                pubmed_refs=node_data['pubmed_refs'],
                summary=node_data['summary'],
                children=node_data.get('children', [])
            )
            
            self.nodes[node.id] = node
            self.hierarchy_graph.add_node(node.id, **asdict(node))
            
            # 親子関係を追加
            if node.parent_id:
                self.hierarchy_graph.add_edge(node.parent_id, node.id)
        
        print(f"Loaded {len(self.nodes)} immune cell nodes")
        
    def build_embeddings(self):
        """すべてのノードのベクトル化を実行"""
        
        print("Building cell node embeddings...")
        embeddings = []
        node_ids = []
        
        for node_id, node in self.nodes.items():
            embedding = self.embedder.encode_cell_node(node)
            node.embedding = embedding
            embeddings.append(embedding)
            node_ids.append(node_id)
        
        # FAISS インデックス構築
        embeddings_matrix = np.vstack(embeddings).astype('float32')
        
        self.faiss_index = faiss.IndexFlatIP(embeddings_matrix.shape[1])  # Inner Product
        self.faiss_index.add(embeddings_matrix)
        
        # IDマッピング
        self.id_to_node_map = {i: node_id for i, node_id in enumerate(node_ids)}
        
        print(f"Built FAISS index with {len(embeddings)} embeddings")
        
    def integrate_pubmed_knowledge(self, max_articles_per_query: int = 30):
        """PubMed知識を統合"""
        
        print("Retrieving PubMed literature...")
        literature_results = self.pubmed_retriever.retrieve_immune_literature(max_articles_per_query)
        
        # 全論文を統合
        all_articles = {}
        for query, articles in literature_results.items():
            for article in articles:
                if article.pmid not in all_articles:
                    all_articles[article.pmid] = article
        
        self.pubmed_articles = all_articles
        
        # 論文ベクトル化
        print("Encoding PubMed articles...")
        for pmid, article in self.pubmed_articles.items():
            embedding = self.embedder.encode_pubmed_article(article)
            self.article_embeddings[pmid] = embedding
        
        print(f"Integrated {len(self.pubmed_articles)} PubMed articles")
    
    def integrate_pubmed_knowledge_parallel_optimized(self, max_articles_per_query: int = 30, workers: int = 8):
        """最適化された8コア並列処理でPubMed知識を統合"""
        
        import time
        from concurrent.futures import as_completed
        
        workers = min(workers, mp.cpu_count())
        print(f"\n🚀 OPTIMIZED PARALLEL PROCESSING with {workers} workers")
        print("=" * 50)
        
        start_time = time.time()
        
        # 並列でPubMed検索（最適化版）
        queries = self.pubmed_retriever.immune_queries
        
        def fetch_query_optimized(query_data):
            query, index = query_data
            thread_start = time.time()
            try:
                # PubMed検索実行
                pmids = self.pubmed_retriever.pubmed_api.search_articles(query, max_articles_per_query)
                
                if pmids:
                    # 論文詳細取得
                    articles = self.pubmed_retriever.pubmed_api.fetch_article_details(pmids)
                    
                    # 関連度スコア計算
                    for article in articles:
                        article.relevance_score = self.pubmed_retriever.pubmed_api.calculate_relevance_score(
                            article, self.pubmed_retriever.target_terms
                        )
                else:
                    articles = []
                
                thread_time = time.time() - thread_start
                print(f"✓ Worker {index+1}: '{query[:50]}...' - {len(articles)}本 ({thread_time:.1f}s)")
                return index, query, articles, thread_time
            except Exception as e:
                print(f"✗ Worker {index+1} error: {e}")
                return index, query, [], 0
        
        # クエリにインデックスを付加
        query_data = [(query, i) for i, query in enumerate(queries)]
        
        # 並列実行（as_completed使用で完了順に処理）
        print("🔄 Executing PubMed queries in parallel...")
        all_articles = {}
        total_retrieved = 0
        query_times = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_query = {executor.submit(fetch_query_optimized, qd): qd for qd in query_data}
            
            for future in as_completed(future_to_query):
                index, query, articles, thread_time = future.result()
                query_times.append(thread_time)
                total_retrieved += len(articles)
                
                for article in articles:
                    if article.pmid not in all_articles:
                        all_articles[article.pmid] = article
        
        pubmed_time = time.time() - start_time
        self.pubmed_articles = all_articles
        
        print(f"📊 PubMed retrieval completed: {len(all_articles)} unique articles in {pubmed_time:.1f}s")
        print(f"   Total retrieved: {total_retrieved}, Average query time: {sum(query_times)/len(query_times):.1f}s")
        
        # 並列ベクトル化（チャンク処理で最適化）
        print("\n⚡ Encoding articles with optimized parallel processing...")
        
        articles_list = list(all_articles.items())
        chunk_size = max(1, len(articles_list) // workers)
        
        def encode_chunk_parallel(chunk_data):
            chunk, chunk_index = chunk_data
            chunk_start = time.time()
            results = []
            
            for pmid, article in chunk:
                embedding = self.embedder.encode_pubmed_article(article)
                results.append((pmid, embedding))
            
            chunk_time = time.time() - chunk_start
            print(f"✓ Encoding chunk {chunk_index+1}: {len(chunk)} articles ({chunk_time:.1f}s)")
            return results
        
        # チャンクに分割
        chunks = []
        for i in range(0, len(articles_list), chunk_size):
            chunk = articles_list[i:i + chunk_size]
            chunks.append((chunk, len(chunks)))
        
        encoding_start = time.time()
        
        # 並列でエンコーディング実行
        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_futures = [executor.submit(encode_chunk_parallel, chunk_data) for chunk_data in chunks]
            
            all_embeddings = {}
            for future in as_completed(chunk_futures):
                chunk_results = future.result()
                for pmid, embedding in chunk_results:
                    all_embeddings[pmid] = embedding
        
        encoding_time = time.time() - encoding_start
        self.article_embeddings = all_embeddings
        
        total_time = time.time() - start_time
        
        print(f"📊 Encoding completed: {len(all_embeddings)} embeddings in {encoding_time:.1f}s")
        print(f"🎯 TOTAL PARALLEL TIME: {total_time:.1f}s")
        print(f"   PubMed retrieval: {pubmed_time:.1f}s ({pubmed_time/total_time*100:.0f}%)")
        print(f"   Article encoding: {encoding_time:.1f}s ({encoding_time/total_time*100:.0f}%)")
        
        # 効率性レポート
        sequential_estimate = len(queries) * (sum(query_times)/len(query_times)) + len(all_articles) * 0.1
        speedup = sequential_estimate / total_time if total_time > 0 else 1
        
        print(f"\n🚀 PERFORMANCE METRICS:")
        print(f"   Estimated sequential time: {sequential_estimate:.1f}s")
        print(f"   Actual parallel time: {total_time:.1f}s")
        print(f"   Speedup achieved: {speedup:.1f}x")
        print(f"   Parallel efficiency: {speedup/workers*100:.0f}%")
        
        return {
            'total_time': total_time,
            'pubmed_time': pubmed_time,
            'encoding_time': encoding_time,
            'speedup': speedup,
            'articles_count': len(all_articles),
            'workers_used': workers
        }
        
    def build_faiss_index_parallel(self, workers: int = None):
        """並列処理でFAISSインデックス構築"""
        
        if workers is None:
            workers = min(4, mp.cpu_count())
        
        print(f"Building FAISS index with {workers} workers...")
        
        def encode_node_parallel(item):
            node_id, node = item
            text = self._create_node_text(node)
            embedding = self.embedder._encode_text(text)
            return node_id, embedding
        
        # 並列でノードベクトル化
        with ThreadPoolExecutor(max_workers=workers) as executor:
            embedding_results = list(executor.map(encode_node_parallel, self.nodes.items()))
        
        # 埋め込みを設定
        embeddings = []
        node_ids = []
        
        for node_id, embedding in embedding_results:
            self.nodes[node_id].embedding = embedding
            embeddings.append(embedding)
            node_ids.append(node_id)
        
        # FAISS インデックス構築
        embeddings_matrix = np.vstack(embeddings)
        
        # 正規化してコサイン類似度用に準備
        faiss.normalize_L2(embeddings_matrix)
        
        self.faiss_index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        self.faiss_index.add(embeddings_matrix)
        
        # IDマッピング
        self.id_to_node_map = {i: node_id for i, node_id in enumerate(node_ids)}
        
        print(f"Built FAISS index with {len(embeddings)} embeddings (parallel processing)")
    
    def _create_node_text(self, node: ImmuneCellNode) -> str:
        """ノードからテキスト生成"""
        return f"""
        Cell Type: {node.cell_type}
        Subtype: {node.subtype}
        Markers: {', '.join(node.markers)}
        Functions: {', '.join(node.functions)}
        Location: {node.location}
        Differentiation Factors: {', '.join(node.differentiation_factors or [])}
        Suppression Mechanisms: {', '.join(node.suppression_mechanisms or [])}
        Summary: {node.summary}
        """
        
    def find_relevant_literature(self, node_id: str, top_k: int = 5) -> List[Tuple[PubMedArticle, float]]:
        """ノードに関連する文献を検索"""
        
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        if node.embedding is None:
            return []
        
        # ノードと論文の類似度計算
        similarities = []
        for pmid, article_embedding in self.article_embeddings.items():
            similarity = cosine_similarity(
                node.embedding.reshape(1, -1),
                article_embedding.reshape(1, -1)
            )[0, 0]
            similarities.append((pmid, similarity))
        
        # トップK論文を取得
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for pmid, similarity in similarities[:top_k]:
            article = self.pubmed_articles[pmid]
            results.append((article, similarity))
        
        return results
    
    def hierarchical_search(self, query: str, top_k: int = 10, max_depth: int = 4) -> List[Tuple[str, float]]:
        """階層検索：クエリに最も関連するノードを階層的に検索"""
        
        # クエリをベクトル化
        query_embedding = self.embedder._encode_text(query).astype('float32')
        
        # FAISS検索
        scores, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1), 
            min(len(self.nodes), top_k)
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.id_to_node_map):  # 有効なインデックスチェック
                node_id = self.id_to_node_map[idx]
                results.append((node_id, float(score)))
        
        return results
    
    def trace_differentiation_path(self, source_cell_type: str, target_cell_type: str) -> Optional[ImmuneDifferentiationPath]:
        """分化経路をトレース"""
        
        # 対象ノードを検索
        source_node = None
        target_node = None
        
        for node in self.nodes.values():
            if source_cell_type.lower() in node.cell_type.lower():
                source_node = node
            if target_cell_type.lower() in node.cell_type.lower():
                target_node = node
        
        if not source_node or not target_node:
            return None
        
        # 最短経路を検索
        try:
            path = nx.shortest_path(self.hierarchy_graph, source_node.id, target_node.id)
        except nx.NetworkXNoPath:
            return None
        
        # 経路の詳細を収集
        pathway_nodes = []
        key_factors = []
        regulatory_mechanisms = []
        clinical_relevance = []
        
        for node_id in path:
            node = self.nodes[node_id]
            pathway_nodes.append(f"{node.cell_type} ({node.subtype})")
            key_factors.extend(node.differentiation_factors)
            if node.suppression_mechanisms:
                regulatory_mechanisms.extend(node.suppression_mechanisms)
        
        # Treg関連の臨床的意義
        if "treg" in target_cell_type.lower():
            clinical_relevance = [
                "Type 1 diabetes",
                "Multiple sclerosis", 
                "Rheumatoid arthritis",
                "Inflammatory bowel disease",
                "Cancer immunotherapy"
            ]
        
        path_obj = ImmuneDifferentiationPath(
            path_id=f"path_{source_node.id}_to_{target_node.id}",
            source_cell=source_node.cell_type,
            target_cell=target_node.cell_type,
            pathway_nodes=pathway_nodes,
            key_factors=list(set(key_factors)),
            regulatory_mechanisms=list(set(regulatory_mechanisms)),
            clinical_relevance=clinical_relevance,
            confidence_score=1.0  # 基本経路は確実
        )
        
        return path_obj
    
    def generate_cell_summary(self, node_id: str, include_literature: bool = True) -> str:
        """細胞ノードの詳細サマリーを生成"""
        
        if node_id not in self.nodes:
            return "Node not found"
        
        node = self.nodes[node_id]
        
        summary_lines = []
        summary_lines.append(f"# {node.cell_type} ({node.subtype})")
        summary_lines.append("")
        summary_lines.append(f"**Level**: {node.level}")
        summary_lines.append(f"**Location**: {node.location}")
        summary_lines.append("")
        
        summary_lines.append("## Markers")
        for marker in node.markers:
            summary_lines.append(f"- {marker}")
        summary_lines.append("")
        
        summary_lines.append("## Functions")
        for function in node.functions:
            summary_lines.append(f"- {function}")
        summary_lines.append("")
        
        if node.differentiation_factors:
            summary_lines.append("## Differentiation Factors")
            for factor in node.differentiation_factors:
                summary_lines.append(f"- {factor}")
            summary_lines.append("")
        
        if node.suppression_mechanisms:
            summary_lines.append("## Suppression Mechanisms")
            for mechanism in node.suppression_mechanisms:
                summary_lines.append(f"- {mechanism}")
            summary_lines.append("")
        
        summary_lines.append("## Summary")
        summary_lines.append(node.summary)
        summary_lines.append("")
        
        # 関連文献を追加
        if include_literature:
            relevant_lit = self.find_relevant_literature(node_id, top_k=3)
            if relevant_lit:
                summary_lines.append("## Relevant Literature")
                for i, (article, similarity) in enumerate(relevant_lit, 1):
                    summary_lines.append(f"\n{i}. **{article.title}**")
                    summary_lines.append(f"   - PMID: {article.pmid}")
                    summary_lines.append(f"   - Similarity: {similarity:.3f}")
                    summary_lines.append(f"   - Journal: {article.journal} ({article.pub_date})")
        
        return "\n".join(summary_lines)
    
    def visualize_hierarchy(self, output_file: str = None):
        """免疫細胞階層の可視化"""
        
        plt.figure(figsize=(15, 10))
        
        # グラフのレイアウト計算
        pos = nx.spring_layout(self.hierarchy_graph, k=3, iterations=50)
        
        # レベル別色分け
        level_colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4'}
        node_colors = []
        node_sizes = []
        
        for node_id in self.hierarchy_graph.nodes():
            node = self.nodes[node_id]
            node_colors.append(level_colors.get(node.level, '#CCCCCC'))
            # Tregノードを大きく表示
            if 'treg' in node.cell_type.lower():
                node_sizes.append(2000)
            else:
                node_sizes.append(1000)
        
        # ノード描画
        nx.draw_networkx_nodes(
            self.hierarchy_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # エッジ描画
        nx.draw_networkx_edges(
            self.hierarchy_graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )
        
        # ラベル追加
        labels = {}
        for node_id in self.hierarchy_graph.nodes():
            node = self.nodes[node_id]
            labels[node_id] = f"{node.cell_type}\n({node.subtype})"
        
        nx.draw_networkx_labels(
            self.hierarchy_graph, pos,
            labels, font_size=10,
            font_weight='bold'
        )
        
        plt.title("Immune Cell Differentiation Hierarchy (HSC → Treg)", fontsize=16, fontweight='bold')
        
        # 凡例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=level_colors[1], 
                      markersize=10, label='Level 1: HSC'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=level_colors[2], 
                      markersize=10, label='Level 2: CLP'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=level_colors[3], 
                      markersize=10, label='Level 3: CD4+ T cell'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=level_colors[4], 
                      markersize=10, label='Level 4: Treg cells')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.axis('off')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Hierarchy visualization saved to: {output_file}")
        
        plt.show()
    
    def save_raptor_tree(self, output_file: str):
        """RAPTOR Treeをファイルに保存"""
        
        # numpy配列を除外してnode情報を保存
        nodes_data = {}
        for node_id, node in self.nodes.items():
            node_dict = asdict(node)
            # embeddingはNone or numpy配列なので除外
            node_dict['embedding'] = None
            nodes_data[node_id] = node_dict
        
        tree_data = {
            'nodes': nodes_data,
            'pubmed_articles': {pmid: asdict(article) for pmid, article in self.pubmed_articles.items()},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_nodes': len(self.nodes),
                'total_articles': len(self.pubmed_articles)
            }
        }
        
        # ベクトルデータは別ファイルに保存（pickle使用）
        embeddings_data = {
            'node_embeddings': {node_id: node.embedding for node_id, node in self.nodes.items()},
            'article_embeddings': self.article_embeddings
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
        
        embeddings_file = output_file.replace('.json', '_embeddings.pkl')
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        # FAISS インデックスも保存
        if self.faiss_index:
            faiss_file = output_file.replace('.json', '_faiss.index')
            faiss.write_index(self.faiss_index, faiss_file)
        
        print(f"RAPTOR Tree saved to: {output_file}")
        print(f"Embeddings saved to: {embeddings_file}")
        if self.faiss_index:
            print(f"FAISS index saved to: {faiss_file}")

def main():
    """メイン実行関数"""
    print("🧬 Immune Cell Differentiation RAPTOR Tree System")
    print("=" * 60)
    
    # パス設定
    base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
    cache_dir = base_dir / "data/immune_cell_differentiation"
    hierarchy_file = cache_dir / "immune_cell_hierarchy.json"
    
    # RAPTOR Tree初期化
    raptor_tree = ImmuneCellRAPTORTree(cache_dir=str(cache_dir))
    
    # 階層データ読み込み
    print("Loading immune cell hierarchy...")
    raptor_tree.load_immune_hierarchy(str(hierarchy_file))
    
    # ベクトル化実行
    print("Building embeddings...")
    raptor_tree.build_embeddings()
    
    # PubMed知識統合
    print("Integrating PubMed knowledge...")
    raptor_tree.integrate_pubmed_knowledge(max_articles_per_query=20)
    
    # サンプル検索テスト
    print("\n" + "=" * 40)
    print("Sample Queries:")
    
    test_queries = [
        "FOXP3+細胞はどこから分化するか？",
        "Treg細胞の主要な免疫抑制機能は何か？",
        "CTLA-4はTreg細胞でどのような役割を果たすか？"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = raptor_tree.hierarchical_search(query, max_depth=4)
        
        for i, (node, score) in enumerate(results[:2], 1):
            print(f"  {i}. {node.cell_type} ({node.subtype}) - Score: {score:.3f}")
    
    # 分化経路トレース
    print("\n" + "=" * 40)
    print("Differentiation Path Tracing:")
    
    path = raptor_tree.trace_differentiation_path("HSC", "Treg")
    if path:
        print(f"Path: {' → '.join(path.pathway_nodes)}")
        print(f"Key Factors: {', '.join(path.key_factors[:5])}")
    
    # 可視化
    print("\nGenerating hierarchy visualization...")
    output_viz = cache_dir / "raptor_trees/immune_hierarchy_visualization.png"
    raptor_tree.visualize_hierarchy(str(output_viz))
    
    # RAPTOR Tree保存
    print("Saving RAPTOR Tree...")
    output_tree = cache_dir / "raptor_trees/immune_cell_raptor_tree.json"
    raptor_tree.save_raptor_tree(str(output_tree))
    
    print("\n" + "=" * 60)
    print("🎯 Immune Cell RAPTOR Tree construction completed!")
    print(f"Tree file: {output_tree}")
    print(f"Visualization: {output_viz}")

if __name__ == "__main__":
    import sys
    
    # コマンドライン引数で並列処理を制御
    if "--parallel" in sys.argv:
        print("🚀 Running with parallel processing")
        
        # 並列処理版メイン関数
        print("🧬 Immune Cell Differentiation RAPTOR Tree System (PARALLEL)")
        print("=" * 60)
        
        start_time = time.time()
        workers = min(8, mp.cpu_count())
        
        # RAPTOR Tree初期化（絶対パス使用）
        base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
        cache_dir = base_dir / "data/immune_cell_differentiation"
        raptor_tree = ImmuneCellRAPTORTree(str(cache_dir))
        
        print(f"\n免疫細胞階層データを読み込んでいます...")
        immune_file = cache_dir / "immune_cell_hierarchy.json"
        raptor_tree.load_immune_hierarchy(str(immune_file))
        print(f"✓ {len(raptor_tree.nodes)}個の免疫細胞ノードを読み込み完了")
        
        print(f"\nRAPTOR Tree用のFAISSインデックスを構築しています（{workers}並列）...")
        raptor_tree.build_faiss_index_parallel(workers)
        print("✓ ベクトルデータベース構築完了")
        
        print(f"\nPubMed文献を検索・統合しています（最適化{workers}並列）...")
        parallel_metrics = raptor_tree.integrate_pubmed_knowledge_parallel_optimized(
            max_articles_per_query=30, 
            workers=workers
        )
        print("✓ PubMed知識統合完了")
        
        # 検索デモ
        print("\n階層検索デモを実行中...")
        query = "FOXP3+ regulatory T cell differentiation"
        results = raptor_tree.hierarchical_search(query, top_k=3)
        
        print(f"\nクエリ: '{query}'")
        for i, (node_id, score) in enumerate(results, 1):
            node = raptor_tree.nodes[node_id]
            print(f"  {i}. {node.cell_type} ({node.subtype}) - スコア: {score:.3f}")
        
        # 分化経路トレース
        print("\n" + "=" * 40)
        print("Differentiation Path Tracing:")
        
        path = raptor_tree.trace_differentiation_path("HSC", "Treg")
        if path:
            print(f"Path: {' → '.join(path.pathway_nodes)}")
            print(f"Key Factors: {', '.join(path.key_factors[:5])}")
        
        # 可視化
        print("\n階層構造を可視化しています...")
        output_viz = cache_dir / "raptor_trees/immune_hierarchy_visualization.png"
        raptor_tree.visualize_hierarchy(str(output_viz))
        print(f"✓ 可視化ファイル生成: {output_viz.name}")
        
        # RAPTOR Tree保存
        print("\nRAPTOR Treeを保存しています...")
        output_tree = cache_dir / "raptor_trees/immune_cell_raptor_tree.json"
        raptor_tree.save_raptor_tree(str(output_tree))
        
        # 埋め込みベクトルをPickle形式で保存
        embeddings_file = cache_dir / "raptor_trees/immune_cell_raptor_tree_embeddings.pkl"
        embeddings_data = {
            'node_embeddings': {node_id: node.embedding for node_id, node in raptor_tree.nodes.items()},
            'article_embeddings': raptor_tree.article_embeddings
        }
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"✓ 埋め込み保存: {embeddings_file.name}")
        
        # FAISS インデックス保存
        if raptor_tree.faiss_index is not None:
            faiss_file = cache_dir / "raptor_trees/immune_cell_raptor_tree_faiss.index"
            faiss.write_index(raptor_tree.faiss_index, str(faiss_file))
            print(f"✓ FAISSインデックス保存: {faiss_file.name}")
        
        execution_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎯 Immune Cell RAPTOR Tree construction completed!")
        print(f"📊 実行時間: {execution_time:.2f}秒")
        print(f"⚡ 並列処理: {workers}ワーカー使用")
        print(f"📁 Tree file: {output_tree.name}")
        print(f"🖼️  Visualization: {output_viz.name}")
        
    else:
        # 通常実行
        main()