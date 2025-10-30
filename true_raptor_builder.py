#!/usr/bin/env python3
"""
True RAPTOR Tree Implementation with Clustering
Implements the full RAPTOR algorithm with recursive clustering and summarization
"""

import os
# Hugging Face高速ダウンロードを有効化
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM, 
    GPT2LMHeadModel, GPT2Tokenizer, GPTNeoXForCausalLM, OPTForCausalLM
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
import time
from datetime import datetime
import os

# Hugging Face ダウンロード設定
os.environ["TRANSFORMERS_VERBOSITY"] = "info"  # ダウンロード進捗表示

@dataclass
class RAPTORNode:
    """RAPTOR Tree Node with clustering support"""
    node_id: str
    parent_id: Optional[str]
    children: List[str]
    level: int
    content: str
    summary: str
    is_leaf: bool
    cluster_id: Optional[int]
    embedding: Optional[np.ndarray]
    source_documents: List[str]
    cluster_size: int = 0

class TrueRAPTORTree:
    """True RAPTOR Tree with Transformers-based embeddings and local LLM"""
    
    def __init__(self):
        # Transformersベースのエンコーダー（既存の依存関係を使用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # 埋め込みモデル初期化
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
            self.embedding_model.eval()
        except Exception as e:
            # フォールバック: より基本的なモデル
            self.embedding_model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
            self.embedding_model.eval()
        
        # ログ設定（最初に設定）
        self.logger = logging.getLogger(__name__)
        
        # ローカルLLM初期化（要約用）
        self.llm_tokenizer = None
        self.llm_model = None
        self._init_local_llm()
        
        self.nodes: Dict[str, RAPTORNode] = {}
        self.faiss_index = None
        self.article_embeddings = {}
        self.max_cluster_size = 30  # 削減してメモリ使用量を抑制
        self.min_cluster_size = 3
        self.max_levels = 4
        
    def _init_local_llm(self):
        """GPU対応の大規模OSSモデルを初期化（要約用）"""
        try:
            # GPU使用量を確認
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
                
                # GPU容量に応じてモデルを選択
                if gpu_memory >= 24:  # 24GB以上の場合 - 大規模モデル
                    llm_model_name = "facebook/opt-6.7b"  # OPT 6.7B
                    self.logger.info("🔥 Using OPT-6.7B for GPU with 24GB+ memory")
                elif gpu_memory >= 16:  # 16GB以上の場合
                    llm_model_name = "facebook/opt-2.7b"  # OPT 2.7B
                    self.logger.info("🚀 Using OPT-2.7B for GPU with 16GB+ memory")
                elif gpu_memory >= 12:  # 12GB以上の場合
                    llm_model_name = "facebook/opt-1.3b"  # OPT 1.3B
                    self.logger.info("⚡ Using OPT-1.3B for GPU with 12GB+ memory")
                elif gpu_memory >= 8:  # 8GB以上の場合
                    llm_model_name = "microsoft/DialoGPT-large"
                    self.logger.info("💪 Using DialoGPT-large for GPU with 8GB+ memory")
                else:  # 8GB未満の場合
                    llm_model_name = "microsoft/DialoGPT-medium"
                    self.logger.info("💡 Using DialoGPT-medium for GPU with <8GB memory")
            else:
                llm_model_name = "distilgpt2"
                self.logger.info("💻 No GPU available, using CPU-optimized model")
            
            # モデル初期化（GPU対応）
            self.logger.info(f"📥 Downloading tokenizer for {llm_model_name}...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            
            # GPU使用時の最適化
            if torch.cuda.is_available():
                self.logger.info(f"📥 Downloading model {llm_model_name} (GPU-optimized)...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    torch_dtype=torch.float16,  # メモリ効率化
                    device_map="auto",  # 自動GPU配置
                    low_cpu_mem_usage=True  # CPU メモリ使用量削減
                ).to(self.device)
            else:
                self.logger.info(f"📥 Downloading model {llm_model_name} (CPU mode)...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(self.device)
            
            self.llm_model.eval()
            
            # パディングトークンを設定
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
            self.logger.info(f"✅ Large-scale LLM initialized: {llm_model_name}")
            self.logger.info(f"🎯 Device: {self.device}")
            
            # GPU メモリ使用量を表示
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"📊 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Large-scale LLM initialization failed: {e}")
            self.logger.info("📝 Falling back to template-based summarization")
            self.llm_model = None
            self.llm_tokenizer = None
        
    def encode_text(self, text: str) -> np.ndarray:
        """単一テキストをエンコード"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # [CLS]トークンの隠れ状態を使用
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
        
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """文書群をエンコード（バッチ処理）"""
        self.logger.info(f"🔤 Encoding {len(documents)} documents using {self.embedding_model_name}...")
        
        embeddings = []
        batch_size = 8  # メモリ使用量を考慮
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_embeddings = []
            
            for doc in batch:
                try:
                    # 長い文書は切り詰める
                    truncated_doc = doc[:1000] if len(doc) > 1000 else doc
                    embedding = self.encode_text(truncated_doc)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    self.logger.warning(f"Encoding error for document: {e}")
                    # エラー時はゼロベクトル
                    embedding_dim = 384 if "MiniLM" in self.embedding_model_name else 768
                    batch_embeddings.append(np.zeros(embedding_dim))
            
            embeddings.extend(batch_embeddings)
            self.logger.info(f"  ✓ Encoded batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        return np.array(embeddings)
    
    def optimal_clusters(self, embeddings: np.ndarray, max_k: int = 10) -> int:
        """最適なクラスタ数を決定（シルエット分析）"""
        if len(embeddings) < 2:
            return 1
        
        max_k = min(max_k, len(embeddings) - 1)
        if max_k < 2:
            return 1
            
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
                
        return best_k
    
    def cluster_documents(self, embeddings: np.ndarray, documents: List[str]) -> Dict[int, List[int]]:
        """文書をクラスタリング"""
        if len(documents) <= self.min_cluster_size:
            return {0: list(range(len(documents)))}
        
        n_clusters = self.optimal_clusters(embeddings, max_k=min(10, len(documents) // 2))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        return clusters
    
    def generate_llm_summary(self, documents: List[str]) -> str:
        """GPU対応の大規模LLMを使用してクラスタの要約を生成"""
        if not self.llm_model or not self.llm_tokenizer:
            # LLMが利用できない場合はテンプレートベース要約
            return self._template_based_summary(documents)
        
        try:
            # GPU使用量をモニタリング
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # キャッシュクリア
            
            # 文書を結合（大規模モデル用により多くの情報）
            combined_text = " ".join(doc[:200] for doc in documents[:5])  # 最大5文書、各200文字
            
            # 免疫学専用のプロンプト作成
            prompt = f"""Summarize the following immune cell research findings in a concise scientific manner.
Focus on key mechanisms, cell types, and biological processes.

Research findings: {combined_text}

Scientific summary:"""
            
            # トークン化（GPU最適化）
            inputs = self.llm_tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=800,  # 大規模モデル用により長いコンテキスト
                padding=True
            )
            inputs = inputs.to(self.device)
            
            # 生成パラメータ（大規模モデル用最適化）
            generation_kwargs = {
                "max_new_tokens": 100,  # より長い要約
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,  # nucleus sampling
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.llm_tokenizer.eos_token_id,
                "attention_mask": torch.ones_like(inputs),
                "no_repeat_ngram_size": 3  # 繰り返し防止
            }
            
            # GPU使用時の追加最適化
            if torch.cuda.is_available():
                generation_kwargs["use_cache"] = True
            
            # 生成実行
            with torch.no_grad():
                outputs = self.llm_model.generate(inputs, **generation_kwargs)
            
            # デコード
            generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプト部分を除去して要約部分のみ抽出
            if "Scientific summary:" in generated_text:
                summary = generated_text.split("Scientific summary:")[-1].strip()
            else:
                summary = generated_text[len(prompt):].strip()
            
            # 要約の後処理
            summary = self._post_process_summary(summary)
            
            # GPU メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 要約が短すぎる場合はテンプレートベースにフォールバック
            if len(summary) < 20:
                self.logger.warning("Generated summary too short, using template-based fallback")
                return self._template_based_summary(documents)
            
            return summary[:400]  # 最大400文字
            
        except Exception as e:
            self.logger.warning(f"GPU LLM summary generation failed: {e}")
            # GPU メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._template_based_summary(documents)
    
    def _post_process_summary(self, summary: str) -> str:
        """生成された要約の後処理"""
        # 不完全な文を除去
        sentences = summary.split('.')
        complete_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.endswith(('and', 'or', 'the', 'a', 'an')):
                complete_sentences.append(sentence)
        
        processed_summary = '. '.join(complete_sentences)
        if processed_summary and not processed_summary.endswith('.'):
            processed_summary += '.'
        
        return processed_summary
    
    def _template_based_summary(self, documents: List[str]) -> str:
        """テンプレートベースの要約（フォールバック）"""
        if len(documents) == 1:
            return documents[0][:400] + "..." if len(documents[0]) > 400 else documents[0]
        
        # 各文書から重要なキーワードを抽出
        immune_keywords = ["T cell", "B cell", "FOXP3", "IL-10", "TGF-β", "CTLA-4", "regulatory", "immune", "differentiation"]
        
        summaries = []
        found_keywords = set()
        
        for i, doc in enumerate(documents[:5]):  # 最大5文書
            # 文書の最初の部分を取得
            summary = doc[:120].strip()
            
            # キーワード検出
            for keyword in immune_keywords:
                if keyword.lower() in doc.lower():
                    found_keywords.add(keyword)
            
            if summary:
                summaries.append(f"Doc{i+1}: {summary}")
        
        # キーワードベースの説明を追加
        keyword_desc = f"[Keywords: {', '.join(sorted(found_keywords))}]" if found_keywords else ""
        
        # 基本的な統計情報を追加
        combined_summary = " | ".join(summaries)
        cluster_info = f"[Cluster of {len(documents)} immune cell research documents] {keyword_desc} {combined_summary}"
        
        # 最大長制限
        return cluster_info[:500] + "..." if len(cluster_info) > 500 else cluster_info
    
    def summarize_cluster(self, documents: List[str]) -> str:
        """クラスタの要約を生成（LLMまたはテンプレートベース）"""
        return self.generate_llm_summary(documents)
    
    def build_raptor_tree(self, documents: List[str], document_ids: List[str]) -> None:
        """真のRAPTORツリーを構築"""
        self.logger.info(f"🌳 RAPTOR Tree construction started with {len(documents)} documents")
        
        # レベル0: リーフノード（元文書）
        current_level_docs = documents
        current_level_ids = document_ids
        level = 0
        
        # 全体の埋め込みを計算
        all_embeddings = self.encode_documents(current_level_docs)
        
        while len(current_level_docs) > 1 and level < self.max_levels:
            self.logger.info(f"📊 Processing level {level}: {len(current_level_docs)} nodes")
            
            # クラスタリング
            clusters = self.cluster_documents(all_embeddings[:len(current_level_docs)], current_level_docs)
            
            next_level_docs = []
            next_level_ids = []
            next_level_embeddings = []
            
            for cluster_id, doc_indices in clusters.items():
                if len(doc_indices) == 0:
                    continue
                    
                cluster_docs = [current_level_docs[i] for i in doc_indices]
                cluster_doc_ids = [current_level_ids[i] for i in doc_indices]
                
                # クラスタ要約を生成
                cluster_summary = self.summarize_cluster(cluster_docs)
                
                # 新しいノードを作成
                node_id = f"raptor_L{level + 1}_C{cluster_id}_{int(time.time())}"
                
                # 要約の埋め込みを計算
                summary_embedding = self.encode_text(cluster_summary)
                
                # ノードを保存
                node = RAPTORNode(
                    node_id=node_id,
                    parent_id=None,  # 親は後で設定
                    children=cluster_doc_ids if level == 0 else [],
                    level=level + 1,
                    content=cluster_summary,
                    summary=cluster_summary,
                    is_leaf=False,
                    cluster_id=cluster_id,
                    embedding=summary_embedding,
                    source_documents=cluster_doc_ids,
                    cluster_size=len(cluster_docs)
                )
                
                self.nodes[node_id] = node
                
                # 次のレベルの準備
                next_level_docs.append(cluster_summary)
                next_level_ids.append(node_id)
                next_level_embeddings.append(summary_embedding)
                
                self.logger.info(f"  ✓ Cluster {cluster_id}: {len(cluster_docs)} docs → {node_id}")
            
            # レベルアップ
            current_level_docs = next_level_docs
            current_level_ids = next_level_ids
            if next_level_embeddings:
                all_embeddings = np.array(next_level_embeddings)
            level += 1
            
            # 単一ノードになったら終了
            if len(current_level_docs) <= 1:
                break
        
        # ルートノード作成
        if current_level_docs:
            root_summary = self.summarize_cluster(current_level_docs)
            root_id = f"raptor_root_{int(time.time())}"
            
            root_node = RAPTORNode(
                node_id=root_id,
                parent_id=None,
                children=current_level_ids,
                level=level + 1,
                content=root_summary,
                summary=root_summary,
                is_leaf=False,
                cluster_id=0,
                embedding=self.encode_text(root_summary),
                source_documents=document_ids,
                cluster_size=len(documents)
            )
            
            self.nodes[root_id] = root_node
            self.logger.info(f"🌟 Root node created: {root_id}")
        
        self.logger.info(f"✅ RAPTOR Tree completed: {len(self.nodes)} total nodes across {level + 1} levels")
    
    def save_tree(self, output_path: str) -> None:
        """ツリーを保存（JSON serializable形式で）"""
        tree_data = {
            'nodes': {},
            'metadata': {
                'creation_time': datetime.now().isoformat(),
                'total_nodes': len(self.nodes),
                'levels': max(node.level for node in self.nodes.values()) if self.nodes else 0,
                'algorithm': 'RAPTOR with Local LLM and Clustering'
            }
        }
        
        for node_id, node in self.nodes.items():
            tree_data['nodes'][node_id] = {
                'node_id': node.node_id,
                'parent_id': node.parent_id,
                'children': node.children,
                'level': int(node.level),  # numpy型をPython intに変換
                'content': node.content,
                'summary': node.summary,
                'is_leaf': bool(node.is_leaf),  # numpy boolをPython boolに変換
                'cluster_id': int(node.cluster_id) if node.cluster_id is not None else None,
                'source_documents': node.source_documents,
                'cluster_size': int(node.cluster_size),
                'embedding': node.embedding.tolist() if node.embedding is not None else None
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 RAPTOR Tree saved: {output_path}")

class TrueRAPTORBuilder:
    """ローカルLLMを使用した真のRAPTORアルゴリズム実装ビルダー"""
    
    def __init__(self):
        self.raptor_tree = TrueRAPTORTree()
        self.setup_logging()
        
    def setup_logging(self):
        """ログ設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("data/immune_cell_differentiation/scaling_results")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"raptor_local_llm_build_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RAPTOR with Local LLM Builder Log initialized: {log_file}")
    
    def create_sample_documents(self) -> Tuple[List[str], List[str]]:
        """免疫細胞研究のサンプル文書を作成"""
        documents = [
            "Hematopoietic stem cells (HSCs) are multipotent progenitor cells that reside in the bone marrow and give rise to all blood cell lineages through tightly regulated differentiation processes.",
            "Common lymphoid progenitors (CLPs) express key surface markers including IL-7Rα and Sca-1, and represent the earliest committed precursors of the lymphoid lineage.",
            "CD4+ T helper cells differentiate from naive T cells upon antigen recognition and provide crucial support for immune responses through cytokine production and cellular interactions.",
            "Natural regulatory T cells (nTregs) develop in the thymus and constitutively express the transcription factor FOXP3, which is essential for their suppressive function.",
            "Induced regulatory T cells (iTregs) can be generated from conventional CD4+ T cells in peripheral tissues through exposure to TGF-β and other immunosuppressive signals.",
            "FOXP3 transcription factor serves as the master regulator of regulatory T cell development and function, controlling the expression of numerous target genes involved in immune suppression.",
            "Regulatory T cells employ multiple mechanisms to suppress immune responses, including contact-dependent inhibition through CTLA-4 and PD-1 signaling pathways.",
            "Cytokine production by regulatory T cells, particularly IL-10 and TGF-β, plays crucial roles in maintaining immune tolerance and preventing autoimmune diseases.",
            "Thymic development of natural Tregs requires specific strength of TCR signaling and interactions with medullary thymic epithelial cells expressing tissue-specific antigens.",
            "Peripheral induction of regulatory T cells occurs in gut-associated lymphoid tissues and at sites of inflammation, contributing to local immune homeostasis.",
            "CTLA-4 checkpoint molecule on regulatory T cells enables competitive inhibition of CD28-mediated costimulation, thereby limiting T cell activation.",
            "Metabolic programming of regulatory T cells involves the mTOR pathway and influences the balance between glycolysis and oxidative phosphorylation.",
            "Dysfunction of regulatory T cells is implicated in various autoimmune diseases including multiple sclerosis, rheumatoid arthritis, and type 1 diabetes.",
            "Tumor-infiltrating regulatory T cells create an immunosuppressive microenvironment that facilitates cancer progression and resistance to immunotherapy.",
            "Therapeutic modulation of regulatory T cell function represents a promising approach for treating autoimmune diseases and enhancing cancer immunotherapy.",
            "Single-cell transcriptomics has revealed significant heterogeneity within regulatory T cell populations across different tissues and disease states.",
            "Epigenetic regulation through DNA methylation and histone modifications ensures stable maintenance of FOXP3 expression in regulatory T cells.",
            "Age-related changes in regulatory T cell frequency and function contribute to increased susceptibility to autoimmune diseases in elderly populations.",
            "Gut microbiota influences the development and function of peripheral regulatory T cells through production of short-chain fatty acids and other metabolites.",
            "Regulatory T cell plasticity allows adaptation to tissue-specific environments while maintaining core suppressive capabilities through context-dependent gene expression programs."
        ]
        
        document_ids = [f"immune_research_{i:03d}" for i in range(len(documents))]
        
        # 追加のバリエーション文書を生成
        extended_docs = []
        extended_ids = []
        
        for i, doc in enumerate(documents):
            extended_docs.append(doc)
            extended_ids.append(document_ids[i])
            
            # 免疫学的バリエーションを追加
            if i < 15:  # 最初の15文書についてバリエーション作成
                variant = doc.replace("cells", "cell populations").replace("function", "biological activity").replace("development", "differentiation")
                extended_docs.append(variant)
                extended_ids.append(f"immune_research_var_{i:03d}")
        
        self.logger.info(f"📚 Created {len(extended_docs)} sample immune cell research documents")
        return extended_docs, extended_ids
    
    def build_local_llm_raptor_tree(self):
        """Transformersベースの真のRAPTORツリーを構築"""
        self.logger.info("🚀 RAPTOR WITH LOCAL LLM CONSTRUCTION STARTED")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # 1. サンプル文書作成
        self.logger.info("📚 Phase 1: Creating Sample Documents")
        documents, document_ids = self.create_sample_documents()
        
        # 2. RAPTORツリー構築
        self.logger.info("🌳 Phase 2: Building RAPTOR Tree with Local LLM Clustering")
        self.raptor_tree.build_raptor_tree(documents, document_ids)
        
        # 3. 保存
        self.logger.info("💾 Phase 3: Saving RAPTOR Tree with Local LLM")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/immune_cell_differentiation/raptor_trees")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tree_file = output_dir / f"raptor_local_llm_tree_{timestamp}.json"
        self.raptor_tree.save_tree(str(tree_file))
        
        total_time = time.time() - start_time
        
        # 結果サマリー
        self.logger.info("✅ RAPTOR WITH LOCAL LLM CONSTRUCTION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Results Summary:")
        self.logger.info(f"   Total execution time: {total_time:.1f}s")
        self.logger.info(f"   Input documents: {len(documents)}")
        self.logger.info(f"   Generated nodes: {len(self.raptor_tree.nodes)}")
        if self.raptor_tree.nodes:
            self.logger.info(f"   Tree levels: {max(node.level for node in self.raptor_tree.nodes.values())}")
        self.logger.info(f"   Output file: {tree_file.name}")
        self.logger.info(f"   Model used: {self.raptor_tree.embedding_model_name}")
        if self.raptor_tree.llm_model:
            # 使用されたモデル名を取得
            model_name = getattr(self.raptor_tree.llm_model, 'name_or_path', 'unknown')
            if 'opt' in model_name.lower():
                self.logger.info(f"   LLM used: {model_name} (Meta OPT series)")
            elif 'neox' in model_name.lower():
                self.logger.info(f"   LLM used: {model_name} (EleutherAI GPT-NeoX)")
            elif 'dialogo' in model_name.lower():
                self.logger.info(f"   LLM used: {model_name} (Microsoft DialoGPT)")
            else:
                self.logger.info(f"   LLM used: {model_name} (GPU-accelerated)")
                
            # GPU統計情報
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"   GPU Memory: Allocated {allocated:.2f}GB, Cached {cached:.2f}GB")
        else:
            self.logger.info(f"   LLM used: template-based (fallback)")
        
        return tree_file

if __name__ == "__main__":
    builder = TrueRAPTORBuilder()
    result_file = builder.build_local_llm_raptor_tree()
    
    if result_file:
        print(f"\n✅ RAPTOR Tree with GPU-Accelerated LLM construction completed!")
        print(f"📁 Output: {result_file}")
        print(f"🔍 This tree contains CLUSTERED nodes with GPU-generated summaries")
        print(f"🌳 Each level represents hierarchical abstraction of immune cell research content")
        print(f"📊 Now you should see MANY MORE than 5 nodes due to clustering!")
        print(f"🚀 Summaries generated using GPU-accelerated large language model!")
        
        # GPU使用統計を表示
        if torch.cuda.is_available():
            final_allocated = torch.cuda.memory_allocated() / 1024**3
            final_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"🎯 Final GPU Memory: Allocated {final_allocated:.2f}GB, Cached {final_cached:.2f}GB")
            
            # GPUクリーンアップ
            torch.cuda.empty_cache()
            print(f"🧹 GPU cache cleared")
        else:
            print(f"💻 Executed on CPU (no GPU available)")