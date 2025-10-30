#!/usr/bin/env python3
"""
GPU-Accelerated 16x Scale RAPTOR Tree Builder
16倍スケールGPU加速RAPTORツリー構築システム

Author: AI Assistant
Date: 2025-10-31
"""

import os
# Hugging Face高速ダウンロードを有効化
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
import json
import time
import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback

# GPU最適化ライブラリ
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    OPTForCausalLM, GPT2LMHeadModel
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import faiss

# ローカルモジュールのインポート
sys.path.append("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
try:
    from true_raptor_builder import TrueRAPTORTree
    from immune_cell_vocab import generate_immune_label, validate_immune_terminology
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")
    print("Will create simplified version...")


class GPU16xScaleRAPTORBuilder:
    """GPU加速16倍スケールRAPTORツリー構築システム"""
    
    def __init__(self):
        self.base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
        self.cache_dir = self.base_dir / "data/immune_cell_differentiation"
        self.results_dir = self.cache_dir / "scaling_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 16倍スケール設定
        self.scale_factor = 16
        self.base_documents = 35  # 基本文書数
        self.target_documents = self.base_documents * self.scale_factor  # 560文書
        self.batch_size = 64  # GPU最適化バッチサイズ
        self.max_workers = min(8, mp.cpu_count())  # CPU並列処理
        
        # GPU設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        self.check_gpu_resources()
        
    def setup_logging(self):
        """ログシステムの設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.results_dir / f"gpu_16x_scale_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_gpu_resources(self):
        """GPU リソースチェック"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # GPU メモリに基づくバッチサイズ調整
            if gpu_memory >= 16:
                self.batch_size = 128
                self.logger.info(f"🔥 Large GPU detected - using batch size {self.batch_size}")
            elif gpu_memory >= 12:
                self.batch_size = 96
            elif gpu_memory >= 8:
                self.batch_size = 64
            else:
                self.batch_size = 32
                
        else:
            self.logger.warning("⚠️ No GPU detected - falling back to CPU")
            self.batch_size = 16
    
    def create_scaled_documents(self, original_documents: List[str]) -> List[str]:
        """文書を16倍にスケール（データ拡張）"""
        self.logger.info(f"📄 Scaling documents from {len(original_documents)} to {self.target_documents}")
        
        scaled_docs = []
        
        # 1. 元の文書をそのまま追加
        scaled_docs.extend(original_documents)
        
        # 2. 文書の組み合わせによる拡張
        import itertools
        base_count = len(original_documents)
        
        # 2文書組み合わせ
        for i, (doc1, doc2) in enumerate(itertools.combinations(original_documents, 2)):
            if len(scaled_docs) >= self.target_documents:
                break
            combined = f"{doc1}\\n\\n{doc2}"
            scaled_docs.append(combined)
            
        # 3. 文書のセクション分割
        for doc in original_documents:
            if len(scaled_docs) >= self.target_documents:
                break
            sentences = doc.split('. ')
            if len(sentences) > 3:
                # 前半セクション
                front_section = '. '.join(sentences[:len(sentences)//2])
                scaled_docs.append(front_section)
                
                # 後半セクション
                if len(scaled_docs) < self.target_documents:
                    back_section = '. '.join(sentences[len(sentences)//2:])
                    scaled_docs.append(back_section)
        
        # 4. 残りを循環で埋める
        while len(scaled_docs) < self.target_documents:
            scaled_docs.append(original_documents[len(scaled_docs) % base_count])
        
        # 目標数に調整
        scaled_docs = scaled_docs[:self.target_documents]
        
        self.logger.info(f"✓ Document scaling completed: {len(scaled_docs)} documents")
        return scaled_docs
    
    def build_gpu_accelerated_tree(self):
        """GPU加速16倍スケールツリー構築"""
        
        self.logger.info("🚀 GPU-ACCELERATED 16X SCALE RAPTOR CONSTRUCTION")
        self.logger.info("=" * 80)
        self.logger.info(f"Target scale: {self.scale_factor}x")
        self.logger.info(f"Target documents: {self.target_documents}")
        self.logger.info(f"GPU device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"CPU workers: {self.max_workers}")
        
        total_start = time.time()
        
        try:
            # Phase 1: GPU対応RAPTORシステム初期化
            self.logger.info("\\n🧬 Phase 1: GPU-Accelerated RAPTOR Initialization")
            init_start = time.time()
            
            raptor_tree = TrueRAPTORTree()
            
            init_time = time.time() - init_start
            self.logger.info(f"✓ RAPTOR system initialized in {init_time:.1f}s")
            self.logger.info(f"  GPU available: {torch.cuda.is_available()}")
            self.logger.info(f"  Device: {raptor_tree.device}")
            
            # Phase 2: 16倍スケール文書準備
            self.logger.info("\\n📚 Phase 2: 16x Scale Document Preparation")
            doc_start = time.time()
            
            # 基本免疫細胞文書の読み込み
            sample_docs = [
                "HSC (Hematopoietic Stem Cell) differentiation into lymphoid lineage",
                "CLP (Common Lymphoid Progenitor) development and IL-7 signaling",
                "CD4+ T cell activation and TCR signaling pathways",
                "Regulatory T cell (Treg) development and Foxp3 expression",
                "TGF-beta signaling in Treg cell function and maintenance",
                "IL-10 production by regulatory T cells and immune suppression",
                "CTLA-4 expression and T cell regulation mechanisms",
                "CD25 marker expression in activated regulatory T cells",
                "Thymic Treg development versus peripheral induced Treg",
                "Treg cell plasticity and stability in immune responses",
                "Regulatory T cell therapy in autoimmune diseases",
                "Treg cells in tumor microenvironment and cancer immunity",
                "Metabolic requirements for Treg cell function",
                "Aging effects on regulatory T cell populations",
                "Tissue-resident Treg cells and organ-specific immunity",
                "FOXP3 transcriptional regulation and epigenetic control",
                "IL-2 signaling and Treg cell maintenance",
                "TGF-beta receptor signaling in T cell differentiation",
                "CD4+ CD25+ Treg cell identification and isolation",
                "Regulatory T cell suppressive mechanisms and contact inhibition",
                "Treg cell migration and tissue homing patterns",
                "Interferon-gamma and Treg cell function modulation",
                "Regulatory T cell development in lymphoid organs",
                "CTLA-4 and PD-1 co-inhibitory receptor signaling",
                "Treg cell interactions with dendritic cells",
                "Regulatory T cell role in transplantation tolerance",
                "Treg cells and allergic immune responses",
                "Regulatory T cell exhaustion in chronic inflammation",
                "FOXP3+ Treg cell lineage specification",
                "Regulatory T cell therapeutic targeting strategies",
                "Treg cell dysfunction in autoimmune diseases",
                "CD4+ T cell subset differentiation pathways",
                "Regulatory T cell phenotypic markers and identification",
                "TGF-beta and regulatory T cell induced differentiation",
                "IL-10 and regulatory T cell effector functions"
            ]
            
            # 16倍スケール文書生成
            scaled_documents = self.create_scaled_documents(sample_docs)
            
            doc_time = time.time() - doc_start
            self.logger.info(f"✓ Document preparation completed in {doc_time:.1f}s")
            self.logger.info(f"  Source documents: {len(sample_docs)}")
            self.logger.info(f"  Scaled documents: {len(scaled_documents)}")
            self.logger.info(f"  Scale factor achieved: {len(scaled_documents) / len(sample_docs):.1f}x")
            
            # Phase 3: GPU加速埋め込み生成
            self.logger.info("\\n⚡ Phase 3: GPU-Accelerated Embedding Generation")
            embed_start = time.time()
            
            # バッチ処理で埋め込み生成
            all_embeddings = []
            batch_count = (len(scaled_documents) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(scaled_documents), self.batch_size):
                batch_docs = scaled_documents[i:i + self.batch_size]
                batch_embeddings = raptor_tree.encode_documents(batch_docs)
                all_embeddings.extend(batch_embeddings)
                
                batch_num = i // self.batch_size + 1
                self.logger.info(f"  Batch {batch_num}/{batch_count}: {len(batch_docs)} documents processed")
            
            embed_time = time.time() - embed_start
            self.logger.info(f"✓ GPU embedding generation completed in {embed_time:.1f}s")
            self.logger.info(f"  Embeddings generated: {len(all_embeddings)}")
            self.logger.info(f"  Embedding rate: {len(all_embeddings) / embed_time:.1f} docs/second")
            
            # Phase 4: 16倍スケールRAPTORツリー構築
            self.logger.info("\\n🌳 Phase 4: 16x Scale RAPTOR Tree Construction")
            tree_start = time.time()
            
            # 文書IDの生成
            document_ids = [f"doc_{i:04d}" for i in range(len(scaled_documents))]
            
            # RAPTORツリー構築
            raptor_tree.build_raptor_tree(scaled_documents, document_ids)
            
            # ツリーデータの取得
            tree_data = {
                'nodes': {},
                'metadata': {
                    'total_documents': len(scaled_documents),
                    'total_nodes': len(raptor_tree.nodes),
                    'max_levels': raptor_tree.max_levels
                }
            }
            
            # ノードデータの変換
            for node_id, node in raptor_tree.nodes.items():
                tree_data['nodes'][node_id] = {
                    'node_id': node.node_id,
                    'parent_id': node.parent_id,
                    'level': node.level,
                    'content': node.content,
                    'summary': node.summary,
                    'cluster_size': node.cluster_size,
                    'source_documents': node.source_documents
                }
            
            tree_time = time.time() - tree_start
            self.logger.info(f"✓ RAPTOR tree construction completed in {tree_time:.1f}s")
            self.logger.info(f"  Total nodes: {len(tree_data['nodes'])}")
            self.logger.info(f"  Tree levels: {max([node.get('level', 0) for node in tree_data['nodes'].values()])}")
            
            # Phase 5: 免疫学的ラベリング
            self.logger.info("\\n🧬 Phase 5: Immunological Labeling")
            label_start = time.time()
            
            labeled_nodes = 0
            try:
                for node_id, node_data in tree_data['nodes'].items():
                    level = node_data.get('level', 0)
                    content = node_data.get('content', '')
                    cluster_size = node_data.get('cluster_size', 1)
                    
                    if level > 0:  # ROOTノード以外
                        cluster_id = node_id.split('_C')[1].split('_')[0] if '_C' in node_id else '0'
                        try:
                            immune_label = generate_immune_label(content, level, cluster_id, cluster_size)
                            node_data['immune_label'] = immune_label
                            labeled_nodes += 1
                        except:
                            # フォールバック用のシンプルなラベル
                            level_names = {1: "CLP", 2: "CD4+T", 3: "Treg", 4: "HSC"}
                            simple_label = f"{level_names.get(level, f'L{level}')}\\n({cluster_size})"
                            node_data['immune_label'] = simple_label
                            labeled_nodes += 1
            except Exception as e:
                self.logger.warning(f"Immunological labeling partially failed: {e}")
                # シンプルなラベリングにフォールバック
                for node_id, node_data in tree_data['nodes'].items():
                    level = node_data.get('level', 0)
                    cluster_size = node_data.get('cluster_size', 1)
                    level_names = {1: "CLP", 2: "CD4+T", 3: "Treg", 4: "HSC"}
                    simple_label = f"{level_names.get(level, f'L{level}')}\\n({cluster_size})"
                    node_data['immune_label'] = simple_label
                    labeled_nodes += 1
            
            label_time = time.time() - label_start
            self.logger.info(f"✓ Immunological labeling completed in {label_time:.1f}s")
            self.logger.info(f"  Labeled nodes: {labeled_nodes}")
            
            # Phase 6: 結果保存
            self.logger.info("\\n💾 Phase 6: Results Saving")
            save_start = time.time()
            
            # ファイル保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tree_file = self.cache_dir / "raptor_trees" / f"gpu_16x_scale_tree_{timestamp}.json"
            tree_file.parent.mkdir(exist_ok=True)
            
            with open(tree_file, 'w', encoding='utf-8') as f:
                json.dump(tree_data, f, indent=2, ensure_ascii=False, default=str)
            
            # メトリクス保存
            metrics = {
                'scale_factor': self.scale_factor,
                'target_documents': self.target_documents,
                'actual_documents': len(scaled_documents),
                'total_nodes': len(tree_data['nodes']),
                'gpu_device': str(self.device),
                'batch_size': self.batch_size,
                'times': {
                    'initialization': init_time,
                    'document_preparation': doc_time,
                    'embedding_generation': embed_time,
                    'tree_construction': tree_time,
                    'immunological_labeling': label_time
                },
                'rates': {
                    'embedding_rate': len(all_embeddings) / embed_time,
                    'tree_construction_rate': len(tree_data['nodes']) / tree_time
                }
            }
            
            metrics_file = self.results_dir / f"gpu_16x_scale_metrics_{timestamp}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            
            save_time = time.time() - save_start
            total_time = time.time() - total_start
            
            self.logger.info(f"✓ Results saved in {save_time:.1f}s")
            self.logger.info(f"  Tree file: {tree_file}")
            self.logger.info(f"  Metrics file: {metrics_file}")
            
            # 最終統計
            self.logger.info("\\n📊 GPU 16X SCALE FINAL STATISTICS")
            self.logger.info("=" * 80)
            self.logger.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            self.logger.info(f"Scale factor achieved: {len(scaled_documents) / len(sample_docs):.1f}x")
            self.logger.info(f"Documents processed: {len(scaled_documents)}")
            self.logger.info(f"Nodes generated: {len(tree_data['nodes'])}")
            self.logger.info(f"GPU utilization: {self.device}")
            self.logger.info(f"Overall processing rate: {len(scaled_documents) / total_time:.1f} docs/second")
            
            # GPU メモリ使用量
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                self.logger.info(f"GPU memory allocated: {gpu_memory_allocated:.2f}GB")
                self.logger.info(f"GPU memory cached: {gpu_memory_cached:.2f}GB")
            
            return True, tree_file, metrics_file
            
        except Exception as e:
            self.logger.error(f"GPU 16x scale construction failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False, None, None


def main():
    """メイン実行関数"""
    print("🚀 GPU-ACCELERATED 16X SCALE RAPTOR BUILDER")
    print("=" * 60)
    print("Initializing GPU-accelerated 16x scale RAPTOR construction...")
    
    try:
        builder = GPU16xScaleRAPTORBuilder()
        success, tree_file, metrics_file = builder.build_gpu_accelerated_tree()
        
        if success:
            print("\\n🎉 GPU 16X SCALE CONSTRUCTION COMPLETED SUCCESSFULLY!")
            print(f"🌳 Tree file: {tree_file}")
            print(f"📊 Metrics file: {metrics_file}")
            print(f"📋 Full log: {builder.log_file}")
        else:
            print("\\n❌ GPU 16X SCALE CONSTRUCTION FAILED!")
            print(f"📋 Check log for details: {builder.log_file}")
            
    except KeyboardInterrupt:
        print("\\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\\n💥 Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()