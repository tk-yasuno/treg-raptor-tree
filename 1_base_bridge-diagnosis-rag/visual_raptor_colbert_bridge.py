"""
Bridge Diagnosis RAPTOR with ColBERT Integration
橋梁診断ロジックに特化したRAGシステム

橋梁診断文書の損傷メカニズム・対策工法を抽出する視覚的文書検索システム
- ColVBERT (Transformers base) による画像・文書統合検索
- 橋梁診断ロジック専用のサマリー生成
- OCRテキスト付き画像処理
- 複雑なレイアウト文書（グラフ、表、スキャン、スクリーンショット）への対応

Version: 2.0 - Bridge Diagnosis Edition
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    VisionEncoderDecoderModel, BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration, Blip2VisionModel
)
from PIL import Image
import pytesseract
import cv2
import fitz  # PyMuPDF for PDF text extraction
import numpy as np
import faiss
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import time
from dataclasses import dataclass
from langchain_core.documents import Document

# ベースクラスをインポート
import sys
sys.path.append(str(Path(__file__).parent / "0_base_tsunami-lesson-rag"))
from tsunami_lesson_raptor import TsunamiLessonRAPTOR


@dataclass
class VisualDocument:
    """ビジュアル文書を表現するデータクラス"""
    image_path: str
    text_content: str = ""  # OCRで抽出されたテキスト
    layout_elements: List[Dict] = None  # レイアウト要素（表、グラフなど）
    metadata: Dict = None
    
    def __post_init__(self):
        if self.layout_elements is None:
            self.layout_elements = []
        if self.metadata is None:
            self.metadata = {}


class ColModernVBERTEncoder(nn.Module):
    """
    ColModernVBERT エンコーダー（最新アーキテクチャ）
    SigLIPを使用した高性能なマルチモーダル表現学習
    
    特徴:
    - SigLIP (Sigmoid Loss for Language-Image Pre-training)
    - 改善されたコントラスト学習
    - より効率的なトークン表現
    - バッチ単位の最適化
    """
    
    def __init__(
        self,
        text_model_name: str = "google/siglip-base-patch16-224",
        vision_model_name: str = "google/siglip-base-patch16-224",
        embedding_dim: int = 768,
        use_cross_attention: bool = True,
        device: str = None
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cross_attention = use_cross_attention
        
        # GPU最適化設定
        if self.device == "cuda":
            print(f"🚀 GPU加速が有効です: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.backends.cudnn.benchmark = True  # cuDNN最適化
        
        try:
            # SigLIP統合モデル（テキスト+ビジョン）
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(
                text_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(vision_model_name)
            
            # SigLIPの埋め込み次元を取得
            siglip_dim = self.model.config.projection_dim if hasattr(self.model.config, 'projection_dim') else 768
            
            # クロスアテンション層（オプション）
            if self.use_cross_attention:
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=siglip_dim,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                )
            
            # 次元調整レイヤー
            self.projection = nn.Sequential(
                nn.Linear(siglip_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            
            # 統合レイヤー（テキスト+画像）
            self.fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim * 2),
                nn.LayerNorm(embedding_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
            
            self.embedding_dim = embedding_dim
            self.to(self.device)
            
            # FP16混合精度の有効化
            if self.device == "cuda":
                # 自動混合精度の設定
                self.scaler = torch.amp.GradScaler()
                print(f"   ✅ FP16混合精度が有効")
                print(f"   ✅ cuDNN最適化が有効")
            
            print(f"✅ ColModernVBERT initialized with SigLIP")
            print(f"   Device: {self.device}")
            print(f"   Embedding dim: {embedding_dim}")
            print(f"   Cross-attention: {use_cross_attention}")
            print(f"   Data type: {'FP16' if self.device == 'cuda' else 'FP32'}")
            
        except ImportError:
            raise ImportError(
                "SigLIP requires transformers>=4.35.0. "
                "Please upgrade: pip install --upgrade transformers"
            )
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """テキストをエンコード（SigLIP使用）- GPU最適化バッチ処理"""
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        all_embeddings = []
        
        # バッチ処理でGPUメモリ効率を向上
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda"), dtype=torch.float16):
                    outputs = self.model.get_text_features(**inputs)
                    # プロジェクション
                    embeddings = self.projection(outputs)
                    # L2正規化を追加
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.float())  # FP32に戻す
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.embedding_dim, device=self.device)
    
    def encode_image(self, images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """画像をエンコード（SigLIP使用）- GPU最適化バッチ処理"""
        if not images:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        all_embeddings = []
        
        # 画像は大きいのでバッチサイズを小さく
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            inputs = self.processor(
                images=batch_images,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda"), dtype=torch.float16):
                    outputs = self.model.get_image_features(**inputs)
                    # プロジェクション
                    embeddings = self.projection(outputs)
                    # L2正規化を追加
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.float())  # FP32に戻す
            
            # GPUメモリクリア
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.embedding_dim, device=self.device)
    
    def encode_multimodal(
        self,
        texts: List[str],
        images: List[Image.Image]
    ) -> torch.Tensor:
        """テキストと画像の統合エンベディング（改善版）"""
        text_embeddings = self.encode_text(texts)
        image_embeddings = self.encode_image(images)
        
        # クロスアテンションによる相互作用（オプション）
        if self.use_cross_attention:
            # テキスト→画像のアテンション
            text_attended, _ = self.cross_attention(
                text_embeddings.unsqueeze(1),
                image_embeddings.unsqueeze(1),
                image_embeddings.unsqueeze(1)
            )
            text_embeddings = text_attended.squeeze(1)
        
        # 結合と融合
        combined = torch.cat([text_embeddings, image_embeddings], dim=1)
        fused_embeddings = self.fusion_layer(combined)
        
        # 最終的にL2正規化
        fused_embeddings = nn.functional.normalize(fused_embeddings, p=2, dim=1)
        
        return fused_embeddings
    
    def compute_similarity(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        SigLIP風のシグモイド類似度計算
        
        従来のコントラスト学習（softmax）ではなく、
        sigmoid lossを使用した類似度
        """
        # コサイン類似度
        text_norm = nn.functional.normalize(text_embeddings, p=2, dim=1)
        image_norm = nn.functional.normalize(image_embeddings, p=2, dim=1)
        
        similarity = torch.matmul(text_norm, image_norm.t())
        
        # SigLIPの温度パラメータ（学習可能にすることも可能）
        temperature = 10.0
        similarity = similarity * temperature
        
        return similarity
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """GPU メモリ使用量を取得"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - allocated
            
            return {
                "allocated_gb": allocated,
                "cached_gb": cached, 
                "free_gb": free,
                "total_gb": total,
                "utilization_percent": (allocated / total) * 100
            }
        return {"message": "CUDA not available"}
    
    def clear_gpu_cache(self):
        """GPUキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")


class ColBERT2BLIP2Encoder(nn.Module):
    """
    ColBERT v2.0 + BLIP2 統合エンコーダー
    
    アーキテクチャ:
    - Vision: BLIP2 with ViT-G/14 (giant vision encoder)
    - Language: Flan-T5-Large (780M parameters)
    - Text Retrieval: ColBERTv2.0 (colbert-ir/colbertv2.0)
    
    特徴:
    - BLIP2のQ-Formerで画像とテキストを統合
    - ColBERTv2.0の後期相互作用(late interaction)で高速検索
    - クエリ専用モード(query-only)とドキュメント事前計算をサポート
    - オプションの量子化でメモリ効率を向上
    """
    
    def __init__(
        self,
        blip2_model_name: str = "Salesforce/blip2-flan-t5-xl",  # BLIP2 with Flan-T5-XL
        colbert_model_name: str = "colbert-ir/colbertv2.0",
        embedding_dim: int = 768,
        use_quantization: bool = False,  # 量子化を使用するか
        image_batch_size: int = 8,  # 画像バッチサイズ
        device: str = None
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.use_quantization = use_quantization
        self.image_batch_size = image_batch_size  # インスタンス変数として保存
        
        # GPU最適化設定
        if self.device == "cuda":
            print(f"🚀 GPU加速が有効です: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.backends.cudnn.benchmark = True
        
        # BLIP2モデル（Vision + Q-Former + Language Model）
        print(f"📥 Loading BLIP2 model: {blip2_model_name}")
        
        # 量子化設定
        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if use_quantization and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            load_kwargs["quantization_config"] = quantization_config
            print("   ⚙️  8-bit quantization enabled")
        
        self.blip2_processor = Blip2Processor.from_pretrained(blip2_model_name)
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_model_name,
            **load_kwargs
        )
        
        # BLIP2の埋め込み次元を取得
        # Q-Formerの出力次元
        blip2_hidden_size = self.blip2_model.config.qformer_config.hidden_size
        
        # ColBERTv2.0モデル（テキスト検索用）
        print(f"📥 Loading ColBERTv2.0 model: {colbert_model_name}")
        self.colbert_tokenizer = AutoTokenizer.from_pretrained(colbert_model_name)
        self.colbert_model = AutoModel.from_pretrained(
            colbert_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        # ColBERTの埋め込み次元
        colbert_hidden_size = self.colbert_model.config.hidden_size
        
        # プロジェクション層
        self.blip2_projection = nn.Linear(blip2_hidden_size, embedding_dim)
        self.colbert_projection = nn.Linear(colbert_hidden_size, embedding_dim)
        
        # マルチモーダル融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # ドキュメント埋め込みキャッシュ（事前計算用）
        self.document_embeddings_cache = {}
        
        self.to(self.device)
        
        if self.device == "cuda":
            print(f"   ✅ FP16混合精度が有効")
            print(f"   ✅ cuDNN最適化が有効")
        
        print(f"✅ ColBERT v2.0 + BLIP2 initialized")
        print(f"   Device: {self.device}")
        print(f"   BLIP2 Model: {blip2_model_name}")
        print(f"   ColBERT Model: {colbert_model_name}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Quantization: {'Enabled' if use_quantization else 'Disabled'}")
        print(f"   Data type: {'FP16' if self.device == 'cuda' else 'FP32'}")
    
    def encode_text_colbert(self, texts: List[str], batch_size: int = 32, mode: str = "query") -> torch.Tensor:
        """
        ColBERTv2.0でテキストをエンコード
        
        Args:
            texts: エンコードするテキストのリスト
            batch_size: バッチサイズ
            mode: "query"（クエリモード）または "doc"（ドキュメントモード）
        """
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # ColBERTv2.0のトークナイゼーション
            if mode == "query":
                # クエリモード: [Q]トークンを追加
                batch_texts = [f"[Q] {text}" for text in batch_texts]
            elif mode == "doc":
                # ドキュメントモード: [D]トークンを追加
                batch_texts = [f"[D] {text}" for text in batch_texts]
            
            inputs = self.colbert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda"), dtype=torch.float16):
                    outputs = self.colbert_model(**inputs)
                    # 最後の隠れ状態の平均プーリング
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    # プロジェクション
                    embeddings = self.colbert_projection(embeddings)
                    # L2正規化
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.float())
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.embedding_dim, device=self.device)
    
    def encode_image_blip2(self, images: List[Image.Image], batch_size: int = None) -> torch.Tensor:
        """
        BLIP2で画像をエンコード
        
        BLIP2のQ-Formerを使用して画像から意味的特徴を抽出
        """
        if not images:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        # batch_sizeが指定されていない場合はインスタンス変数を使用
        if batch_size is None:
            batch_size = self.image_batch_size
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # BLIP2のプロセッサで前処理
            inputs = self.blip2_processor(
                images=batch_images,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda"), dtype=torch.float16):
                    # Vision Encoder + Q-Former
                    vision_outputs = self.blip2_model.vision_model(inputs['pixel_values'])
                    image_embeds = vision_outputs[0]  # [batch, num_patches, hidden_size]
                    
                    # Q-Formerでクエリ埋め込みを生成
                    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
                    query_tokens = self.blip2_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    
                    query_outputs = self.blip2_model.qformer(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_attention_mask,
                        return_dict=True
                    )
                    
                    # クエリ出力の平均プーリング
                    image_features = query_outputs.last_hidden_state.mean(dim=1)
                    
                    # プロジェクション
                    image_features = self.blip2_projection(image_features)
                    # L2正規化
                    image_features = nn.functional.normalize(image_features, p=2, dim=1)
            
            all_embeddings.append(image_features.float())
            
            # GPUメモリクリア
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.embedding_dim, device=self.device)
    
    def encode_multimodal(
        self,
        texts: List[str],
        images: List[Image.Image],
        mode: str = "doc"
    ) -> torch.Tensor:
        """
        テキストと画像の統合エンベディング
        
        Args:
            texts: テキストのリスト
            images: 画像のリスト
            mode: "query"（検索時）または "doc"（インデックス構築時）
        """
        # ColBERTでテキストをエンコード
        text_embeddings = self.encode_text_colbert(texts, mode=mode)
        
        # BLIP2で画像をエンコード
        image_embeddings = self.encode_image_blip2(images)
        
        # マルチモーダル融合
        combined = torch.cat([text_embeddings, image_embeddings], dim=1)
        fused_embeddings = self.fusion_layer(combined)
        
        # 最終L2正規化
        fused_embeddings = nn.functional.normalize(fused_embeddings, p=2, dim=1)
        
        return fused_embeddings
    
    def encode_image(self, images: List[Image.Image], batch_size: int = None) -> torch.Tensor:
        """
        画像をエンコード（encode_image_blip2のエイリアス）
        他のエンコーダーとのインターフェース統一のため
        """
        return self.encode_image_blip2(images, batch_size)
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        テキストをエンコード（encode_text_colbertのエイリアス、ドキュメントモード）
        他のエンコーダーとのインターフェース統一のため
        """
        return self.encode_text_colbert(texts, batch_size, mode="doc")
    
    def precompute_document_embeddings(
        self,
        documents: List[Dict],  # {'text': str, 'image': Image.Image}
        doc_ids: List[str],
        batch_size: int = 16
    ):
        """
        ドキュメント埋め込みを事前計算してキャッシュに保存
        
        Args:
            documents: ドキュメントのリスト（テキストと画像を含む）
            doc_ids: ドキュメントID のリスト
            batch_size: バッチサイズ
        """
        print(f"📊 Pre-computing document embeddings for {len(documents)} documents...")
        
        texts = [doc['text'] for doc in documents]
        images = [doc['image'] for doc in documents]
        
        # バッチ処理で埋め込みを生成
        embeddings = self.encode_multimodal(texts, images, mode="doc")
        
        # キャッシュに保存
        for doc_id, embedding in zip(doc_ids, embeddings):
            self.document_embeddings_cache[doc_id] = embedding.cpu().numpy()
        
        print(f"✅ Document embeddings cached: {len(self.document_embeddings_cache)} documents")
    
    def save_document_cache(self, cache_path: str):
        """ドキュメント埋め込みキャッシュをファイルに保存"""
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(self.document_embeddings_cache, f)
        print(f"💾 Document cache saved to: {cache_path}")
    
    def load_document_cache(self, cache_path: str):
        """ドキュメント埋め込みキャッシュをファイルから読み込み"""
        import pickle
        with open(cache_path, 'rb') as f:
            self.document_embeddings_cache = pickle.load(f)
        print(f"📂 Document cache loaded: {len(self.document_embeddings_cache)} documents")
    
    def late_interaction_score(
        self,
        query_embedding: torch.Tensor,
        doc_id: str
    ) -> float:
        """
        ColBERTの後期相互作用スコアリング
        
        Args:
            query_embedding: クエリの埋め込み
            doc_id: ドキュメントID
        
        Returns:
            類似度スコア
        """
        if doc_id not in self.document_embeddings_cache:
            raise ValueError(f"Document {doc_id} not found in cache")
        
        doc_embedding = torch.tensor(
            self.document_embeddings_cache[doc_id],
            device=self.device
        )
        
        # コサイン類似度
        score = torch.dot(query_embedding, doc_embedding).item()
        
        return score


class ColVBERTEncoder(nn.Module):
    """
    ColVBERT エンコーダー（Transformers base）
    テキストと画像の統合表現を学習
    """
    
    def __init__(
        self,
        text_model_name: str = "intfloat/multilingual-e5-large",
        vision_model_name: str = "Salesforce/blip-image-captioning-base",
        embedding_dim: int = 768,
        device: str = None
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU最適化設定
        if self.device == "cuda":
            print(f"🚀 GPU加速が有効です: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.backends.cudnn.benchmark = True  # cuDNN最適化
        
        # テキストエンコーダー
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # テキスト埋め込み次元を取得
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # ビジョンエンコーダー
        self.vision_processor = BlipProcessor.from_pretrained(vision_model_name)
        self.vision_encoder = BlipForConditionalGeneration.from_pretrained(
            vision_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # ビジョン埋め込み次元を取得
        vision_hidden_size = self.vision_encoder.config.vision_config.hidden_size
        
        # テキストとビジョンの投影層
        self.text_projection = nn.Linear(text_hidden_size, embedding_dim)
        self.vision_projection = nn.Linear(vision_hidden_size, embedding_dim)
        
        # 統合レイヤー
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
        self.to(self.device)
        
        # GPU最適化設定の表示
        if self.device == "cuda":
            print(f"   ✅ FP16混合精度が有効")
            print(f"   ✅ cuDNN最適化が有効")
            
        print(f"✅ ColVBERT initialized with BLIP")
        print(f"   Device: {self.device}")
        print(f"   Text Model: {text_model_name}")
        print(f"   Vision Model: {vision_model_name}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Data type: {'FP16' if self.device == 'cuda' else 'FP32'}")
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """テキストをエンコード - GPU最適化バッチ処理"""
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        all_embeddings = []
        
        # バッチ処理でGPUメモリ効率を向上
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.text_tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda"), dtype=torch.float16):
                    outputs = self.text_encoder(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    # 投影層を通して埋め込み次元に変換
                    embeddings = self.text_projection(embeddings)
                    # L2正規化を追加
                    embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.float())  # FP32に戻す
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.embedding_dim, device=self.device)
    
    def encode_image(self, images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """画像をエンコード - GPU最適化バッチ処理"""
        if not images:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        all_embeddings = []
        
        # 画像は大きいのでバッチサイズを小さく
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            inputs = self.vision_processor(
                images=batch_images,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda"), dtype=torch.float16):
                    # BLIPForConditionalGeneration uses vision_model for image features
                    outputs = self.vision_encoder.vision_model(inputs['pixel_values'])
                    # CLSトークンではなく平均プーリングを使用
                    image_features = outputs.last_hidden_state.mean(dim=1)
                    # 投影層を通して埋め込み次元に変換
                    image_features = self.vision_projection(image_features)
                    # L2正規化を追加
                    image_features = nn.functional.normalize(image_features, p=2, dim=1)
            
            all_embeddings.append(image_features.float())  # FP32に戻す
            
            # GPUメモリクリア
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, self.embedding_dim, device=self.device)
    
    def encode_multimodal(
        self,
        texts: List[str],
        images: List[Image.Image]
    ) -> torch.Tensor:
        """テキストと画像の統合エンベディング"""
        text_embeddings = self.encode_text(texts)
        image_embeddings = self.encode_image(images)
        
        # 次元を合わせる
        if image_embeddings.size(1) != text_embeddings.size(1):
            image_embeddings = nn.functional.adaptive_avg_pool1d(
                image_embeddings.unsqueeze(0), 
                text_embeddings.size(1)
            ).squeeze(0)
        
        # 結合
        combined = torch.cat([text_embeddings, image_embeddings], dim=1)
        
        # 融合
        fused_embeddings = self.fusion_layer(combined)
        
        # 最終的にL2正規化
        fused_embeddings = nn.functional.normalize(fused_embeddings, p=2, dim=1)
        
        return fused_embeddings
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """GPU メモリ使用量を取得"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - allocated
            
            return {
                "allocated_gb": allocated,
                "cached_gb": cached, 
                "free_gb": free,
                "total_gb": total,
                "utilization_percent": (allocated / total) * 100
            }
        return {"message": "CUDA not available"}
    
    def clear_gpu_cache(self):
        """GPUキャッシュをクリア"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")


class VisualDocumentProcessor:
    """ビジュアル文書処理クラス"""
    
    def __init__(self, ocr_config: Dict = None, pdf_dir: str = None):
        self.ocr_config = ocr_config or {
            'lang': 'jpn+eng',
            'config': '--psm 6'
        }
        self.pdf_dir = pdf_dir  # PDF元ファイルのディレクトリ
        self.pdf_text_cache = {}  # PDFテキストのキャッシュ
        
        # PDFディレクトリが指定されている場合、テキストを事前抽出
        if pdf_dir:
            self._extract_pdf_texts(pdf_dir)
    
    def _extract_pdf_texts(self, pdf_dir: str):
        """PDFディレクトリから全テキストを抽出してキャッシュ"""
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            return
        
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                doc = fitz.open(str(pdf_file))
                pdf_name = pdf_file.stem
                
                # 各ページのテキストを抽出
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # キャッシュキー: ファイル名_ページ番号
                    cache_key = f"{pdf_name}_page{page_num+1:03d}"
                    self.pdf_text_cache[cache_key] = text.strip()
                
                print(f"  ✅ PDF抽出: {pdf_name} ({len(doc)}ページ)")
                doc.close()  # ループ終了後にクローズ
                
            except Exception as e:
                print(f"  ⚠️ PDF抽出エラー ({pdf_file.name}): {e}")
            except Exception as e:
                print(f"  ⚠️ PDF抽出エラー ({pdf_file.name}): {e}")
    
    def get_text_from_cache(self, image_filename: str) -> str:
        """画像ファイル名からキャッシュされたPDFテキストを取得"""
        # ファイル名から.pngを除去してキャッシュキーを作成
        cache_key = image_filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        return self.pdf_text_cache.get(cache_key, "")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """OCRでテキストを抽出 (フォールバック)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            # 前処理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # ノイズ除去
            denoised = cv2.fastNlMeansDenoising(gray)
            # コントラスト向上
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # OCR実行
            text = pytesseract.image_to_string(
                enhanced,
                lang=self.ocr_config['lang'],
                config=self.ocr_config['config']
            )
            
            return text.strip()
        except Exception as e:
            # OCRエラーは無視（Tesseract未インストールの場合）
            return ""
    
    def detect_layout_elements(self, image_path: str) -> List[Dict]:
        """レイアウト要素を検出"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            elements = []
            
            # 表の検出（単純な矩形検出）
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 50:  # 最小サイズ閾値
                    elements.append({
                        'type': 'table_or_figure',
                        'bbox': [x, y, w, h],
                        'area': w * h
                    })
            
            return elements
        except Exception as e:
            # レイアウト検出エラーは無視
            return []
    
    def process_visual_document(self, image_path: str) -> VisualDocument:
        """ビジュアル文書を処理"""
        # まずPDFキャッシュからテキストを取得
        image_filename = Path(image_path).name
        text_content = self.get_text_from_cache(image_filename)
        
        # キャッシュにない場合はOCRでテキスト抽出（フォールバック）
        if not text_content:
            text_content = self.extract_text_from_image(image_path)
        
        # レイアウト要素検出
        layout_elements = self.detect_layout_elements(image_path)
        
        # メタデータ
        metadata = {
            'image_size': self._get_image_size(image_path),
            'processing_time': time.time()
        }
        
        return VisualDocument(
            image_path=image_path,
            text_content=text_content,
            layout_elements=layout_elements,
            metadata=metadata
        )
    
    def _get_image_size(self, image_path: str) -> Tuple[int, int]:
        """画像サイズを取得"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except:
            return (0, 0)


class BridgeRAPTORColBERT(TsunamiLessonRAPTOR):
    """
    橋梁診断ロジック専用 RAPTOR + ColBERT システム
    
    機能:
    - OCRテキスト付き画像処理
    - ColVBERT / ColModernVBERT による統合検索
    - 橋梁診断文書の損傷メカニズム・対策工法の抽出
    - 複雑なレイアウト文書への対応
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        colbert_config: Dict = None,
        visual_config: Dict = None,
        use_modern_vbert: bool = False,
        use_blip2_colbert: bool = False,  # 新しいBLIP2+ColBERTv2.0を使用
        pdf_source_dir: str = None,  # PDF元ファイルのディレクトリ
        use_multimodal: bool = True,  # マルチモーダル埋め込みを使用
        multimodal_weight: float = 0.3,  # 画像埋め込みの重み
        **kwargs
    ):
        # ColVBERT / ColModernVBERT / ColBERT2BLIP2 エンコーダー選択
        colbert_config = colbert_config or {}
        encoder_type = colbert_config.get('encoder_type', 'modern' if use_modern_vbert else 'standard')
        
        if use_blip2_colbert or encoder_type == 'blip2_colbert':
            # ColBERT v2.0 + BLIP2（最新アーキテクチャ）
            self.colbert_encoder = ColBERT2BLIP2Encoder(
                blip2_model_name=colbert_config.get('blip2_model', 'Salesforce/blip2-flan-t5-xl'),
                colbert_model_name=colbert_config.get('colbert_model', 'colbert-ir/colbertv2.0'),
                embedding_dim=colbert_config.get('embedding_dim', 768),
                use_quantization=colbert_config.get('use_quantization', False),
                image_batch_size=colbert_config.get('image_batch_size', 8)  # image_batch_sizeを渡す
            )
            self.encoder_type = 'ColBERT v2.0 + BLIP2'
        elif encoder_type == 'modern' or use_modern_vbert:
            # ColModernVBERT（SigLIP使用）
            self.colbert_encoder = ColModernVBERTEncoder(
                text_model_name=colbert_config.get('text_model', 'google/siglip-base-patch16-224'),
                vision_model_name=colbert_config.get('vision_model', 'google/siglip-base-patch16-224'),
                embedding_dim=colbert_config.get('embedding_dim', 768),
                use_cross_attention=colbert_config.get('use_cross_attention', True)
            )
            self.encoder_type = 'ColModernVBERT (SigLIP)'
        else:
            # 従来のColVBERT（BLIP使用）
            self.colbert_encoder = ColVBERTEncoder(
                text_model_name=colbert_config.get('text_model', 'intfloat/multilingual-e5-large'),
                vision_model_name=colbert_config.get('vision_model', 'Salesforce/blip-image-captioning-base'),
                embedding_dim=colbert_config.get('embedding_dim', 768)
            )
            self.encoder_type = 'ColVBERT (BLIP)'
        
        # 親クラスの初期化（マルチモーダルパラメータを渡す）
        super().__init__(
            embeddings_model=embeddings_model,
            llm=llm,
            visual_encoder=self.colbert_encoder,
            use_multimodal=use_multimodal,
            multimodal_weight=multimodal_weight,
            **kwargs
        )
        
        # ビジュアル文書プロセッサー（PDF元ディレクトリを渡す）
        visual_config = visual_config or {}
        self.visual_processor = VisualDocumentProcessor(
            ocr_config=visual_config.get('ocr_config'),
            pdf_dir=pdf_source_dir
        )
        
        # ビジュアル文書のストレージ
        self.visual_documents: List[VisualDocument] = []
        self.visual_embeddings: Optional[np.ndarray] = None
        self.visual_index: Optional[faiss.Index] = None
        
        print(f"🖼️ Visual RAPTOR ColBERT initialized")
        print(f"   Encoder: {self.encoder_type}")
        print(f"   Device: {self.colbert_encoder.device}")
        if use_multimodal:
            print(f"   🎨 Multimodal embeddings: ENABLED (weight: {multimodal_weight})")
        if encoder_type == 'modern' or use_modern_vbert:
            print(f"   ✨ Using ColModernVBERT with SigLIP")
        else:
            print(f"   Text Model: {colbert_config.get('text_model', 'intfloat/multilingual-e5-large')}")
            print(f"   Vision Model: {colbert_config.get('vision_model', 'Salesforce/blip-image-captioning-base')}")
    
    def load_visual_documents(
        self,
        image_directory: str,
        supported_formats: List[str] = None
    ) -> List[VisualDocument]:
        """ビジュアル文書を読み込み"""
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_dir = Path(image_directory)
        image_files = []
        
        for ext in supported_formats:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"📸 Processing {len(image_files)} visual documents...")
        
        visual_docs = []
        for i, image_path in enumerate(image_files, 1):
            if i % 20 == 0 or i == 1:
                print(f"   Progress: {i}/{len(image_files)}")
            visual_doc = self.visual_processor.process_visual_document(str(image_path))
            visual_docs.append(visual_doc)
        
        self.visual_documents = visual_docs
        print(f"✅ Loaded {len(visual_docs)} visual documents")
        
        return visual_docs
    
    def summarize_cluster(self, documents: List[Document]) -> str:
        """クラスタを要約（橋梁診断ロジック用）"""
        print(f"   🔄 Summarizing {len(documents)} documents...", end=" ", flush=True)
        
        import time
        start_time = time.time()
        
        # ドキュメントからテキストを抽出
        texts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            elif hasattr(doc, 'cached_text'):
                texts.append(doc.cached_text)
            elif hasattr(doc, 'text_content'):
                texts.append(doc.text_content)
            else:
                texts.append(str(doc))
        
        # テキストを結合し、8000文字まで（高速化）
        combined_text = "\n\n".join(texts)
        max_input_length = 8000  # num_ctx=16384で余裕あり
        
        if len(combined_text) > max_input_length:
            # 均等にサンプリング
            sample_ratio = max_input_length / len(combined_text)
            sampled_texts = []
            for text in texts:
                sample_len = int(len(text) * sample_ratio)
                if sample_len > 0:
                    sampled_texts.append(text[:sample_len])
            combined_text = "\n\n".join(sampled_texts)[:max_input_length]
        
        # 橋梁診断ロジック用のプロンプト
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template(
            "以下は橋梁診断ロジックに関する複数のドキュメントです。\n\n"
            "【要約タスク】\n"
            "- 部材と損傷の状態、その原因と補修工法を要約してください\n"
            "- 400-500文字で簡潔にまとめてください\n"
            "- 箇条書きではなく、段落形式で記述してください\n\n"
            "【テキスト】\n{text}\n\n"
            "【要約】"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            summary = chain.invoke({"text": combined_text})
            elapsed = time.time() - start_time
            print(f"✅ ({len(summary)} chars, {elapsed:.1f}s)")
            return summary
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"⚠️ Error ({elapsed:.1f}s): {str(e)[:100]}")
            # エラー時は先頭1000文字を返す
            return combined_text[:1000]
    
    def build_visual_index(self) -> faiss.Index:
        """ビジュアル文書のインデックスを構築"""
        if not self.visual_documents:
            raise ValueError("ビジュアル文書が読み込まれていません")
        
        print(f"🔍 Building visual document index...")
        
        # テキストと画像を準備
        texts = [doc.text_content for doc in self.visual_documents]
        images = []
        
        for doc in self.visual_documents:
            try:
                img = Image.open(doc.image_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"画像読み込みエラー {doc.image_path}: {e}")
                # ダミー画像を作成
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        # エンベディング生成
        embeddings = self.colbert_encoder.encode_multimodal(texts, images)
        embeddings_np = embeddings.detach().cpu().numpy()  # .detach()を追加
        
        # FAISSインデックス構築
        embedding_dim = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for similarity
        index.add(embeddings_np.astype('float32'))
        
        self.visual_embeddings = embeddings_np
        self.visual_index = index
        
        print(f"✅ Visual index built with {len(self.visual_documents)} documents")
        return index
    
    def search_visual_documents(
        self,
        query: str,
        query_image: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[VisualDocument, float]]:
        """ビジュアル文書を検索"""
        if self.visual_index is None:
            raise ValueError("ビジュアルインデックスが構築されていません")
        
        # クエリエンベディング生成
        if query_image:
            # マルチモーダルクエリ
            query_img = Image.open(query_image).convert('RGB')
            query_embedding = self.colbert_encoder.encode_multimodal([query], [query_img])
        else:
            # テキストのみクエリ
            query_embedding = self.colbert_encoder.encode_text([query])
        
        query_vec = query_embedding.cpu().numpy().astype('float32')
        
        # 検索実行
        scores, indices = self.visual_index.search(query_vec, top_k)
        
        # 結果をまとめる
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.visual_documents):
                results.append((self.visual_documents[idx], float(score)))
        
        return results
    
    def create_hybrid_documents(self) -> List[Document]:
        """ビジュアル文書をLangChainのDocumentに変換"""
        hybrid_docs = []
        
        for i, visual_doc in enumerate(self.visual_documents):
            # テキストコンテンツを結合
            content = f"画像パス: {visual_doc.image_path}\n"
            if visual_doc.text_content:
                content += f"抽出テキスト:\n{visual_doc.text_content}\n"
            
            # レイアウト情報を追加
            if visual_doc.layout_elements:
                content += f"レイアウト要素: {len(visual_doc.layout_elements)}個の要素を検出\n"
                for elem in visual_doc.layout_elements:
                    content += f"- {elem['type']}: 面積{elem['area']}\n"
            
            # メタデータ
            metadata = {
                'source': visual_doc.image_path,
                'type': 'visual_document',
                'image_size': visual_doc.metadata.get('image_size'),
                'layout_elements_count': len(visual_doc.layout_elements),
                'visual_doc_index': i
            }
            
            doc = Document(page_content=content, metadata=metadata)
            hybrid_docs.append(doc)
        
        return hybrid_docs
    
    def build_hybrid_tree(
        self,
        text_documents: List[Document],
        visual_documents: List[VisualDocument] = None,
        save_dir: str = "saved_models/visual_raptor"
    ) -> Dict:
        """テキストとビジュアル文書の統合ツリーを構築"""
        # ビジュアル文書をDocumentに変換
        if visual_documents is None:
            visual_documents = self.visual_documents
        
        if visual_documents:
            hybrid_docs = self.create_hybrid_documents()
            all_documents = text_documents + hybrid_docs
        else:
            all_documents = text_documents
        
        print(f"🌲 Building hybrid tree with {len(text_documents)} text docs and {len(visual_documents)} visual docs")
        
        # 親クラスのツリー構築メソッドを呼び出し
        tree = self.build_disaster_tree(all_documents, save_dir)
        
        return tree
    
    def save_visual_components(self, save_dir: str):
        """ビジュアルコンポーネントを保存"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # ビジュアル文書メタデータ
        visual_docs_data = []
        for doc in self.visual_documents:
            visual_docs_data.append({
                'image_path': doc.image_path,
                'text_content': doc.text_content,
                'layout_elements': doc.layout_elements,
                'metadata': doc.metadata
            })
        
        with open(save_path / 'visual_documents.json', 'w', encoding='utf-8') as f:
            json.dump(visual_docs_data, f, ensure_ascii=False, indent=2)
        
        # ビジュアルエンベディング
        if self.visual_embeddings is not None:
            np.save(save_path / 'visual_embeddings.npy', self.visual_embeddings)
        
        # FAISSインデックス
        if self.visual_index is not None:
            faiss.write_index(self.visual_index, str(save_path / 'visual_index.faiss'))
        
        print(f"✅ Visual components saved to {save_dir}")
    
    def load_visual_components(self, save_dir: str):
        """ビジュアルコンポーネントを読み込み"""
        save_path = Path(save_dir)
        
        # ビジュアル文書メタデータ
        visual_docs_file = save_path / 'visual_documents.json'
        if visual_docs_file.exists():
            with open(visual_docs_file, 'r', encoding='utf-8') as f:
                visual_docs_data = json.load(f)
            
            self.visual_documents = []
            for data in visual_docs_data:
                visual_doc = VisualDocument(
                    image_path=data['image_path'],
                    text_content=data['text_content'],
                    layout_elements=data['layout_elements'],
                    metadata=data['metadata']
                )
                self.visual_documents.append(visual_doc)
        
        # ビジュアルエンベディング
        embeddings_file = save_path / 'visual_embeddings.npy'
        if embeddings_file.exists():
            self.visual_embeddings = np.load(embeddings_file)
        
        # FAISSインデックス
        index_file = save_path / 'visual_index.faiss'
        if index_file.exists():
            self.visual_index = faiss.read_index(str(index_file))
        
        print(f"✅ Visual components loaded from {save_dir}")


def main():
    """メイン実行関数"""
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    
    print("="*80)
    print("Bridge Diagnosis RAPTOR ColBERT System")
    print("橋梁診断ロジック専用システム")
    print("="*80)
    
    # モデル初期化
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    llm = ChatOllama(
        model="granite-code:8b",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=300
    )
    
    # BridgeRAPTORColBERT初期化
    bridge_raptor = BridgeRAPTORColBERT(
        embeddings_model=embeddings,
        llm=llm,
        colbert_config={
            'text_model': 'intfloat/multilingual-e5-large',
            'vision_model': 'Salesforce/blip2-opt-2.7b',
            'embedding_dim': 768
        },
        visual_config={
            'ocr_config': {
                'lang': 'jpn+eng',
                'config': '--psm 6'
            }
        },
        min_clusters=2,
        max_clusters=5,
        max_depth=2,
        chunk_size=500,
        chunk_overlap=100
    )
    
    print("✅ Visual RAPTOR ColBERT システムが初期化されました")


if __name__ == "__main__":
    main()