"""
MIRAGE Evaluation System for Immune Cell Differentiation RAG
免疫細胞分化系譜特化型RAG評価システム
- Retriever精度、Summarizer精度、QA整合性の評価
- JQaRA拡張による分化経路・マーカー・機能の整合性評価

Author: AI Assistant
Date: 2024-10-30
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# RAPTOR Treeシステムをインポート
from immune_raptor_tree import ImmuneCellRAPTORTree, ImmuneCellNode, ImmuneDifferentiationPath

@dataclass
class EvaluationQuery:
    """評価用クエリデータクラス"""
    query_id: str
    query_text: str
    query_type: str  # 'lineage', 'function', 'mechanism', 'comparison', 'clinical'
    expected_nodes: List[str]
    expected_path: Optional[List[str]] = None
    expected_mechanisms: Optional[List[str]] = None
    expected_markers: Optional[List[str]] = None
    ground_truth: Optional[str] = None
    difficulty_level: int = 1  # 1-5

@dataclass
class RetrievalResult:
    """検索結果データクラス"""
    query_id: str
    retrieved_nodes: List[Tuple[str, float]]  # (node_id, score)
    retrieved_articles: List[Tuple[str, float]]  # (pmid, score)
    execution_time: float
    success: bool = True

@dataclass
class EvaluationMetrics:
    """評価メトリクスデータクラス"""
    precision: float
    recall: float
    f1_score: float
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    ndcg: float  # Normalized Discounted Cumulative Gain

@dataclass
class MIRAGEEvaluationResult:
    """MIRAGE評価結果データクラス"""
    evaluation_id: str
    timestamp: str
    retrieval_metrics: EvaluationMetrics
    summarization_metrics: Dict[str, float]
    qa_consistency_metrics: Dict[str, float]
    overall_score: float
    detailed_results: List[Dict]

class ImmuneCellEvaluationDataset:
    """免疫細胞評価データセット"""
    
    def __init__(self):
        self.evaluation_queries = []
        self._build_evaluation_dataset()
    
    def _build_evaluation_dataset(self):
        """評価用データセットを構築"""
        
        # 1. 分化系譜クエリ
        lineage_queries = [
            EvaluationQuery(
                query_id="lineage_001",
                query_text="FOXP3+細胞はどこから分化するか？",
                query_type="lineage",
                expected_nodes=["cell_hsc_001", "cell_clp_001", "cell_cd4_001", "cell_treg_ntreg_001"],
                expected_path=["HSC", "CLP", "CD4+ T cell", "Treg"],
                difficulty_level=2
            ),
            EvaluationQuery(
                query_id="lineage_002", 
                query_text="造血幹細胞からTreg細胞への分化経路を教えてください",
                query_type="lineage",
                expected_nodes=["cell_hsc_001", "cell_clp_001", "cell_cd4_001", "cell_treg_ntreg_001", "cell_treg_itreg_001"],
                expected_path=["HSC", "CLP", "CD4+ T cell", "Treg"],
                difficulty_level=3
            ),
            EvaluationQuery(
                query_id="lineage_003",
                query_text="Common Lymphoid Progenitorは何に分化しますか？",
                query_type="lineage",
                expected_nodes=["cell_clp_001", "cell_cd4_001"],
                expected_path=["CLP", "CD4+ T cell"],
                difficulty_level=1
            )
        ]
        
        # 2. 機能クエリ
        function_queries = [
            EvaluationQuery(
                query_id="function_001",
                query_text="Treg細胞の主要な免疫抑制機能は何か？",
                query_type="function",
                expected_nodes=["cell_treg_ntreg_001", "cell_treg_itreg_001"],
                expected_mechanisms=["CTLA-4", "IL-10", "TGF-β", "Perforin/granzyme"],
                difficulty_level=2
            ),
            EvaluationQuery(
                query_id="function_002",
                query_text="CD4+ T細胞の主な機能を教えてください",
                query_type="function", 
                expected_nodes=["cell_cd4_001"],
                expected_mechanisms=["Helper T cell responses", "Cytokine production"],
                difficulty_level=1
            ),
            EvaluationQuery(
                query_id="function_003",
                query_text="造血幹細胞の特徴的な機能は何ですか？",
                query_type="function",
                expected_nodes=["cell_hsc_001"],
                expected_mechanisms=["Self-renewal", "Multilineage differentiation"],
                difficulty_level=1
            )
        ]
        
        # 3. メカニズムクエリ
        mechanism_queries = [
            EvaluationQuery(
                query_id="mechanism_001",
                query_text="CTLA-4はTreg細胞でどのような役割を果たすか？",
                query_type="mechanism",
                expected_nodes=["cell_treg_ntreg_001"],
                expected_mechanisms=["Costimulation blockade", "Immune suppression"],
                expected_markers=["CTLA-4"],
                difficulty_level=3
            ),
            EvaluationQuery(
                query_id="mechanism_002", 
                query_text="FOXP3の転写調節機能について説明してください",
                query_type="mechanism",
                expected_nodes=["cell_treg_ntreg_001", "cell_treg_itreg_001"],
                expected_markers=["FOXP3"],
                difficulty_level=4
            ),
            EvaluationQuery(
                query_id="mechanism_003",
                query_text="TGF-βがTreg分化に果たす役割は？",
                query_type="mechanism",
                expected_nodes=["cell_treg_itreg_001"],
                expected_mechanisms=["TGF-β", "Peripheral tolerance"],
                difficulty_level=3
            )
        ]
        
        # 4. 比較クエリ
        comparison_queries = [
            EvaluationQuery(
                query_id="comparison_001",
                query_text="nTregとiTregの分化場所と条件の違いは？",
                query_type="comparison",
                expected_nodes=["cell_treg_ntreg_001", "cell_treg_itreg_001"],
                ground_truth="nTreg: Thymus, High-affinity TCR; iTreg: Periphery, TGF-β",
                difficulty_level=4
            ),
            EvaluationQuery(
                query_id="comparison_002",
                query_text="HSCとCLPの分化ポテンシャルの違いを説明してください",
                query_type="comparison", 
                expected_nodes=["cell_hsc_001", "cell_clp_001"],
                ground_truth="HSC: Multilineage; CLP: Lymphoid-restricted",
                difficulty_level=3
            )
        ]
        
        # 5. 臨床関連クエリ
        clinical_queries = [
            EvaluationQuery(
                query_id="clinical_001",
                query_text="自己免疫疾患におけるTreg細胞の役割は？",
                query_type="clinical",
                expected_nodes=["cell_treg_ntreg_001", "cell_treg_itreg_001"],
                expected_mechanisms=["Immune tolerance", "Autoimmune suppression"],
                difficulty_level=4
            ),
            EvaluationQuery(
                query_id="clinical_002",
                query_text="Treg細胞の機能不全が引き起こす疾患は？",
                query_type="clinical",
                expected_nodes=["cell_treg_ntreg_001"],
                ground_truth="Type 1 diabetes, Multiple sclerosis, Rheumatoid arthritis",
                difficulty_level=3
            )
        ]
        
        # 全クエリを統合
        self.evaluation_queries.extend(lineage_queries)
        self.evaluation_queries.extend(function_queries)
        self.evaluation_queries.extend(mechanism_queries)
        self.evaluation_queries.extend(comparison_queries)
        self.evaluation_queries.extend(clinical_queries)

class MIRAGEEvaluator:
    """MIRAGE評価システム"""
    
    def __init__(self, raptor_tree: ImmuneCellRAPTORTree):
        self.raptor_tree = raptor_tree
        self.dataset = ImmuneCellEvaluationDataset()
        
    def evaluate_retrieval(self, top_k: int = 5) -> Dict[str, EvaluationMetrics]:
        """検索性能を評価"""
        
        print("Evaluating retrieval performance...")
        
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_mrrs = []
        all_maps = []
        all_ndcgs = []
        
        query_type_results = {}
        
        for query in self.dataset.evaluation_queries:
            # 検索実行
            start_time = datetime.now()
            search_results = self.raptor_tree.hierarchical_search(query.query_text, max_depth=4)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            retrieved_node_ids = [node.id for node, score in search_results[:top_k]]
            
            # メトリクス計算
            metrics = self._calculate_retrieval_metrics(
                retrieved_node_ids, 
                query.expected_nodes,
                top_k
            )
            
            all_precisions.append(metrics.precision)
            all_recalls.append(metrics.recall)
            all_f1s.append(metrics.f1_score)
            all_mrrs.append(metrics.mrr)
            all_maps.append(metrics.map_score)
            all_ndcgs.append(metrics.ndcg)
            
            # クエリタイプ別結果
            if query.query_type not in query_type_results:
                query_type_results[query.query_type] = []
            query_type_results[query.query_type].append(metrics)
        
        # 全体メトリクス
        overall_metrics = EvaluationMetrics(
            precision=np.mean(all_precisions),
            recall=np.mean(all_recalls),
            f1_score=np.mean(all_f1s),
            mrr=np.mean(all_mrrs),
            map_score=np.mean(all_maps),
            ndcg=np.mean(all_ndcgs)
        )
        
        # クエリタイプ別メトリクス
        type_metrics = {}
        for query_type, metrics_list in query_type_results.items():
            type_metrics[query_type] = EvaluationMetrics(
                precision=np.mean([m.precision for m in metrics_list]),
                recall=np.mean([m.recall for m in metrics_list]),
                f1_score=np.mean([m.f1_score for m in metrics_list]),
                mrr=np.mean([m.mrr for m in metrics_list]),
                map_score=np.mean([m.map_score for m in metrics_list]),
                ndcg=np.mean([m.ndcg for m in metrics_list])
            )
        
        return {
            'overall': overall_metrics,
            **type_metrics
        }
    
    def _calculate_retrieval_metrics(self, retrieved: List[str], expected: List[str], k: int) -> EvaluationMetrics:
        """検索メトリクスを計算"""
        
        # Precision@K
        relevant_retrieved = len(set(retrieved) & set(expected))
        precision = relevant_retrieved / len(retrieved) if retrieved else 0.0
        
        # Recall@K
        recall = relevant_retrieved / len(expected) if expected else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, item in enumerate(retrieved):
            if item in expected:
                mrr = 1.0 / (i + 1)
                break
        
        # MAP (Mean Average Precision)
        map_score = 0.0
        relevant_count = 0
        for i, item in enumerate(retrieved):
            if item in expected:
                relevant_count += 1
                map_score += relevant_count / (i + 1)
        map_score = map_score / len(expected) if expected else 0.0
        
        # NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        for i, item in enumerate(retrieved):
            if item in expected:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            mrr=mrr,
            map_score=map_score,
            ndcg=ndcg
        )
    
    def evaluate_summarization(self) -> Dict[str, float]:
        """要約品質を評価"""
        
        print("Evaluating summarization quality...")
        
        coherence_scores = []
        completeness_scores = []
        accuracy_scores = []
        
        for query in self.dataset.evaluation_queries:
            if not query.expected_nodes:
                continue
                
            # 関連ノードの要約生成
            for node_id in query.expected_nodes:
                if node_id in self.raptor_tree.nodes:
                    summary = self.raptor_tree.generate_cell_summary(node_id, include_literature=True)
                    
                    # 要約品質スコア計算
                    coherence = self._calculate_coherence_score(summary)
                    completeness = self._calculate_completeness_score(summary, query)
                    accuracy = self._calculate_accuracy_score(summary, node_id)
                    
                    coherence_scores.append(coherence)
                    completeness_scores.append(completeness)
                    accuracy_scores.append(accuracy)
        
        return {
            'coherence': np.mean(coherence_scores),
            'completeness': np.mean(completeness_scores),
            'accuracy': np.mean(accuracy_scores),
            'overall_summarization': np.mean([
                np.mean(coherence_scores),
                np.mean(completeness_scores), 
                np.mean(accuracy_scores)
            ])
        }
    
    def _calculate_coherence_score(self, summary: str) -> float:
        """要約の一貫性スコアを計算"""
        # 簡易的な一貫性評価（実際にはより複雑な手法を使用）
        lines = summary.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # 基本的な構造チェック
        has_title = any('# ' in line for line in non_empty_lines[:3])
        has_markers = any('marker' in line.lower() for line in non_empty_lines)
        has_functions = any('function' in line.lower() for line in non_empty_lines)
        
        structure_score = (has_title + has_markers + has_functions) / 3.0
        
        # 長さによる評価（適切な長さかどうか）
        length_score = min(len(summary) / 1000, 1.0)  # 1000文字を理想とする
        
        return (structure_score + length_score) / 2.0
    
    def _calculate_completeness_score(self, summary: str, query: EvaluationQuery) -> float:
        """要約の完全性スコアを計算"""
        summary_lower = summary.lower()
        
        # 期待される要素のカバレッジチェック
        coverage_score = 0.0
        total_elements = 0
        
        if query.expected_markers:
            for marker in query.expected_markers:
                if marker.lower() in summary_lower:
                    coverage_score += 1
                total_elements += 1
        
        if query.expected_mechanisms:
            for mechanism in query.expected_mechanisms:
                if mechanism.lower() in summary_lower:
                    coverage_score += 1
                total_elements += 1
        
        return coverage_score / total_elements if total_elements > 0 else 0.5
    
    def _calculate_accuracy_score(self, summary: str, node_id: str) -> float:
        """要約の正確性スコアを計算"""
        if node_id not in self.raptor_tree.nodes:
            return 0.0
        
        node = self.raptor_tree.nodes[node_id]
        summary_lower = summary.lower()
        
        # ノード情報との整合性チェック
        accuracy_elements = []
        
        # 細胞タイプ
        if node.cell_type.lower() in summary_lower:
            accuracy_elements.append(1.0)
        else:
            accuracy_elements.append(0.0)
        
        # マーカー
        marker_matches = sum(1 for marker in node.markers if marker.lower() in summary_lower)
        marker_accuracy = marker_matches / len(node.markers) if node.markers else 0.5
        accuracy_elements.append(marker_accuracy)
        
        # 機能
        function_matches = sum(1 for func in node.functions if any(word in summary_lower for word in func.lower().split()))
        function_accuracy = function_matches / len(node.functions) if node.functions else 0.5
        accuracy_elements.append(function_accuracy)
        
        return np.mean(accuracy_elements)
    
    def evaluate_qa_consistency(self) -> Dict[str, float]:
        """QA整合性を評価"""
        
        print("Evaluating QA consistency...")
        
        lineage_consistency = []
        functional_consistency = []
        mechanistic_consistency = []
        
        for query in self.dataset.evaluation_queries:
            search_results = self.raptor_tree.hierarchical_search(query.query_text, max_depth=4)
            
            if query.query_type == "lineage":
                consistency = self._evaluate_lineage_consistency(query, search_results)
                lineage_consistency.append(consistency)
            elif query.query_type == "function":
                consistency = self._evaluate_functional_consistency(query, search_results)
                functional_consistency.append(consistency)
            elif query.query_type == "mechanism":
                consistency = self._evaluate_mechanistic_consistency(query, search_results)
                mechanistic_consistency.append(consistency)
        
        return {
            'lineage_consistency': np.mean(lineage_consistency) if lineage_consistency else 0.0,
            'functional_consistency': np.mean(functional_consistency) if functional_consistency else 0.0,
            'mechanistic_consistency': np.mean(mechanistic_consistency) if mechanistic_consistency else 0.0,
            'overall_qa_consistency': np.mean([
                np.mean(lineage_consistency) if lineage_consistency else 0.0,
                np.mean(functional_consistency) if functional_consistency else 0.0,
                np.mean(mechanistic_consistency) if mechanistic_consistency else 0.0
            ])
        }
    
    def _evaluate_lineage_consistency(self, query: EvaluationQuery, search_results: List[Tuple]) -> float:
        """分化系譜の整合性を評価"""
        if not query.expected_path:
            return 0.5
        
        # 検索結果に含まれるノードの階層順序をチェック
        retrieved_nodes = [node for node, score in search_results[:5]]
        retrieved_levels = [node.level for node in retrieved_nodes]
        
        # 階層順序の正しさを評価
        if len(retrieved_levels) >= 2:
            ordered_pairs = sum(1 for i in range(len(retrieved_levels)-1) 
                              if retrieved_levels[i] <= retrieved_levels[i+1])
            consistency = ordered_pairs / (len(retrieved_levels) - 1)
        else:
            consistency = 1.0
        
        return consistency
    
    def _evaluate_functional_consistency(self, query: EvaluationQuery, search_results: List[Tuple]) -> float:
        """機能的整合性を評価"""
        if not query.expected_mechanisms:
            return 0.5
        
        # 検索結果のノードが期待される機能を持つかチェック
        retrieved_nodes = [node for node, score in search_results[:3]]
        
        mechanism_coverage = 0
        for node in retrieved_nodes:
            node_mechanisms = node.functions + (node.suppression_mechanisms or [])
            for expected_mech in query.expected_mechanisms:
                if any(expected_mech.lower() in mech.lower() for mech in node_mechanisms):
                    mechanism_coverage += 1
                    break
        
        return mechanism_coverage / len(query.expected_mechanisms)
    
    def _evaluate_mechanistic_consistency(self, query: EvaluationQuery, search_results: List[Tuple]) -> float:
        """メカニズム的整合性を評価"""
        if not (query.expected_mechanisms or query.expected_markers):
            return 0.5
        
        retrieved_nodes = [node for node, score in search_results[:3]]
        
        consistency_score = 0
        total_checks = 0
        
        # マーカーチェック
        if query.expected_markers:
            for marker in query.expected_markers:
                for node in retrieved_nodes:
                    if marker in node.markers:
                        consistency_score += 1
                        break
                total_checks += 1
        
        # メカニズムチェック
        if query.expected_mechanisms:
            for mechanism in query.expected_mechanisms:
                for node in retrieved_nodes:
                    node_mechanisms = (node.suppression_mechanisms or []) + (node.differentiation_factors or [])
                    if any(mechanism.lower() in mech.lower() for mech in node_mechanisms):
                        consistency_score += 1
                        break
                total_checks += 1
        
        return consistency_score / total_checks if total_checks > 0 else 0.5
    
    def run_full_evaluation(self) -> MIRAGEEvaluationResult:
        """完全なMIRAGE評価を実行"""
        
        print("🔬 Running full MIRAGE evaluation...")
        print("=" * 50)
        
        timestamp = datetime.now().isoformat()
        
        # 各評価実行
        retrieval_results = self.evaluate_retrieval()
        summarization_results = self.evaluate_summarization()
        qa_consistency_results = self.evaluate_qa_consistency()
        
        # 全体スコア計算
        overall_score = np.mean([
            retrieval_results['overall'].f1_score,
            summarization_results['overall_summarization'],
            qa_consistency_results['overall_qa_consistency']
        ])
        
        # 詳細結果
        detailed_results = []
        for query in self.dataset.evaluation_queries:
            search_results = self.raptor_tree.hierarchical_search(query.query_text, max_depth=4)
            detailed_results.append({
                'query_id': query.query_id,
                'query_text': query.query_text,
                'query_type': query.query_type,
                'top_results': [(node.id, score) for node, score in search_results[:3]],
                'expected_nodes': query.expected_nodes
            })
        
        evaluation_result = MIRAGEEvaluationResult(
            evaluation_id=f"mirage_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            retrieval_metrics=retrieval_results['overall'],
            summarization_metrics=summarization_results,
            qa_consistency_metrics=qa_consistency_results,
            overall_score=overall_score,
            detailed_results=detailed_results
        )
        
        return evaluation_result
    
    def generate_evaluation_report(self, evaluation_result: MIRAGEEvaluationResult, 
                                 output_file: str = None) -> str:
        """評価レポートを生成"""
        
        report_lines = []
        report_lines.append("# MIRAGE Evaluation Report: Immune Cell Differentiation RAG")
        report_lines.append(f"Evaluation ID: {evaluation_result.evaluation_id}")
        report_lines.append(f"Timestamp: {evaluation_result.timestamp}")
        report_lines.append("")
        
        # 全体スコア
        report_lines.append(f"## Overall Performance Score: {evaluation_result.overall_score:.3f}")
        report_lines.append("")
        
        # 検索性能
        ret_metrics = evaluation_result.retrieval_metrics
        report_lines.append("## Retrieval Performance")
        report_lines.append(f"- Precision@5: {ret_metrics.precision:.3f}")
        report_lines.append(f"- Recall@5: {ret_metrics.recall:.3f}")
        report_lines.append(f"- F1 Score: {ret_metrics.f1_score:.3f}")
        report_lines.append(f"- MRR: {ret_metrics.mrr:.3f}")
        report_lines.append(f"- MAP: {ret_metrics.map_score:.3f}")
        report_lines.append(f"- NDCG: {ret_metrics.ndcg:.3f}")
        report_lines.append("")
        
        # 要約性能
        sum_metrics = evaluation_result.summarization_metrics
        report_lines.append("## Summarization Performance")
        report_lines.append(f"- Coherence: {sum_metrics['coherence']:.3f}")
        report_lines.append(f"- Completeness: {sum_metrics['completeness']:.3f}")
        report_lines.append(f"- Accuracy: {sum_metrics['accuracy']:.3f}")
        report_lines.append(f"- Overall: {sum_metrics['overall_summarization']:.3f}")
        report_lines.append("")
        
        # QA整合性
        qa_metrics = evaluation_result.qa_consistency_metrics
        report_lines.append("## QA Consistency Performance")
        report_lines.append(f"- Lineage Consistency: {qa_metrics['lineage_consistency']:.3f}")
        report_lines.append(f"- Functional Consistency: {qa_metrics['functional_consistency']:.3f}")
        report_lines.append(f"- Mechanistic Consistency: {qa_metrics['mechanistic_consistency']:.3f}")
        report_lines.append(f"- Overall: {qa_metrics['overall_qa_consistency']:.3f}")
        report_lines.append("")
        
        # サンプル結果
        report_lines.append("## Sample Query Results")
        for result in evaluation_result.detailed_results[:5]:
            report_lines.append(f"\n### Query: {result['query_text']}")
            report_lines.append(f"Type: {result['query_type']}")
            report_lines.append("Top Results:")
            for i, (node_id, score) in enumerate(result['top_results'], 1):
                report_lines.append(f"  {i}. {node_id} (score: {score:.3f})")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Evaluation report saved to: {output_file}")
        
        return report_content

def main():
    """メイン実行関数"""
    print("🔬 MIRAGE Evaluation System for Immune Cell RAG")
    print("=" * 60)
    
    # パス設定  
    base_dir = Path("C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree")
    cache_dir = base_dir / "data/immune_cell_differentiation"
    hierarchy_file = cache_dir / "immune_cell_hierarchy.json"
    
    # RAPTOR Tree読み込み（簡易版）
    print("Loading RAPTOR Tree...")
    raptor_tree = ImmuneCellRAPTORTree(cache_dir=str(cache_dir))
    raptor_tree.load_immune_hierarchy(str(hierarchy_file))
    raptor_tree.build_embeddings()
    
    # 評価実行
    evaluator = MIRAGEEvaluator(raptor_tree)
    evaluation_result = evaluator.run_full_evaluation()
    
    # レポート生成
    output_file = cache_dir / "evaluation_results" / f"mirage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report = evaluator.generate_evaluation_report(evaluation_result, str(output_file))
    
    # 結果表示
    print("\n" + "=" * 60)
    print("🎯 MIRAGE Evaluation Completed!")
    print(f"Overall Score: {evaluation_result.overall_score:.3f}")
    print(f"Report saved to: {output_file}")
    
    # 主要メトリクス表示
    print("\nKey Metrics:")
    print(f"- Retrieval F1: {evaluation_result.retrieval_metrics.f1_score:.3f}")
    print(f"- Summarization: {evaluation_result.summarization_metrics['overall_summarization']:.3f}")
    print(f"- QA Consistency: {evaluation_result.qa_consistency_metrics['overall_qa_consistency']:.3f}")

if __name__ == "__main__":
    main()