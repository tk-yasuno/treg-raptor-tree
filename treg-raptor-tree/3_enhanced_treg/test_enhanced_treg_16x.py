#!/usr/bin/env python3
"""
Enhanced Treg Differentiation 16x Scale Integration Test
拡張Treg分化経路 16倍スケール統合テスト

新機能テスト:
1. CD127低発現（IL-7Rα）マーカー検出
2. Foxp3安定性識別（TSDR脱メチル化 vs 一過性発現）
3. 7層階層構造の正確な判定
4. ヒトTreg臨床マーカー対応（CD4+CD25+CD127low）

Author: AI Assistant
Date: 2025-11-02
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# パス設定
sys.path.insert(0, str(Path(__file__).parent))

# 拡張Treg語彙システムをインポート
from enhanced_treg_vocab import (
    ENHANCED_LEVEL_COLOR_MAPPING,
    TREG_DIFFERENTIATION_VOCAB,
    determine_treg_level,
    generate_enhanced_treg_label
)

class EnhancedTregIntegrationTest:
    """拡張Treg分化システム統合テスト"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "data/enhanced_treg_test_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = {}
        
    def create_test_documents(self) -> List[Dict[str, Any]]:
        """7層階層をカバーするテスト文書セット"""
        
        test_docs = [
            # Level 0: HSC
            {
                "content": "Hematopoietic stem cells (HSC) reside in bone marrow niches with multipotent self-renewal capacity. "
                          "Lin- Sca-1+ c-Kit+ CD34- CD150+ SLAM markers identify long-term HSC populations.",
                "expected_level": 0,
                "expected_name": "HSC"
            },
            
            # Level 1: CLP
            {
                "content": "Common lymphoid progenitors (CLP) express IL-7R and Flt3, committing to lymphoid lineage. "
                          "IL-7 signaling is critical for CLP maintenance and B/T/NK cell potential.",
                "expected_level": 1,
                "expected_name": "CLP"
            },
            
            # Level 2: CD4+ T
            {
                "content": "CD4+ T helper cells recognize MHC class II presented antigens via TCR. "
                          "Thymic selection produces CD4+CD8- single positive T cells with diverse TCR repertoire.",
                "expected_level": 2,
                "expected_name": "CD4+T"
            },
            
            # Level 3: CD4+CD25+CD127low (ヒトTreg同定マーカー)
            {
                "content": "Human Treg identification uses CD4+CD25high CD127low phenotype. "
                          "IL-2Rα high expression with IL-7Rα low distinguishes Treg from activated T cells. "
                          "CD127 low expression is a critical marker for human Treg purity.",
                "expected_level": 3,
                "expected_name": "CD4+CD25+CD127low"
            },
            
            # Level 4: nTreg (thymic origin)
            {
                "content": "Natural Treg (nTreg) develop in thymus with Helios+ Nrp1+ phenotype. "
                          "Thymic selection by AIRE-expressing medullary epithelial cells drives nTreg generation. "
                          "High-affinity TCR recognition of self-antigens is required.",
                "expected_level": 4,
                "expected_name": "nTreg/iTreg"
            },
            
            # Level 4: iTreg (peripheral origin)
            {
                "content": "Peripheral induced Treg (iTreg) convert from naive CD4+ T cells in gut mucosa. "
                          "TGF-beta and retinoic acid drive peripheral Treg conversion. "
                          "iTreg are typically Helios- and critical for mucosal tolerance.",
                "expected_level": 4,
                "expected_name": "nTreg/iTreg"
            },
            
            # Level 5: Foxp3+ stable Treg
            {
                "content": "Stable Foxp3+ Treg show TSDR demethylation at CNS2 region ensuring epigenetic stability. "
                          "CD45RA+ resting Treg represent bona fide suppressive population. "
                          "TSDR demethylation distinguishes stable Treg from transient Foxp3 expression.",
                "expected_level": 5,
                "expected_name": "Foxp3+Treg"
            },
            
            # Level 5: Transient Foxp3 (一過性発現)
            {
                "content": "Activated CD4+ T cells transiently express Foxp3 upon TCR stimulation. "
                          "CD45RO+ effector T cells with TSDR methylated lack suppressive function. "
                          "Transient Foxp3 expression does not confer regulatory phenotype.",
                "expected_level": 5,
                "expected_name": "Foxp3+Treg"
            },
            
            # Level 6: Functional Treg (サイトカイン産生)
            {
                "content": "Functional Treg suppress via IL-10, TGF-beta, and IL-35 production. "
                          "CTLA-4 mediates contact-dependent immunosuppression by competing for CD80/CD86. "
                          "LAG-3 and PD-1 contribute to multi-mechanism suppressive function.",
                "expected_level": 6,
                "expected_name": "Functional Treg"
            },
            
            # 複合マーカー検出テスト
            {
                "content": "Human Treg are CD4+CD25+CD127low Foxp3+ with TSDR demethylation. "
                          "These cells produce IL-10 and TGF-beta for immunosuppression. "
                          "CTLA-4 expression enables contact-dependent suppression.",
                "expected_level": 6,  # 最高レベル（機能的Treg）
                "expected_name": "Functional Treg"
            }
        ]
        
        return test_docs
    
    def test_level_determination(self):
        """階層判定ロジックのテスト"""
        print("\n" + "="*80)
        print("TEST 1: Level Determination Accuracy")
        print("="*80)
        
        test_docs = self.create_test_documents()
        passed = 0
        failed = 0
        
        results = []
        
        for i, doc in enumerate(test_docs):
            content = doc["content"]
            expected_level = doc["expected_level"]
            expected_name = doc["expected_name"]
            
            # 階層判定
            detected_level = determine_treg_level(content)
            detected_name = ENHANCED_LEVEL_COLOR_MAPPING[detected_level]["name"]
            
            # 結果判定
            is_correct = (detected_level == expected_level)
            status = "✓ PASS" if is_correct else "✗ FAIL"
            
            if is_correct:
                passed += 1
            else:
                failed += 1
            
            # 詳細出力
            print(f"\nTest Case {i+1}: {status}")
            print(f"  Content: {content[:80]}...")
            print(f"  Expected: Level {expected_level} ({expected_name})")
            print(f"  Detected: Level {detected_level} ({detected_name})")
            
            results.append({
                "test_case": i+1,
                "content_preview": content[:100],
                "expected_level": expected_level,
                "detected_level": detected_level,
                "expected_name": expected_name,
                "detected_name": detected_name,
                "passed": is_correct
            })
        
        # 統計サマリー
        accuracy = (passed / len(test_docs)) * 100
        print(f"\n{'='*80}")
        print(f"TEST 1 SUMMARY:")
        print(f"  Total Cases: {len(test_docs)}")
        print(f"  Passed: {passed} ({accuracy:.1f}%)")
        print(f"  Failed: {failed}")
        print(f"{'='*80}")
        
        self.test_results["level_determination"] = {
            "total": len(test_docs),
            "passed": passed,
            "failed": failed,
            "accuracy": accuracy,
            "details": results
        }
        
        return accuracy >= 80  # 80%以上で合格
    
    def test_label_generation(self):
        """ラベル生成機能のテスト"""
        print("\n" + "="*80)
        print("TEST 2: Enhanced Label Generation")
        print("="*80)
        
        test_cases = [
            {
                "content": "CD4+CD25high CD127low IL-7Rα low expression in human Treg",
                "level": 3,
                "cluster_id": 1,
                "cluster_size": 42,
                "expected_keywords": ["CD127low", "IL-2Rα", "IL-7Rα"]
            },
            {
                "content": "Stable Foxp3+ Treg with TSDR demethylation and CD45RA expression",
                "level": 5,
                "cluster_id": 2,
                "cluster_size": 28,
                "expected_keywords": ["stable", "TSDR"]
            },
            {
                "content": "Transient Foxp3 in CD45RO+ activated effector T cells",
                "level": 5,
                "cluster_id": 3,
                "cluster_size": 15,
                "expected_keywords": ["transient", "CD45RO"]
            },
            {
                "content": "IL-10 and TGF-beta producing suppressive Treg with CTLA-4",
                "level": 6,
                "cluster_id": 4,
                "cluster_size": 35,
                "expected_keywords": ["IL-10", "TGF-β", "CTLA-4"]
            }
        ]
        
        passed = 0
        results = []
        
        for i, case in enumerate(test_cases):
            label = generate_enhanced_treg_label(
                case["content"],
                case["level"],
                case["cluster_id"],
                case["cluster_size"]
            )
            
            # キーワード検出チェック
            label_lower = label.lower()
            keywords_found = [kw for kw in case["expected_keywords"] 
                            if kw.lower() in label_lower]
            
            is_valid = len(keywords_found) > 0
            status = "✓ PASS" if is_valid else "✗ FAIL"
            
            if is_valid:
                passed += 1
            
            print(f"\nTest Case {i+1}: {status}")
            print(f"  Level {case['level']}: {ENHANCED_LEVEL_COLOR_MAPPING[case['level']]['name']}")
            print(f"  Generated Label:\n{label}")
            print(f"  Expected Keywords: {case['expected_keywords']}")
            print(f"  Found Keywords: {keywords_found}")
            
            results.append({
                "test_case": i+1,
                "level": case["level"],
                "label": label,
                "expected_keywords": case["expected_keywords"],
                "found_keywords": keywords_found,
                "passed": is_valid
            })
        
        accuracy = (passed / len(test_cases)) * 100
        print(f"\n{'='*80}")
        print(f"TEST 2 SUMMARY:")
        print(f"  Total Cases: {len(test_cases)}")
        print(f"  Passed: {passed} ({accuracy:.1f}%)")
        print(f"{'='*80}")
        
        self.test_results["label_generation"] = {
            "total": len(test_cases),
            "passed": passed,
            "accuracy": accuracy,
            "details": results
        }
        
        return accuracy >= 75
    
    def test_vocabulary_coverage(self):
        """語彙体系の網羅性テスト"""
        print("\n" + "="*80)
        print("TEST 3: Vocabulary Coverage Analysis")
        print("="*80)
        
        coverage_stats = {}
        
        for level_name, vocab_data in TREG_DIFFERENTIATION_VOCAB.items():
            print(f"\n{level_name}:")
            
            if isinstance(vocab_data, dict):
                if "japanese" in vocab_data and "english" in vocab_data:
                    jp_count = len(vocab_data["japanese"])
                    en_count = len(vocab_data["english"])
                    print(f"  Japanese terms: {jp_count}")
                    print(f"  English terms: {en_count}")
                    coverage_stats[level_name] = {"japanese": jp_count, "english": en_count}
                else:
                    # ネストされた辞書の場合
                    total_terms = 0
                    for sub_key, sub_data in vocab_data.items():
                        if isinstance(sub_data, dict):
                            if "japanese" in sub_data:
                                total_terms += len(sub_data["japanese"])
                            if "english" in sub_data:
                                total_terms += len(sub_data["english"])
                    print(f"  Total terms: {total_terms}")
                    coverage_stats[level_name] = {"total": total_terms}
        
        self.test_results["vocabulary_coverage"] = coverage_stats
        
        return True
    
    def test_gpu_performance(self):
        """GPU性能テスト（簡易版）"""
        print("\n" + "="*80)
        print("TEST 4: GPU Performance Check")
        print("="*80)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"  GPU: {gpu_name}")
            print(f"  Total Memory: {gpu_memory:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch Version: {torch.__version__}")
            
            # メモリテスト
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            
            # ダミーテンソル作成
            test_tensor = torch.randn(1000, 1000).to(self.device)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            
            del test_tensor
            torch.cuda.empty_cache()
            
            print(f"  Initial Memory: {initial_memory:.3f} GB")
            print(f"  Peak Memory: {peak_memory:.3f} GB")
            
            self.test_results["gpu_performance"] = {
                "gpu_name": gpu_name,
                "total_memory_gb": gpu_memory,
                "cuda_available": True
            }
            
            return True
        else:
            print("  ⚠️ No GPU available - running on CPU")
            self.test_results["gpu_performance"] = {"cuda_available": False}
            return False
    
    def run_all_tests(self):
        """全テストを実行"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("="*80)
        print("Enhanced Treg Differentiation - 16x Scale Integration Test")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        test_suite = [
            ("Level Determination", self.test_level_determination),
            ("Label Generation", self.test_label_generation),
            ("Vocabulary Coverage", self.test_vocabulary_coverage),
            ("GPU Performance", self.test_gpu_performance)
        ]
        
        results_summary = []
        
        for test_name, test_func in test_suite:
            try:
                start_time = time.time()
                passed = test_func()
                elapsed = time.time() - start_time
                
                results_summary.append({
                    "test": test_name,
                    "passed": passed,
                    "elapsed_seconds": elapsed
                })
            except Exception as e:
                print(f"\n✗ ERROR in {test_name}: {e}")
                import traceback
                traceback.print_exc()
                results_summary.append({
                    "test": test_name,
                    "passed": False,
                    "error": str(e)
                })
        
        # 最終サマリー
        print("\n" + "="*80)
        print("FINAL TEST SUMMARY")
        print("="*80)
        
        total_tests = len(results_summary)
        passed_tests = sum(1 for r in results_summary if r.get("passed", False))
        
        for result in results_summary:
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            elapsed = result.get("elapsed_seconds", 0)
            print(f"  {status} {result['test']: <30} ({elapsed:.2f}s)")
        
        print(f"\n  Overall: {passed_tests}/{total_tests} tests passed")
        print("="*80)
        
        # 結果を保存
        output_file = self.results_dir / f"test_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "summary": results_summary,
                "detailed_results": self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Test results saved to: {output_file}")
        
        return passed_tests == total_tests


def main():
    """メイン実行関数"""
    tester = EnhancedTregIntegrationTest()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
