#!/usr/bin/env python3
"""
免疫学用語検証ツール
Immune Terminology Validation Tool

生成されたRAPTORツリーのラベルが免疫学的に正しいかを検証
"""

import json
from pathlib import Path
from datetime import datetime
from immune_cell_vocab import (
    generate_immune_label, 
    validate_immune_terminology,
    extract_level_keywords,
    IMMUNE_HIERARCHY_VOCAB,
    LEVEL_COLOR_MAPPING
)

class ImmuneTerminologyValidator:
    """免疫学用語検証システム"""
    
    def __init__(self):
        self.validation_results = []
        self.corrections = []
    
    def load_latest_tree(self):
        """最新のRAPTORツリーを読み込み"""
        data_dir = Path("data/immune_cell_differentiation/raptor_trees")
        if not data_dir.exists():
            data_dir = Path(".")
            
        json_files = list(data_dir.glob("*raptor*tree*.json"))
        if not json_files:
            raise FileNotFoundError("RAPTORツリーファイルが見つかりません")
            
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Loading: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f), latest_file.name
    
    def validate_all_nodes(self, tree_data):
        """全ノードの免疫学用語を検証"""
        
        print("\n🔬 免疫学用語検証開始")
        print("=" * 60)
        
        validation_summary = {
            'total_nodes': 0,
            'valid_nodes': 0,
            'invalid_nodes': 0,
            'warnings': 0,
            'level_analysis': {}
        }
        
        for node_id, node_data in tree_data['nodes'].items():
            if any(x in node_id for x in ['_L', 'root']):  # クラスター化されたノードのみ
                validation_summary['total_nodes'] += 1
                
                # レベル判定
                if 'root' in node_id:
                    level = 4
                elif '_L3_' in node_id:
                    level = 3
                elif '_L2_' in node_id:
                    level = 2
                elif '_L1_' in node_id:
                    level = 1
                else:
                    level = 0
                
                # レベル別統計初期化
                if level not in validation_summary['level_analysis']:
                    validation_summary['level_analysis'][level] = {
                        'count': 0, 'valid': 0, 'invalid': 0, 'keywords_found': []
                    }
                
                validation_summary['level_analysis'][level]['count'] += 1
                
                # コンテンツとクラスターサイズ取得
                content = node_data.get('content', '')
                cluster_size = node_data.get('cluster_size', 1)
                cluster_id = node_id.split('_C')[1].split('_')[0] if '_C' in node_id else '0'
                
                # 免疫学的ラベル生成
                immune_label = generate_immune_label(content, level, cluster_id, cluster_size)
                
                # 用語妥当性検証
                is_valid, validation_msg = validate_immune_terminology(immune_label)
                
                # レベル固有キーワード抽出
                level_keywords = extract_level_keywords(content, level)
                validation_summary['level_analysis'][level]['keywords_found'].extend(level_keywords)
                
                # 結果記録
                result = {
                    'node_id': node_id,
                    'level': level,
                    'original_content_preview': content[:100] + "..." if len(content) > 100 else content,
                    'generated_label': immune_label,
                    'is_valid': is_valid,
                    'validation_message': validation_msg,
                    'keywords_found': level_keywords,
                    'cluster_size': cluster_size
                }
                
                self.validation_results.append(result)
                
                if is_valid:
                    validation_summary['valid_nodes'] += 1
                    validation_summary['level_analysis'][level]['valid'] += 1
                    status = "✅"
                else:
                    validation_summary['invalid_nodes'] += 1
                    validation_summary['level_analysis'][level]['invalid'] += 1
                    status = "❌"
                
                print(f"{status} {node_id}")
                print(f"   レベル: {level} ({LEVEL_COLOR_MAPPING.get(level, {}).get('description', 'Unknown')})")
                print(f"   ラベル: {immune_label}")
                print(f"   検証: {validation_msg}")
                if level_keywords:
                    print(f"   キーワード: {', '.join(level_keywords[:5])}")
                print("-" * 60)
        
        return validation_summary
    
    def generate_corrections(self):
        """修正提案の生成"""
        
        print("\n🔧 修正提案")
        print("=" * 60)
        
        corrections_made = 0
        
        for result in self.validation_results:
            if not result['is_valid']:
                corrections_made += 1
                
                # 修正提案の生成
                level = result['level']
                content = result['original_content_preview']
                cluster_size = result['cluster_size']
                
                # より適切なキーワードの提案
                suggested_keywords = self._suggest_keywords(content, level)
                
                # 修正されたラベル生成
                if suggested_keywords:
                    corrected_label = self._generate_corrected_label(level, suggested_keywords, cluster_size)
                else:
                    corrected_label = f"{LEVEL_COLOR_MAPPING.get(level, {}).get('name', f'L{level}')}\n({cluster_size})"
                
                correction = {
                    'node_id': result['node_id'],
                    'level': level,
                    'original_label': result['generated_label'],
                    'corrected_label': corrected_label,
                    'suggested_keywords': suggested_keywords,
                    'reason': result['validation_message']
                }
                
                self.corrections.append(correction)
                
                print(f"🔧 {result['node_id']}")
                print(f"   修正前: {result['generated_label']}")
                print(f"   修正後: {corrected_label}")
                print(f"   理由: {result['validation_message']}")
                print("-" * 60)
        
        if corrections_made == 0:
            print("✅ 修正が必要なノードはありません")
        
        return corrections_made
    
    def _suggest_keywords(self, content, level):
        """レベルに応じた適切なキーワードを提案"""
        
        if level == 0:
            vocab_set = IMMUNE_HIERARCHY_VOCAB["hsc_level"]
        elif level == 1:
            vocab_set = IMMUNE_HIERARCHY_VOCAB["clp_level"]
        elif level == 2:
            vocab_set = IMMUNE_HIERARCHY_VOCAB["cd4_t_level"]
        elif level == 3:
            vocab_set = IMMUNE_HIERARCHY_VOCAB["treg_level"]
        else:
            return []
        
        # 日本語・英語キーワードから関連用語を検索
        all_keywords = set()
        all_keywords.update(vocab_set["japanese"])
        all_keywords.update(vocab_set["english"])
        
        # コンテンツに含まれるキーワードを抽出
        found_keywords = []
        content_lower = content.lower()
        
        for keyword in all_keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:3]  # 最大3つ
    
    def _generate_corrected_label(self, level, keywords, cluster_size):
        """修正されたラベルを生成"""
        
        level_info = LEVEL_COLOR_MAPPING.get(level, {"name": f"L{level}", "description": "不明"})
        
        if keywords:
            keyword_text = "・".join(keywords)
            if len(keyword_text) > 20:
                keyword_text = keyword_text[:17] + "..."
            return f"{level_info['name']}\n{keyword_text}\n({cluster_size})"
        else:
            return f"{level_info['name']}\n{level_info['description']}\n({cluster_size})"
    
    def print_validation_summary(self, summary):
        """検証結果サマリーの表示"""
        
        print(f"\n📊 検証結果サマリー")
        print("=" * 60)
        print(f"総ノード数: {summary['total_nodes']}")
        print(f"✅ 適切: {summary['valid_nodes']}")
        print(f"❌ 不適切: {summary['invalid_nodes']}")
        print(f"正解率: {summary['valid_nodes']/summary['total_nodes']*100:.1f}%")
        
        print(f"\n📈 レベル別分析:")
        for level, analysis in summary['level_analysis'].items():
            level_info = LEVEL_COLOR_MAPPING.get(level, {"name": f"L{level}", "description": "不明"})
            accuracy = analysis['valid'] / analysis['count'] * 100 if analysis['count'] > 0 else 0
            
            print(f"  {level_info['name']} ({level_info['description']}):")
            print(f"    ノード数: {analysis['count']}")
            print(f"    正解率: {accuracy:.1f}%")
            
            # 発見されたキーワードの頻度分析
            if analysis['keywords_found']:
                unique_keywords = list(set(analysis['keywords_found']))[:5]
                print(f"    主要キーワード: {', '.join(unique_keywords)}")
    
    def save_validation_report(self, summary, filename=None):
        """検証レポートの保存"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"immune_terminology_validation_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'validation_results': self.validation_results,
            'corrections': self.corrections
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 検証レポート保存: {filename}")

def main():
    print("🧬 免疫学用語検証システム")
    print("=" * 60)
    
    validator = ImmuneTerminologyValidator()
    
    try:
        # ツリーデータ読み込み
        tree_data, filename = validator.load_latest_tree()
        print(f"📊 Tree loaded: {len(tree_data['nodes'])} nodes")
        
        # 全ノード検証
        summary = validator.validate_all_nodes(tree_data)
        
        # 修正提案生成
        corrections_count = validator.generate_corrections()
        
        # サマリー表示
        validator.print_validation_summary(summary)
        
        # レポート保存
        validator.save_validation_report(summary)
        
        print(f"\n✅ 検証完了!")
        print(f"📊 総ノード数: {summary['total_nodes']}")
        print(f"🎯 正解率: {summary['valid_nodes']/summary['total_nodes']*100:.1f}%")
        print(f"🔧 修正提案: {corrections_count}件")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main()