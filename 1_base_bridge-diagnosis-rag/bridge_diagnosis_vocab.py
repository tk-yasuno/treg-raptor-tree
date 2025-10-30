"""
橋梁診断ドメイン専門語彙
Bridge Diagnosis Domain Vocabulary

橋梁点検・診断に特化した専門用語と翻訳辞書を提供
"""

# ストップワード（ノードラベルから除外する語）
STOP_WORDS = {
    # 一般的なストップワード
    '本資料', '本研究', '主な数', '橋梁点検', '橋梁の', '橋梁に', '橋台や',
    '設計', '施工', '残念な', '申し訳', '提示い', 'いただ',
    'Grider', 'Rust', 'Damage',
    
    # 追加のストップワード
    '資料', '研究', '報告', '説明', '記載', '図', '表', '写真',
    'ページ', '参照', '以下', '上記', '前述', '後述',
    '場合', '状態', '程度', '箇所', '部分', '全体',
    'こと', 'もの', 'ため', 'など', 'による', 'について',
    '診断',  # 診断セット文書で頻出するため除外
}

# 橋梁診断ドメインキーワード（日本語）
BRIDGE_DOMAIN_KEYWORDS = {
    # 橋梁構造部材
    '床版', '桁', '主桁', '横桁', '縦桁', '鋼桁', 'RC桁', 'PC桁',
    '支承', 'ゴム支承', 'パッド型支承', '鋼製支承', 'ピン支承', 'ローラー支承',
    '橋台', '橋脚', '下部構造', '上部構造', '基礎', 'フーチング',
    '伸縮装置', '排水装置', '高欄', '地覆', '舗装', '道路舗装',
    'アーチ', 'トラス', 'ラーメン', '吊橋', '斜張橋',
    '溝橋', '床版橋', 'ボックスカルバート', '連結桁',
    
    # 橋の種類
    'コンクリート橋', 'RC橋', 'PC橋', '鋼橋', '鋼コンクリート合成橋',
    'プレテンション橋', 'ポストテンション橋', 'ポステン橋', 'プレテン橋',
    'トラス橋', 'アーチ橋', '吊橋', '斜張橋', 'ラーメン橋',
    '単純桁橋', '連続桁橋', 'ゲルバー桁橋', '張出桁橋',
    
    # 材料
    'コンクリート', '鉄筋', '鉄筋露出', '鋼材', 'PC鋼材', 'アスファルト',
    'モルタル', 'グラウト', '塗装', '防水材', '補強材',
    '高力ボルト', 'リベット', '溶接', 'ボルト', 'ナット',
    
    # 劣化・損傷
    '腐食', 'ひび割れ', '剥離', '剥落', '変色', '遊離石灰',
    '疲労', '亀裂', '疲労亀裂', 'クラック', '鉄筋腐食', '塩害', 'ASR',
    '凍害', '中性化', 'アルカリシリカ反応', '土砂化',
    '錆', '錆汁', '鋼材腐食', '板厚減少', '断面欠損',
    '変形', 'たわみ', 'ずれ', '段差', '目地開き',
    '滞水', '漏水', '遊間不足', '肩当り', '逸脱',
    'ボイド管浮き上がり', '陥没', '破断', '座屈', '緩み',
    
    # 劣化要因
    '凍結融解', '塩分', '塩化物', '酸性雨', '排気ガス',
    '洗堀', '側方流動', '地すべり', '沈下', '地盤変動',
    '地震', '振動', '荷重', '過荷重', '衝撃', '摩耗',
    'オゾン劣化', '紫外線', '化学的侵食', '浸食',
    
    # 点検・診断
    '定期点検', '詳細点検', '近接目視', '打音検査', '触診',
    '健全性診断', '判定区分', '健全度', '要因分析',
    '損傷程度', '進行性', '緊急性', '重要度',
    '初期点検', '中間点検', '事後点検',
    
    # 評価
    '損傷評価', 'ランク', '段階', 'レベル', '区分',
    '判定', '要注意', '要対策', '要監視', '経過観察',
    'I区分', 'II区分', 'III区分', 'IV区分',
    '健全', '予防保全段階', '早期措置段階', '緊急措置段階',
    
    # 対策・補修
    '補修', '補強', '更新', '修繕', '保全', '措置',
    '応急措置', '恒久対策', '予防保全', '事後保全',
    'ひび割れ補修', '断面修復', '表面被覆', '含浸材',
    '炭素繊維シート', '鋼板接着', '増桁', '増厚',
    '塗装塗替え', '再塗装', 'ボルト締直し', '支承取替え',
    '床版取替え', '桁取替え', '落橋防止', '耐震補強',
    
    # 検査技術
    '非破壊検査', '超音波探傷', '磁粉探傷', '浸透探傷',
    'X線検査', 'コア抜き', 'はつり調査', '鉄筋探査',
    '塩分測定', '中性化深さ', '強度試験', '載荷試験',
    
    # 記録・書類
    '点検調書', '診断カルテ', '損傷図', '写真管理',
    '変状展開図', '橋梁台帳', '履歴', '記録',
    
    # 管理
    '維持管理', 'アセットマネジメント', 'LCC', 'ライフサイクルコスト',
    '長寿命化', '予防保全型管理', '計画的管理',
    'PDCA', '優先順位', '予算', 'コスト縮減',
    
    # 基準・規定
    '道路橋示方書', '点検要領', '診断基準', '設計基準',
    '規準', '仕様書', 'マニュアル', 'ガイドライン',
    '告示', '通達', '指針', '要領',
    
    # 品質・性能
    '耐久性', '耐荷力', '耐震性', '走行性', '安全性',
    '機能', '性能', '品質', '劣化抵抗性',
    
    # 張出し部特有
    '張出し部', '張出床版', 'キャンチレバー',
    
    # 継手・接合部
    '継手', '添接', '連結部', '接合部', '結合部',
    '現場継手', '工場継手', 'ボルト継手', '溶接継手',
    
    # 付属物
    '高欄', '防護柵', '遮音壁', '照明', '標識',
    
    # その他専門用語
    '活荷重', '死荷重', '設計荷重', '許容応力',
    'かぶり', '鉄筋かぶり', '縁端距離', '有効高',
    '付着', '定着', '定着長', '開口部', '貫通孔',
}

# 日本語→英語翻訳辞書（橋梁診断専門用語）
BRIDGE_TRANSLATION_DICT = {
    # 橋梁構造部材
    '床版': 'Deck Slab',
    '桁': 'Girder',
    '主桁': 'Main Girder',
    '横桁': 'Cross Beam',
    '縦桁': 'Stringer',
    '鋼桁': 'Steel Girder',
    'RC桁': 'RC Girder',
    'PC桁': 'PC Girder',
    '支承': 'Bearing',
    'ゴム支承': 'Rubber Bearing',
    'パッド型支承': 'Pad Bearing',
    '鋼製支承': 'Steel Bearing',
    'ピン支承': 'Pin Bearing',
    'ローラー支承': 'Roller Bearing',
    '橋台': 'Abutment',
    '橋脚': 'Pier',
    '下部構造': 'Substructure',
    '上部構造': 'Superstructure',
    '基礎': 'Foundation',
    'フーチング': 'Footing',
    '伸縮装置': 'Expansion Joint',
    '排水装置': 'Drainage',
    '高欄': 'Railing',
    '地覆': 'Curb',
    '舗装': 'Pavement',
    '道路舗装': 'Road Pavement',
    'アーチ': 'Arch',
    'トラス': 'Truss',
    'ラーメン': 'Rahmen',
    '吊橋': 'Suspension Bridge',
    '斜張橋': 'Cable-Stayed',
    '溝橋': 'Culvert Bridge',
    '床版橋': 'Slab Bridge',
    'ボックスカルバート': 'Box Culvert',
    '連結桁': 'Connected Girder',
    '張出し部': 'Cantilever',
    '張出床版': 'Cantilever Slab',
    'キャンチレバー': 'Cantilever',
    
    # 材料
    'コンクリート': 'Concrete',
    '鉄筋': 'Rebar',
    '鉄筋露出': 'Rebar Exposure',
    '鋼材': 'Steel',
    'PC鋼材': 'PC Steel',
    'アスファルト': 'Asphalt',
    'モルタル': 'Mortar',
    'グラウト': 'Grout',
    '塗装': 'Coating',
    '防水材': 'Waterproofing',
    '補強材': 'Reinforcement',
    '高力ボルト': 'High Strength Bolt',
    'リベット': 'Rivet',
    '溶接': 'Welding',
    'ボルト': 'Bolt',
    'ナット': 'Nut',
    
    # 劣化・損傷
    '腐食': 'Corrosion',
    'ひび割れ': 'Crack',
    '剥離': 'Delamination',
    '剥落': 'Spalling',
    '変色': 'Discoloration',
    '遊離石灰': 'Free Lime',
    '疲労': 'Fatigue',
    '亀裂': 'Crack',
    '疲労亀裂': 'Fatigue Crack',
    'クラック': 'Crack',
    '鉄筋腐食': 'Rebar Corrosion',
    '塩害': 'Salt Damage',
    'ASR': 'ASR',
    '凍害': 'Frost Damage',
    '中性化': 'Carbonation',
    'アルカリシリカ反応': 'ASR',
    '土砂化': 'Disintegration',
    '錆': 'Rust',
    '錆汁': 'Rust Stain',
    '鋼材腐食': 'Steel Corrosion',
    '板厚減少': 'Plate Thinning',
    '断面欠損': 'Section Loss',
    '変形': 'Deformation',
    'たわみ': 'Deflection',
    'ずれ': 'Displacement',
    '段差': 'Step',
    '目地開き': 'Joint Opening',
    '滞水': 'Water Ponding',
    '漏水': 'Water Leakage',
    '遊間不足': 'Gap Shortage',
    '肩当り': 'Shoulder Contact',
    '逸脱': 'Deviation',
    'ボイド管浮き上がり': 'Void Uplift',
    '陥没': 'Subsidence',
    '破断': 'Fracture',
    '座屈': 'Buckling',
    '緩み': 'Loosening',
    
    # 劣化要因
    '凍結融解': 'Freeze-Thaw',
    '塩分': 'Salt',
    '塩化物': 'Chloride',
    '酸性雨': 'Acid Rain',
    '排気ガス': 'Exhaust Gas',
    '洗堀': 'Scouring',
    '側方流動': 'Lateral Flow',
    '地すべり': 'Landslide',
    '沈下': 'Settlement',
    '地盤変動': 'Ground Movement',
    '地震': 'Earthquake',
    '振動': 'Vibration',
    '荷重': 'Load',
    '過荷重': 'Overload',
    '衝撃': 'Impact',
    '摩耗': 'Wear',
    'オゾン劣化': 'Ozone Degradation',
    '紫外線': 'UV',
    '化学的侵食': 'Chemical Attack',
    '浸食': 'Erosion',
    
    # 点検・診断
    '定期点検': 'Regular Inspection',
    '詳細点検': 'Detailed Inspection',
    '近接目視': 'Close Visual',
    '打音検査': 'Hammer Sounding',
    '触診': 'Touch Test',
    '健全性診断': 'Soundness Diagnosis',
    '判定区分': 'Judgment Category',
    '健全度': 'Soundness Level',
    '要因分析': 'Factor Analysis',
    '損傷程度': 'Damage Degree',
    '進行性': 'Progression',
    '緊急性': 'Urgency',
    '重要度': 'Importance',
    '初期点検': 'Initial Inspection',
    '中間点検': 'Interim Inspection',
    '事後点検': 'Post Inspection',
    
    # 評価
    '損傷評価': 'Damage Assessment',
    'ランク': 'Rank',
    '段階': 'Stage',
    'レベル': 'Level',
    '区分': 'Category',
    '判定': 'Judgment',
    '要注意': 'Caution Required',
    '要対策': 'Action Required',
    '要監視': 'Monitoring Required',
    '経過観察': 'Follow-up',
    'I区分': 'Category I',
    'II区分': 'Category II',
    'III区分': 'Category III',
    'IV区分': 'Category IV',
    '健全': 'Sound',
    '予防保全段階': 'Preventive Stage',
    '早期措置段階': 'Early Action Stage',
    '緊急措置段階': 'Emergency Stage',
    
    # 対策・補修
    '補修': 'Repair',
    '補強': 'Strengthening',
    '更新': 'Renewal',
    '修繕': 'Maintenance',
    '保全': 'Preservation',
    '措置': 'Measure',
    '応急措置': 'Emergency Measure',
    '恒久対策': 'Permanent Measure',
    '予防保全': 'Preventive Maintenance',
    '事後保全': 'Corrective Maintenance',
    'ひび割れ補修': 'Crack Repair',
    '断面修復': 'Section Restoration',
    '表面被覆': 'Surface Coating',
    '含浸材': 'Impregnation',
    '炭素繊維シート': 'CFRP Sheet',
    '鋼板接着': 'Steel Plate Bonding',
    '増桁': 'Girder Addition',
    '増厚': 'Thickening',
    '塗装塗替え': 'Recoating',
    '再塗装': 'Repainting',
    'ボルト締直し': 'Bolt Retightening',
    '支承取替え': 'Bearing Replacement',
    '床版取替え': 'Deck Replacement',
    '桁取替え': 'Girder Replacement',
    '落橋防止': 'Anti-Collapse',
    '耐震補強': 'Seismic Retrofitting',
    
    # 検査技術
    '非破壊検査': 'NDT',
    '超音波探傷': 'UT',
    '磁粉探傷': 'MT',
    '浸透探傷': 'PT',
    'X線検査': 'RT',
    'コア抜き': 'Core Sampling',
    'はつり調査': 'Chipping Test',
    '鉄筋探査': 'Rebar Detection',
    '塩分測定': 'Chloride Test',
    '中性化深さ': 'Carbonation Depth',
    '強度試験': 'Strength Test',
    '載荷試験': 'Load Test',
    
    # 記録・書類
    '点検調書': 'Inspection Report',
    '診断カルテ': 'Diagnosis Record',
    '損傷図': 'Damage Map',
    '写真管理': 'Photo Management',
    '変状展開図': 'Damage Layout',
    '橋梁台帳': 'Bridge Register',
    '履歴': 'History',
    '記録': 'Record',
    
    # 管理
    '維持管理': 'Maintenance Management',
    'アセットマネジメント': 'Asset Management',
    'LCC': 'LCC',
    'ライフサイクルコスト': 'Life Cycle Cost',
    '長寿命化': 'Life Extension',
    '予防保全型管理': 'Preventive Management',
    '計画的管理': 'Planned Management',
    'PDCA': 'PDCA',
    '優先順位': 'Priority',
    '予算': 'Budget',
    'コスト縮減': 'Cost Reduction',
    
    # 基準・規定
    '道路橋示方書': 'Bridge Specifications',
    '点検要領': 'Inspection Manual',
    '診断基準': 'Diagnosis Standard',
    '設計基準': 'Design Standard',
    '規準': 'Code',
    '仕様書': 'Specification',
    'マニュアル': 'Manual',
    'ガイドライン': 'Guideline',
    '告示': 'Notification',
    '通達': 'Circular',
    '指針': 'Guideline',
    '要領': 'Manual',
    
    # 品質・性能
    '耐久性': 'Durability',
    '耐荷力': 'Load Capacity',
    '耐震性': 'Seismic Resistance',
    '走行性': 'Rideability',
    '安全性': 'Safety',
    '機能': 'Function',
    '性能': 'Performance',
    '品質': 'Quality',
    '劣化抵抗性': 'Degradation Resistance',
    
    # 継手・接合部
    '継手': 'Joint',
    '添接': 'Splice',
    '連結部': 'Connection',
    '接合部': 'Joint',
    '結合部': 'Joint',
    '現場継手': 'Field Joint',
    '工場継手': 'Shop Joint',
    'ボルト継手': 'Bolted Joint',
    '溶接継手': 'Welded Joint',
    
    # 付属物
    '防護柵': 'Guard Fence',
    '遮音壁': 'Sound Barrier',
    '照明': 'Lighting',
    '標識': 'Sign',
    
    # その他専門用語
    '活荷重': 'Live Load',
    '死荷重': 'Dead Load',
    '設計荷重': 'Design Load',
    '許容応力': 'Allowable Stress',
    'かぶり': 'Cover',
    '鉄筋かぶり': 'Rebar Cover',
    '縁端距離': 'Edge Distance',
    '有効高': 'Effective Depth',
    '付着': 'Bond',
    '定着': 'Anchorage',
    '定着長': 'Anchorage Length',
    '開口部': 'Opening',
    '貫通孔': 'Through Hole',
    
    # 共通一般語彙
    '診断': 'Diagnosis',
    'セット': 'Set',
    '部': 'Part',
    '部材': 'Member',
    '構造': 'Structure',
    '変状': 'Defect',
    '損傷': 'Damage',
    '劣化': 'Deterioration',
    '状態': 'Condition',
    '原因': 'Cause',
    '対策': 'Countermeasure',
    '特有': 'Specific',
    '起因': 'Origin',
    '移動': 'Movement',
    '不足': 'Shortage',
    '不良': 'Deficiency',
    '遅れ破壊': 'Delayed Fracture',
    '充填': 'Filling',
    '浮き上がり': 'Uplift',
}

def is_bridge_keyword(word: str) -> bool:
    """単語が橋梁診断ドメインキーワードか判定"""
    return word in BRIDGE_DOMAIN_KEYWORDS

def is_stop_word(word: str) -> bool:
    """単語がストップワードか判定"""
    return word in STOP_WORDS

def filter_bridge_keywords(words: list) -> list:
    """橋梁診断ドメインキーワードのみをフィルタリング（ストップワードを除外）"""
    return [w for w in words if is_bridge_keyword(w) and not is_stop_word(w)]

def translate_bridge_term(term: str) -> str:
    """
    橋梁診断用語を英語に翻訳
    
    Args:
        term: 日本語用語
        
    Returns:
        英訳（辞書にない場合はローマ字化）
    """
    # 既に英語の場合はそのまま返す
    if term.isascii():
        return term
    
    # 辞書から翻訳
    if term in BRIDGE_TRANSLATION_DICT:
        return BRIDGE_TRANSLATION_DICT[term]
    
    # ローマ字化フォールバック
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        result = kks.convert(term)
        romaji_parts = []
        for item in result:
            if 'hepburn' in item and item['hepburn']:
                romaji_parts.append(item['hepburn'].capitalize())
        
        if romaji_parts:
            return ' '.join(romaji_parts)
    except ImportError:
        pass
    
    # 完全なフォールバック
    return f"Term({term})"

# 階層別キーワードカテゴリ（上位から下位へ）
# depth=0: ROOT（最上位）
# depth=1: 部材カテゴリ
COMPONENT_KEYWORDS = {
    '床版', '桁', '主桁', '横桁', '縦桁', '鋼桁', 'RC桁', 'PC桁',
    '支承', 'ゴム支承', 'パッド型支承', '鋼製支承', 'ピン支承', 'ローラー支承',
    '橋台', '橋脚', '下部構造', '上部構造', '基礎', 'フーチング',
    '伸縮装置', '排水装置', '高欄', '地覆', '舗装',
    'アーチ', 'トラス', 'ラーメン', '吊橋', '斜張橋',
    '溝橋', '床版橋', 'ボックスカルバート', '連結桁',
    'コンクリート', '鉄筋', '鋼材', 'PC鋼材',
}

# depth=2: 損傷カテゴリ
DAMAGE_KEYWORDS = {
    '腐食', 'ひび割れ', '剥離', '剥落', '変色', '遊離石灰',
    '疲労', '亀裂', '疲労亀裂', 'クラック', '鉄筋腐食', '塩害', 'ASR',
    '凍害', '中性化', 'アルカリシリカ反応', '土砂化',
    '錆', '錆汁', '鋼材腐食', '板厚減少', '断面欠損',
    '変形', 'たわみ', 'ずれ', '段差', '目地開き',
    '滞水', '漏水', '遊間不足', '肩当り', '逸脱',
    'ボイド管浮き上がり', '陥没', '破断', '座屈', '緩み',
    '鉄筋露出',
}

# depth=3: 原因カテゴリ
CAUSE_KEYWORDS = {
    '凍結融解', '塩分', '塩化物', '酸性雨', '排気ガス',
    '洗堀', '側方流動', '地すべり', '沈下', '地盤変動',
    '地震', '振動', '荷重', '過荷重', '衝撃', '摩耗',
    'オゾン劣化', '紫外線', '化学的侵食', '浸食',
    '凍結防止剤', '飛来塩', '初期内在塩',
}

# depth=4以降: 補修工法カテゴリ
REPAIR_KEYWORDS = {
    '補修', '補強', '更新', '修繕', '保全', '措置',
    '応急措置', '恒久対策', '予防保全', '事後保全',
    'ひび割れ補修', '断面修復', '表面被覆', '含浸材',
    '炭素繊維シート', '鋼板接着', '増桁', '増厚',
    '塗装塗替え', '再塗装', 'ボルト締直し', '支承取替え',
    '床版取替え', '桁取替え', '落橋防止', '耐震補強',
    '電気防食', 'グラウト再充填', '部分打替え', '鋼材防錆',
    '橋面防水', '伸縮装置改良', '排水装置改良',
}

def get_priority_keywords_by_depth(depth: int) -> set:
    """
    深さに応じた優先キーワードセットを返す
    
    Args:
        depth: ノードの深さ（0=ROOT, 1=部材, 2=損傷, 3=原因, 4+=補修工法）
    
    Returns:
        優先するキーワードのセット
    """
    if depth == 0:
        return set()  # ROOTは特別扱い
    elif depth == 1:
        return COMPONENT_KEYWORDS
    elif depth == 2:
        return DAMAGE_KEYWORDS
    elif depth == 3:
        return CAUSE_KEYWORDS
    else:  # depth >= 4
        return REPAIR_KEYWORDS

# エクスポート
__all__ = [
    'STOP_WORDS',
    'BRIDGE_DOMAIN_KEYWORDS',
    'BRIDGE_TRANSLATION_DICT',
    'is_stop_word',
    'is_bridge_keyword',
    'filter_bridge_keywords',
    'translate_bridge_term',
    'COMPONENT_KEYWORDS',
    'DAMAGE_KEYWORDS',
    'CAUSE_KEYWORDS',
    'REPAIR_KEYWORDS',
    'get_priority_keywords_by_depth',
]
