"""
PubMed API Integration for Immune Cell Differentiation Knowledge
FOXP3, CTLA-4, IL-10関連文献の自動取得・要約機能

Author: AI Assistant
Date: 2024-10-30
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class PubMedArticle:
    """PubMed論文データクラス"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    keywords: List[str]
    mesh_terms: List[str]
    relevance_score: float = 0.0

class PubMedAPI:
    """PubMed API連携クラス"""
    
    def __init__(self, email: str = "researcher@example.com", cache_dir: str = None):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.cache_dir = cache_dir or "./pubmed_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def search_articles(self, query: str, max_results: int = 100, 
                       retmode: str = "xml") -> List[str]:
        """
        PubMedで論文を検索しPMIDリストを取得
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            retmode: 戻り値形式
            
        Returns:
            PMIDリスト
        """
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': retmode,
            'email': self.email,
            'tool': 'immune_cell_rag'
        }
        
        try:
            response = requests.get(f"{self.base_url}esearch.fcgi", params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            
            print(f"Found {len(pmids)} articles for query: {query}")
            return pmids
            
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def fetch_article_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        PMIDリストから論文詳細を取得
        
        Args:
            pmids: PMIDリスト
            
        Returns:
            PubMedArticleリスト
        """
        if not pmids:
            return []
            
        # キャッシュチェック
        cache_key = hashlib.md5(','.join(sorted(pmids)).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"articles_{cache_key}.json")
        
        if os.path.exists(cache_file):
            print(f"Loading {len(pmids)} articles from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return [PubMedArticle(**article) for article in cached_data]
        
        # APIから取得
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': self.email,
            'tool': 'immune_cell_rag'
        }
        
        try:
            response = requests.get(f"{self.base_url}efetch.fcgi", params=params)
            response.raise_for_status()
            
            articles = self._parse_pubmed_xml(response.content)
            
            # キャッシュに保存
            cache_data = [article.__dict__ for article in articles]
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"Fetched and cached {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"Error fetching article details: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: bytes) -> List[PubMedArticle]:
        """XML形式のPubMedデータをパース"""
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article_elem in root.findall('.//PubmedArticle'):
                try:
                    # PMID
                    pmid_elem = article_elem.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # タイトル
                    title_elem = article_elem.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ""
                    
                    # アブストラクト
                    abstract_elems = article_elem.findall('.//AbstractText')
                    abstract_parts = []
                    for abs_elem in abstract_elems:
                        label = abs_elem.get('Label', '')
                        text = abs_elem.text or ""
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                    abstract = " ".join(abstract_parts)
                    
                    # 著者
                    authors = []
                    for author_elem in article_elem.findall('.//Author'):
                        lastname_elem = author_elem.find('LastName')
                        forename_elem = author_elem.find('ForeName')
                        if lastname_elem is not None and forename_elem is not None:
                            authors.append(f"{forename_elem.text} {lastname_elem.text}")
                    
                    # ジャーナル
                    journal_elem = article_elem.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # 出版日
                    pub_date_elem = article_elem.find('.//PubDate/Year')
                    pub_date = pub_date_elem.text if pub_date_elem is not None else ""
                    
                    # キーワード
                    keywords = []
                    for keyword_elem in article_elem.findall('.//Keyword'):
                        if keyword_elem.text:
                            keywords.append(keyword_elem.text)
                    
                    # MeSH用語
                    mesh_terms = []
                    for mesh_elem in article_elem.findall('.//DescriptorName'):
                        if mesh_elem.text:
                            mesh_terms.append(mesh_elem.text)
                    
                    article = PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        pub_date=pub_date,
                        keywords=keywords,
                        mesh_terms=mesh_terms
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error parsing XML: {e}")
            
        return articles
    
    def calculate_relevance_score(self, article: PubMedArticle, 
                                 target_terms: List[str]) -> float:
        """
        論文の関連度スコアを計算
        
        Args:
            article: PubMedArticle
            target_terms: 対象用語リスト
            
        Returns:
            関連度スコア (0.0-1.0)
        """
        text_content = f"{article.title} {article.abstract} {' '.join(article.keywords)}"
        text_lower = text_content.lower()
        
        score = 0.0
        max_score = len(target_terms)
        
        for term in target_terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                # タイトルでの言及は高得点
                if term_lower in article.title.lower():
                    score += 0.8
                # アブストラクトでの言及
                elif term_lower in article.abstract.lower():
                    score += 0.5
                # キーワードでの言及
                elif any(term_lower in kw.lower() for kw in article.keywords):
                    score += 0.3
                else:
                    score += 0.1
        
        return min(score / max_score, 1.0)

class ImmuneCellPubMedRetriever:
    """免疫細胞特化型PubMed論文検索システム"""
    
    def __init__(self, cache_dir: str = None):
        self.pubmed_api = PubMedAPI(cache_dir=cache_dir)
        self.cache_dir = cache_dir or "./pubmed_cache"
        self.immune_queries = [
            "FOXP3 AND Treg differentiation",
            "nTreg vs iTreg function", 
            "CTLA-4 AND immune suppression",
            "TGF-beta Treg induction",
            "Treg autoimmune disease",
            "thymic Treg development",
            "peripheral Treg conversion",
            "FOXP3 transcriptional regulation",
            "Treg suppression mechanisms",
            "Treg metabolic regulation"
        ]
        
        self.target_terms = [
            "FOXP3", "CTLA-4", "CD25", "IL-10", "TGF-β", "Treg", "regulatory T cell",
            "immune suppression", "autoimmune", "tolerance", "thymus", "peripheral",
            "differentiation", "lineage", "HSC", "CLP", "CD4"
        ]
    
    def _get_cached_articles(self, query: str) -> List[PubMedArticle]:
        """キャッシュから論文を取得"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"query_{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    return [PubMedArticle(**article) for article in cached_data]
            except Exception as e:
                print(f"Error loading cache for '{query}': {e}")
        
        return []
    
    def retrieve_immune_literature(self, max_articles_per_query: int = 50) -> Dict[str, List[PubMedArticle]]:
        """
        免疫関連文献を検索・取得
        
        Args:
            max_articles_per_query: クエリあたりの最大論文数
            
        Returns:
            クエリ別論文辞書
        """
        results = {}
        
        for query in self.immune_queries:
            print(f"\nSearching for: {query}")
            
            # 検索実行
            pmids = self.pubmed_api.search_articles(query, max_articles_per_query)
            
            if pmids:
                # 論文詳細取得
                articles = self.pubmed_api.fetch_article_details(pmids)
                
                # 関連度スコア計算
                for article in articles:
                    article.relevance_score = self.pubmed_api.calculate_relevance_score(
                        article, self.target_terms
                    )
                
                # 関連度でソート
                articles.sort(key=lambda x: x.relevance_score, reverse=True)
                
                results[query] = articles
                print(f"Retrieved {len(articles)} articles, top relevance: {articles[0].relevance_score:.3f}")
            
            # API制限回避
            time.sleep(0.5)
        
        return results
    
    def generate_summary_report(self, literature_results: Dict[str, List[PubMedArticle]], 
                              output_file: str = None) -> str:
        """
        文献検索結果のサマリーレポート生成
        
        Args:
            literature_results: 文献検索結果
            output_file: 出力ファイルパス
            
        Returns:
            サマリーレポート文字列
        """
        report_lines = []
        report_lines.append("# Immune Cell Differentiation Literature Summary")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_articles = sum(len(articles) for articles in literature_results.values())
        report_lines.append(f"Total articles retrieved: {total_articles}")
        report_lines.append("")
        
        for query, articles in literature_results.items():
            report_lines.append(f"## Query: {query}")
            report_lines.append(f"Articles found: {len(articles)}")
            
            if articles:
                # トップ3論文を表示
                report_lines.append("\n### Top Relevant Articles:")
                for i, article in enumerate(articles[:3], 1):
                    report_lines.append(f"\n{i}. **{article.title}**")
                    report_lines.append(f"   - PMID: {article.pmid}")
                    report_lines.append(f"   - Journal: {article.journal} ({article.pub_date})")
                    report_lines.append(f"   - Relevance Score: {article.relevance_score:.3f}")
                    report_lines.append(f"   - Authors: {', '.join(article.authors[:3])}{'...' if len(article.authors) > 3 else ''}")
                    
                    # アブストラクト要約
                    abstract_preview = article.abstract[:200] + "..." if len(article.abstract) > 200 else article.abstract
                    report_lines.append(f"   - Abstract: {abstract_preview}")
            
            report_lines.append("")
        
        # 高関連度論文の統計
        all_articles = []
        for articles in literature_results.values():
            all_articles.extend(articles)
        
        high_relevance = [a for a in all_articles if a.relevance_score > 0.5]
        report_lines.append(f"## High Relevance Articles (score > 0.5): {len(high_relevance)}")
        
        # 頻出キーワード分析
        all_keywords = []
        for article in all_articles:
            all_keywords.extend(article.keywords)
        
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report_lines.append("\n## Top Keywords:")
        for kw, count in top_keywords:
            report_lines.append(f"- {kw}: {count}")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Summary report saved to: {output_file}")
        
        return report_content

def main():
    """メイン実行関数"""
    print("🧬 Immune Cell Differentiation PubMed Retriever")
    print("=" * 50)
    
    # キャッシュディレクトリ設定
    cache_dir = "C:/Users/yasun/LangChain/learning-langchain/treg-raptor-tree/data/immune_cell_differentiation/pubmed_cache"
    
    # 検索システム初期化
    retriever = ImmuneCellPubMedRetriever(cache_dir=cache_dir)
    
    # 文献検索実行
    print("Starting literature retrieval...")
    literature_results = retriever.retrieve_immune_literature(max_articles_per_query=30)
    
    # サマリーレポート生成
    output_file = os.path.join(cache_dir, f"literature_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    summary = retriever.generate_summary_report(literature_results, output_file)
    
    print("\n" + "=" * 50)
    print("Literature retrieval completed!")
    print(f"Summary report: {output_file}")
    
    # 簡易統計表示
    total_articles = sum(len(articles) for articles in literature_results.values())
    high_relevance_count = sum(
        len([a for a in articles if a.relevance_score > 0.5]) 
        for articles in literature_results.values()
    )
    
    print(f"Total articles: {total_articles}")
    print(f"High relevance articles (>0.5): {high_relevance_count}")

if __name__ == "__main__":
    main()