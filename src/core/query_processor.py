"""
Query processing module for expanding and optimizing search queries.
Handles multi-query generation, synonym expansion, and bilingual support.
"""

import re
import logging
from typing import List, Dict
import unicodedata

from ..models.search_models import QueryExpansion
from ..config.settings import settings

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles query preprocessing, expansion, and optimization."""
    
    def __init__(self):
        # Synonym mappings for query expansion
        self.query_synonyms = {
            'sürdürülebilirlik': ['sustainability', 'çevre', 'environment', 'yeşil', 'green'],
            'ESG': ['environmental social governance', 'çevresel sosyal yönetişim'],
            'karbon': ['carbon', 'CO2', 'emisyon', 'emission'],
            'hedef': ['target', 'goal', 'amaç', 'objective'],
            'çalışan': ['employee', 'personel', 'staff', 'workforce'],
            'enerji': ['energy', 'güç', 'power'],
            'teknoloji': ['technology', 'dijital', 'digital'],
            'rapor': ['report', 'raporlama', 'reporting'],
            'strateji': ['strategy', 'plan', 'yaklaşım', 'approach'],
            'yönetim': ['management', 'governance', 'yönetişim'],
            'iklim': ['climate', 'weather', 'hava'],
            'yenilenebilir': ['renewable', 'clean', 'temiz'],
            'atık': ['waste', 'çöp', 'garbage'],
            'su': ['water', 'aqua'],
            'toplum': ['community', 'society', 'sosyal'],
            'etki': ['impact', 'effect', 'sonuç'],
            'risk': ['risk', 'tehlike', 'danger'],
            'fırsat': ['opportunity', 'chance', 'şans']
        }
        
        # Turkish to English translations for sustainability terms
        self.turkish_to_english = {
            'sürdürülebilirlik': 'sustainability',
            'çevre': 'environment',
            'karbon': 'carbon',
            'enerji': 'energy',
            'hedef': 'target',
            'strateji': 'strategy',
            'çalışan': 'employee',
            'teknoloji': 'technology',
            'yönetim': 'management',
            'iklim': 'climate',
            'yenilenebilir': 'renewable',
            'atık': 'waste',
            'su': 'water',
            'toplum': 'community',
            'etki': 'impact',
            'risk': 'risk',
            'fırsat': 'opportunity',
            'raporlama': 'reporting',
            'performans': 'performance',
            'verimlilik': 'efficiency',
            'inovasyon': 'innovation',
            'gelişim': 'development',
            'büyüme': 'growth',
            'azaltma': 'reduction',
            'artırma': 'increase'
        }
        
        # Sustainability keywords for detection
        self.sustainability_keywords = list(self.query_synonyms.keys()) + [
            'ESG', 'CDP', 'GRI', 'TCFD', 'Paris Agreement', 'SDG',
            'scope 1', 'scope 2', 'scope 3', 'net zero', 'carbon neutral',
            'circular economy', 'biodiversity', 'stakeholder'
        ]
    
    def clean_and_normalize_query(self, query: str) -> str:
        """Clean and normalize query text."""
        try:
            # Unicode normalization
            query = unicodedata.normalize('NFKC', query)
            
            # Remove extra whitespace
            query = re.sub(r'\s+', ' ', query)
            
            # Remove special characters but keep Turkish characters
            query = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', query)
            
            # Remove multiple spaces
            query = re.sub(r'\s{2,}', ' ', query)
            
            return query.strip()
            
        except Exception as e:
            logger.error(f"❌ Error cleaning query: {e}")
            return query
    
    def detect_sustainability_keywords(self, query: str) -> List[str]:
        """Detect sustainability-related keywords in the query."""
        try:
            query_lower = query.lower()
            detected = []
            
            for keyword in self.sustainability_keywords:
                if keyword.lower() in query_lower:
                    detected.append(keyword)
            
            return detected
            
        except Exception as e:
            logger.error(f"❌ Error detecting keywords: {e}")
            return []
    
    def expand_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms."""
        try:
            if not settings.rag.enable_synonym_expansion:
                return query
            
            expanded_query = query.lower()
            added_synonyms = set()
            
            for term, synonyms in self.query_synonyms.items():
                if term in expanded_query:
                    for synonym in synonyms:
                        if synonym not in added_synonyms and synonym not in expanded_query:
                            expanded_query += f" {synonym}"
                            added_synonyms.add(synonym)
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"❌ Error expanding with synonyms: {e}")
            return query
    
    def create_english_version(self, query: str) -> str:
        """Create English version of Turkish query."""
        try:
            if not settings.rag.enable_bilingual_support:
                return query
            
            english_version = query.lower()
            
            # Replace Turkish terms with English equivalents
            for turkish, english in self.turkish_to_english.items():
                pattern = r'\b' + re.escape(turkish) + r'\b'
                english_version = re.sub(pattern, english, english_version, flags=re.IGNORECASE)
            
            # Only return if translation was made
            if english_version != query.lower():
                return english_version
            
            return ""
            
        except Exception as e:
            logger.error(f"❌ Error creating English version: {e}")
            return ""
    
    def process_query(self, query: str) -> QueryExpansion:
        """Process query and create all variations."""
        try:
            logger.debug(f"🔄 Processing query: '{query}'")
            
            # Clean and normalize
            cleaned_query = self.clean_and_normalize_query(query)
            
            # Detect keywords
            detected_keywords = self.detect_sustainability_keywords(cleaned_query)
            
            # Expand with synonyms
            expanded_query = self.expand_with_synonyms(cleaned_query)
            
            # Create English version
            english_query = self.create_english_version(cleaned_query)
            
            result = QueryExpansion(
                original_query=query,
                cleaned_query=cleaned_query,
                expanded_query=expanded_query,
                english_query=english_query if english_query else None,
                detected_keywords=detected_keywords
            )
            
            logger.debug(f"📝 Query expansion completed: {len(detected_keywords)} keywords detected")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            # Return minimal expansion on error
            return QueryExpansion(
                original_query=query,
                cleaned_query=query,
                expanded_query=query,
                english_query=None,
                detected_keywords=[]
            )
    
    def generate_multi_queries(self, query: str) -> List[str]:
        """Generate multiple query variations for multi-query search."""
        try:
            if not settings.rag.enable_multi_query:
                return [query]
            
            # Process the query
            expansion = self.process_query(query)
            
            queries = []
            
            # Add original cleaned query
            queries.append(expansion.cleaned_query)
            
            # Add expanded query if different
            if expansion.expanded_query != expansion.cleaned_query:
                queries.append(expansion.expanded_query)
            
            # Add English version if available
            if expansion.english_query:
                queries.append(expansion.english_query)
            
            # Create keyword-focused queries if sustainability keywords detected
            if expansion.detected_keywords:
                keyword_query = " ".join(expansion.detected_keywords)
                if keyword_query not in queries:
                    queries.append(keyword_query)
            
            # Remove duplicates while preserving order
            unique_queries = []
            for q in queries:
                if q and q not in unique_queries:
                    unique_queries.append(q)
            
            logger.debug(f"🔍 Generated {len(unique_queries)} query variations")
            
            return unique_queries
            
        except Exception as e:
            logger.error(f"❌ Error generating multi-queries: {e}")
            return [query]
    
    def optimize_query_for_search(self, query: str) -> str:
        """Optimize query for vector similarity search."""
        try:
            # Clean and normalize
            optimized = self.clean_and_normalize_query(query)
            
            # Remove stop words that might interfere with semantic search
            turkish_stop_words = ['ve', 'ile', 'bir', 'bu', 'şu', 'o', 'için', 'gibi', 'kadar', 'daha']
            english_stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
            
            all_stop_words = turkish_stop_words + english_stop_words
            
            words = optimized.split()
            filtered_words = [word for word in words if word.lower() not in all_stop_words or len(word) > 3]
            
            # Rejoin words
            optimized = " ".join(filtered_words)
            
            logger.debug(f"🎯 Query optimized: '{query}' -> '{optimized}'")
            
            return optimized
            
        except Exception as e:
            logger.error(f"❌ Error optimizing query: {e}")
            return query