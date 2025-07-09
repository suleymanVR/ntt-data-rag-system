"""
Text processing module for PDF loading, cleaning, and chunking.
Handles document preprocessing and chunk analysis.
"""

import re
import unicodedata
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from pypdf import PdfReader

from ..models.chunk_models import DocumentChunk, ChunkMetadata, ChunkType, ChunkAnalysis, DocumentInfo
from ..config.settings import settings

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text processing, cleaning, and chunking operations."""
    
    def __init__(self):
        self.sustainability_keywords = [
            'sustainability', 'sürdürülebilirlik', 'ESG', 'environment', 'çevre',
            'carbon', 'karbon', 'emission', 'emisyon', 'energy', 'enerji',
            'target', 'hedef', 'goal', 'objective', 'strategy', 'strateji',
            'renewable', 'yenilenebilir', 'climate', 'iklim', 'green', 'yeşil'
        ]
        
        # Pattern for numerical data detection
        self.numerical_patterns = [
            r'\d+\.?\d*\s*%',  # Percentages
            r'\d+\.?\d*\s*(ton|kg|MW|GW|milyon|milyar|million|billion)',  # Units
            r'\d{4}\s*(yıl|year)',  # Years
            r'\d+\.?\d*\s*(azalt|reduce|artır|increase)',  # Reduction/increase targets
        ]
    
    def clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize text content."""
        try:
            # Unicode normalization
            text = unicodedata.normalize('NFKC', text)
            
            # Remove extra whitespace and special characters
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)%]', ' ', text)
            
            # Fix common OCR errors in Turkish
            ocr_fixes = {
                'ﬁ': 'fi',
                'ﬂ': 'fl',
                '—': '-',
                '…': '...',
                '"': '"',
                '"': '"',
                ''': "'",
                ''': "'"
            }
            
            for wrong, correct in ocr_fixes.items():
                text = text.replace(wrong, correct)
            
            # Remove extra spaces around punctuation
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
            
            # Remove multiple spaces
            text = re.sub(r'\s{2,}', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Error cleaning text: {e}")
            return text
    
    def analyze_chunk(self, text: str) -> ChunkAnalysis:
        """Analyze a text chunk to determine its type and characteristics."""
        try:
            text_lower = text.lower()
            
            # Check for numerical data
            has_numbers = any(re.search(pattern, text) for pattern in self.numerical_patterns)
            found_patterns = [pattern for pattern in self.numerical_patterns if re.search(pattern, text)]
            
            # Check for sustainability keywords
            found_keywords = [kw for kw in self.sustainability_keywords if kw.lower() in text_lower]
            has_keywords = len(found_keywords) > 0
            
            # Determine chunk type
            chunk_type = self._determine_chunk_type(text, has_numbers, has_keywords)
            
            # Count words
            word_count = len(text.split())
            
            return ChunkAnalysis(
                text=text,
                chunk_type=chunk_type,
                has_numbers=has_numbers,
                has_keywords=has_keywords,
                word_count=word_count,
                sustainability_keywords=found_keywords,
                numerical_patterns=found_patterns
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing chunk: {e}")
            return ChunkAnalysis(
                text=text,
                chunk_type=ChunkType.GENERAL,
                has_numbers=False,
                has_keywords=False,
                word_count=len(text.split()),
                sustainability_keywords=[],
                numerical_patterns=[]
            )
    
    def _determine_chunk_type(self, text: str, has_numbers: bool, has_keywords: bool) -> ChunkType:
        """Determine the type of a text chunk."""
        text_lower = text.lower()
        
        # Check for metrics (numbers + keywords)
        if has_numbers and has_keywords:
            return ChunkType.METRICS
        
        # Check for sustainability content
        if has_keywords:
            return ChunkType.SUSTAINABILITY
        
        # Check for visual references
        if any(visual_term in text_lower for visual_term in ['tablo', 'şekil', 'grafik', 'chart', 'figure', 'table']):
            return ChunkType.VISUAL
        
        # Check for titles (short text)
        if len(text.split()) < 20:
            return ChunkType.TITLE
        
        return ChunkType.GENERAL
    
    def chunk_text(self, text: str, page_num: int, 
                   chunk_size: int = None, overlap: int = None) -> List[ChunkAnalysis]:
        """Split text into optimized chunks."""
        try:
            chunk_size = chunk_size or settings.rag.chunk_size
            overlap = overlap or settings.rag.chunk_overlap
            
            chunks = []
            
            # Split by paragraphs first
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If adding this paragraph exceeds chunk size
                if len(current_chunk + " " + paragraph) > chunk_size and current_chunk:
                    # Process current chunk
                    chunk_analysis = self.analyze_chunk(current_chunk)
                    chunks.append(chunk_analysis)
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk += " " + paragraph if current_chunk else paragraph
            
            # Add remaining chunk
            if current_chunk.strip():
                chunk_analysis = self.analyze_chunk(current_chunk)
                chunks.append(chunk_analysis)
            
            logger.debug(f"📄 Page {page_num}: Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error chunking text on page {page_num}: {e}")
            return []
    
    def load_pdf_document(self, pdf_path: str) -> Tuple[List[DocumentChunk], DocumentInfo]:
        """Load and process a single PDF document."""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"📖 Loading PDF: {pdf_path.name}")
            
            chunks = []
            chunk_distribution = {chunk_type: 0 for chunk_type in ChunkType}
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if not page_text.strip():
                            continue
                        
                        # Clean and normalize text
                        cleaned_text = self.clean_and_normalize_text(page_text)
                        
                        # Create chunks
                        page_chunks = self.chunk_text(cleaned_text, page_num + 1)
                        
                        # Convert to DocumentChunk objects
                        for chunk_idx, chunk_analysis in enumerate(page_chunks):
                            chunk_id = f"{pdf_path.stem}_p{page_num + 1}_c{chunk_idx}"
                            
                            metadata = ChunkMetadata(
                                source=pdf_path.name,
                                page=page_num + 1,
                                chunk_id=chunk_id,
                                chunk_index=chunk_idx,
                                total_chunks=len(page_chunks),
                                chunk_type=chunk_analysis.chunk_type,
                                has_numbers=chunk_analysis.has_numbers,
                                has_keywords=chunk_analysis.has_keywords
                            )
                            
                            chunk = DocumentChunk(
                                text=chunk_analysis.text,
                                metadata=metadata
                            )
                            
                            chunks.append(chunk)
                            chunk_distribution[chunk_analysis.chunk_type] += 1
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing page {page_num + 1}: {e}")
                        continue
            
            # Create document info
            doc_info = DocumentInfo(
                filename=pdf_path.name,
                total_pages=total_pages,
                total_chunks=len(chunks),
                chunk_distribution=chunk_distribution,
                file_size=pdf_path.stat().st_size
            )
            
            logger.info(f"✅ Successfully loaded {pdf_path.name}: {len(chunks)} chunks from {total_pages} pages")
            logger.info(f"📊 Chunk distribution: {dict(chunk_distribution)}")
            
            return chunks, doc_info
            
        except Exception as e:
            logger.error(f"❌ Error loading PDF {pdf_path}: {e}")
            raise
    
    def load_reports_directory(self, reports_dir: str = None) -> Tuple[List[DocumentChunk], List[DocumentInfo]]:
        """Load all PDF reports from a directory."""
        try:
            reports_dir = reports_dir or settings.directories.reports_dir
            reports_path = Path(reports_dir)
            
            if not reports_path.exists():
                logger.warning(f"⚠️ Reports directory {reports_dir} does not exist")
                return [], []
            
            pdf_files = list(reports_path.glob("*.pdf"))
            logger.info(f"📁 Found {len(pdf_files)} PDF files in {reports_dir}")
            
            all_chunks = []
            all_doc_info = []
            
            for pdf_file in pdf_files:
                try:
                    chunks, doc_info = self.load_pdf_document(str(pdf_file))
                    all_chunks.extend(chunks)
                    all_doc_info.append(doc_info)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {pdf_file.name}: {e}")
                    continue
            
            total_chunks = len(all_chunks)
            total_docs = len(all_doc_info)
            
            logger.info(f"🎉 Successfully loaded {total_docs} documents with {total_chunks} total chunks")
            
            return all_chunks, all_doc_info
            
        except Exception as e:
            logger.error(f"❌ Error loading reports directory: {e}")
            raise