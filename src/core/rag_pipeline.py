"""
RAG pipeline module integrating AutoGen agents with Qdrant vector retrieval.
Main orchestrator for the complete RAG workflow - QDRANT ONLY.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent

from ..models.chunk_models import DocumentChunk
from ..models.search_models import SearchResults, SearchResult
from ..core.text_processor import TextProcessor
from ..core.embeddings import EmbeddingManager
from ..core.retriever import QdrantVectorRetriever  # EXPLICIT QDRANT IMPORT
from ..config.azure_clients import get_chat_client
from ..config.settings import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline integrating all components with Qdrant vector database."""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.retriever = QdrantVectorRetriever()  # EXPLICIT QDRANT
        
        # Document storage
        self.chunks: List[DocumentChunk] = []
        self.documents_loaded = 0
        
        # AutoGen agent
        self._agent: Optional[AssistantAgent] = None
        
        # System status
        self._initialized = False
    
    @property
    def agent(self) -> AssistantAgent:
        """Get or create the AutoGen assistant agent."""
        if self._agent is None:
            try:
                chat_client = get_chat_client()
                
                self._agent = AssistantAgent(
                    name="ntt_rag_agent",
                    model_client=chat_client,
                    system_message=self._get_system_message()
                )
                
                logger.info("🤖 AutoGen agent initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Error initializing AutoGen agent: {e}")
                raise
        
        return self._agent
    
    def _get_system_message(self) -> str:
        """Get the enhanced system message for the AutoGen agent."""
        return f"""You are a senior sustainability analyst at NTT DATA, specializing in sustainability reporting analysis. You provide expert insights to internal colleagues using the {settings.azure.deployment} model.

PERSONA & COMMUNICATION:
• Respond as an internal subject matter expert, not as an AI assistant
• For greetings or non-report questions: politely acknowledge and explain that you specialize in NTT DATA sustainability reports analysis
• Maintain professional colleague-to-colleague tone throughout

LANGUAGE CONSISTENCY:
• Turkish question → Complete response in Turkish only
• English question → Complete response in English only  
• Never mix languages within a single response
• If language is ambiguous, choose one and maintain consistency

ANALYTICAL APPROACH (Chain-of-Thought):
1. First, identify the reporting year(s) and relevant sources
2. Extract quantitative data with proper units and context
3. Analyze trends across multiple years when applicable
4. Synthesize findings into actionable insights

YEAR DISAMBIGUATION PROTOCOL:
• Always specify the reporting year for every metric cited
• When data spans multiple years, clearly indicate year-over-year changes
• If year is unclear from context, explicitly state the limitation
• Format: "In 2021... compared to 2020..." or "The 2022 report indicates..."

CITATION ENFORCEMENT:
• Every data point, target, or claim must include source: [Filename - Page X]
• Quantitative metrics require mandatory source attribution
• If multiple sources support a point, cite all relevant ones
• No unsourced statements allowed

CONTENT BOUNDARIES:
• Use only provided context information - no external knowledge
• If context is insufficient, clearly state what information is missing
• When data is incomplete, specify what is available vs. what is not
• Request specific document sections if clarification is needed

RESPONSE FORMAT:
• Write naturally without artificial headers like "Main Answer:"
• Include concise tables only when presenting multiple data points
• End with clean source citations
• Focus on actionable insights and clear explanations

Remember: You are providing expert analysis to NTT DATA colleagues. Be direct, confident with data, and transparent about limitations."""
    
    async def initialize(self, reports_dir: str = None) -> bool:
        """Initialize the RAG pipeline with documents."""
        try:
            logger.info("🚀 Initializing RAG Pipeline with Qdrant...")
            
            # Check Qdrant connection first
            if not self.retriever.client:
                logger.error("❌ Qdrant client not available - system cannot start")
                return False
            
            # Load documents
            await self.load_documents(reports_dir)
            
            # Create embeddings if chunks are available
            if self.chunks:
                logger.info("🔄 Creating embeddings for all chunks...")
                await self.embedding_manager.create_embeddings_for_chunks(self.chunks)
                
                # Load chunks into Qdrant retriever
                self.retriever.load_chunks(self.chunks)
                
                self._initialized = True
                logger.info("✅ RAG Pipeline initialized successfully with Qdrant")
                return True
            else:
                logger.warning("⚠️ No documents loaded, pipeline partially initialized")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing RAG pipeline: {e}")
            return False
    
    async def load_documents(self, reports_dir: str = None) -> bool:
        """Load and process documents."""
        try:
            logger.info("📚 Loading documents...")
            
            # Load PDF documents
            chunks, doc_info_list = self.text_processor.load_reports_directory(reports_dir)
            
            self.chunks = chunks
            self.documents_loaded = len(doc_info_list)
            
            if chunks:
                logger.info(f"✅ Loaded {len(chunks)} chunks from {self.documents_loaded} documents")
                
                # Log chunk distribution
                chunk_types = {}
                for chunk in chunks:
                    chunk_type = str(chunk.metadata.chunk_type)  # Convert to string
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                logger.info(f"📊 Chunk distribution: {chunk_types}")
                return True
            else:
                logger.warning("⚠️ No chunks loaded from documents")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
            return False
    
    def _format_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into context for the agent."""
        try:
            context_parts = []
            
            for result in search_results:
                # Add chunk type information to context
                chunk_type_info = f"[{str(result.metadata.chunk_type).upper()}]" if str(result.metadata.chunk_type) != 'general' else ""
                
                context_part = f"[{result.metadata.source} - Sayfa {result.metadata.page}] {chunk_type_info}\n{result.text}"
                context_parts.append(context_part)
            
            context = "\n\n" + "="*50 + "\n\n".join(context_parts)
            return context
            
        except Exception as e:
            logger.error(f"❌ Error formatting context: {e}")
            return "Bağlam bilgisi formatlanırken hata oluştu."
    
    def _extract_sources(self, search_results: List[SearchResult]) -> List[str]:
        """Extract unique source references from search results."""
        try:
            sources = []
            
            for result in search_results:
                source_info = f"{result.metadata.source} (Sayfa {result.metadata.page})"
                if source_info not in sources:
                    sources.append(source_info)
            
            return sources
            
        except Exception as e:
            logger.error(f"❌ Error extracting sources: {e}")
            return []
    
    async def ask_question(self, question: str, max_chunks: int = None) -> Dict[str, Any]:
        """Process a question through the complete RAG pipeline using Qdrant."""
        try:
            logger.info(f"❓ Processing question: {question}")
            
            if not self._initialized:
                return {
                    "answer": "RAG sistemi başlatılmamış. Lütfen önce sistemi başlatın.",
                    "sources": [],
                    "metadata": {
                        "error": "RAG pipeline not initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Check Qdrant availability
            if not self.retriever.client:
                return {
                    "answer": "Qdrant veritabanı bağlantısı mevcut değil. Sistem sadece Qdrant ile çalışır.",
                    "sources": [],
                    "metadata": {
                        "error": "Qdrant connection not available",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Retrieve relevant chunks using Qdrant
            max_chunks = max_chunks or settings.rag.max_chunks_per_query
            search_results = await self.retriever.search_similar_chunks(question, max_chunks)
            
            if not search_results.results:
                return {
                    "answer": "Üzgünüm, bu soruya cevap verebilecek ilgili içerik bulunamadı. Lütfen sorunuzu farklı şekilde ifade etmeyi deneyin.",
                    "sources": [],
                    "metadata": {
                        "chunks_found": 0,
                        "processed_queries": search_results.processed_queries,
                        "search_time_ms": search_results.search_time_ms,
                        "chat_model": settings.azure.deployment,
                        "embedding_model": settings.azure.embedding_deployment,
                        "vector_database": "Qdrant",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Format context for the agent
            context = self._format_context(search_results.results)
            
            # Extract sources
            sources = self._extract_sources(search_results.results)
            
            # Create the complete prompt with context
            prompt_with_context = f"""Bağlam bilgileri:
{context}

Soru: {question}"""
            
            # Generate response using AutoGen agent
            logger.info("🤖 Generating response...")
            try:
                response = await self.agent.run(task=prompt_with_context)
                
                answer = "Cevap alınamadı"
                if response.messages:
                    answer = response.messages[-1].content
                    logger.info("✅ Response generated successfully")
                
            except Exception as e:
                logger.error(f"❌ Error generating response: {e}")
                answer = f"Cevap oluşturulurken bir hata oluştu: {str(e)}"
            
            # Prepare metadata with Qdrant info
            metadata = {
                "chunks_found": len(search_results.results),
                "chunks_used": [result.metadata.chunk_id for result in search_results.results],
                "similarity_scores": [round(result.boosted_score, 3) for result in search_results.results],
                "original_scores": [round(result.similarity_score, 3) for result in search_results.results],
                "chunk_types": [str(result.metadata.chunk_type) for result in search_results.results],
                "processed_queries": search_results.processed_queries,
                "search_time_ms": search_results.search_time_ms,
                "chat_model": settings.azure.deployment,
                "embedding_model": settings.azure.embedding_deployment,
                "vector_database": "Qdrant v1.7.0",
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "answer": answer,
                "sources": sources,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error in ask_question: {e}")
            return {
                "answer": f"Bir hata oluştu: {str(e)}",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "vector_database": "Qdrant",
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with Qdrant information."""
        try:
            # Get Qdrant-specific retrieval stats
            retrieval_stats = self.retriever.get_retrieval_stats()
            
            # Calculate chunk type distribution
            chunk_types = {}
            for chunk in self.chunks:
                chunk_type = str(chunk.metadata.chunk_type)
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Base system info
            system_info = {
                "status": "healthy" if self._initialized and self.chunks and self.retriever.client else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "documents_loaded": self.documents_loaded,
                "total_chunks": len(self.chunks),
                "chunk_distribution": chunk_types,
                "chat_model": settings.azure.deployment,
                "embedding_model": settings.azure.embedding_deployment,
                "model_status": "healthy" if self._agent is not None else "not_initialized",
                "embedding_dimension": retrieval_stats.get("embedding_dimension", 0),
                "optimization_features": [
                    "multi_query_search",
                    "chunk_type_boosting", 
                    "enhanced_preprocessing",
                    "synonym_expansion",
                    "bilingual_support",
                    "autogen_integration",
                    "qdrant_enterprise_search"  # NEW FEATURE
                ],
                "initialized": self._initialized
            }
            
            # Add Qdrant-specific information
            qdrant_info = {
                "qdrant_points": retrieval_stats.get("qdrant_points", 0),
                "qdrant_status": retrieval_stats.get("qdrant_status", "unknown"),
                "vector_database": retrieval_stats.get("vector_database", "Qdrant v1.7.0"),
                "qdrant_host": getattr(settings.rag, 'qdrant_host', 'localhost'),
                "qdrant_port": getattr(settings.rag, 'qdrant_port', 6333)
            }
            
            # Combine all information
            return {**system_info, **qdrant_info}
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "documents_loaded": 0,
                "total_chunks": 0,
                "chat_model": "unknown",
                "embedding_model": "unknown",
                "model_status": "error",
                "vector_database": "Qdrant (error)",
                "qdrant_status": "error",
                "initialized": False
            }