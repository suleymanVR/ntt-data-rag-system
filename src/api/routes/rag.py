"""
RAG endpoints for question answering functionality.
Handles question processing and response generation.
"""

import logging
import uuid
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from ...models.api_models import QuestionRequest, AnswerResponse, ErrorResponse
from ...exceptions.api_exceptions import RAGNotInitializedError, QuestionProcessingError
from ...utils.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)

router = APIRouter()
health_monitor = HealthMonitor()


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Ask a question to the RAG system.
    
    This endpoint:
    1. Validates the input question
    2. Retrieves relevant document chunks
    3. Generates an answer using AutoGen + Azure OpenAI
    4. Returns the answer with sources and metadata
    
    Args:
        request: Question request with text and optional parameters
        background_tasks: Background tasks for logging/monitoring
        http_request: HTTP request object for accessing app state
    
    Returns:
        AnswerResponse with answer, sources, and detailed metadata
    
    Raises:
        HTTPException: If RAG system is not initialized or processing fails
    """
    
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    
    try:
        logger.info(f"[{request_id}] ❓ Processing question: {request.question}")
        
        # Get RAG pipeline from app state
        rag_pipeline = getattr(http_request.app.state, 'rag_pipeline', None)
        
        if not rag_pipeline:
            raise RAGNotInitializedError("RAG pipeline not initialized")
        
        # Validate question length and content
        if len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Question too short. Please provide a more detailed question."
            )
        
        # Process the question through RAG pipeline
        try:
            result = await rag_pipeline.ask_question(
                question=request.question,
                max_chunks=request.max_chunks
            )
            
            # Add request ID to metadata
            if request.include_metadata:
                result["metadata"]["request_id"] = request_id
                result["metadata"]["question_length"] = len(request.question)
                result["metadata"]["max_chunks_requested"] = request.max_chunks
            else:
                # Simplified metadata
                result["metadata"] = {
                    "request_id": request_id,
                    "chunks_found": result["metadata"].get("chunks_found", 0),
                    "timestamp": result["metadata"].get("timestamp")
                }
            
            # Create response
            response = AnswerResponse(
                answer=result["answer"],
                sources=result["sources"],
                metadata=result["metadata"]
            )
            
            # Schedule background task for metrics collection
            background_tasks.add_task(
                _log_question_metrics,
                request_id=request_id,
                question=request.question,
                chunks_found=result["metadata"].get("chunks_found", 0),
                processing_time=result["metadata"].get("search_time_ms", 0)
            )
            
            logger.info(f"[{request_id}] ✅ Question processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error processing question: {e}")
            raise QuestionProcessingError(f"Failed to process question: {str(e)}")
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except RAGNotInitializedError as e:
        logger.error(f"[{request_id}] ❌ RAG system not initialized: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except QuestionProcessingError as e:
        logger.error(f"[{request_id}] ❌ Question processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred while processing your question. Request ID: {request_id}"
        )


@router.post("/ask/batch")
async def ask_questions_batch(
    questions: list[QuestionRequest],
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Process multiple questions in batch.
    
    Useful for bulk processing or testing multiple queries.
    Maximum 10 questions per batch to prevent abuse.
    
    Args:
        questions: List of question requests
        background_tasks: Background tasks for monitoring
        http_request: HTTP request object
    
    Returns:
        List of answers with metadata
    """
    
    batch_id = str(uuid.uuid4())[:8]
    
    try:
        logger.info(f"[{batch_id}] 📦 Processing batch of {len(questions)} questions")
        
        # Validate batch size
        if len(questions) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 questions allowed per batch"
            )
        
        if len(questions) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one question required"
            )
        
        # Get RAG pipeline
        rag_pipeline = getattr(http_request.app.state, 'rag_pipeline', None)
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Process each question
        results = []
        for i, question_request in enumerate(questions):
            try:
                request_id = f"{batch_id}-{i+1}"
                logger.info(f"[{request_id}] Processing question {i+1}/{len(questions)}")
                
                result = await rag_pipeline.ask_question(
                    question=question_request.question,
                    max_chunks=question_request.max_chunks
                )
                
                # Add batch information to metadata
                result["metadata"]["batch_id"] = batch_id
                result["metadata"]["batch_index"] = i + 1
                result["metadata"]["batch_total"] = len(questions)
                
                results.append({
                    "question": question_request.question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "metadata": result["metadata"],
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"[{batch_id}-{i+1}] ❌ Error processing question: {e}")
                results.append({
                    "question": question_request.question,
                    "answer": f"Error processing question: {str(e)}",
                    "sources": [],
                    "metadata": {"error": str(e), "batch_id": batch_id, "batch_index": i + 1},
                    "success": False
                })
        
        # Schedule background task for batch metrics
        background_tasks.add_task(
            _log_batch_metrics,
            batch_id=batch_id,
            total_questions=len(questions),
            successful_questions=sum(1 for r in results if r["success"])
        )
        
        logger.info(f"[{batch_id}] ✅ Batch processing completed")
        
        return {
            "batch_id": batch_id,
            "total_questions": len(questions),
            "successful_questions": sum(1 for r in results if r["success"]),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{batch_id}] ❌ Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.get("/ask/examples")
async def get_example_questions():
    """
    Get example questions that work well with the RAG system.
    
    Useful for testing and understanding the system capabilities.
    """
    
    examples = {
        "sustainability_general": [
            "NTT DATA'nın sürdürülebilirlik hedefleri nelerdir?",
            "Şirketin çevresel etkilerini azaltmak için aldığı önlemler nelerdir?",
            "ESG konularında hangi stratejiler benimsenmiştir?"
        ],
        "metrics_and_targets": [
            "2020 yılında karbon emisyonu azaltma hedefleri nelerdi?",
            "Enerji verimliliği konusunda hangi sayısal hedefler belirlenmiş?",
            "Sürdürülebilirlik performans metrikleri nasıl ölçülüyor?"
        ],
        "employees_and_social": [
            "Çalışanlar için hangi sürdürülebilirlik programları var?",
            "Sosyal sorumluluk projeleri nelerdir?",
            "Toplumsal etki yaratmak için hangi girişimler yapıldı?"
        ],
        "technology_and_innovation": [
            "Dijital teknolojiler sürdürülebilirlik için nasıl kullanılıyor?",
            "Temiz teknoloji yatırımları nelerdir?",
            "İnovasyon ve sürdürülebilirlik arasındaki bağlantı nedir?"
        ],
        "comparative": [
            "2020 ve 2021 yılları arasında sürdürülebilirlik performansı nasıl değişti?",
            "Farklı yıllardaki karbon emisyon hedefleri karşılaştırıldığında ne görülüyor?",
            "Zaman içinde ESG stratejilerinde hangi gelişmeler oldu?"
        ]
    }
    
    return {
        "message": "Example questions for testing the NTT DATA RAG system",
        "categories": examples,
        "tips": [
            "Specific questions about metrics and numbers work well",
            "Ask about sustainability topics for best results", 
            "Compare different years for trend analysis",
            "Use Turkish or English - both are supported"
        ]
    }


async def _log_question_metrics(request_id: str, question: str, chunks_found: int, processing_time: float):
    """Background task to log question processing metrics."""
    try:
        health_monitor.record_question_metric(
            request_id=request_id,
            question_length=len(question),
            chunks_found=chunks_found,
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"❌ Error logging question metrics: {e}")


async def _log_batch_metrics(batch_id: str, total_questions: int, successful_questions: int):
    """Background task to log batch processing metrics."""
    try:
        health_monitor.record_batch_metric(
            batch_id=batch_id,
            total_questions=total_questions,
            successful_questions=successful_questions
        )
    except Exception as e:
        logger.error(f"❌ Error logging batch metrics: {e}")