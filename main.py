"""
Main entry point for the NTT DATA RAG System.
Provides both API server and standalone command-line interface.
"""

import asyncio
import logging
import sys
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')  

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn
from src.api.app import create_app
from src.core.rag_pipeline import RAGPipeline
from src.config.settings import settings
from src.utils.logger import setup_logging
from src.utils.port_manager import get_available_port, get_process_info_on_port

logger = logging.getLogger(__name__)


async def run_standalone():
    """Run the RAG system in standalone mode for testing and development."""
    print("🚀 NTT DATA RAG System - Standalone Mode")
    print("="*50)
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Initialize the system
        print("🔄 Initializing RAG system...")
        success = await rag.initialize(settings.directories.reports_dir)
        
        if not success:
            print("❌ Failed to initialize RAG system")
            print("Please ensure PDF files are in the reports directory")
            return
        
        # Get system status
        status = rag.get_system_status()
        
        print(f"✅ RAG system initialized successfully!")
        print(f"📚 Documents loaded: {status['documents_loaded']}")
        print(f"📄 Total chunks: {status['total_chunks']}")
        print(f"🧠 Embedding dimension: {status['embedding_dimension']}")
        print(f"📊 Chunk distribution: {status['chunk_distribution']}")
        print()
        
        # Test questions
        test_questions = [
            "NTT DATA'nın sürdürülebilirlik hedefleri nelerdir?",
            "2020 yılında hangi ESG konularına odaklanılmış?",
            "Karbon ayak izi azaltma stratejileri nelerdir?",
            "Çalışanlar için hangi sürdürülebilirlik programları var?",
            "NTT DATA'nın çevresel etkilerini azaltmak için aldığı önlemler nelerdir?"
        ]
        
        print("🎯 Running test questions...\n")
        
        for i, question in enumerate(test_questions, 1):
            print(f"{'='*80}")
            print(f"❓ TEST {i}/5: {question}")
            print(f"{'='*80}")
            
            result = await rag.ask_question(question)
            
            print(f"💡 ANSWER:\n{result['answer']}")
            print(f"\n📚 SOURCES: {', '.join(result['sources'])}")
            
            metadata = result['metadata']
            print(f"\n📊 METADATA:")
            print(f"   🔢 Chunks found: {metadata['chunks_found']}")
            print(f"   📈 Similarity scores: {metadata['similarity_scores']}")
            print(f"   🏷️  Chunk types: {metadata.get('chunk_types', 'N/A')}")
            print()
        
        # Interactive mode
        print(f"{'='*80}")
        print("🎯 Interactive mode started!")
        print("Type your questions or 'quit' to exit")
        print(f"{'='*80}")
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'çıkış', 'q']:
                    break
                
                if not question:
                    continue
                
                result = await rag.ask_question(question)
                print(f"\n💡 ANSWER:\n{result['answer']}")
                print(f"\n📚 SOURCES: {', '.join(result['sources'])}")
                
                metadata = result['metadata']
                print(f"\n📊 STATS:")
                print(f"   📈 Scores: {metadata['similarity_scores']}")
                print(f"   ⏱️  Time: {metadata.get('search_time_ms', 0):.1f}ms")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Standalone mode ended!")
        
    except Exception as e:
        logger.error(f"❌ Error in standalone mode: {e}")
        print(f"❌ Error: {e}")


def run_api_server():
    """Run the FastAPI server with automatic port management."""
    print("🌐 Starting NTT DATA RAG API Server...")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🏠 Host: {settings.api.host}")
    print(f"� Reports directory: {settings.directories.reports_dir}")
    
    # Check and handle port conflicts
    original_port = settings.api.port
    print(f"� Checking port {original_port}...")
    
    # Get process info if port is in use
    processes = get_process_info_on_port(original_port)
    if processes:
        print(f"⚠️  Port {original_port} is in use by:")
        for proc in processes:
            print(f"   - PID {proc['pid']}: {proc['name']}")
        
        print(f"🔍 Finding alternative port...")
    
    # Get available port (this will handle the conflict)
    available_port = get_available_port(original_port, auto_kill=False)
    
    if available_port != original_port:
        print(f"✅ Using port {available_port} instead of {original_port}")
    else:
        print(f"✅ Port {original_port} is available")
    
    # Update settings with available port
    settings.api.port = available_port
    
    # Create FastAPI app
    app = create_app()
    
    print(f"🚀 Server starting on http://{settings.api.host}:{available_port}")
    print(f"📖 API Documentation: http://{settings.api.host}:{available_port}/docs")
    print(f"🔍 Interactive API: http://{settings.api.host}:{available_port}/redoc")
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=settings.api.host,
        port=available_port,
        log_level=settings.logging.log_level.lower(),
        access_log=True,
        server_header=False,
        date_header=False
    )


def run_health_check():
    """Run a health check against the API server."""
    import requests
    import json
    
    try:
        print("🏥 Running health check...")
        
        base_url = f"http://{settings.api.host}:{settings.api.port}"
        
        # Check root endpoint
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint: OK")
        else:
            print(f"❌ Root endpoint: {response.status_code}")
        
        # Check health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint: OK")
            print(f"   📊 Status: {health_data.get('status')}")
            print(f"   📚 Documents: {health_data.get('documents_loaded')}")
            print(f"   📄 Chunks: {health_data.get('total_chunks')}")
        else:
            print(f"❌ Health endpoint: {response.status_code}")
        
        # Test question endpoint
        test_question = {"question": "NTT DATA'nın sürdürülebilirlik hedefleri nelerdir?"}
        response = requests.post(f"{base_url}/ask", json=test_question, timeout=30)
        
        if response.status_code == 200:
            print("✅ Question endpoint: OK")
            result = response.json()
            print(f"   📝 Answer length: {len(result.get('answer', ''))}")
            print(f"   📚 Sources: {len(result.get('sources', []))}")
        else:
            print(f"❌ Question endpoint: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Is it running?")
    except Exception as e:
        print(f"❌ Health check failed: {e}")


def main():
    """Main entry point with command-line argument handling."""
    # Setup logging first
    setup_logging()
    
    if len(sys.argv) < 2:
        print("🤖 NTT DATA RAG System")
        print("Usage:")
        print("  python main.py api          - Start API server")
        print("  python main.py standalone   - Run in standalone mode")
        print("  python main.py health      - Run health check")
        print("  python main.py test        - Run test questions")
        return
    
    command = sys.argv[1].lower()
    
    if command == "api":
        run_api_server()
    elif command in ["standalone", "test"]:
        asyncio.run(run_standalone())
    elif command == "health":
        run_health_check()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: api, standalone, health, test")


if __name__ == "__main__":
    main()