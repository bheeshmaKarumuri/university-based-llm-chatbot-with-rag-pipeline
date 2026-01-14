from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
from agent import (
    initialize_components, 
    process_query,
    AmritaQdrantClient,
    SentenceTransformer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Amrita University RAG System",
    description="AI-powered search and question answering for Amrita University",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for initialized components
qdrant_client = None
embedding_model = None
enhancer_agent = None
response_agent = None

# Pydantic models for request/response
class ChatQuery(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    answer_type: str
    key_points: List[str]
    suggested_questions: List[str]
    query: str
    enhanced_query: str
    intent: str
    results_count: int
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    components: dict

@app.on_event("startup")
async def startup_event():
    """Initialize components when the app starts."""
    global qdrant_client, embedding_model, enhancer_agent, response_agent
    
    logger.info("Starting up Amrita University RAG System...")
    
    try:
        qdrant_client, embedding_model, enhancer_agent, response_agent = initialize_components()
        logger.info("All components initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the app shuts down."""
    logger.info("Shutting down Amrita University RAG System...")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Amrita University RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify system status."""
    try:
        # Check if components are initialized
        components_status = {
            "qdrant_client": qdrant_client is not None,
            "embedding_model": embedding_model is not None,
            "enhancer_agent": enhancer_agent is not None,
            "response_agent": response_agent is not None
        }
        
        # Try to get collection info if qdrant client is available
        if qdrant_client:
            try:
                info = qdrant_client.get_collection_info()
                components_status["database_connection"] = True
                components_status["documents_count"] = info.get("points_count", 0)
            except Exception as e:
                components_status["database_connection"] = False
                components_status["database_error"] = str(e)
        
        all_healthy = all(components_status.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            message="All systems operational" if all_healthy else "Some components may have issues",
            components=components_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_query(request: ChatQuery):
    """
    Main chat endpoint for processing user queries.
    
    Args:
        request: ChatQuery object containing the user's question
        
    Returns:
        ChatResponse with the AI-generated answer and metadata
    """
    try:
        # Validate components are initialized
        if not all([qdrant_client, embedding_model, enhancer_agent, response_agent]):
            raise HTTPException(
                status_code=503, 
                detail="System components not properly initialized"
            )
        
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        logger.info(f"Processing query: {request.query}")
        
        # Process the query using existing agent logic
        result = process_query(
            request.query,
            enhancer_agent,
            response_agent,
            qdrant_client,
            embedding_model
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to process query"
            )
        
        # Create response
        chat_response = ChatResponse(
            answer=result['response']['answer'],
            confidence=result['response']['confidence'],
            answer_type=result['response']['answer_type'],
            key_points=result['response']['key_points'],
            suggested_questions=result['response']['suggested_questions'],
            query=result['query'],
            enhanced_query=result['enhanced_query'],
            intent=result['intent'],
            results_count=result['results_count'],
            session_id=request.session_id
        )
        
        logger.info(f"Query processed successfully. Results: {result['results_count']}")
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/collection-info")
async def get_collection_info():
    """Get information about the vector database collection."""
    try:
        if not qdrant_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not initialized"
            )
        
        info = qdrant_client.get_collection_info()
        return {
            "collection_name": info.get("name", "unknown"),
            "points_count": info.get("points_count", 0),
            "status": "connected"
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection info: {str(e)}"
        )

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions that users can ask."""
    return {
        "sample_questions": [
            "What programs does Amrita University offer?",
            "Tell me about computer science admission requirements",
            "What are the campus facilities available?",
            "Research opportunities in engineering",
            "MBA program details and placement statistics",
            "What are the upcoming events at Amrita University?",
            "Tell me about student life at Amrita",
            "What scholarships are available?",
            "How to apply for PhD programs?",
            "Campus locations and contact information"
        ]
    }

# Development server runner
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "orchestrator:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Amrita University RAG System API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Get the actual IP address for network access
    if args.host == "0.0.0.0":
        import socket
        try:
            # Get local IP address
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"Starting Amrita University RAG System on:")
            print(f"  Local: http://localhost:{args.port}")
            print(f"  Network: http://{local_ip}:{args.port}")
            print(f"API Documentation available at:")
            print(f"  Local: http://localhost:{args.port}/docs")
            print(f"  Network: http://{local_ip}:{args.port}/docs")
        except:
            print(f"Starting Amrita University RAG System on http://{args.host}:{args.port}")
            print(f"API Documentation available at: http://localhost:{args.port}/docs")
    else:
        print(f"Starting Amrita University RAG System on http://{args.host}:{args.port}")
        print(f"API Documentation available at: http://{args.host}:{args.port}/docs")
    
    run_server(host=args.host, port=args.port, reload=args.reload)