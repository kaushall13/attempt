from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from .rag_orchestrator import RAGOrchestrator

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    confidence: float

class AddDocumentsRequest(BaseModel):
    documents: List[str]

# --- FastAPI App ---
app = FastAPI(
    title="RAG API",
    description="A simple API for a RAG application.",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- RAG Orchestrator Initialization ---
rag_orchestrator = RAGOrchestrator()

# --- API Endpoints ---
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/add_documents")
def add_documents(request: AddDocumentsRequest):
    """Endpoint to add new documents to the knowledge base."""
    try:
        rag_orchestrator.add_documents(request.documents)
        return {"message": "Documents added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Main RAG query endpoint."""
    try:
        result = rag_orchestrator.query(request.query, request.session_id)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
