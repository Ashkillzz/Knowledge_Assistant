import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import uuid
import shutil

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel

from assistant import EnterpriseKnowledgeAssistant, AnswerResponse
from document_ingestion import EnterpriseKnowledgeIngestor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="Production-ready RAG API for internal document querying.",
    version="2.0.0",
)

# Singletons — shared across all requests
assistant = EnterpriseKnowledgeAssistant()
ingestor = EnterpriseKnowledgeIngestor()

# Temp upload directory (already exists in project)
TEMP_UPLOAD_DIR = "./temp_files"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# In-memory job tracker  {job_id: {status, filename, message, chunks_added}}
ingestion_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []


class QueryResponse(AnswerResponse):
    pass


# ---------------------------------------------------------------------------
# /ask  — RAG query endpoint
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=QueryResponse, summary="Ask a question to the knowledge base")
async def ask_question(payload: QueryRequest):
    """
    Receives an employee question, runs the full RAG pipeline
    (Router → Hybrid Retrieval → Cross-Encoder Reranking → LLM),
    and returns a structured answer with sources and confidence.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = assistant.ask(payload.question, payload.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# ---------------------------------------------------------------------------
# /ingest  — File upload endpoint
# ---------------------------------------------------------------------------
def _run_ingestion(job_id: str, file_path: str, filename: str):
    """
    Background task: ingest a single file, then reload the assistant's
    BM25 retriever so the new content is immediately queryable.
    """
    try:
        ingestion_jobs[job_id]["status"] = "processing"
        docs = ingestor.ingest_file(file_path, filename)
        # Rebuild BM25 + hybrid retriever with the new chunks
        assistant.reload_retriever()
        if len(docs) == 0:
            ingestion_jobs[job_id]["status"] = "failed"
            ingestion_jobs[job_id]["chunks_added"] = 0
            ingestion_jobs[job_id]["message"] = f"Ingestion warning: 0 chunks were extracted from '{filename}'. Check file format or text readability."
        else:
            ingestion_jobs[job_id]["status"] = "done"
            ingestion_jobs[job_id]["chunks_added"] = len(docs)
            ingestion_jobs[job_id]["message"] = f"Successfully ingested '{filename}' ({len(docs)} chunks added)."
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["message"] = f"Ingestion failed: {str(e)}"
    finally:
        # Clean up the temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/ingest", summary="Upload and ingest a document")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accepts a document upload, saves it temporarily, then runs ingestion in
    the background. Returns a job_id to poll for status.
    """
    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ingestor.supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(ingestor.supported_extensions)}"
        )

    job_id = str(uuid.uuid4())
    save_path = os.path.join(TEMP_UPLOAD_DIR, f"{job_id}_{file.filename}")

    # Save upload to disk (async-safe)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Register job and dispatch background task
    ingestion_jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "chunks_added": 0,
        "message": "File received, waiting to be processed.",
    }
    background_tasks.add_task(
        _run_ingestion, job_id, save_path, file.filename
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "detail": "Ingestion started in background. Poll /ingest/status/{job_id} for updates.",
    }


# ---------------------------------------------------------------------------
# /ingest/status/{job_id}  — Poll ingestion progress
# ---------------------------------------------------------------------------
@app.get("/ingest/status/{job_id}", summary="Check ingestion job status")
async def ingest_status(job_id: str):
    """
    Returns the current status of an ingestion job.
    Statuses: queued → processing → done | failed
    """
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return ingestion_jobs[job_id]


# ---------------------------------------------------------------------------
# /ingest/jobs  — List all ingestion jobs (useful for debugging)
# ---------------------------------------------------------------------------
@app.get("/ingest/jobs", summary="List all ingestion jobs")
async def list_jobs():
    return {"total": len(ingestion_jobs), "jobs": ingestion_jobs}


# ---------------------------------------------------------------------------
# Document Management Endpoints
# ---------------------------------------------------------------------------
@app.get("/documents", summary="List all indexed documents")
async def get_documents():
    """Returns a list of all documents currently stored in the knowledge base."""
    docs = ingestor.get_indexed_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{filename}", summary="Delete a document and refresh retriever")
async def delete_document(filename: str):
    """
    Deletes all chunks belonging to a document from ChromaDB,
    then triggers retriever reload so it is removed from BM25 and vector search.
    """
    deleted_count = ingestor.delete_document(filename)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in knowledge base."
        )
    # Reload BM25 + Hybrid Retriever
    assistant.reload_retriever()
    return {
        "status": "deleted",
        "filename": filename,
        "chunks_deleted": deleted_count,
        "message": f"Successfully removed '{filename}' ({deleted_count} chunks deleted)."
    }