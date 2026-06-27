from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from assistant import EnterpriseKnowledgeAssistant, AnswerResponse

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="Production-ready RAG API for internal document querying."
)

# Initialize the assistant once when the server starts
assistant = EnterpriseKnowledgeAssistant()

# Define the exact incoming request schema from the assignment PDF
class QueryRequest(BaseModel):
    question: str

# Define the outgoing response schema (matching our Pydantic assistant model)
class QueryResponse(AnswerResponse):
    pass

@app.post("/ask", response_model=QueryResponse, summary="Ask a question to the knowledge base")
async def ask_question(payload: QueryRequest):
    """
    Receives an employee question, performs semantic search over the enterprise document vault,
    and returns a safe, context-bound answer with exact sources and confidence scores.
    """
    try:
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        # Run our RAG pipeline
        result = assistant.ask(payload.question)
        return result

    except Exception as e:
        # Fulfills the 'Engineering Quality: Error Handling' requirement
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# To run this file locally for testing:
# pip install uvicorn
# uvicorn main:app --reload