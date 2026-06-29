from typing import Literal
from pydantic import BaseModel, Field

# --- 1. Define the Strict JSON Schema ---
class Source(BaseModel):
    document: str = Field(description="The name of the source document")
    page: int = Field(description="The page number where the information was found")

class AnswerResponse(BaseModel):
    answer: str = Field(description="Factual summary matching context, or a polite conversational apology if the context lacks the answer.")
    sources: list[Source] = Field(description="List of exact sources used to answer the question, extracted from the context.")
    confidence: float = Field(description="Confidence score of the answer between 0.0 and 1.0")

class RouteDecision(BaseModel):
    route: Literal["casual", "search", "ambiguous"] = Field(
        description="Classify the intent: 'casual' for greetings, 'search' for clear queries about company data, 'ambiguous' if the query lacks specific context to search effectively."
    )
    category_filter: str = Field(
        description="If route is 'search', infer the document category (e.g., 'HR', 'IT', 'Customer'). If unknown, output 'ALL'."
    )
    reply: str = Field(
        description="If route is 'casual', put the friendly reply here. If 'ambiguous', put a clarifying question here (e.g., 'Are you looking for HR or IT policies?'). If 'search', leave empty."
    )