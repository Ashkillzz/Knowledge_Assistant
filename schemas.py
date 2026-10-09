from pydantic import BaseModel, Field

class Source(BaseModel):
    document: str = Field(description="The name of the source document")
    page: int = Field(description="The page number where the information was found. Use 1 if unknown.")

class AnswerResponse(BaseModel):
    answer: str = Field(description="Factual summary matching context, or a polite conversational apology if the context lacks the answer.")
    sources: list[Source] = Field(description="List of exact sources used to answer the question.")
    confidence: float = Field(description="Confidence score of the answer between 0.0 and 1.0")