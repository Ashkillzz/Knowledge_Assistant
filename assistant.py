import os
import json
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# Load API keys from .env
load_dotenv()

# --- 1. Define the Strict JSON Schema ---
class Source(BaseModel):
    document: str = Field(description="The name of the source document")
    page: int = Field(description="The page number where the information was found")

class AnswerResponse(BaseModel):
    answer: str = Field(description="The concise answer to the user's question. If not found in context, output exactly 'Information unavailable'.")
    sources: list[Source] = Field(description="List of exact sources used to answer the question, extracted from the context.")
    confidence: float = Field(description="Confidence score of the answer between 0.0 and 1.0")

class RouteDecision(BaseModel):
    route: Literal["casual", "search", "ambiguous"] = Field(
        description="Classify the intent: 'casual' for greetings, 'search' for clear queries about company data, 'ambiguous' if the query lacks specific context to search effectively."
    )
    reply: str = Field(
        description="If route is 'casual', put the friendly reply here. If 'ambiguous', put a clarifying question here (e.g., 'Are you looking for HR or IT policies?'). If 'search', leave empty."
    )

# --- 2. Build the Generation Engine ---
class EnterpriseKnowledgeAssistant:
    def __init__(self, persist_directory="./chroma_db"):
        """Connects to the existing database and initializes the LLM."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Connect to the local database you just built
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Initialize the LLM and bind it to our strict Pydantic schema

        # llm to generate answers based on retrieved context
        self.rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnswerResponse)
        
        # llm to decide if the user query requires a search or is just casual conversation
        self.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).with_structured_output(RouteDecision)

    def ask(self, question: str) -> dict:
        """Evaluates intent, routes the query, and generates a structured answer."""
        
        # ------------- STEP A: The Router ---------------
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the intelligent routing brain of an Enterprise Knowledge Assistant. 
                       Classify the user's input into one of three categories:
                       1. 'casual': Greetings, thanks, or casual conversation. Provide a friendly 'reply'.
                       2. 'ambiguous': The question is unclear or lacks context. Provide a clarifying 'reply' to guide the user.
                       3. 'search': A clear question to perform a search. Leave 'reply' empty."""),
            ("user", "{question}")
        ])
        
        router_chain = router_prompt | self.router_llm
        decision = router_chain.invoke({"question": question})

        # ------------- STEP B: Handle Casual/Ambiguous Requests -------------

        if decision.route in ["casual", "ambiguous"]:
            # Return the conversational/clarifying response directly
            return {
                "answer": decision.reply,
                "sources": [],
                "confidence": 1.0 
            }

        #  ------------- STEP C: The RAG Pipeline -------------
        
        # A. Retrieve the top 3 most relevant chunks from ChromaDB
        docs = self.vector_store.similarity_search(question, k=3)
        
        # B. Format the chunks into a readable context block, injecting the metadata
        context_text = "\n\n".join(
            [f"Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})\nContent: {doc.page_content}" 
             for doc in docs]
        )
        
        # C. The Guardrail Prompt (Crucial for the evaluation criteria)
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an Enterprise Knowledge Assistant. Answer the user's question ONLY using the provided Context, converse with them and be polite at all times.\nIf the Context does not contain the answer, your answer MUST be 'Information unavailable'.\nNever make up facts through assumptions or partial awareness. Extract the correct source document and page number strictly from the Context provided."),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        # D. Execute the pipeline
        rag_chain = rag_prompt | self.rag_llm
        response_obj = rag_chain.invoke({"context": context_text, "question": question})
        
        # Return as a standard Python dictionary for easy API integration later
        return response_obj.model_dump()

# Example Usage:
if __name__ == "__main__":
    assistant = EnterpriseKnowledgeAssistant()
    
    # Try testing it with a question related to the 3 PDFs you just ingested!
    result = assistant.ask("What is the Raspberry Pi model?")

    print(json.dumps(result, indent=2))