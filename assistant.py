import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever    ## Note: Import for EnsembleRetriever moved from langchain_community.retrievers to langchain_classic.retrievers 
from sentence_transformers import CrossEncoder

from schemas import AnswerResponse, RouteDecision
from prompts import CONDENSE_PROMPT, ROUTER_PROMPT, RAG_PROMPT
    
# Load API keys from .env
load_dotenv()

# --- 2. Build the Generation Engine ---
class EnterpriseKnowledgeAssistant:
    def __init__(self, persist_directory="./chroma_db", raw_docs=None):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Retrieval Setup
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        if raw_docs:
            self.keyword_retriever = BM25Retriever.from_documents(raw_docs)
            self.keyword_retriever.k = 10
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.keyword_retriever],
                weights=[0.5, 0.5]
            )
        else:
            self.hybrid_retriever = self.vector_retriever
            
        # Re-ranking and LLM Setup
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.condense_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnswerResponse)
        self.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouteDecision)
        
    def ask(self, question: str, history: list[dict] = None) -> dict:
        history = history or []
        standalone_question = question
        
        # STEP 1: Coreference Shorthand Correction (Memory Processing)
        if history:
            history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
            standalone_question = (CONDENSE_PROMPT | self.condense_llm).invoke(
                {"history": history_text, "question": question}
            ).content

        # STEP 2: Intent Classification Routing
        decision = (ROUTER_PROMPT | self.router_llm).invoke({"question": standalone_question})
        
        if decision.route in ["casual", "ambiguous"]:
            return {"answer": decision.reply, "sources": [], "confidence": 1.0}
            
        # STEP 3: Metadata Target Extraction Filter (Gathers up to 15 nodes)
        search_kwargs = {"k": 15}
        if decision.category_filter != "ALL":
            search_kwargs["filter"] = {"category": decision.category_filter}
            
        initial_docs = self.vector_store.similarity_search(standalone_question, **search_kwargs)
        
        if not initial_docs:
            # Bypass Re-ranking and pass an empty context to the LLM
            context_text = "No documents found."
        
        else:
            # STEP C: Re-ranking
            doc_texts = [doc.page_content for doc in initial_docs]
            pairs = [[question, text] for text in doc_texts]
            scores = self.cross_encoder.predict(pairs)
            
            scored_docs = sorted(list(zip(initial_docs, scores)), key=lambda x: x[1], reverse=True)
            best_docs = [doc for doc, score in scored_docs[:3]]
            
            # STEP D: Generation
            context_text = "\n\n".join(
                [f"Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})\nContent: {doc.page_content}" 
                for doc in best_docs]
            )
        
        response_obj = (RAG_PROMPT | self.rag_llm).invoke({"context": context_text, "question": question})
        return response_obj.model_dump()

# Example Usage:
if __name__ == "__main__":
    assistant = EnterpriseKnowledgeAssistant()
    
    # Try testing it with a question related to the 3 PDFs you just ingested!
    result = assistant.ask("What is the Raspberry Pi model?")

    print(json.dumps(result, indent=2))