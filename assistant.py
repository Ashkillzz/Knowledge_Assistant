import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import sentence_transformers first to prevent Windows C++ runtime / DLL initialization conflicts
from sentence_transformers import CrossEncoder

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

from schemas import AnswerResponse
from prompts import AGENT_SYSTEM_PROMPT, FORMATTER_PROMPT

load_dotenv()


class HybridRetriever:
    """
    Combines dense vector retrieval and sparse BM25 keyword retrieval
    using standard Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vector_retriever, keyword_retriever, weights=(0.5, 0.5), k=60):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.weights = weights
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        try:
            vec_docs = self.vector_retriever.invoke(query)
        except Exception:
            vec_docs = []
            
        try:
            kw_docs = self.keyword_retriever.invoke(query)
        except Exception:
            kw_docs = []

        scores = {}
        doc_map = {}
        for rank, doc in enumerate(vec_docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + self.weights[0] * (1.0 / (self.k + rank + 1))

        for rank, doc in enumerate(kw_docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + self.weights[1] * (1.0 / (self.k + rank + 1))

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys]


class EnterpriseKnowledgeAssistant:
    def __init__(self, persist_directory="./chroma_db"):
        # ------------------------------------------------------------------
        # Vector store + embeddings
        # ------------------------------------------------------------------
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # ------------------------------------------------------------------
        # Build hybrid retriever by loading existing docs from ChromaDB
        # ------------------------------------------------------------------
        self._build_hybrid_retriever()

        # ------------------------------------------------------------------
        # Reranker + LLMs
        # ------------------------------------------------------------------
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # The core reasoning LLM (Agent)
        self.agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # The formatting LLM to ensure structured output
        self.formatter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnswerResponse)

        # ------------------------------------------------------------------
        # Agent Tools
        # ------------------------------------------------------------------
        search_tool = StructuredTool.from_function(
            func=self.search_knowledge_bank,
            name="search_knowledge_bank"
        )

        list_tool = StructuredTool.from_function(
            func=self.list_available_documents,
            name="list_available_documents",
            description="Returns a list of all documents currently indexed in the knowledge bank, including brief overviews and statistics for each file."
        )

        self.tools = [search_tool, list_tool]

        # Compile the Tool-Calling Agent
        self.agent = create_react_agent(
            self.agent_llm, 
            self.tools,
            prompt=AGENT_SYSTEM_PROMPT
        )

    # ------------------------------------------------------------------
    # Retriever builder — called on startup AND after each new ingest
    # ------------------------------------------------------------------
    def _load_raw_docs_from_store(self) -> list[Document]:
        """Pull every stored chunk out of ChromaDB as LangChain Documents."""
        data = self.vector_store.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        return [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]

    def _build_hybrid_retriever(self):
        """Build (or rebuild) BM25 + dense hybrid retriever from currently stored ChromaDB documents."""
        raw_docs = self._load_raw_docs_from_store()
        if raw_docs:
            print(f"[Assistant] Building BM25 over {len(raw_docs)} chunks.")
            keyword_retriever = BM25Retriever.from_documents(raw_docs)
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_retriever,
                keyword_retriever=keyword_retriever,
                weights=[0.5, 0.5]
            )
        else:
            print("[Assistant] ChromaDB is empty — using vector-only retrieval.")
            self.hybrid_retriever = self.vector_retriever

    def reload_retriever(self):
        """Public hook called by the API after a new document is ingested."""
        print("[Assistant] Reloading hybrid retriever after ingestion...")
        self._build_hybrid_retriever()

    # ------------------------------------------------------------------
    # Tool Implementations
    # ------------------------------------------------------------------
    def list_available_documents(self) -> str:
        """Returns a list of all documents currently indexed in the knowledge bank, including brief overviews and statistics for each file."""
        data = self.vector_store.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        
        if not metas:
            return "The knowledge bank is currently empty. No documents are indexed."

        # Aggregate document-level metrics and extract opening previews
        catalog = {}
        for text, m in zip(texts, metas):
            source = m.get("source")
            if not source:
                continue
            
            page = m.get("page", 1)
            raw_text = m.get("raw_text", text)

            if source not in catalog:
                catalog[source] = {
                    "chunks": 0,
                    "min_page": page,
                    "preview": "",
                }

            catalog[source]["chunks"] += 1

            # Extract the earliest page snippet as the document's introductory overview
            if not catalog[source]["preview"] or page <= catalog[source]["min_page"]:
                catalog[source]["min_page"] = page
                clean_snippet = " ".join(raw_text.strip().split())
                if len(clean_snippet) > 250:
                    clean_snippet = clean_snippet[:247] + "..."
                catalog[source]["preview"] = clean_snippet

        if not catalog:
            return "The knowledge bank is currently empty. No documents are indexed."

        report = [f"Indexed Documents ({len(catalog)} files available):\n"]
        for source, info in sorted(catalog.items()):
            report.append(
                f"- **{source}** ({info['chunks']} chunks)\n"
                f"  *Overview:* \"{info['preview']}\"\n"
            )

        return "\n".join(report)

    def search_knowledge_bank(self, query: str, target_document: str = None) -> str:
        """Search the enterprise knowledge bank for facts, policies, data, or documents. 
        Pass a precise search query string. 
        Optionally pass `target_document` (exact filename) to restrict the search to ONLY that document and prevent information bleed.
        """
        print(f"[Assistant Tool] search_knowledge_bank: query='{query}', target_document='{target_document}'")
        try:
            if target_document:
                # Strict isolation: use vector search with metadata filtering
                initial_docs = self.vector_store.similarity_search(
                    query, k=15, filter={"source": target_document}
                )
            else:
                initial_docs = self.hybrid_retriever.invoke(query)
        except Exception as e:
            # Fallback if Ensemble fails due to empty BM25
            initial_docs = self.vector_retriever.invoke(query)
            
        if not initial_docs:
            if target_document:
                return f"No documents found matching the query inside '{target_document}'."
            return "No documents found matching the query."
            
        # Rerank with Cross-Encoder
        doc_texts = [doc.page_content for doc in initial_docs]
        pairs = [[query, text] for text in doc_texts]
        scores = self.cross_encoder.predict(pairs)
        
        scored_docs = list(zip(initial_docs, scores))
        RELEVANCE_THRESHOLD = 0.3 # Slightly lower for safety
        
        # Filter and sort
        valid_docs = [(doc, s) for doc, s in scored_docs if s > RELEVANCE_THRESHOLD]
        if not valid_docs:
            # Fallback if strict threshold removes everything
            valid_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:3]
        else:
            valid_docs = sorted(valid_docs, key=lambda x: x[1], reverse=True)[:5]
            
        context = []
        for doc, score in valid_docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 1)
            context.append(f"Source: {source} (Page {page})\nRelevance Score: {score:.2f}\nContent: {doc.page_content}\n")
            
        return "\n".join(context)

    # ------------------------------------------------------------------
    # Main RAG pipeline (Agent Loop)
    # ------------------------------------------------------------------
    def ask(self, question: str, history: list[dict] = None) -> dict:
        history = history or []
        
        messages = []
        # Load conversation history so the agent remembers context
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=question))
        
        print(f"[Assistant] Starting Agent loop for query: '{question}'")
        # Run agent loop
        result = self.agent.invoke({"messages": messages})
        final_message = result["messages"][-1].content
        
        print("[Assistant] Agent loop completed. Formatting final response...")
        # Format the unstructured text into AnswerResponse schema
        formatted_response = self.formatter_llm.invoke(
            FORMATTER_PROMPT.format_messages(agent_response=final_message)
        )
        
        return formatted_response.model_dump()


# ---------------------------------------------------------------------------
# Manual test: python assistant.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    assistant = EnterpriseKnowledgeAssistant()
    result = assistant.ask("What documents do you have?")
    print(json.dumps(result, indent=2))