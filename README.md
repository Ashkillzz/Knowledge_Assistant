# 🏢 Enterprise Knowledge Assistant (Advanced RAG Pipeline)

An enterprise-grade Retrieval-Augmented Generation (RAG) system built to securely and intelligently query company documents (HR policies, Manuals, Techincal guides, Product documentation etc).  

This project tackles real-world edge cases: conversational memory loss, duplicate document ingestion, hallucinations(partially enforced), and retrieval precision across acronyms. It features a decoupled architecture using FastAPI for the backend and Streamlit for the frontend.

## ✨ Key Features & Architecture Highlights

### 1. Smart Ingestion & Deduplication
* **Content Hashing:** Uses SHA-256 to hash text chunks during ingestion, natively preventing ChromaDB from storing duplicate vectors even if duplicate files are uploaded.
* **Path-Based Metadata:** Automatically extracts domain categories (e.g., `HR`, `IT`) from the local directory structure at runtime to enable highly targeted metadata filtering during search.

### 2. Conversational Memory & Query Condensation
* **Coreference Resolution:** Automatically translates conversational shorthand (e.g., "Where did *he* study?") into fully resolved, standalone search queries using chat history.
* **Topic Bleed Prevention:** Able to realsie when a user abruptly changes the subject, dropping old context to prevent mixed-domain hallucinations. (More enhancements necessary)

### 3. Semantic Intent Routing
Implements a 3-way LLM router at the entry point to classify queries before they hit the database:
* **Search:** Extracts categorical metadata and routes to the RAG pipeline.
* **Casual:** Handles greetings, casual conversationional questions with assistant.
* **Ambiguous:** Asks the user clarifying questions about missing parameters/keywords for performing quality search.

### 4. Advanced Retrieval & Re-ranking
* **Cross-Encoder Re-ranking:** Passes the top 15 retrieved chunks through a local `ms-marco-MiniLM-L-6-v2` cross-encoder. It scores the direct logical alignment between the query and context, surfacing the top 3 highest-fidelity chunks to the generation LLM.

### 5. Graceful Degradation
* Bypasses robotic responses such as`Information unavailable` hardcodes. If the database returns no relevant context, the LLM dynamically generates polite, conversational apologies based on system guardrails.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Pydantic (Structured Outputs)
* **Frontend:** Streamlit
* **AI/NLP:** LangChain, OpenAI (`gpt-4o-mini`, `text-embedding-3-small`), SentenceTransformers (Cross-Encoders)
* **Vector Database:** ChromaDB
* **Document Parsing:** PDFPlumber

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/enterprise-knowledge-assistant.git](https://github.com/yourusername/enterprise-knowledge-assistant.git)
   cd enterprise-knowledge-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Application

1. **Load your documents**
   Place your PDFs into the `./documents` folder (use subfolders like `./documents/HR` for automatic tagging). Then run the ingestion script:
   ```bash
   python document_ingestion.py
   ```

2. **Start the FastAPI Backend**
   ```bash
   uvicorn main:app --reload
   ```

3. **Start the Streamlit Frontend**
   Open a new terminal and run:
   ```bash
   streamlit run app.py
   ```

## 🚧 Limitations & Future Work

While this architecture is robust for mid-sized document pools, scaling to enterprise-level datasets requires architectural shifts:
* **Entity & Acronym Resolution:** The system initially struggled to link abbreviations (e.g., "SPIL") with full company names (e.g., "Sirca Paints India Ltd") if the specific chunk lacked both. While we mitigated this using LLM-based Query Expansion, a production system should implement a dedicated Knowledge Graph or a custom synonym dictionary injection at the embedding level.
* **Document Lifecycle & State Management:** Implemented in-memory hash tracking to prevent `DuplicateIDError` during batch ingestion. However, the current system lacks full CRUD capabilities for documents (e.g., detecting when a PDF is updated or deleted in the local folder and automatically syncing those changes to ChromaDB).
* **Topic Bleed in Conversational Memory:** When users abruptly changed subjects, the query condenser occasionally dragged old context into the new query, causing retrieval misses. While patched with strict "Topic Shift" prompt directives, a more robust solution would involve a state-machine architecture (like LangGraph) to mathematically detect and manage context boundary resets.
* **Persona Stability ("Identity Theft"):** During testing, the assistant occasionally adopted the persona of candidates whose CVs were stored in the database when asked "Tell me about yourself." We enforced strict routing guardrails to intercept identity questions, but future iterations should separate generic chat routing from database retrieval more cleanly at the middleware layer.
* **Dynamic Document Categorization:** We optimized metadata tagging by deriving categories from folder paths (e.g., `./documents/HR`). In a messy, real-world enterprise data lake where folder structures are unreliable, this should be replaced with a lightweight, asynchronous classification model that categorizes documents dynamically upon upload.

## 📜 License
No licenses included.


## 📂 Project Structure

```text
├── documents/               # Drop your PDFs here (organize by subfolders for metadata)
│   ├── HR/                  # Scenario
│   ├── IT/
├── document_ingestion.py                # Offline batch processor & deduplicator
├── assistant.py             # Core RAG engine (Memory -> Router -> Retrieval -> Generation)
├── prompts.py               # Prompt ledger & system instructions
├── schemas.py               # Pydantic models for structured output validation
├── main.py                  # FastAPI server endpoint
├── app.py                   # Streamlit user interface
├── requirements.txt         
└── .env                     # API keys
