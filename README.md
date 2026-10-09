# 🏢 Enterprise Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built to query enterprise documents (PDFs, Word documents, Excel sheets, and Presentations). 

The system features a decoupled architecture with a **FastAPI** backend and an interactive **Streamlit** frontend, combining hybrid search (Dense Vector + Sparse BM25), Cross-Encoder reranking, and an autonomous **LangGraph ReAct agent**.

---

## ✨ System Architecture & Key Features

### 1. Ingestion Pipeline & Contextual Chunking
* **Multi-Format Extraction:** Parses `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`, and `.csv` files using `UnstructuredFileLoader` and `PyPDFLoader`.
* **Contextual Metadata Enrichment:** Text is split using `RecursiveCharacterTextSplitter` (chunk size: `1,000` characters, overlap: `200` characters). Each chunk is pre-pended with a contextual header containing the filename, page number, category, and an introductory summary to retain document identity.
* **Deduplication:** Computes a SHA-256 hash for every chunk, ensuring idempotency and preventing duplicate vectors from being stored in ChromaDB.
* **Embeddings & Storage:** Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and stored in a persistent local ChromaDB instance (`./chroma_db`).

### 2. Hybrid Retrieval (Dense + Sparse)
* **Dense Semantic Search:** Queries ChromaDB for semantic similarity using cosine distance.
* **Sparse Keyword Search:** Builds an in-memory `BM25Retriever` across all indexed chunks to capture exact terminology, numbers, and acronyms.
* **Reciprocal Rank Fusion (RRF):** Blends vector and keyword results using standard RRF (50/50 weighting) to produce a combined candidate pool.

### 3. Cross-Encoder Re-Ranking
* **Model:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score candidate chunks against the user's query.
* **Noise Pruning:** Applies a relevance score threshold (`0.30`) to filter out superficially matched chunks, passing only the top 3–5 highest-scoring passages to the LLM to prevent prompt bloat and context distraction.

### 4. Agentic Orchestration & Structured Output
* **LangGraph ReAct Agent:** Powered by `gpt-4o-mini`, the agent executes iterative reasoning using two tools:
  * `search_knowledge_bank`: Runs the hybrid retrieval and reranking pipeline. Supports an optional `target_document` parameter for strict document isolation.
  * `list_available_documents`: Returns the catalog of currently indexed files, chunk counts, and introductory previews.
* **Structured Outputs:** A secondary formatter LLM enforces the `AnswerResponse` Pydantic schema, delivering a clean answer, specific source citations (document name and page number), and a confidence score (`0.0`–`1.0`).
* **Hallucination Prevention:** When the retrieved context does not contain the answer, the assistant responds with a polite disclaimer and assigns a `0.0` confidence score instead of hallucinating.

### 5. Vector Database Visualizer
* Includes an interactive 3D embeddings visualizer (`DB_visualizer.py`) powered by `chromaviz`, allowing you to inspect PCA and t-SNE reduced vector clusters in the browser.

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit
* **AI & Orchestration:** LangGraph, LangChain, OpenAI (`gpt-4o-mini`)
* **Embeddings & Reranker:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim), `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Keyword Search:** `rank-bm25`
* **Vector Store:** ChromaDB
* **Document Parsing:** Unstructured, PyPDF, pdf2image, python-docx, openpyxl, python-pptx

---

## 📂 Project Structure

```text
├── app.py                   # Streamlit interactive frontend
├── main.py                  # FastAPI server & REST API endpoints (/ask, /ingest, /documents)
├── assistant.py             # Core RAG engine (Hybrid Retriever, Reranker, LangGraph ReAct Agent)
├── document_ingestion.py    # Document parsing, contextual header enrichment & deduplication
├── prompts.py               # Agent system prompts and formatting instructions
├── schemas.py               # Pydantic models for API request/response validation
├── DB_visualizer.py         # 3D vector database visualizer (ChromaViz)
├── requirements.txt         # Project dependencies
├── docs/                    # Default folder for local document ingestion
├── Ingested_Docs/           # Indexed enterprise reference documents
├── chroma_db/               # Persistent ChromaDB storage directory
└── temp_files/              # Staging directory for API/UI file uploads
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ashkillzz/Knowledge_Assistant.git
   cd Knowledge_Assistant
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

### Running the Application

1. **Start the FastAPI Backend**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000` (Swagger docs at `http://127.0.0.1:8000/docs`).

2. **Start the Streamlit Frontend**
   In a separate terminal:
   ```bash
   streamlit run app.py
   ```
   The UI will open in your browser at `http://localhost:8501`.

3. **(Optional) Launch the 3D Database Visualizer**
   ```bash
   python DB_visualizer.py
   ```
   Opens a 3D point cloud of your indexed vector embeddings at `http://127.0.0.1:5000`.

---

### Ingesting Documents

* **Via the Web UI:** Use the drag-and-drop file uploader in the Streamlit sidebar. Files are uploaded to `/ingest`, processed asynchronously in the background, and the hybrid retriever's BM25 index automatically reloads upon completion.
* **Via Python Script:**
  ```python
  from document_ingestion import EnterpriseKnowledgeIngestor

  ingestor = EnterpriseKnowledgeIngestor(persist_directory="./chroma_db")
  ingestor.process_directory("./docs")
  ```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `POST /ask` | POST | Submits a user question to the RAG pipeline and returns answer, sources, and confidence. |
| `POST /ingest` | POST | Uploads and processes a document in a background worker task. |
| `GET /ingest/status/{job_id}` | GET | Checks the real-time status of an ingestion job. |
| `GET /documents` | GET | Returns a list of all currently indexed documents and chunk statistics. |
| `DELETE /documents/{filename}` | DELETE | Removes all chunks associated with a specific file from ChromaDB. |

---

## 📜 License
Internal / Proprietary project.
