import os
import hashlib
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader

load_dotenv()


class EnterpriseKnowledgeIngestor:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        # Supported formats for the general-purpose data bank
        self.supported_extensions = {
            ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", 
            ".txt", ".md", ".png", ".jpg", ".jpeg", ".csv"
        }

    # ------------------------------------------------------------------
    # Public: Ingest an entire directory of mixed documents
    # ------------------------------------------------------------------
    def process_directory(self, doc_dir="./docs") -> list[Document]:
        """
        Traverses a directory, hashes chunks, and adds unique documents to ChromaDB.
        Returns the list of newly added Document objects.
        """
        if not os.path.exists(doc_dir):
            print(f"[Ingestor] Target folder '{doc_dir}' not found.")
            return []

        all_docs: list[Document] = []
        all_ids: list[str] = []

        for root, _, files in os.walk(doc_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.supported_extensions:
                    file_path = os.path.join(root, filename)

                    print(f"[Ingestor] Indexing: {filename}")
                    docs, ids = self._extract_chunks_from_file(file_path, filename)
                    all_docs.extend(docs)
                    all_ids.extend(ids)

        return self._add_unique_to_store(all_docs, all_ids)

    # ------------------------------------------------------------------
    # Public: Ingest a single file (used by the API upload endpoint)
    # ------------------------------------------------------------------
    def ingest_file(self, file_path: str, filename: str) -> list[Document]:
        """
        Ingests a single file (any supported format) into ChromaDB.
        Returns the list of newly added Document objects.
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"File format {ext} is not currently supported.")
            
        print(f"[Ingestor] Single-file ingest: {filename}")
        docs, ids = self._extract_chunks_from_file(file_path, filename)
        return self._add_unique_to_store(docs, ids)

    # ------------------------------------------------------------------
    # Public: Load all stored documents back as LangChain Documents
    # ------------------------------------------------------------------
    def load_all_raw_docs(self) -> list[Document]:
        """
        Reads every chunk currently stored in ChromaDB and returns them as
        LangChain Document objects so the BM25 retriever can be rebuilt.
        """
        data = self.vector_store.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metas = data.get("metadatas") or []
        if not texts:
            print("[Ingestor] ChromaDB is empty — no documents loaded for BM25.")
            return []
        print(f"[Ingestor] Loaded {len(texts)} chunks from ChromaDB for BM25.")
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metas)
        ]

    # ------------------------------------------------------------------
    # Public: Query & Delete Document Metadata
    # ------------------------------------------------------------------
    def get_indexed_documents(self) -> list[dict]:
        """
        Returns a list of unique documents with their total chunk count.
        """
        data = self.vector_store.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        docs = {}
        for m in metas:
            source = m.get("source")
            if source:
                if source not in docs:
                    docs[source] = {"filename": source, "chunks": 0}
                docs[source]["chunks"] += 1
        return list(docs.values())

    def delete_document(self, filename: str) -> int:
        """
        Removes all chunks associated with `filename` from ChromaDB.
        Returns the number of deleted chunks.
        """
        data = self.vector_store.get(where={"source": filename})
        chunk_ids = data.get("ids", [])
        if chunk_ids:
            self.vector_store.delete(ids=chunk_ids)
            print(f"[Ingestor] 🗑️ Deleted {len(chunk_ids)} chunks for '{filename}'.")
            return len(chunk_ids)
        return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _extract_chunks_from_file(
        self, file_path: str, filename: str
    ) -> tuple[list[Document], list[str]]:
        """
        Opens a file (PDF, Word, Excel, PPT), splits into chunks,
        enriches each chunk with contextual headers, and returns (list[Document], list[sha256_ids]).
        """
        docs: list[Document] = []
        ids: list[str] = []

        try:
            ext = os.path.splitext(filename)[1].lower()
            raw_documents = []

            # For PDFs, use PyPDFLoader for fast, robust digital text extraction without OCR dependencies
            if ext == ".pdf":
                try:
                    loader = PyPDFLoader(file_path)
                    raw_documents = loader.load()
                    
                    # Fallback to OCR if document is scanned (no text extracted)
                    if not raw_documents or all(not doc.page_content.strip() for doc in raw_documents):
                        print(f"[Ingestor] '{filename}' appears to be a scanned PDF. Attempting OCR...")
                        from pdf2image import convert_from_path
                        from rapidocr_onnxruntime import RapidOCR
                        import numpy as np
                        
                        ocr = RapidOCR()
                        images = convert_from_path(file_path)
                        ocr_docs = []
                        for i, img in enumerate(images):
                            img_array = np.array(img)
                            result, _ = ocr(img_array)
                            text = '\n'.join([line[1] for line in result]) if result else ""
                            if text.strip():
                                ocr_docs.append(Document(page_content=text, metadata={"source": file_path, "page_number": i + 1}))
                        
                        if ocr_docs:
                            raw_documents = ocr_docs
                        else:
                            print(f"[Ingestor] OCR extracted no text for {filename}.")
                except Exception as pe:
                    print(f"[Ingestor] PyPDFLoader notice ({pe}), falling back to Unstructured...")

            # Fallback or other formats (Word, Excel, PPTX, etc.)
            if not raw_documents:
                try:
                    loader = UnstructuredFileLoader(file_path, strategy="fast")
                    raw_documents = loader.load()
                except Exception:
                    loader = UnstructuredFileLoader(file_path, strategy="auto")
                    raw_documents = loader.load()

            if not raw_documents:
                print(f"[Ingestor] ⚠️ No text could be extracted from {filename}!")
                return docs, ids

            # ----- Contextual Chunk Enrichment -----
            # Build a document-level preview from the first ~500 chars
            # so every chunk "knows" what document it belongs to.
            full_text_preview = ""
            for raw_doc in raw_documents:
                full_text_preview += raw_doc.page_content + " "
                if len(full_text_preview) >= 500:
                    break
            full_text_preview = full_text_preview[:500].strip()

            for raw_doc in raw_documents:
                text = raw_doc.page_content
                if not text.strip():
                    continue

                page_num = raw_doc.metadata.get("page_number", 1)

                for chunk in self.text_splitter.split_text(text):
                    # Prepend contextual header — this gets embedded with the chunk
                    # so the vector encodes document identity + location + content.
                    enriched_content = (
                        f"Document: {filename} | Page: {page_num}\n"
                        f"Document Preview: {full_text_preview[:200]}...\n"
                        f"---\n"
                        f"{chunk}"
                    )

                    doc = Document(
                        page_content=enriched_content,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "raw_text": chunk,  # preserve clean text for LLM display
                        }
                    )
                    chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    docs.append(doc)
                    ids.append(chunk_hash)

        except Exception as e:
            print(f"[Ingestor] Error extracting from {filename}: {e}")

        return docs, ids

    def _add_unique_to_store(
        self, docs: list[Document], ids: list[str]
    ) -> list[Document]:
        """
        Deduplicates against already-seen hashes and adds only new chunks
        to ChromaDB. Returns the unique Document objects that were added.
        """
        if not docs:
            return []

        unique_docs: list[Document] = []
        unique_ids: list[str] = []
        seen: set[str] = set()

        for doc, chunk_id in zip(docs, ids):
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_docs.append(doc)
                unique_ids.append(chunk_id)

        # Add to ChromaDB (texts + metadatas extracted from Document objects)
        self.vector_store.add_texts(
            texts=[d.page_content for d in unique_docs],
            metadatas=[d.metadata for d in unique_docs],
            ids=unique_ids,
        )
        print(f"[Ingestor] ✅ Added {len(unique_docs)} unique chunks to ChromaDB.")
        return unique_docs


# ---------------------------------------------------------------------------
# Manual run: python document_ingestion.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingestor = EnterpriseKnowledgeIngestor()
    ingestor.process_directory("./docs")