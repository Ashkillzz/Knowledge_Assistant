from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import hashlib
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

class EnterpriseKnowledgeIngestor:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )

    def process_directory(self, doc_dir="./documents"):
        """Traverses directories, extracts path-based category tags and hashes chunks."""
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        if not os.path.exists(doc_dir):
            print(f"Target folder {doc_dir} missing.")
            return

        for root, _, files in os.walk(doc_dir):
            for filename in files:
                if filename.endswith(".pdf"):
                    file_path = os.path.join(root, filename)
                    
                    # Deterministic, free path-based categories (e.g. ./documents/HR -> 'HR')
                    relative_path = os.path.relpath(root, doc_dir)
                    category = "GENERAL" if relative_path == "." else relative_path.upper()
                    
                    print(f"Indexing: {filename} under category: [{category}]")
                    chunks, metadatas, ids = self._extract_chunk_and_hash_pdf(file_path, filename, category)
                    
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)
                    all_ids.extend(ids)
                    
        if all_chunks:

            # 1. Create clean lists and a tracker set
            unique_chunks = []
            unique_metadatas = []
            unique_ids = []
            seen_ids = set()

            # 2. Filter the batch before sending it to ChromaDB
            for chunk, metadata, chunk_id in zip(all_chunks, all_metadatas, all_ids):
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_chunks.append(chunk)
                    unique_metadatas.append(metadata)
                    unique_ids.append(chunk_id)

            # 3. Add the purely unique batch to the database
            self.vector_store.add_texts(texts=unique_chunks, metadatas=unique_metadatas, ids=unique_ids)
            print(f"Successfully synced database. Unique active fragments stored.")

    def _extract_chunk_and_hash_pdf(self, file_path, filename, category):
        chunks, metadatas, ids = [], [], []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    page_chunks = self.text_splitter.split_text(text)
                    for chunk in page_chunks:
                        chunks.append(chunk)
                        metadatas.append({
                            "source": filename, 
                            "page": page_num + 1,
                            "category": category
                        })
                        # Generates unique fingerprint to intercept duplicates
                        chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
                        ids.append(chunk_hash)
        return chunks, metadatas, ids

if __name__ == "__main__":
    ingestor = EnterpriseKnowledgeIngestor()
    ingestor.process_directory(r"D:/Aswin/Enterprise_Document_Assistant/docs/data")