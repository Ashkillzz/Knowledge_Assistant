from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import pdfplumber

from dotenv import load_dotenv

load_dotenv()

class EnterpriseKnowledgeIngestor:
    def __init__(self, persist_directory="./chroma_db"):
        """Initializes the embedding model and local vector database."""
        self.persist_directory = persist_directory
        # Using OpenAI's small embedding model for high accuracy and low cost
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 1000 characters with 200 overlap to maintain context across paragraphs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize the local Chroma vector store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def process_directory(self, doc_dir="./documents"):
        """Loops through the 'library' of different PDFs and ingests them."""
        all_chunks = []
        all_metadatas = []
        
        print(f"Scanning directory: {doc_dir}...")
        
        for filename in os.listdir(doc_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(doc_dir, filename)
                print(f"Processing: {filename}")
                
                chunks, metadatas = self._extract_and_chunk_pdf(file_path, filename)
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                
        # Store all extracted chunks and metadata into ChromaDB
        if all_chunks:
            self.vector_store.add_texts(texts=all_chunks, metadatas=all_metadatas)
            self.vector_store.persist() # Save to disk
            print(f"Successfully indexed {len(all_chunks)} chunks into the database!")

    def _extract_and_chunk_pdf(self, file_path, filename):
        """Extracts text page-by-page and attaches exact source metadata."""
        chunks = []
        metadatas = []
        
        # pdfplumber is excellent at maintaining reading order and layout
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if text:
                    # Split the page text into manageable chunks
                    page_chunks = self.text_splitter.split_text(text)
                    chunks.extend(page_chunks)
                    
                    # Create metadata for every single chunk to satisfy the citation requirement
                    chunk_metadata = {"source": filename, "page": page_num + 1}
                    metadatas.extend([chunk_metadata] * len(page_chunks))
                    
        return chunks, metadatas

# Example Usage:
if __name__ == "__main__":
    ingestor = EnterpriseKnowledgeIngestor()
    ingestor.process_directory(r"D:/Aswin/Enterprise_Document_Assistant/docs")