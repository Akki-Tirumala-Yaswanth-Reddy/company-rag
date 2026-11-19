import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION", "acme_policies")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=CHROMA_DIR)

def load_pdfs():
    docs = []
    for pdf in DOCS_DIR.glob("*.pdf"):
        docs.extend(PyMuPDF4LLMLoader(str(pdf)).load())
    return docs

def split(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def chunk_id(chunk):
    key = str(chunk.metadata.get("source", "")) + str(chunk.metadata.get("page", "")) + chunk.page_content
    return hashlib.sha256(key.encode()).hexdigest()

def build_index(chunks):
    ids = [chunk_id(chunk) for chunk in chunks]
    sources = [chunk.metadata.get("source", "") for chunk in chunks]
    existing_ids = set(vectorstore.get(ids=ids, include=[])["ids"]) 
    new_chunks = []
    new_ids = []

    unique_sources = list(set(sources))
    stored_ids = vectorstore.get(where={"source": {"$in": unique_sources}}, include=[])["ids"]

    changed_chunk_ids = list(set(stored_ids) - existing_ids)
    if changed_chunk_ids:
        vectorstore.delete(ids=changed_chunk_ids)


    for chunk, id in zip(chunks, ids):
        if id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(id)

    if not new_chunks:
        print("No new docs to embed")
        return

    vectorstore.add_documents(documents=new_chunks, ids=new_ids)
    print(f"Embedded and stored {len(new_chunks)}")

