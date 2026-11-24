import time
from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_milvus import Milvus

from app.config import (
    COLLECTION,
    DOCS_DIR,
    MILVUS_URI,
    USE_MODEL,
    chunker,
    converter,
    embeddings,
)

pdfs = [str(p) for p in sorted(Path(DOCS_DIR).glob("*.pdf"))]
docs = DoclingLoader(file_path=pdfs, converter=converter, chunker=chunker).load()
print(f"{len(docs)} chunks from {len(pdfs)} PDFs")

store = Milvus(
    embeddings,
    collection_name=COLLECTION,
    connection_args={"uri": MILVUS_URI},
    index_params={"index_type": "FLAT", "metric_type": "COSINE"}, # instead of flat we can use HNSW(greedy)
    drop_old=True,                      
)

if USE_MODEL == "QWEN":
    print("Using qwen's embedding model")
    store.add_documents(docs)
else:
    print("Using qwen google's model")
    BATCH, PAUSE = 50, 60                    
    for i in range(0, len(docs), BATCH):
        store.add_documents(docs[i:i + BATCH])
        print(f"Stored {min(i + BATCH, len(docs))}/{len(docs)}")
        if i + BATCH < len(docs):
            time.sleep(PAUSE)
print("Stored in Milvus")