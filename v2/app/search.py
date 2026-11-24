from pathlib import Path

from langchain_milvus import Milvus

from app.config import COLLECTION, MILVUS_URI, embeddings

store = Milvus(embeddings, collection_name=COLLECTION, connection_args={"uri": MILVUS_URI})

for question in [
    "Which leave types are unpaid?",                           
    "When does the higher annual leave accrual start?",        
    "What is the hotel limit per night in Hyderabad?",       
]:
    print(f"\nQ: {question}")
    for d in store.similarity_search(question, k=3):
        print(f"  [{Path(d.metadata['source']).name}] {d.page_content}")