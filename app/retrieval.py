import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.ingest import vectorstore

STOPWORDS = {"do", "i", "what", "should", "if", "my", "how", "many", "the", "a", "an",
             "is", "are", "of", "to", "in", "for", "and", "or", "can", "get"}

def _tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    words = [word for word in words if word not in STOPWORDS]
    return words

def build_bm25():
    global docs, bm25
    store = vectorstore.get(include=["documents", "metadatas"])
    docs = [Document(page_content=page_content, metadata=metadata) 
            for page_content, metadata in zip(store["documents"], store["metadatas"])]
    corpus = [_tokenize(doc.page_content) for doc in docs]
    # BM25Okapi divides by corpus size, so an empty index must be left as None
    bm25 = BM25Okapi(corpus=corpus) if corpus else None
build_bm25()

def keyword_search(query: str, k: int) -> list[Document]:
    if bm25 is None:
        return []
    scores = bm25.get_scores(query=_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top if scores[i] > 0]

def vector_search(query: str, k: int) -> list[Document]:
    return vectorstore.similarity_search(query=query, k=k)

def rrf(ranked_lists: list[list[Document]], k: int = 60) -> list[Document]:
    scores: dict[tuple, float] = {}
    docs_by_key: dict[tuple, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank + 1)
            docs_by_key[key] = doc

    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ordered]


def hybrid_search(query: str, k: int = 4, candidates: int = 10) -> list[Document]:
    return rrf([vector_search(query, candidates), keyword_search(query, candidates)])[:k]
