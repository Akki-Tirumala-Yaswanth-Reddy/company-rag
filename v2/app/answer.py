from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama

from app.config import COLLECTION, MILVUS_URI, USE_MODEL, embeddings

FALLBACK = "I couldn't find this in ACME's policies. Please contact hr@acme.example."

store = Milvus(embeddings, collection_name=COLLECTION, connection_args={"uri": MILVUS_URI})
retriever = store.as_retriever(search_kwargs={"k": 4})

if USE_MODEL == "GOOGLE":
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
else:
    llm = ChatOllama(
        model="qwen3.5:9b",
        keep_alive="10m",
        reasoning=False,
        temperature=0.1,
    )

PROMPT = f"""
- You are Acme Corp's internal policy assistant for employees.
- Answer ONLY from the numbered policy excerpts. Cite the ones you used, eg: [1], [3].
- Policy tables are shown in Markdown, read across the row to match the right column.
- Include relevant details such as amounts, deadlines and conditions.
- Never mention excerpts. Don't add a Sources section; cite inline only. Answer concisely.
- If the excerpts do not answer the question, reply with exactly: "{FALLBACK}"
- Do not mention about things like "tables, provided context" to the user.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT),
    ("human", "Policy excerpts:\n{context}\n\nQuestion: {question}"),
])

chain = prompt | llm


def describe(doc) -> str:
    """'Acme_Leave_Policy_Extended.pdf, 4. Annual Leave > 4.1 ..., p2' from Docling's metadata."""
    meta = doc.metadata.get("dl_meta", {})
    pages = sorted({p["page_no"] for item in meta.get("doc_items", []) for p in item.get("prov", [])})
    section = " > ".join(meta.get("headings") or [])
    return f"{Path(doc.metadata['source']).name}, {section}, p{pages[0] if pages else '?'}"


def answer(question: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{i}] ({describe(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))
    reply = chain.invoke({"context": context, "question": question, "fallback": FALLBACK})
    return {
        "answer": reply.text,
        "contexts": [d.page_content for d in docs], # RAGAS need the data to be like this
        "sources": [describe(d) for d in docs],
    }