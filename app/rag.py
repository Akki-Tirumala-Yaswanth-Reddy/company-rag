from pathlib import Path

from app.llm import ask
from app.retrieval import hybrid_search

TOP_K = 4

# TODO 2: the system prompt. It must make the model:
   
FALLBACK = "I couldn't find this in Acme's policies. Please contact hr@acme.example."

SYSTEM_PROMPT = """
    You are Acme Corp's internal policy assistant for employees.

    - Answer ONLY from the provided context (no outside knowledge)
    - Reply with "I can't help you, as I do not have the required information." when the context doesn't contain the answer.
    - Cite the chunk labels used, e.g. [1], [3]
    - Never mention 'context' or 'chunks'; just answer.
    - Include all relevant details such as deadlines, amounts and contacts.
"""


def format_context(docs: list) -> str:
    """Turn retrieved chunks into one string Gemini can read AND cite.
    TODO 1: number each chunk ([1], [2], ...) and include its file name + page.
    Example of ONE formatted chunk:
        [1] (Acme_Leave_Policy.pdf, p1)
        Employees with 5 or more years of completed service receive 24 days...
    """
    context = f"\n{"-"*8}\n".join([f"[{i+1}]\n{docs[i].page_content}" for i in range(len(docs))])
    return context
    

def answer_question(question: str) -> dict:
    docs = hybrid_search(question, k=TOP_K)
    context = format_context(docs)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    result = ask(user_msg, SYSTEM_PROMPT)
    if "can't help" in result["response"]:
        result["sources"] = " "
        result["response"] = FALLBACK
        return result
    result["sources"] = sorted({f"{Path(d.metadata['source']).name} (p{d.metadata['page'] + 1})" for d in docs})
    return result
