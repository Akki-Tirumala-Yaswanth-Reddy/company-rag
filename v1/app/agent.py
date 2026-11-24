import operator
from pathlib import Path
from typing import Annotated, Required, TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm import ask
from app.rag import FALLBACK, TOP_K, format_context
from app.retrieval import hybrid_search
from app.sql import text_to_sql


class AgentState(TypedDict, total=False):
    question: Required[str]                        # always supplied by the caller
    route: str                                     # "docs" | "sql" | "both"
    docs: list
    sql_result: dict
    answer: str
    context: str
    sources: list[str]
    input_tokens: Annotated[int, operator.add]     # reducer: every node's tokens get ADDED
    output_tokens: Annotated[int, operator.add]


# ---------------------------------------------------------------- TODO 1
ROUTER_PROMPT = """
    <rules>
    - You are a router who routes a request to two agents.
    - You must reply with exactly ONE word: docs, sql or both.
    - Reply "docs" when the request requires company policies or company policy PDFs.
    - employees sql table schema - (name, email, department, level, location, hire_date, manager).
    - Reply "sql" when the request requires the employees information or might need to use the information in the employees table.
    - Reply "both" when the request requires company policies and also need to use the information in the employees table.
    </rules>
    <examples>
        <example-1>
            <user>
            How many engineers work in Hyderabad?
            </user>
            <model>
            sql
            </model>
        </example-1>
        <example-2>
            <user>
            What is the company policy on remote work and home office stipends?
            </user>
            <model>
            docs
            </model>
        </example-2>
        <example-3>
            <user>
            Is Kavya Reddy eligible for a sabbatical based on her hire date?
            </user>
            <model>
            both
            </model>
        </example-3>
        <example-4>
            <user>
            Who reports to Kavya Reddy?
            </user>
            <model>
            sql
            </model>
        </example-4>
        <example-5>
            <user>
            What is the travel expense limit for L5 employees based in Hyderabad?
            </user>
            <model>
            both
            </model>
        </example-5>
        <example-6>
            <user>
            How many days of paternity leave do employees get?
            </user>
            <model>
            docs
            </model>
        </example-6>
        <example-7>
            <user>
            When did Priya join, and how many years has she worked here?
            </user>
            <model>
            sql
            </model>
        </example-7>
        <example-8>
            <user>
            Which employees in the Sales department have been with the company long enough to qualify for the annual bonus?
            </user>
            <model>
            both
            </model>
        </example-8>
    </examples>

"""
# Must make the model reply with exactly ONE word: docs, sql, or both.
# Tell it what each source contains:
#   docs = company policy PDFs (leave, travel, remote work, IT security, handbook)
#   sql  = employees table (name, email, department, level, location, hire_date, manager)
# Give 1 example per route. The "both" example matters most (e.g. a named person + a policy rule).

def route_question(state: AgentState) -> dict:
    r = ask(state["question"], ROUTER_PROMPT)
    # ------------------------------------------------------------ TODO 2
    # Turn r["response"] into a clean route. Models add spaces, capitals, full stops...
    # and sometimes say something unexpected. What should the SAFE DEFAULT be, and why?
    response = r["response"].strip().lower()
    if response not in ("docs", "sql", "both"):
        route = "both"
    else:
        route = response
    return {"route": route, "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}


def retrieve_docs(state: AgentState) -> dict:
    return {"docs": hybrid_search(state["question"], k=TOP_K)}


def query_db(state: AgentState) -> dict:
    question = state["question"]
    if state.get("route") == "both":
        question = ("Look up the employee(s) named in the text below and return their "
                    "name, level, location, hire_date and years of service. "
                    f"Ignore everything else in the text.\nText: {state['question']}")
    r = text_to_sql(question)
    return {"sql_result": r, "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}

def format_rows(columns: list[str], rows: list[tuple]) -> str:
    """Turn SQL results into one readable line per row, e.g.
    '- name: Vikram Singh | department: Sales | level: L4 | ...'
    TODO: ~3 lines. zip() each row with columns.
    """
    lines = []
    for row in rows:
        fields = [f"{c}: {r}" for c, r in zip(columns, row)]
        lines.append("- " + " | ".join(fields))
    return "\n".join(lines)

ANSWER_PROMPT = f"""You are Acme Corp's internal assistant for employees.
Answer using ONLY the policy excerpts and database results provided.
- Cite policy excerpts by their labels, e.g. [1]. Cite database facts as [DB].
- Combine sources when needed (e.g. a hire date from [DB] with a rule from a policy).
- Include relevant details such as deadlines, amounts and contacts.
- Never mention excerpts, context, databases or SQL to the user; just answer.
- If the information provided does not answer the question, reply with exactly: "{FALLBACK}" """


def synthesize(state: AgentState) -> dict:
    parts = []
    if state.get("docs"):
        parts.append("Policy excerpts:\n" + format_context(state.get("docs", [])))
    sql = state.get("sql_result") or {}
    if sql.get("rows"):
        parts.append(f"Database query executed: {sql["sql"]}")
        parts.append(f"Database results [DB]:\n{format_rows(sql["columns"], sql["rows"])}")
        parts.append(f"Use ")
    if not parts:
        return {"answer": FALLBACK, "sources": []}

    context = "\n\n".join(parts)
    r = ask(f"{context}\n\nQuestion: {state['question']}", ANSWER_PROMPT)
    answer = r["response"].strip()

    sources = sorted({f"{Path(d.metadata['source']).name} (p{d.metadata['page'] + 1})"
                      for d in state.get("docs", [])})
    if sql.get("rows"):
        sources.append("employees database")
    if answer == FALLBACK:
        sources = []
    return {"answer": answer, "context": context, "sources": sources,
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}


def pick_first_step(state: AgentState) -> str:
    return "query_db" if state.get("route", "both") == "sql" else "retrieve_docs"


def after_retrieve(state: AgentState) -> str:
    return "query_db" if state.get("route", "both") == "both" else "synthesize"


builder = StateGraph(AgentState)
builder.add_node("route_question", route_question)
builder.add_node("retrieve_docs", retrieve_docs)
builder.add_node("query_db", query_db)
builder.add_node("synthesize", synthesize)
builder.add_edge(START, "route_question")
builder.add_conditional_edges("route_question", pick_first_step, ["retrieve_docs", "query_db"])
builder.add_conditional_edges("retrieve_docs", after_retrieve, ["query_db", "synthesize"])
builder.add_edge("query_db", "synthesize")
builder.add_edge("synthesize", END)
graph = builder.compile()
