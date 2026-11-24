import re
import sqlite3

from app.llm import ask

DB_PATH = "data/acme.db"
MAX_ROWS = 50

# The schema the LLM sees. salary_inr is deliberately NOT listed.
SCHEMA = """Table employees(
  id INTEGER,
  name TEXT,
  email TEXT,
  department TEXT,   -- one of: Engineering, Sales, HR, Finance, IT level TEXT,        -- 'L3' (junior) to 'L7' (senior)
  location TEXT,     -- e.g. Bengaluru, Hyderabad, Mumbai, Delhi NCR
  hire_date TEXT,    -- 'YYYY-MM-DD'
  manager_id INTEGER -- references employees.id
)"""

SQL_PROMPT = f"""You translate employee questions into ONE SQLite SELECT query.

{SCHEMA}

Rules:
- Output ONLY the SQL. No explanation, no markdown.
- Always list columns explicitly. Never use SELECT *.
- Years of service = (julianday('now') - julianday(hire_date)) / 365.25
- Match names with LIKE '%...%' so partial names work.
- If the request names an employee, ALWAYS return that employee's row, even if other parts of the request are not about the table.
- Otherwise, if the request cannot be answered from this table, output exactly: NONE

"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)   # layer 1: read-only

    def authorizer(action, arg1, arg2, db_name, source):               # layer 2: column guard
        if action == sqlite3.SQLITE_READ and arg2 == "salary_inr":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorizer)
    return conn


def clean_sql(text: str) -> str:
    """LLMs often wrap SQL in ```sql fences despite instructions. Strip them."""
    return re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()


def validate_sql(sql: str) -> None:
    # A 3rd layer: fails fast with a clear message, and still protects us
    # if someone later opens the database read-write by mistake.
    words = re.findall(r"[A-Za-z_]+", sql.upper())
    if not words or words[0] not in {"SELECT"}:
        raise ValueError("Only SELECT queries are allowed.")
    not_allowed = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                 "REPLACE", "PRAGMA", "ATTACH", "DETACH", "VACUUM"}
    if bad := not_allowed & set(words):
        raise ValueError(f"The following commands are not allowed: {', '.join(sorted(bad))}")


def run_sql(sql: str) -> dict:
    with _connect() as conn:
        cursor = conn.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchmany(MAX_ROWS)                                  # never return unbounded rows
    return {"columns": columns, "rows": rows}


def text_to_sql(question: str) -> dict:
    """Question → SQL → rows. Returns the SQL, the data, and LLM metrics."""
    result = ask(question, SQL_PROMPT)
    sql = clean_sql(result["response"])
    out = {"sql": None, "columns": [], "rows": [], "error": None,
           "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"]}
    if sql.upper() == "NONE":
        return out
    out["sql"] = sql
    try:
        validate_sql(sql)
        out.update(run_sql(sql))
    except (ValueError, sqlite3.Error) as e:
        out["error"] = str(e)
    return out
