# Acme Enterprise Knowledge Assistant

An internal Q&A assistant for a fictional company, **Acme Corp**. Employees ask questions in plain English and get cited answers drawn from two sources:

- **Policy documents** (PDFs: employee handbook, leave, IT security, remote work and travel), searched with hybrid RAG.
- **The employees database** (SQLite), queried with LLM-generated, read-only SQL.

A small LangGraph agent decides which source(s) a question needs, gathers the information and writes one answer with citations.

```text
"Is Kavya eligible for a sabbatical based on her hire date?"
  → route: both
  → hire date from the employees DB + sabbatical rule from the handbook
  → "Yes. Kavya joined on 2018-04-02 [DB], and employees with ... [2]"
```

---

## Tech stack

| Layer          | Tool                                                   |
| -------------- | ------------------------------------------------------ |
| API            | FastAPI + Uvicorn                                      |
| Orchestration  | LangGraph (`StateGraph`)                               |
| LLM            | Google Gemini via `langchain-google-genai`             |
| Embeddings     | `models/gemini-embedding-001`                          |
| Vector store   | Chroma (persisted on disk in `chroma_db/`)             |
| Keyword search | BM25 (`rank-bm25`)                                     |
| PDF parsing    | PyMuPDF4LLM                                            |
| Structured data| SQLite (`data/acme.db`)                                |
| Packaging      | `uv`, Docker (works on Cloud Run and AWS Lambda)       |

---

## Project structure

```text
app/
  main.py        FastAPI app: /health and /ask, API-key auth, metrics
  agent.py       LangGraph agent: router → retrieve/query → synthesize
  llm.py         Gemini chat wrapper (tracks tokens, cost and latency)
  ingest.py      PDF loading, chunking, embedding into Chroma
  retrieval.py   Hybrid search: vector + BM25, merged with RRF
  rag.py         Context formatting and a standalone docs-only RAG helper
  sql.py         Text-to-SQL with read-only safety layers
scripts/
  seed_db.py     Creates data/acme.db with sample employees
  ingest_docs.py Embeds the PDFs in data/docs into Chroma
data/
  docs/          Source policy PDFs
  acme.db        SQLite employees database
chroma_db/       Persisted vector index
Dockerfile, start.sh
```

---

## Setup

### Prerequisites

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/)
- A Google AI Studio API key for Gemini

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Then fill in `GOOGLE_API_KEY` and `APP_API_KEY` in `.env` (see [Environment variables](#environment-variables)).

### 3. Build the data (optional: both are already committed)

```bash
# Recreate the SQLite employees table
uv run python scripts/seed_db.py

# Embed the PDFs in data/docs into Chroma (only new or changed chunks are embedded)
uv run python -m scripts.ingest_docs
```

Re-run the ingest step whenever you add or edit a PDF in `data/docs/`, then restart the server: the BM25 index is built from Chroma when the app starts.

### 4. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8080
```

Interactive docs are at <http://localhost:8080/docs>.

### Run with Docker

```bash
docker run --env-file .env -p 8080:8080 atyaswanth/company-rag-aws
```

The image includes the seeded database and the committed `chroma_db/`. `start.sh` honours `$PORT` (Cloud Run) and, on AWS Lambda, copies the data to `/tmp` because that is the only writable directory there.

---

## Usage

### `GET /health`

No auth. Returns `{"status": "ok", "model": "<model name>"}`.

### `POST /ask`

Requires the `X-API-Key` header to match `APP_API_KEY`. The question must be 3 to 500 characters long.

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"question": "How many days of paternity leave do employees get?"}'
```

Response:

```json
{
  "answer": "Employees get ... days of paternity leave [1].",
  "route": "docs",
  "sources": ["Acme_Leave_Policy.pdf (p2)"],
  "metrics": {
    "latency_ms": 1834.2,
    "input_tokens": 1520,
    "output_tokens": 64,
    "est_cost_usd": 0.000616
  }
}
```

| Status | Meaning                                                   |
| ------ | --------------------------------------------------------- |
| 401    | `X-API-Key` header missing or wrong                       |
| 422    | Question too short or too long                            |
| 500    | Server has no `APP_API_KEY` set, or an unexpected error   |
| 503    | Gemini rate limit hit (429 / `RESOURCE_EXHAUSTED`); retry |

Example questions:

- **docs:** "What is the company policy on remote work and home office stipends?"
- **sql:** "How many engineers work in Hyderabad?" / "Who reports to Kavya Reddy?"
- **both:** "What is the travel expense limit for L5 employees based in Hyderabad?"

---

## Environment variables

| Variable            | Default                       | Description                                                       |
| ------------------- | ----------------------------- | ----------------------------------------------------------------- |
| `GOOGLE_API_KEY`    | (required)                    | Gemini API key, used for both chat and embeddings.                |
| `APP_API_KEY`       | (required)                    | Secret clients send in `X-API-Key`. If unset, `/ask` refuses every request. |
| `MODEL`             | `gemini-3.5-flash-lite`       | Gemini chat model.                                                |
| `EMBEDDING_MODEL`   | `models/gemini-embedding-001` | Embedding model. Changing it requires re-ingesting into a fresh `chroma_db/`. |
| `TEMPERATURE`       | `0`                           | LLM temperature. `0` keeps routing and SQL deterministic.          |
| `INPUT_TOKEN_COST`  | `0`                           | USD per 1M input tokens, used for `est_cost_usd`.                  |
| `OUTPUT_TOKEN_COST` | `0`                           | USD per 1M output tokens, used for `est_cost_usd`.                 |
| `DOCS_DIR`          | `data/docs`                   | Folder of PDFs to ingest.                                         |
| `CHROMA_DIR`        | `chroma_db`                   | Where the Chroma index is stored.                                 |
| `COLLECTION`        | `acme_policies`               | Chroma collection name.                                           |
| `CHUNK_SIZE`        | `400`                         | Characters per chunk.                                             |
| `CHUNK_OVERLAP`     | `150` (`.env.example`: `130`) | Overlap between neighbouring chunks.                              |
| `PORT`              | `8080`                        | Port used by `start.sh` in Docker.                                |

Never commit `.env`; it is already in `.gitignore` and `.dockerignore`.

---

## How it works

### 1. Ingestion pipeline (offline)

```text
PDFs ──► PyMuPDF4LLM loader ──► RecursiveCharacterTextSplitter ──► Gemini embeddings ──► Chroma
                                 (CHUNK_SIZE / CHUNK_OVERLAP)
```

- Each chunk gets a deterministic ID: `sha256(source + page + text)`.
- On re-runs, chunks whose ID already exists are skipped, so unchanged documents aren't re-embedded (no extra API cost).
- Stored chunks from the same files that no longer match are deleted, so edited PDFs don't leave stale text behind.

### 2. Query pipeline (the agent)

```text
                ┌──────────────────┐
  question ───► │  route_question  │  LLM replies: docs | sql | both
                └────────┬─────────┘  (anything else → "both", the safe default)
             docs / both │        │ sql
                         ▼        │
                ┌──────────────┐  │
                │ retrieve_docs│  │   hybrid search, top 4 chunks
                └──────┬───────┘  │
            docs │     │ both     │
                 │     ▼          ▼
                 │   ┌──────────────┐
                 │   │   query_db   │  text-to-SQL on employees
                 │   └──────┬───────┘
                 ▼          ▼
                ┌──────────────────┐
                │    synthesize    │  answer with [n] / [DB] citations
                └──────────────────┘
```

**Router.** A few-shot prompt makes Gemini reply with one word. The reply is cleaned up, and anything unexpected falls back to `both` so no relevant source is skipped.

**Hybrid retrieval** (`app/retrieval.py`):
- *Vector search* in Chroma finds chunks that are similar in meaning ("time off for a new baby" → paternity leave).
- *BM25 keyword search* finds exact terms such as "L5" or "VPN", which embeddings often blur together.
- The two ranked lists are merged with **Reciprocal Rank Fusion**: `score = Σ 1 / (60 + rank)`. A chunk ranked well by both methods comes out on top, and the raw scores of the two methods never need to be compared. The top 4 chunks are kept.

**Text-to-SQL** (`app/sql.py`): Gemini writes one SQLite `SELECT` from a schema description, then the query is run. For `both` questions, the question is first narrowed to "look up the named employee(s)" so the SQL step doesn't try to answer the policy part.

**Synthesis.** The numbered policy chunks and the DB rows are passed to Gemini with strict rules: answer only from what was provided, cite `[1]`, `[2]` or `[DB]`, and if the answer isn't there, reply with a fixed fallback message pointing to HR. Sources (file + page, and/or "employees database") are returned alongside the answer.

**Metrics.** Every LLM call reports its token usage. LangGraph adds them up across nodes (an `operator.add` reducer on the state), and the API returns latency, tokens and estimated cost with each answer.

### 3. Safety measures

- **API-key auth** on `/ask`, compared with `secrets.compare_digest` (constant-time), and fail-closed if the server has no key configured.
- **Input limits** (3–500 characters) to cap token cost and long prompt-injection payloads.
- **Three layers of SQL protection:**
  1. The database is opened **read-only** (`mode=ro`).
  2. A SQLite **authorizer** denies reads of the restricted `salary_inr` column, which is also left out of the schema the LLM sees.
  3. A **validator** rejects anything that isn't a `SELECT` or contains write/admin keywords.
- Results are capped at 50 rows.
- Errors are logged on the server only; clients get a generic message.
