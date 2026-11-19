import logging
import os
import secrets
import time
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, StringConstraints

from app.agent import graph
from app.llm import INPUT_PRICE, MODEL, OUTPUT_PRICE

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant")
APP_API_KEY = os.getenv("APP_API_KEY")

app = FastAPI(title="Acme Enterprise Knowledge Assistant", version="1.0")


class AskRequest(BaseModel):
    # TODO 2: choose sensible limits. Think: what's the shortest real question?
    # Why cap the length at all? (hint: tokens = money, and prompt-injection payloads are long)
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class Metrics(BaseModel):
    latency_ms: float
    input_tokens: int
    output_tokens: int
    est_cost_usd: float


class AskResponse(BaseModel):
    answer: str
    route: str
    sources: list[str]
    metrics: Metrics


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI runs this BEFORE the endpoint (it's a 'dependency').
    Header(...) reads the 'X-API-Key' request header into x_api_key.
    TODO 1:
      - if the server has no APP_API_KEY configured → fail CLOSED (which status code?)
      - if the header is missing or wrong → 401
      - compare with secrets.compare_digest, not ==
    Raise HTTPException(status_code=..., detail="...") to reject.
    """
    key_server = os.getenv("APP_API_KEY")
    if key_server is None:
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="X-API-KEY is missing.")
    if not secrets.compare_digest(key_server, x_api_key):
        raise HTTPException(status_code=401, detail="X-API-KEY is wrong")


@app.get("/health")
def health() -> dict:
    """No auth: Cloud Run and uptime checkers call this to see if the app is alive."""
    return {"status": "ok", "model": MODEL}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask_endpoint(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    try:
        out = graph.invoke({"question": req.question})
    except Exception as e:
        logger.exception("Agent failed")                    # full details go to the server log only
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(status_code=503, detail="AI service is busy. Please retry in a minute.")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")

    latency_ms = (time.perf_counter() - start) * 1000
    tin, tout = out.get("input_tokens", 0), out.get("output_tokens", 0)
    cost = tin / 1_000_000 * INPUT_PRICE + tout / 1_000_000 * OUTPUT_PRICE
    logger.info("route=%s latency_ms=%.0f tokens_in=%d tokens_out=%d cost_usd=%.6f",
                out.get("route"), latency_ms, tin, tout, cost)
    return AskResponse(
        answer=out["answer"], route=out["route"], sources=out.get("sources", []),
        metrics=Metrics(latency_ms=round(latency_ms, 1), input_tokens=tin,
                        output_tokens=tout, est_cost_usd=round(cost, 8)),
    )