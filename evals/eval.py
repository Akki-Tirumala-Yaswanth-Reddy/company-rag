"""Run the eval set against the deployed API and print accuracy, latency and cost.

Usage:
    export EVAL_URL=https://your-service.run.app
    export APP_API_KEY=your-key
    python evals/run_eval.py
"""
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

URL = "http://127.0.0.1:8080" # os.environ["URL"].rstrip("/")
API_KEY = os.environ["APP_API_KEY"]
FALLBACK = "I couldn't find this in Acme's policies"   # prefix of app.rag.FALLBACK
PAUSE_S = 5                                           # stay under the Gemini free-tier rate limit
CASES = json.loads(Path("evals/questions.json").read_text())


def call_api(question: str, retries: int = 2) -> tuple[dict | None, float]:
    """POST /ask. Returns (json_body or None, client-side latency in ms). Retries on 503 (rate limit)."""
    for attempt in range(retries + 1):
        start = time.perf_counter()
        r = requests.post(f"{URL}/ask", json={"question": question},
                          headers={"X-API-Key": API_KEY}, timeout=90)
        ms = (time.perf_counter() - start) * 1000
        if r.status_code == 503 and attempt < retries:
            time.sleep(30)
            continue
        return (r.json() if r.ok else None), ms
    return None, 0.0


def grade(case: dict, body: dict | None) -> tuple[bool, str]:
    """Decide whether the API's answer passes this test case.
    Returns (passed, reason). The reason is printed for failures, so make it useful.

    TODO: implement these checks (only the ones present in `case` apply):
      - body is None                         -> fail ("request failed")
      - case["expect_fallback"] is True      -> pass only if the answer starts with FALLBACK
      - case["must_include"]                 -> a list of GROUPS; every group must match.
                                                A group matches if ANY of its strings appears
                                                in the answer (case-insensitive).
                                                e.g. [["Vikram"], ["Sneha"], ["Rohan"]] = all three names
                                                     [["2,000", "2000"]]                = either spelling
      - case["must_not_include"]             -> fail if ANY of these strings appears (data leak!)
    """
    if body is None:
        return (False, "Body is empty")
    elif case.get("expected_route", None) and case["expected_route"] != body["route"]:
        return (False, f"Expected to take {case["expected_route"]} route but instead took {body["route"]}")
    elif case.get("expect_fallback", False) and FALLBACK not in body["answer"]:
        return (False, f"Expected fallback but instead got {body["answer"]}")
    answer = body["answer"].lower()
    for group in case.get("must_include", []):
        if not any(option.lower() in answer for option in group):
            return (False, f"Expected one of {group} but got {body['answer']}")
    for banned in case.get("must_not_include", []):
        if banned.lower() in answer:
            return (False, f"{banned!r} must not be included but got {body['answer']}")
    return (True, "Correct")

def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


if __name__ == "__main__":
    results = []
    for case in CASES:
        body, client_ms = call_api(case["question"])
        passed, reason = grade(case, body)
        route_ok = body is not None and case.get("expected_route") in (None, body["route"])
        source_ok = (case.get("expected_source") is None or
                     (body is not None and any(case["expected_source"] in s for s in body["sources"])))
        m = (body or {}).get("metrics", {})
        results.append({**case, "passed": passed, "reason": reason, "route_ok": route_ok,
                        "source_ok": source_ok, "client_ms": client_ms,
                        "answer": (body or {}).get("answer"), "route": (body or {}).get("route"),
                        "server_ms": m.get("latency_ms"), "input_tokens": m.get("input_tokens", 0),
                        "output_tokens": m.get("output_tokens", 0), "cost": m.get("est_cost_usd", 0.0)})
        print(f"{'✅' if passed else '❌'} #{case['id']:>2} [{case['category']:8}] "
              f"{client_ms/1000:5.1f}s  route={(body or {}).get('route')}  {'' if passed else reason}")
        time.sleep(PAUSE_S)

    ok = [r for r in results if r["server_ms"] is not None]
    lat = [r["client_ms"] for r in ok]
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])

    print("\n================ SUMMARY ================")
    print(f"Answer accuracy:   {sum(r['passed'] for r in results)}/{len(results)} "
          f"({100 * sum(r['passed'] for r in results) / len(results):.0f}%)")
    for cat, vals in by_cat.items():
        print(f"   {cat:9} {sum(vals)}/{len(vals)}")
    routed = [r for r in results if r.get("expected_route")]
    print(f"Routing accuracy:  {sum(r['route_ok'] for r in routed)}/{len(routed)}")
    sourced = [r for r in results if r.get("expected_source")]
    print(f"Source hit rate:   {sum(r['source_ok'] for r in sourced)}/{len(sourced)}")
    if lat:
        print(f"Latency p50 / p95: {percentile(lat, 50)/1000:.1f}s / {percentile(lat, 95)/1000:.1f}s")
        print(f"Avg tokens in/out: {statistics.mean(r['input_tokens'] for r in ok):.0f} / "
              f"{statistics.mean(r['output_tokens'] for r in ok):.0f}")
        print(f"Avg cost/request:  ${statistics.mean(r['cost'] for r in ok):.6f}  "
              f"(≈ ${1000 * statistics.mean(r['cost'] for r in ok):.2f} per 1,000 questions)")

    Path("evals/results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\nFull results saved to evals/results.json")