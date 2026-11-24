import json
import time
from pathlib import Path

from app.answer import FALLBACK, answer
from app.config import USE_EMBEDDING_MODEL, USE_MODEL

testset = json.loads(Path("evals/testset.json").read_text())
rows = []
for item in testset:
    r = answer(item["question"])
    if USE_EMBEDDING_MODEL == "GOOGLE" or USE_MODEL == "GOOGLE":
        time.sleep(2)
    rows.append({"user_input": item["question"], "response": r["answer"],
                 "retrieved_contexts": r["contexts"], "reference": item["reference"]})
Path("evals/answers-gemini.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

refusals = [r for r in rows if r["reference"] == FALLBACK]
ok = sum(r["response"].strip() == FALLBACK for r in refusals)
print(f"Saved {len(rows)} answers. Unanswerable handled correctly: {ok}/{len(refusals)}")