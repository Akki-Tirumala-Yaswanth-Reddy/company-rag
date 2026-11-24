import json
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.answer import FALLBACK
from app.config import JUDGE_LLM, embeddings

# # 1. Run YOUR system on every golden question, and record what RAGAS needs
# golden = json.loads(Path("evals/golden.json").read_text())
# rows = []
# for item in golden:
#     r = answer(item["question"])
#     rows.append({
#         "user_input": item["question"],          # RAGAS's names for the four fields
#         "response": r["answer"],
#         "retrieved_contexts": r["contexts"],
#         "reference": item["reference"],
#     })

rows = json.loads(Path("evals/answers-qwen.json").read_text())
rows = [r for r in rows if r["reference"] != FALLBACK] 
dataset = EvaluationDataset.from_list(rows)

# 2. The JUDGE: an LLM that grades the answers (can be a different model from the one being tested)
if JUDGE_LLM == "QWEN":
    llm_used_for_judging = ChatOllama(
        model="qwen3.5:9b",
        keep_alive="10m",
        reasoning=False,
        temperature=0,
    )
elif JUDGE_LLM == "GROQ":
    # Groq speaks the OpenAI API, so the OpenAI client works with a different base_url
    llm_used_for_judging = ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key=SecretStr(os.environ["GROQ_API_KEY"]),
        temperature=0,
    )
else:
    llm_used_for_judging = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
judge_llm = LangchainLLMWrapper(llm_used_for_judging)
judge_emb = LangchainEmbeddingsWrapper(embeddings)     # only ResponseRelevancy uses embeddings

# 3. Score. max_workers=1 = one request at a time, to stay inside the Gemini free-tier rate limit
result = evaluate(
    dataset=dataset,
    metrics=[LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness(), ResponseRelevancy(strictness=1)],
    llm=judge_llm,
    embeddings=judge_emb,
    run_config=RunConfig(max_workers=1, timeout=180, max_retries=5),
    raise_exceptions=True
)

assert isinstance(result, EvaluationResult)        # evaluate() is typed EvaluationResult | Executor
print(result)                                      # the average of each metric
df = result.to_pandas()
df.to_csv("evals/ragas_qwen_result.csv", index=False)   # per-question scores, for inspecting
print(df[["user_input", "llm_context_precision_with_reference", "context_recall",
          "faithfulness", "answer_relevancy"]].round(2).to_string())
