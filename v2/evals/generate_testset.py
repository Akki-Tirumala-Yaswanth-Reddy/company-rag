from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from ragas import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.testset_schema import Testset

from app.config import DOCS_DIR, converter, embeddings

# 1. One Markdown document per PDF: RAGAS does its own splitting, using the headings
docs = []
for pdf in sorted(Path(DOCS_DIR).glob("*.pdf")):
    md = converter.convert(pdf).document.export_to_markdown()
    docs.append(Document(page_content=md, metadata={"source": pdf.name}))

llm = LangchainLLMWrapper(ChatOllama(model="qwen3.5:9b", reasoning=False, temperature=0))
emb = LangchainEmbeddingsWrapper(embeddings)

# 2. Who is asking: makes the questions realistic
personas = [
    Persona(name="New joiner", role_description="Joined Acme recently; asks about leave, probation and first-month rules."),
    Persona(name="Frequent traveller", role_description="Travels for client work; asks about flights, hotels, per diem and expense claims."),
    Persona(name="People manager", role_description="Manages a team; asks about approvals, notice periods and team members' entitlements."),
    Persona(name="Security-conscious engineer", role_description="Handles code and customer data; asks about data classification, devices and access."),
]

# 3. What kinds of questions, and in what proportion
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.6),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.4),
]

generator = TestsetGenerator(llm=llm, embedding_model=emb, persona_list=personas)
testset = generator.generate_with_langchain_docs(
    docs, testset_size=12,
    query_distribution=query_distribution,
    run_config=RunConfig(max_workers=2, timeout=300, max_retries=3),
)
assert isinstance(testset, Testset)             # generate_* is typed Testset | Executor
df = testset.to_pandas()
df.to_csv("evals/synthetic_raw.csv", index=False)
pd.set_option("display.max_colwidth", 120)
print(df[["user_input", "reference", "synthesizer_name"]].to_string())