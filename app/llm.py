import os
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")
INPUT_PRICE = float(os.getenv("INPUT_TOKEN_COST", '0'))
OUTPUT_PRICE = float(os.getenv("OUTPUT_TOKEN_COST", '0'))
TEMPERATURE = float(os.getenv("TEMPERATURE", '0'))

llm = ChatGoogleGenerativeAI(model=MODEL, temperature=TEMPERATURE)

def ask(question, system_prompt):
    start = time.perf_counter()
    response = llm.invoke([("system", system_prompt), ("human", question)])
    time_consumed = time.perf_counter() - start
    if response.usage_metadata:
        input_usage = response.usage_metadata["input_tokens"]
        output_usage = response.usage_metadata["output_tokens"]
    else:
        input_usage = 0
        output_usage = 0

    cost = (input_usage / 1000000) * INPUT_PRICE + (output_usage / 1000000) * OUTPUT_PRICE

    return {"response": response.text, "input_tokens": input_usage, "output_tokens": output_usage, "cost": cost, "time": time_consumed}
