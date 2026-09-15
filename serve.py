"""FastAPI service serving the fine-tuned text-to-SQL model (base model + LoRA adapter).

Run: uvicorn serve:app --port 8000
"""
import re

from fastapi import FastAPI
from mlx_lm import generate, load
from pydantic import BaseModel

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER_PATH = "adapters"

PROMPT_TEMPLATE = """Given the following database schema, write a SQL query to answer the question.

Schema: {schema}
Question: {question}
SQL:"""

app = FastAPI(title="Text-to-SQL Fine-Tuned Model Service")

print("Loading model + LoRA adapter...")
model, tokenizer = load(MODEL, adapter_path=ADAPTER_PATH)
print("Model loaded and ready.")


class SQLRequest(BaseModel):
    db_schema: str
    question: str


class SQLResponse(BaseModel):
    db_schema: str
    question: str
    sql: str


def clean_generated_sql(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```.*$", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^sql:\s*", "", cleaned, flags=re.IGNORECASE)
    return " ".join(line.strip() for line in cleaned.strip().split("\n") if line.strip())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL, "adapter": ADAPTER_PATH}


@app.post("/generate", response_model=SQLResponse)
async def generate_sql(request: SQLRequest) -> SQLResponse:
    prompt = PROMPT_TEMPLATE.format(schema=request.db_schema, question=request.question)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    raw_output = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=100, verbose=False)
    sql = clean_generated_sql(raw_output)

    return SQLResponse(db_schema=request.db_schema, question=request.question, sql=sql)