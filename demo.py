"""Quick demo: run a handful of schema/question pairs through the fine-tuned
model directly (no server needed).

Run: python3 demo.py
"""
import re

from mlx_lm import generate, load

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER_PATH = "adapters"

PROMPT_TEMPLATE = """Given the following database schema, write a SQL query to answer the question.

Schema: {schema}
Question: {question}
SQL:"""

DEMO_EXAMPLES = [
    {
        "schema": "CREATE TABLE employees (name VARCHAR, department VARCHAR, salary INT)",
        "question": "What is the name of the employee with the highest salary?",
    },
    {
        "schema": "CREATE TABLE orders (order_id INT, customer_name VARCHAR, total FLOAT, status VARCHAR)",
        "question": "How many orders have a status of 'shipped'?",
    },
    {
        "schema": "CREATE TABLE movies (title VARCHAR, genre VARCHAR, release_year INT, rating FLOAT)",
        "question": "List the titles of movies released after 2010 with a rating above 8.0",
    },
]


def clean_generated_sql(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```.*$", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^sql:\s*", "", cleaned, flags=re.IGNORECASE)
    return " ".join(line.strip() for line in cleaned.strip().split("\n") if line.strip())


def main():
    print("Loading base model + LoRA adapter...\n")
    model, tokenizer = load(MODEL, adapter_path=ADAPTER_PATH)

    for i, example in enumerate(DEMO_EXAMPLES, 1):
        prompt = PROMPT_TEMPLATE.format(schema=example["schema"], question=example["question"])
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        raw_output = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=100, verbose=False)
        sql = clean_generated_sql(raw_output)

        print(f"[{i}/{len(DEMO_EXAMPLES)}]")
        print(f"  Schema:   {example['schema']}")
        print(f"  Question: {example['question']}")
        print(f"  SQL:      {sql}\n")


if __name__ == "__main__":
    main()