"""Evaluate a model (base or fine-tuned) on the SQL test set.

Metric: exact-match after normalization (case-insensitive, whitespace-collapsed).
This is a strict metric — semantically equivalent but differently-formatted SQL
will count as wrong — but it's simple, deterministic, and good enough to show
relative improvement from fine-tuning.

Usage:
  python3 evaluate.py                          # evaluate base model
  python3 evaluate.py --adapter-path adapters   # evaluate base model + LoRA adapter
"""
import argparse
import json
import re

from mlx_lm import generate, load

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
TEST_FILE = "data/test.jsonl"


def normalize_sql(sql: str) -> str:
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";").strip()
    sql = sql.replace("'", '"')  # normalize quote style so it doesn't affect scoring
    return sql


def extract_prompt_and_answer(text: str) -> tuple[str, str]:
    """Split the training-format text into the prompt (everything up to 'SQL:')
    and the gold answer (everything after)."""
    prompt, answer = text.rsplit("SQL:", 1)
    return prompt + "SQL:", answer.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter, if evaluating a fine-tuned model")
    parser.add_argument("--limit", type=int, default=50, help="Number of test examples to evaluate")
    args = parser.parse_args()

    print(f"Loading model{' with adapter' if args.adapter_path else ' (base, no adapter)'}...")
    if args.adapter_path:
        model, tokenizer = load(MODEL, adapter_path=args.adapter_path)
    else:
        model, tokenizer = load(MODEL)

    with open(TEST_FILE) as f:
        examples = [json.loads(line) for line in f][: args.limit]

    correct = 0
    results = []

    for i, example in enumerate(examples, 1):
        prompt, gold_sql = extract_prompt_and_answer(example["text"])

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        generated = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=100, verbose=False)

               # Strip markdown code fences if present, then take the first non-empty line
        # as the SQL (models often wrap output in ```sql ... ``` even when asked not to).
        cleaned = generated.strip()
        cleaned = re.sub(r"^```(?:sql)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```.*$", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"^sql:\s*", "", cleaned, flags=re.IGNORECASE)
        generated_sql = " ".join(
            line.strip() for line in cleaned.strip().split("\n") if line.strip()
        )

        is_correct = normalize_sql(generated_sql) == normalize_sql(gold_sql)
        correct += is_correct

        results.append({
            "gold": gold_sql,
            "generated": generated_sql,
            "correct": is_correct,
        })
    
        print(f"[{i}/{len(examples)}] {'✓' if is_correct else '✗'}")
        if i <= 5:
            print(f"    GOLD:      {gold_sql}")
            print(f"    GENERATED: {generated_sql}")

    accuracy = correct / len(examples)
    print(f"\n=== Results ===")
    print(f"Exact-match accuracy: {correct}/{len(examples)} ({accuracy:.1%})")

    return accuracy, results


if __name__ == "__main__":
    main()