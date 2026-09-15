"""Download b-mc2/sql-create-context and reformat it into MLX's expected JSONL
structure for LoRA fine-tuning: train.jsonl, valid.jsonl, test.jsonl, each line
a {"text": "..."} object with a consistent instruction-following format.
"""
import json
import os

from datasets import load_dataset

OUTPUT_DIR = "data"
PROMPT_TEMPLATE = """Given the following database schema, write a SQL query to answer the question.

Schema: {context}
Question: {question}
SQL: {answer}"""


def format_example(example: dict) -> dict:
    text = PROMPT_TEMPLATE.format(
        context=example["context"],
        question=example["question"],
        answer=example["answer"],
    )
    return {"text": text}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading b-mc2/sql-create-context...")
    dataset = load_dataset("b-mc2/sql-create-context")["train"]

    # Use a manageable subset for fast iteration — full dataset is ~78k examples,
    # which is more than needed to demonstrate fine-tuning on a small model quickly.
    dataset = dataset.shuffle(seed=42).select(range(2000))

    # 90/5/5 train/valid/test split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_valid = split["train"]
    test = split["test"]

    valid_test_split = test.train_test_split(test_size=0.5, seed=42)
    valid = valid_test_split["train"]
    test = valid_test_split["test"]

    for name, split_data in [("train", train_valid), ("valid", valid), ("test", test)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        with open(path, "w") as f:
            for example in split_data:
                formatted = format_example(example)
                f.write(json.dumps(formatted) + "\n")
        print(f"Wrote {len(split_data)} examples to {path}")


if __name__ == "__main__":
    main()