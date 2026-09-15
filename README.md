# Domain-Specific LLM Fine-Tuning & Serving Service

A small local model fine-tuned with LoRA to specialize in one narrow task — natural
language to SQL — then served through a FastAPI endpoint exposed over real HTTPS via
an ngrok tunnel. Everything runs on a single Mac, no GPU cloud costs involved.

The point of this project is the full loop: measurable baseline, a real fine-tuning run,
a rigorous before/after comparison, and an actual serving layer at the end — not just a
training script that outputs a loss curve and stops.

## Result

| | Exact-match accuracy (50 held-out test examples) |
|---|---|
| Base model (no fine-tuning) | 40.0% |
| After LoRA fine-tuning | **70.0%** |

+30 percentage points, from a LoRA adapter with **5.3M trainable parameters — 0.34% of
the 1.5B base model** — trained for 300 iterations on 1,800 examples.

## Stack

- **Base model**: `Qwen2.5-1.5B-Instruct` (4-bit quantized), via MLX
- **Fine-tuning**: LoRA, via `mlx-lm`'s built-in training command — no `bitsandbytes`,
  no CUDA; this runs natively on Apple Silicon through Metal
- **Dataset**: `b-mc2/sql-create-context` (2,000-example subset), reformatted into a
  consistent schema/question/SQL instruction prompt
- **Serving**: FastAPI, exposing a `/generate` endpoint
- **Public access**: ngrok tunnel (see note below on why, not a cloud deploy)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install mlx-lm datasets fastapi uvicorn
```

Prepare the dataset:
```bash
python3 prepare_dataset.py
```

## Usage

Evaluate the base model (no fine-tuning):
```bash
python3 evaluate.py --limit 50
```

Fine-tune:
```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data data --iters 300 --adapter-path adapters
```

Evaluate the fine-tuned model:
```bash
python3 evaluate.py --adapter-path adapters --limit 50
```

Run the demo (a few fresh, out-of-training-set examples):
```bash
python3 demo.py
```

Serve it:
```bash
uvicorn serve:app --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"db_schema": "CREATE TABLE employees (name VARCHAR, department VARCHAR, salary INT)", "question": "What is the name of the employee with the highest salary?"}'
```

## What actually went wrong along the way

**The evaluation script was broken before the model was.** The very first baseline run
came back at a suspicious 0% — turned out the extraction logic was only grabbing the
first line of the model's output, but the model wraps its answers in markdown code
fences and sometimes splits a single SQL statement across multiple lines. Fixed by
stripping the fences and joining all non-empty lines into one query. Worth calling out
specifically because a broken eval that reports 0% looks a lot like "the fine-tuning
didn't work," when the real problem was upstream of the model entirely.

**Exact-match is a strict metric, and quote style shouldn't count against it.**
Semantically identical queries were failing to match purely because the model used
single quotes and the gold data used double quotes. Normalizing quote style before
comparison fixed this — otherwise the accuracy numbers would have been measuring
formatting preference, not correctness.

**Fine-tuning taught the model an artifact of the training format.** After training,
the model started prefixing every answer with the literal string `"SQL:"` — because
that's exactly how every training example ends (`...Question: ...\nSQL: <query>`), so
the model learned that "SQL:" is part of what comes next, not a prompt boundary. This
tanked the post-training eval to 0% before it was diagnosed and fixed by stripping a
leading `SQL:` prefix from generated output. Worth noting as a real lesson: the exact
formatting of your training prompt becomes part of what the model learns to produce,
even parts you didn't intend as content.

**MLX and threaded web servers don't mix by default.** The FastAPI `/generate` endpoint
crashed with `RuntimeError: There is no Stream(cpu, 0) in current thread` — FastAPI runs
synchronous route handlers in a background threadpool, but MLX's compute stream is tied
to the thread it was initialized on. Making the endpoint `async def` instead of `def`
keeps it on the main thread and fixed it outright.

## Why ngrok instead of a cloud deployment

MLX only runs on Apple Silicon — it's built directly on Metal, Apple's GPU framework.
Standard cloud VMs (Render, Fly.io, AWS, etc.) run Linux on x86/ARM and simply can't
execute MLX code; this isn't a configuration problem, it's a hardware dependency. A real
cloud deployment would mean converting the fine-tuned weights to a Linux-portable format
(GGUF, served via `llama.cpp`) — a legitimately separate project, not a quick step.
ngrok gets a real public HTTPS endpoint in front of the actual MLX-served model without
pretending the hardware constraint doesn't exist. Note that free-tier ngrok URLs are
ephemeral (they change on every restart) and sessions time out — this is a demo
mechanism, not a persistent deployment.

## Known limitations

- 40%→70% is measured on a 50-example held-out set from the same dataset distribution as
  training — a more rigorous eval would test on a genuinely different SQL dataset to
  measure generalization beyond this specific data source's style and schema patterns.
- Exact-match scoring is strict: a semantically correct query with a different (but
  valid) approach — e.g. a subquery instead of `ORDER BY ... LIMIT 1` — would be marked
  wrong even though it produces the same result. A production eval would ideally execute
  both queries against a real database and compare result sets instead.
- The LoRA adapter is specialized narrowly for this schema-question-SQL prompt format;
  it wasn't tested on the base model's general instruction-following ability, so there's
  no measurement of whether fine-tuning degraded performance on unrelated tasks.