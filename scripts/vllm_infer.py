#!/usr/bin/env python3
"""Run vLLM inference on a partition of a JSONL file.

This script processes only the lines where (index % mod) == start_mod so
multiple copies can run in parallel on different GPUs. For each input record
it runs `repeats` independent generations and extracts the text between
<final>...</final> into answer1..answer{repeats} fields, then writes the
augmented record to the output JSONL.

Note: This script expects `vllm` to be installed and available as a Python
package. Adjust the `generate` call if your installed vLLM API differs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None


FINAL_RE = re.compile(r"<final>(.*?)</final>", re.S)


def extract_final(text: str) -> str:
    m = FINAL_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def run_generation(llm: "LLM", prompt: str, repeats: int = 1, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 1.0) -> list[str]:
    # Wrap generate call to be robust to small API differences.
    # Set n=repeats to let vLLM batch independent generations concurrently.
    params = SamplingParams(n=repeats, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    # vLLM generate yields RequestOutput objects.
    resp_list = llm.generate([prompt], sampling_params=params, use_tqdm=False)
    results = []
    if resp_list:
        resp = resp_list[0]
        if hasattr(resp, "outputs") and resp.outputs:
            for out in resp.outputs:
                results.append(out.text)
        elif hasattr(resp, "generations") and resp.generations:
            for gen in resp.generations:
                results.append(gen.text)
        else:
            results.append(str(resp))
    
    # Pad to expected length just in case
    while len(results) < repeats:
        results.append("")
        
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to vLLM model")
    p.add_argument("--input", required=True, type=Path, help="Input JSONL file")
    p.add_argument("--output", required=True, type=Path, help="Output JSONL file for this worker")
    p.add_argument("--gpu-id", required=True, type=int, help="Local GPU id (for logging only)")
    p.add_argument("--start-mod", required=True, type=int, help="Start modulo index for partitioning (0-based)")
    p.add_argument("--mod", required=True, type=int, help="Number of partitions (e.g., 8)")
    p.add_argument("--repeats", type=int, default=8, help="Number of independent generations per record")
    p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for generation")
    p.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768")),
        help="Max model context length for vLLM initialization",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
        help="GPU memory utilization passed to vLLM",
    )
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p")
    args = p.parse_args()

    if LLM is None:
        print("vLLM package not available. Install vllm in your Python env.", file=sys.stderr)
        sys.exit(2)

    model_path = args.model
    in_path = args.input
    out_path = args.output

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create LLM instance using one device. Depending on your vLLM version you
    # may need to pass different args (e.g., ``device_ids=[0]``). The server
    # will typically detect available GPU(s) based on environment vars.
    # Set ``gpu_memory_utilization`` slightly lower to avoid OOM in parallel runs.
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
        )
    except Exception as e:
        print(f"Failed to initialize vLLM (maybe try reducing GPU memory util further): {e}")
        raise

    with in_path.open("r", encoding="utf-8") as inf, out_path.open("w", encoding="utf-8") as outf:
        for idx, line in enumerate(inf):
            if (idx % args.mod) != args.start_mod:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # preserve invalid lines
                continue

            # Build prompt: combine instruction and input
            instruction = rec.get("instruction", "").strip()
            input_text = rec.get("input", "").strip()
            
            if instruction and input_text:
                prompt_text = f"{instruction}\n\n{input_text}"
            else:
                prompt_text = instruction or input_text or json.dumps(rec, ensure_ascii=False)

            # Use vLLM's tokenizer to format correctly
            tokenizer = llm.get_tokenizer()
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are an expert in solving scientific problems."},
                    {"role": "user", "content": prompt_text}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"<|im_start|>system\nYou are an expert in solving scientific problems.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            print(f"[Worker {args.gpu_id}] Processing record index {idx}...")

            # For reproducibility you can set seeds or sampling params; here we
            # perform `repeats` independent calls.
            try:
                out_texts = run_generation(llm, prompt, repeats=args.repeats, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
            except Exception as e:
                print(f"[Worker {args.gpu_id}] Generation failed for index {idx}: {e}")
                out_texts = [""] * args.repeats
                
            for r, out_text in enumerate(out_texts, start=1):
                final = extract_final(out_text)
                rec[f"response{r}"] = out_text
                rec[f"answer{r}"] = final

            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outf.flush()

if __name__ == "__main__":
    main()
