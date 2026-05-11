import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
REWARD_FN_DIR = REPO_ROOT / "examples" / "grpo_trainer"
if str(REWARD_FN_DIR) not in sys.path:
    sys.path.append(str(REWARD_FN_DIR))

from qwen3_0p6b_em_reward import compute_score, extract_boxed_content  # noqa: E402


DEFAULT_MODEL_PATH = "/mnt/oss/zwl/checkpoints/qwen3_4b_sft_mixed/global_step_124/huggingface"
DEFAULT_DATA_PATH = "/mnt/data/zwl/verl/data/test/aime_2023.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/data/zwl/verl/data/output"
DEFAULT_USER_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AIME 2023 mean@64 for a trained model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the HF model directory.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the AIME 2023 jsonl file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs.")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of samples per question.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum generated tokens per sample.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization for vLLM.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def get_problem(item: dict) -> str:
    for key in ("prompt", "problem", "question", "raw_problem"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Cannot find problem text in item keys: {list(item.keys())}")


def get_ground_truth(item: dict) -> str:
    reward_model = item.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is not None and str(ground_truth).strip():
            return str(ground_truth).strip()

    for key in ("ground_truth", "answer", "gold_answer", "final_answer"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    raise ValueError(f"Cannot find ground truth in item keys: {list(item.keys())}")


def build_prompt(item: dict, tokenizer: AutoTokenizer) -> str:
    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raw_prompt = item.get("prompt")
    if isinstance(raw_prompt, list) and raw_prompt:
        return tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt.strip()

    problem = get_problem(item)
    user_prompt = problem.strip()
    if DEFAULT_USER_SUFFIX.strip() not in user_prompt:
        user_prompt += DEFAULT_USER_SUFFIX

    constructed_messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(constructed_messages, tokenize=False, add_generation_prompt=True)


def extract_prediction(response_text: str) -> str:
    boxed = extract_boxed_content(response_text)
    if boxed:
        return boxed[-1]

    stripped = response_text.strip()
    if not stripped:
        return ""

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return lines[-1] if lines else stripped


def normalize_stop_token_ids(tokenizer: AutoTokenizer) -> list[int] | None:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id
    return None


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    details_path = os.path.join(args.output_dir, "aime23_mean64_details.jsonl")
    summary_path = os.path.join(args.output_dir, "aime23_mean64_summary.json")

    print(f"Loading dataset from: {args.data_path}")
    dataset = load_jsonl(args.data_path)
    if not dataset:
        raise ValueError(f"Dataset is empty: {args.data_path}")
    print(f"Loaded {len(dataset)} questions.")

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts = [build_prompt(item, tokenizer) for item in dataset]
    stop_token_ids = normalize_stop_token_ids(tokenizer)

    print(f"Loading model from: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=stop_token_ids,
    )

    print(
        f"Generating {args.n_samples} samples per problem "
        f"for {len(prompts)} AIME 2023 questions..."
    )
    outputs = llm.generate(prompts, sampling_params)

    total_questions = len(outputs)
    pass_at_n_count = 0
    mean_at_n_sum = 0.0
    all_results = []

    for item, prompt, output in tqdm(
        list(zip(dataset, prompts, outputs, strict=True)),
        total=total_questions,
        desc="Scoring responses",
    ):
        ground_truth = get_ground_truth(item)
        problem = get_problem(item)
        source = item.get("source", "unknown")

        responses = [candidate.text for candidate in output.outputs]
        sample_scores = []
        sample_predictions = []

        for response in responses:
            score = float(compute_score(solution_str=response, ground_truth=ground_truth))
            pred = extract_prediction(response)
            sample_scores.append(score)
            sample_predictions.append(pred)

        correct_flags = [score >= 0.98 for score in sample_scores]
        correct_count = sum(correct_flags)
        any_correct = correct_count > 0
        mean_at_n = correct_count / max(len(sample_scores), 1)

        if any_correct:
            pass_at_n_count += 1
        mean_at_n_sum += mean_at_n

        all_results.append(
            {
                "source": source,
                "problem": problem,
                "ground_truth": ground_truth,
                "prompt": prompt,
                "responses": responses,
                "predictions": sample_predictions,
                "scores": sample_scores,
                "correct_flags": correct_flags,
                "correct_count": correct_count,
                f"mean@{args.n_samples}": mean_at_n,
                f"pass@{args.n_samples}": any_correct,
            }
        )

    mean_at_n_pct = (mean_at_n_sum / total_questions) * 100
    pass_at_n_pct = (pass_at_n_count / total_questions) * 100

    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "num_questions": total_questions,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        f"mean@{args.n_samples}": mean_at_n_pct,
        f"pass@{args.n_samples}": pass_at_n_pct,
        "details_path": details_path,
    }

    with open(details_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Results saved to: {details_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"✅ mean@{args.n_samples}: {mean_at_n_pct:.2f}%")
    print(f"✅ pass@{args.n_samples}: {pass_at_n_pct:.2f}% ({pass_at_n_count}/{total_questions})")


if __name__ == "__main__":
    main()