import os
import sys
import json
import math
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from safetensors.torch import load_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import EmberForCausalLM, EmberConfig


# ─── Core Utilities ──────────────────────────────────────────────────────────

def load_model(model_path: str, device: str):
    print(f"Loading model from {model_path}...")

    # Load tokenizer: try model_path first, then local tokenizer_output, then HF hub
    tokenizer = None
    local_tok = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer_output")
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "tokenizer.json")):
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    elif os.path.isdir(local_tok):
        tokenizer = AutoTokenizer.from_pretrained(local_tok)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Kush26/ember-tokenizer")

    # Manual loading to avoid HF 5.x tied-weight corruption
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model.safetensors")):
        local_model_path = model_path
    else:
        from huggingface_hub import snapshot_download
        local_model_path = snapshot_download(repo_id=model_path)

    config = EmberConfig.from_pretrained(local_model_path)
    model = EmberForCausalLM(config)
    state_dict = load_file(os.path.join(local_model_path, "model.safetensors"))
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
    model.load_state_dict(state_dict, strict=True)
    model.float().to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")
    return model, tokenizer


@torch.no_grad()
def compute_log_likelihood(model, tokenizer, context: str, continuation: str, device: str) -> float:
    context_enc = tokenizer.encode(context, add_special_tokens=False)
    continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
    input_ids = torch.tensor([context_enc + continuation_enc], dtype=torch.long, device=device)

    outputs = model(input_ids)
    logits = outputs.logits[0]

    ctx_len = len(context_enc)
    cont_len = len(continuation_enc)
    cont_logits = logits[ctx_len - 1 : ctx_len + cont_len - 1]
    cont_targets = torch.tensor(continuation_enc, dtype=torch.long, device=device)

    log_probs = F.log_softmax(cont_logits, dim=-1)
    target_log_probs = log_probs[torch.arange(cont_len, device=device), cont_targets]
    return target_log_probs.sum().item()


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    max_ctx = getattr(model.config, "max_position_embeddings", 2048)

    # Truncate text to fit in one forward pass (~4 chars per token)
    max_chars = max_ctx * 4
    if len(text) > max_chars:
        text = text[:max_chars]

    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    targets = input_ids.clone()
    targets[:, 0] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=targets)
    return math.exp(outputs.loss.item())


def format_table(results: dict) -> str:
    lines = []
    lines.append(f"{'Benchmark':<25} {'Metric':<15} {'Score':>10}")
    lines.append("-" * 52)
    for name, data in results.items():
        if isinstance(data, dict):
            for metric, score in data.items():
                if isinstance(score, float):
                    lines.append(f"{name:<25} {metric:<15} {score:>10.4f}")
                else:
                    lines.append(f"{name:<25} {metric:<15} {str(score):>10}")
        else:
            lines.append(f"{name:<25} {'':<15} {str(data):>10}")
    return "\n".join(lines)


# ─── Benchmarks ──────────────────────────────────────────────────────────────

def eval_hellaswag(model, tokenizer, device: str, num_samples: int) -> dict:
    print(f"\n[HellaSwag] Running {num_samples} samples...")
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)

    correct = 0
    total = 0
    pbar = tqdm(total=num_samples, desc="HellaSwag")
    for item in ds:
        if total >= num_samples:
            break
        context = item["ctx_a"] + " " + item["ctx_b"]
        endings = item["endings"]
        label = int(item["label"])

        log_likes = [compute_log_likelihood(model, tokenizer, context, e, device) for e in endings]
        pred = log_likes.index(max(log_likes))
        if pred == label:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()

    acc = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_arc(model, tokenizer, device: str, num_samples: int, split_name: str) -> dict:
    print(f"\n[{split_name}] Running {num_samples} samples...")
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", split_name, split="validation", streaming=True)

    correct = 0
    total = 0
    pbar = tqdm(total=num_samples, desc=split_name)
    for item in ds:
        if total >= num_samples:
            break
        question = item["question"]
        choices = item["choices"]
        text_choices = choices["text"]
        label_choices = choices["label"]
        answer_key = item["answerKey"]

        if answer_key not in label_choices:
            continue
        label = label_choices.index(answer_key)

        log_likes = [compute_log_likelihood(model, tokenizer, question, c, device) for c in text_choices]
        pred = log_likes.index(max(log_likes))
        if pred == label:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()

    acc = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_winoGrande(model, tokenizer, device: str, num_samples: int) -> dict:
    print(f"\n[WinoGrande] Running {num_samples} samples...")
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", streaming=True)

    correct = 0
    total = 0
    pbar = tqdm(total=num_samples, desc="WinoGrande")
    for item in ds:
        if total >= num_samples:
            break
        sentence = item["sentence"]
        answer = item["answer"]  # "1" or "2"
        option1 = item["option1"]
        option2 = item["option2"]

        label = int(answer) - 1  # 0 or 1

        log_likes = [
            compute_log_likelihood(model, tokenizer, sentence, option1, device),
            compute_log_likelihood(model, tokenizer, sentence, option2, device),
        ]
        pred = log_likes.index(max(log_likes))
        if pred == label:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()

    acc = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}


def eval_wikitext2(model, tokenizer, device: str, num_samples: int) -> dict:
    print(f"\n[WikiText-2] Computing perplexity...")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    full_text = "\n".join(ds["text"])

    # Limit to ~20k chars (~5k tokens) for reasonable CPU runtime
    if len(full_text) > 20_000:
        full_text = full_text[:20_000]

    ppl = compute_perplexity(model, tokenizer, full_text, device)
    print(f"  Perplexity: {ppl:.4f}")
    return {"perplexity": ppl}


def eval_mbpp(model, tokenizer, device: str, num_samples: int) -> dict:
    print(f"\n[MBPP] Running {num_samples} samples (generation)...")
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test", streaming=True)

    correct = 0
    total = 0
    bleu_scores = []
    pbar = tqdm(total=num_samples, desc="MBPP")

    for item in ds:
        if total >= num_samples:
            break
        prompt = item["text"]
        test_code = "\n".join(item["test_list"])
        reference = item["code"]
        setup_code = item.get("test_setup_code", "")

        # Generate code
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Compute BLEU
        try:
            from sacrebleu import corpus_bleu
            bleu = corpus_bleu([generated], [[reference]])
            bleu_scores.append(bleu.score)
        except Exception:
            pass

        # Check if generated code passes the test case
        try:
            full_code = (setup_code + "\n" if setup_code else "") + generated + "\n" + test_code
            exec(full_code, {})
            correct += 1
        except Exception:
            pass

        total += 1
        pbar.update(1)
    pbar.close()

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    pass_rate = correct / total if total > 0 else 0.0
    print(f"  BLEU: {avg_bleu:.4f}")
    print(f"  Pass@1: {pass_rate:.4f} ({correct}/{total})")
    return {"bleu": avg_bleu, "pass_at_1": pass_rate, "correct": correct, "total": total}


# ─── Macro F1 ────────────────────────────────────────────────────────────────

def compute_macro_f1(results: dict) -> float:
    accuracies = []
    for name in ["hellaswag", "arc_easy", "arc_challenge", "winoGrande"]:
        if name in results and "accuracy" in results[name]:
            accuracies.append(results[name]["accuracy"])
    if not accuracies:
        return 0.0
    return sum(accuracies) / len(accuracies)


# ─── Main ────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "hellaswag": eval_hellaswag,
    "arc_easy": lambda m, t, d, n: eval_arc(m, t, d, n, "ARC-Easy"),
    "arc_challenge": lambda m, t, d, n: eval_arc(m, t, d, n, "ARC-Challenge"),
    "winoGrande": eval_winoGrande,
    "wikitext2": eval_wikitext2,
    "mbpp": eval_mbpp,
}


def main():
    parser = argparse.ArgumentParser(description="Ember-275M Evaluation Suite")
    parser.add_argument("--model", type=str, default="Kush26/ember", help="Model path or HF repo ID")
    parser.add_argument("--samples", type=int, default=500, help="Samples per benchmark")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated list of benchmarks to run (default: all)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(args.model, device)

    bench_list = list(BENCHMARKS.keys())
    if args.benchmarks:
        bench_list = [b.strip() for b in args.benchmarks.split(",")]

    results = {}
    for name in bench_list:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}, skipping")
            continue
        results[name] = BENCHMARKS[name](model, tokenizer, device, args.samples)

    # Macro F1 (average accuracy across multi-choice benchmarks)
    macro_f1 = compute_macro_f1(results)
    results["macro_f1"] = macro_f1

    # Print summary
    print("\n" + "=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(format_table(results))
    print("=" * 52)
    print(f"  Macro F1 (avg accuracy): {macro_f1:.4f}")

    # Save results
    output_file = os.path.join(args.model, "eval_results.json") if os.path.isdir(args.model) else "eval_results.json"
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {mk: mv for mk, mv in v.items()}
        else:
            serializable[k] = v

    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
