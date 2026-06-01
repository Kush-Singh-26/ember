import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import EmberForCausalLM

def compute_log_likelihood(model, tokenizer, context: str, continuation: str, device: str) -> float:
    """Computes the log-likelihood of the continuation given the context."""
    context_enc = tokenizer.encode(context, add_special_tokens=False)
    continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
    
    # Combined inputs
    input_ids = torch.tensor([context_enc + continuation_enc], dtype=torch.long, device=device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0] # [seq_len, vocab_size]
        
    # We only care about the logits predicting the continuation tokens
    # Shift logits by 1 to align with targets
    ctx_len = len(context_enc)
    cont_len = len(continuation_enc)
    
    cont_logits = logits[ctx_len - 1 : ctx_len + cont_len - 1] # [cont_len, vocab_size]
    cont_targets = torch.tensor(continuation_enc, dtype=torch.long, device=device) # [cont_len]
    
    # Calculate log softmax
    log_probs = F.log_softmax(cont_logits, dim=-1)
    
    # Gather the log probs of target tokens
    target_log_probs = log_probs[torch.arange(cont_len), cont_targets]
    
    # Sum or average of log likelihood
    return target_log_probs.sum().item()

def evaluate_hellaswag(model, tokenizer, device: str, num_samples: int = 100) -> float:
    """Evaluates the model on the HellaSwag common-sense reasoning benchmark."""
    print(f"Evaluating on HellaSwag ({num_samples} samples)...")
    from datasets import load_dataset
    try:
        ds = load_dataset("hellaswag", split="validation", streaming=True)
    except Exception as e:
        print(f"Failed to load HellaSwag: {e}")
        return 0.0
        
    correct = 0
    total = 0
    
    pbar = tqdm(total=num_samples)
    for item in ds:
        if total >= num_samples:
            break
            
        context = item["ctx_a"] + " " + item["ctx_b"]
        endings = item["endings"]
        label = int(item["label"]) # correct ending index (0-3)
        
        # Compute log-likelihood of all 4 endings
        log_likes = []
        for ending in endings:
            log_likes.append(compute_log_likelihood(model, tokenizer, context, ending, device))
            
        pred = log_likes.index(max(log_likes))
        if pred == label:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"HellaSwag Zero-Shot Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def evaluate_arc_easy(model, tokenizer, device: str, num_samples: int = 100) -> float:
    """Evaluates the model on the ARC-Easy science QA benchmark."""
    print(f"Evaluating on ARC-Easy ({num_samples} samples)...")
    from datasets import load_dataset
    try:
        ds = load_dataset("ai2_arc", "ARC-Easy", split="validation", streaming=True)
    except Exception as e:
        print(f"Failed to load ARC-Easy: {e}")
        return 0.0
        
    correct = 0
    total = 0
    
    pbar = tqdm(total=num_samples)
    for item in ds:
        if total >= num_samples:
            break
            
        question = item["question"]
        choices = item["choices"]
        text_choices = choices["text"]
        label_choices = choices["label"] # e.g. ["A", "B", "C", "D"]
        answer_key = item["answerKey"]
        
        # Check if answerKey is in the choices list
        if answer_key not in label_choices:
            continue
            
        label = label_choices.index(answer_key)
        
        log_likes = []
        for choice_text in text_choices:
            # We append the answer to the question
            log_likes.append(compute_log_likelihood(model, tokenizer, question, choice_text, device))
            
        pred = log_likes.index(max(log_likes))
        if pred == label:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"ARC-Easy Zero-Shot Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def run_all_evaluations(model_path: str, num_samples: int = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running zero-shot benchmarks on device: {device}")
    
    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EmberForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    results = {}
    
    # 1. HellaSwag Reasoning
    results["hellaswag_accuracy"] = evaluate_hellaswag(model, tokenizer, device, num_samples=num_samples)
    
    # 2. ARC Science QA
    results["arc_easy_accuracy"] = evaluate_arc_easy(model, tokenizer, device, num_samples=num_samples)
    
    # Save results
    output_file = os.path.join(model_path, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Evaluation results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Ember model on benchmarks")
    parser.add_argument("--model", type=str, default="./outputs/final", help="Path to pre-trained model directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of evaluation samples to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model directory {args.model} does not exist.")
        sys.exit(1)
        
    run_all_evaluations(model_path=args.model, num_samples=args.samples)
