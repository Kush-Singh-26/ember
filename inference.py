import argparse
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from model import EmberForCausalLM


def load_model(model_path: str, device: str) -> tuple:
    config = EmberForCausalLM.config_class.from_pretrained(model_path)
    model = EmberForCausalLM(config)
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 120,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 3,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Ember model inference")
    parser.add_argument("--model-path", type=str, default="Kush26/ember",
                        help="Local path or HF Hub repo ID")
    parser.add_argument("--tokenizer", type=str, default="Kush26/ember-tokenizer",
                        help="Tokenizer path or HF Hub repo ID")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--do-sample", action="store_true",
                        help="Use sampling (nucleus) instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or auto")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Resolve local path vs HF Hub
    model_path = args.model_path
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download
        print(f"Downloading model from {model_path}...")
        local_dir = snapshot_download(repo_id=model_path)
        model_path = local_dir

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(model_path, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Interactive or single prompt mode
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {generate_text(model, tokenizer, args.prompt, device, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)}")
        return

    print("\nEmber Inference (type 'quit' to exit)")
    print("-" * 40)
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ("quit", "exit"):
            break
        if not prompt.strip():
            continue
        output = generate_text(model, tokenizer, prompt, device, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
        print(output)


if __name__ == "__main__":
    main()
