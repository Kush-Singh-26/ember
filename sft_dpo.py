"""
Ember-275M Post-Training Pipeline: SFT → DPO

SoTA techniques applied:
  - Safetensors model loading (avoids HF 5.x tied-weight corruption bug)
  - Special token injection (<|im_start|>, <|im_end|>) with embedding resize
  - Loss masking via DataCollatorForCompletionOnlyLM (only trains on assistant responses)
  - LoRA rank 64 targeting all 7 projection layers
  - Mixed multilingual dataset (English + Hindi + Code)
  - Merged model output (adapter fused into base weights for deployment)

Usage:
  # SFT (run first):
  python sft_dpo.py --model ./checkpoint-9650 --output_dir ./outputs/sft --phase sft

  # DPO (run after SFT, using the merged SFT model):
  python sft_dpo.py --model ./outputs/sft/sft_merged --output_dir ./outputs/dpo --phase dpo

  # Both phases sequentially:
  python sft_dpo.py --model ./checkpoint-9650 --output_dir ./outputs --phase both
"""

import os
import sys
import argparse
import math
import torch
from pathlib import Path

# --- Hotfix: PEFT + torchao version compatibility crash ------------------
# Some versions of PEFT raise an ImportError if an incompatible version
# of torchao is installed (e.g., in Google Colab). Mock this check to False.
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass
# -------------------------------------------------------------------------

from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import TrainingArguments, AutoTokenizer
from trl import DPOTrainer
from datasets import load_dataset, concatenate_datasets, Dataset, IterableDataset

# --- Portable DataCollatorForCompletionOnlyLM ----------------------------
# TRL moved this class between versions. Try three locations before falling
# back to our own implementation so the code works on any installed version.
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer import DataCollatorForCompletionOnlyLM
    except ImportError:
        import torch
        from dataclasses import dataclass
        from typing import Any, Dict, List

        @dataclass
        class DataCollatorForCompletionOnlyLM:
            """
            Portable fallback: masks all prompt tokens with -100 so that loss
            is computed only on the assistant's response tokens.
            """
            tokenizer: Any
            response_template: str
            mlm: bool = False

            def __post_init__(self):
                self._template_ids = self.tokenizer.encode(
                    self.response_template, add_special_tokens=False
                )

            def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                import torch as _t
                input_ids_list = [f["input_ids"] for f in features]
                attention_mask_list = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]

                max_len = max(len(ids) for ids in input_ids_list)
                pad_id = self.tokenizer.pad_token_id or 0

                batch_ids, batch_mask, batch_labels = [], [], []
                for ids, mask in zip(input_ids_list, attention_mask_list):
                    pad_len = max_len - len(ids)
                    padded = ids + [pad_id] * pad_len
                    padded_mask = mask + [0] * pad_len
                    labels = list(padded)  # start as copy

                    # Find the first occurrence of the response template
                    tpl = self._template_ids
                    tpl_len = len(tpl)
                    response_start = None
                    for j in range(len(ids) - tpl_len + 1):
                        if ids[j : j + tpl_len] == tpl:
                            response_start = j + tpl_len
                            break

                    # Mask everything before response start and all padding
                    mask_until = response_start if response_start is not None else len(padded)
                    for j in range(mask_until):
                        labels[j] = -100
                    for j in range(len(ids), max_len):  # padding positions
                        labels[j] = -100

                    batch_ids.append(padded)
                    batch_mask.append(padded_mask)
                    batch_labels.append(labels)

                return {
                    "input_ids": _t.tensor(batch_ids, dtype=_t.long),
                    "attention_mask": _t.tensor(batch_mask, dtype=_t.long),
                    "labels": _t.tensor(batch_labels, dtype=_t.long),
                }
# -------------------------------------------------------------------------

# Ensure project root is in path for model imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import EmberForCausalLM, EmberConfig


# ─── Constants ───────────────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
}
CHATML_SYSTEM = "You are Ember, a helpful multilingual AI assistant fluent in English, Hindi, and Python."

# ─── Model Loading ────────────────────────────────────────────────────────────

def load_base_model(model_path: str, device: str):
    """
    Load Ember base model using safetensors directly to avoid the HF 5.x
    tied-weight race condition that corrupts lm_head with random noise.
    """
    from safetensors.torch import load_file

    print(f"Loading base model from {model_path}...")
    config = EmberConfig.from_pretrained(model_path)
    model = EmberForCausalLM(config)

    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)
    # Manually inject tied lm_head weight before loading state dict
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Loaded: {params:,} parameters")
    return model


def load_tokenizer(model_path: str):
    """
    Load tokenizer from local ./tokenizer_output, or fall back to the model
    checkpoint directory, then HF Hub. Injects ChatML special tokens and
    resizes the model embedding table if new tokens were added.
    """
    local_tok = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_output")
    tok_in_checkpoint = os.path.join(model_path, "tokenizer.json")

    if os.path.isdir(local_tok):
        tokenizer = AutoTokenizer.from_pretrained(local_tok)
        print(f"  Tokenizer loaded from {local_tok}")
    elif os.path.exists(tok_in_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  Tokenizer loaded from checkpoint")
    else:
        tokenizer = AutoTokenizer.from_pretrained("Kush26/ember-tokenizer")
        print(f"  Tokenizer loaded from HF Hub")

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def add_special_tokens(tokenizer, model):
    """
    Add ChatML special tokens to the tokenizer and resize model embedding
    matrix to accommodate them. Returns number of new tokens added.

    Works with both slow tokenizers (have .additional_special_tokens) and
    fast tokenizers (TokenizersBackend, which only exposes .all_special_tokens).
    """
    # Robust access — fast tokenizers raise AttributeError on .additional_special_tokens
    try:
        existing = set(tokenizer.additional_special_tokens)
    except AttributeError:
        existing = set(tokenizer.all_special_tokens)

    new_tokens = [t for t in SPECIAL_TOKENS["additional_special_tokens"] if t not in existing]

    if not new_tokens:
        print("  ChatML special tokens already present — skipping.")
        return 0

    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    n_added = len(new_tokens)

    # Resize embedding matrix — new rows are mean-initialized (better than random)
    old_size = model.model.embed_tokens.weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_size = model.model.embed_tokens.weight.shape[0]

    with torch.no_grad():
        # Keep mean_emb on the same device as the embedding weight
        mean_emb = model.model.embed_tokens.weight[:old_size].mean(0).detach()
        model.model.embed_tokens.weight[old_size:] = mean_emb.unsqueeze(0).expand(new_size - old_size, -1)

    print(f"  ✅ Added {n_added} special tokens. Embedding: {old_size} → {new_size}")
    return n_added


# ─── Dataset Building ─────────────────────────────────────────────────────────

def format_chatml(system: str, user: str, assistant: str) -> str:
    """Format a single turn into ChatML string."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return "\n".join(parts)


def load_openhermes(max_samples: int = 8_000) -> Dataset:
    """
    OpenHermes-2.5: GPT-4 generated, diverse English instructions.
    Schema: {conversations: [{from: 'human'|'gpt'|'system', value: str}]}
    Default: 8K samples (~200 MB RAM) — safe for Colab T4's 12 GB system RAM.
    """
    print(f"  Loading OpenHermes-2.5 ({max_samples:,} samples)...")
    try:
        # Load normally (non-streaming) to utilize memory mapping (Arrow) and save RAM
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
        ds = ds.select(range(min(len(ds), max_samples * 2)))
    except Exception as e:
        print(f"    ⚠️  Failed to load OpenHermes-2.5 ({e})")
        return Dataset.from_list([])

    rows = []
    for item in ds:
        if len(rows) >= max_samples:
            break
        convs = item.get("conversations", [])
        system = next((c["value"] for c in convs if c.get("from") == "system"), CHATML_SYSTEM)
        human = next((c["value"] for c in convs if c.get("from") == "human"), None)
        gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
        if not human or not gpt:
            continue
        # Quality filter: response must be meaningful
        if len(gpt.strip()) < 50 or len(gpt.strip()) > 1800:
            continue
        rows.append({"text": format_chatml(system, human, gpt)})

    print(f"    ✅ OpenHermes-2.5: {len(rows):,} samples")
    return Dataset.from_list(rows)


def load_anudesh(max_samples: int = 3_000) -> Dataset:
    """
    Anudesh (AI4Bharat): Native Hindi instruction-following dataset.
    Falls back to translated Alpaca if unavailable.
    Default: 3K samples — safe for Colab RAM.
    """
    print(f"  Loading Hindi instruction dataset ({max_samples:,} samples)...")
    try:
        # Load normally (non-streaming) to utilize memory mapping (Arrow) and save RAM
        ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", "anudesh", split="hi")
        ds = ds.select(range(min(len(ds), max_samples * 2)))
        rows = []
        for item in ds:
            if len(rows) >= max_samples:
                break
            convs = item.get("messages", [])
            user = next((c["content"] for c in convs if c.get("role") == "user"), None)
            assistant = next((c["content"] for c in convs if c.get("role") in ("assistant", "gpt")), None)
            if not user or not assistant or len(assistant.strip()) < 20:
                continue
            rows.append({"text": format_chatml(CHATML_SYSTEM, user.strip(), assistant.strip())})
        if rows:
            print(f"    ✅ Anudesh: {len(rows):,} samples")
            return Dataset.from_list(rows)
    except Exception as e:
        print(f"    ⚠️  Anudesh failed ({e}), falling back to Hindi Alpaca...")

    # Fallback: filter Hindi samples from multilingual Alpaca
    try:
        ds = load_dataset("iamshnoo/alpaca-cleaned-hindi", split="train")
        ds = ds.select(range(min(len(ds), max_samples * 2)))
        rows = []
        for item in ds:
            if len(rows) >= max_samples:
                break
            instr = (item.get("instruction") or "") + " " + (item.get("input") or "")
            output = item.get("output") or ""
            if not instr.strip() or not output.strip() or len(output.strip()) < 20:
                continue
            rows.append({"text": format_chatml(CHATML_SYSTEM, instr.strip(), output)})
        print(f"    ✅ Hindi Alpaca fallback: {len(rows):,} samples")
        return Dataset.from_list(rows)
    except Exception as e2:
        print(f"    ⚠️  Hindi fallback also failed ({e2}). Skipping Hindi data.")
        return Dataset.from_list([])


def load_code_instructions(max_samples: int = 3_000) -> Dataset:
    """
    Python code instruction dataset (Alpaca format).
    Default: 3K samples — safe for Colab RAM.
    """
    print(f"  Loading code instruction dataset ({max_samples:,} samples)...")
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        ds = ds.select(range(min(len(ds), max_samples * 2)))
        rows = []
        for item in ds:
            if len(rows) >= max_samples:
                break
            instr = item.get("instruction", "") + (" " + item.get("input", "") if item.get("input") else "")
            output = item.get("output", "")
            if not instr.strip() or not output.strip() or len(output.strip()) < 30:
                continue
            rows.append({"text": format_chatml(CHATML_SYSTEM, instr.strip(), output)})
        print(f"    ✅ Code instructions: {len(rows):,} samples")
        return Dataset.from_list(rows)
    except Exception as e:
        print(f"    ⚠️  Code dataset failed ({e}). Skipping code data.")
        return Dataset.from_list([])


def build_sft_dataset(en_samples: int = 8_000, hi_samples: int = 3_000, code_samples: int = 3_000) -> Dataset:
    """
    Build the mixed SFT dataset: English + Hindi + Code, shuffled.
    Defaults are tuned to fit in Colab T4's 12 GB system RAM (~14K total).
    With effective batch 32 and 2000 steps = 64K examples consumed,
    the dataset is repeated ~4-5x — fine for SFT.
    Increase sizes if running on a machine with more RAM (e.g. A100 node).
    """
    import psutil
    ram_gb = psutil.virtual_memory().available / 1e9
    print(f"Building SFT dataset mix... (available RAM: {ram_gb:.1f} GB)")
    parts = []

    en_ds = load_openhermes(max_samples=en_samples)
    if len(en_ds) > 0:
        parts.append(en_ds)

    hi_ds = load_anudesh(max_samples=hi_samples)
    if len(hi_ds) > 0:
        parts.append(hi_ds)

    code_ds = load_code_instructions(max_samples=code_samples)
    if len(code_ds) > 0:
        parts.append(code_ds)

    if not parts:
        raise RuntimeError("All dataset sources failed. Cannot proceed with SFT.")

    combined = concatenate_datasets(parts).shuffle(seed=42)
    print(f"  ✅ Total SFT dataset: {len(combined):,} examples")
    return combined


# ─── SFT ─────────────────────────────────────────────────────────────────────

def run_sft(
    model_path: str,
    output_dir: str,
    lora_r: int = 64,
    max_steps: int = 2000,
    max_seq_length: int = 2048,
):
    print("\n" + "=" * 60)
    print("  Phase 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print(f"Device: {device} | Precision: {'bf16' if use_bf16 else 'fp16'}")

    # 1. Load tokenizer and model
    tokenizer = load_tokenizer(model_path)
    model = load_base_model(model_path, device)

    # 2. Add ChatML special tokens and resize embeddings
    n_new = add_special_tokens(tokenizer, model)

    # Save updated tokenizer alongside adapter
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    # 3. Enable gradient checkpointing (essential for T4 memory)
    # gradient_checkpointing_kwargs only accepted in transformers ≥ 4.36
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    # 4. LoRA config — rank 64, all projection layers
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,       # Standard: alpha = 2 × r
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 5. Build mixed SFT dataset
    dataset = build_sft_dataset()

    # 6. Loss masking — only trains on tokens AFTER <|im_start|>assistant\n
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    print(f"  Loss masking: active  [collator: {type(collator).__name__}]")

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,      # Effective batch = 32
        learning_rate=2e-4,                  # LoRA uses higher LR than full fine-tune
        weight_decay=0.01,
        max_steps=max_steps,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )

    # 8. Pre-tokenize dataset then train with plain HuggingFace Trainer.
    #    Avoids ALL SFTTrainer/SFTConfig API version issues — works on any TRL version.
    print(f"  Tokenizing dataset (max_seq_length={max_seq_length})...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    print(f"  ✅ Tokenized {len(tokenized):,} examples")

    # Apply LoRA manually (SFTTrainer did this internally; we do it explicitly now)
    from peft import get_peft_model as _get_peft_model
    peft_model = _get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA: {trainable:,} / {total:,} trainable params ({100*trainable/total:.1f}%)")

    from transformers import Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print(f"\nLaunching SFT: {max_steps} steps, LR=2e-4, seq_len={max_seq_length}")
    trainer.train()

    # 9. Save adapter
    adapter_path = os.path.join(output_dir, "sft_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ LoRA adapter saved to: {adapter_path}")

    # 10. Merge adapter into base model and save full merged model
    print("Merging LoRA adapter into base weights...")
    merged_model = trainer.model.merge_and_unload()
    merged_path = os.path.join(output_dir, "sft_merged")
    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"✅ Merged model saved to: {merged_path}")

    return adapter_path, merged_path


# ─── DPO ─────────────────────────────────────────────────────────────────────

def build_dpo_dataset(max_samples: int = 10_000) -> Dataset:
    """
    UltraFeedback binarized: chosen (preferred) vs rejected responses.
    Schema: {prompt, chosen: [{role, content}], rejected: [{role, content}]}
    """
    print("Building DPO dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    ds = ds.select(range(min(len(ds), max_samples * 2)))

    rows = []
    for item in ds:
        if len(rows) >= max_samples:
            break
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", [])
        rejected = item.get("rejected", [])

        # Extract assistant text from chosen/rejected message lists
        chosen_text = next((m["content"] for m in chosen if m.get("role") == "assistant"), "")
        rejected_text = next((m["content"] for m in rejected if m.get("role") == "assistant"), "")

        if not prompt or not chosen_text or not rejected_text:
            continue
        if chosen_text == rejected_text:
            continue

        rows.append({
            "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "chosen": chosen_text + "<|im_end|>",
            "rejected": rejected_text + "<|im_end|>",
        })

    ds_out = Dataset.from_list(rows).shuffle(seed=42)
    print(f"  ✅ DPO dataset: {len(ds_out):,} preference pairs")
    return ds_out


def run_dpo(
    model_path: str,
    output_dir: str,
    max_steps: int = 500,
    max_seq_length: int = 2048,
):
    print("\n" + "=" * 60)
    print("  Phase 2: Direct Preference Optimization (DPO)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    # Load tokenizer and merged SFT model
    tokenizer = load_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(model_path, device)

    # Light LoRA on top of the SFT-merged model for DPO
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    dataset = build_dpo_dataset()

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "dpo_checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=5e-7,
        weight_decay=0.01,
        max_steps=max_steps,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # DPOTrainer: 'tokenizer' kwarg renamed to 'processing_class' in TRL ≥ 0.9
    dpo_kwargs = dict(
        model=model,
        ref_model=None,
        beta=0.1,
        args=training_args,
        train_dataset=dataset,
        max_length=max_seq_length,
        max_prompt_length=max_seq_length // 2,
    )
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **dpo_kwargs)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **dpo_kwargs)

    print(f"\nLaunching DPO: {max_steps} steps, beta=0.1")
    trainer.train()

    # Merge and save
    merged_model = trainer.model.merge_and_unload()
    dpo_path = os.path.join(output_dir, "dpo_merged")
    merged_model.save_pretrained(dpo_path, safe_serialization=True)
    tokenizer.save_pretrained(dpo_path)
    print(f"✅ DPO merged model saved to: {dpo_path}")
    return dpo_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ember Post-Training: SFT → DPO")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to base/SFT-merged model checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/post_trained",
                        help="Root output directory for adapters and merged models")
    parser.add_argument("--phase", type=str, default="sft", choices=["sft", "dpo", "both"],
                        help="Which phase(s) to run")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank (default: 64 for SFT, 32 for DPO)")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Training steps (default: 2000 for SFT, 500 for DPO)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Max sequence length (default: 2048)")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"Error: Model directory does not exist: {args.model}")
        sys.exit(1)

    sft_merged = args.model

    if args.phase in ("sft", "both"):
        sft_adapter, sft_merged = run_sft(
            model_path=args.model,
            output_dir=os.path.join(args.output_dir, "sft"),
            lora_r=args.lora_r,
            max_steps=args.max_steps,
            max_seq_length=args.max_seq_length,
        )

    if args.phase in ("dpo", "both"):
        dpo_path = run_dpo(
            model_path=sft_merged,
            output_dir=os.path.join(args.output_dir, "dpo"),
            max_steps=500 if args.phase == "both" else args.max_steps,
            max_seq_length=args.max_seq_length,
        )

    print("\n✅ Post-training pipeline complete.")
