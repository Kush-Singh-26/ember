import os
import sys
import argparse
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer, DPOTrainer

# Ensure current directory is in PYTHONPATH for importing model architecture
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import EmberForCausalLM

def run_sft(model_path: str, dataset_name: str, output_dir: str, lora_r: int = 8, lora_alpha: int = 16):
    print("=== Supervised Fine-Tuning (SFT) Phase ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SFT running on device: {device}")
    
    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EmberForCausalLM.from_pretrained(model_path)
    model.to(device)
    
    # Enable gradient checkpointing to fit in memory
    model.gradient_checkpointing_enable()
    
    # 2. Setup LoRA Config for parameter-efficient fine-tuning (PEFT)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 3. Load dataset
    print(f"Loading SFT dataset: {dataset_name}...")
    from datasets import load_dataset
    try:
        # Standard instruction dataset (e.g. ultrachat_200k or similar)
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}. Falling back to sample dataset.")
        # Create a mock conversational dataset for validation
        from datasets import Dataset
        sample_data = {
            "prompt": ["What is python?", "Tell me a joke."],
            "completion": [
                "Python is a high-level programming language known for its readability.",
                "Why don't scientists trust atoms? Because they make up everything!"
            ]
        }
        dataset = Dataset.from_dict(sample_data)
        
    # Standard ChatML formatting helper
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"<|im_start|>user\n{example['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n{example['completion'][i]}<|im_end|>"
            output_texts.append(text)
        return output_texts

    # 4. Setup SFT Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_steps=500,                  # Short SFT run for demonstrations
        save_steps=100,
        logging_steps=10,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="none",
        remove_unused_columns=False
    )
    
    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=1024,
        args=training_args
    )
    
    print("Launching SFT training...")
    trainer.train()
    
    # Save PEFT adapter
    sft_output = os.path.join(output_dir, "sft_adapter")
    trainer.model.save_pretrained(sft_output)
    tokenizer.save_pretrained(sft_output)
    print(f"✅ SFT Adapter saved to {sft_output}")
    return sft_output

def run_dpo(model_path: str, adapter_path: str, dataset_name: str, output_dir: str):
    print("=== Direct Preference Optimization (DPO) Phase ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DPO running on device: {device}")
    
    # 1. Load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EmberForCausalLM.from_pretrained(model_path)
    model.to(device)
    
    # Load PEFT adapter (which acts as the reference and policy model simultaneously)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    
    # Enable gradient checkpointing to fit in memory
    model.gradient_checkpointing_enable()
    
    # Reference model is the base PEFT model (without active adapter training)
    # TRL DPOTrainer handles PeftModel automatically by creating a reference copy internally.
    
    # 2. Load DPO Dataset (chosen vs rejected formatting)
    print(f"Loading DPO dataset: {dataset_name}...")
    from datasets import load_dataset
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}. Falling back to sample dataset.")
        from datasets import Dataset
        sample_data = {
            "prompt": ["What is python?", "Tell me a joke."],
            "chosen": [
                "Python is a versatile high-level programming language known for readability.",
                "Why don't scientists trust atoms? Because they make up everything!"
            ],
            "rejected": [
                "Python is a big snake that wraps around its prey to suffocate them.",
                "I don't know any jokes. Go search the web yourself."
            ]
        }
        dataset = Dataset.from_dict(sample_data)

    def process_dpo_features(examples):
        # Format prompts and completions using ChatML format
        formatted = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        for i in range(len(examples["prompt"])):
            formatted["prompt"].append(f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n")
            formatted["chosen"].append(f"{examples['chosen'][i]}<|im_end|>")
            formatted["rejected"].append(f"{examples['rejected'][i]}<|im_end|>")
        return formatted

    # Map dataset (if using Dataset object, streaming needs slightly different handling)
    if isinstance(dataset, IterableDataset):
        dataset = dataset.map(process_dpo_features, batched=True)
    else:
        dataset = dataset.map(process_dpo_features, batched=True, remove_columns=dataset.column_names)

    # 3. DPO Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,   # DPO uses more memory, keep batch size small
        gradient_accumulation_steps=32,
        learning_rate=5e-7,              # Low DPO learning rate
        weight_decay=0.01,
        max_steps=300,                   # Short DPO run for demonstration
        save_steps=100,
        logging_steps=10,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="none",
        remove_unused_columns=False
    )
    
    # 4. Initialize DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                  # TRL will automatically handle ref_model when using PEFT
        beta=0.1,                        # DPO temperature parameter
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512
    )
    
    print("Launching DPO training...")
    trainer.train()
    
    dpo_output = os.path.join(output_dir, "dpo_final")
    trainer.model.save_pretrained(dpo_output)
    tokenizer.save_pretrained(dpo_output)
    print(f"✅ DPO Adapter saved to {dpo_output}")
    return dpo_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ember Post-Training Pipeline: SFT -> DPO")
    parser.add_argument("--model", type=str, default="./outputs/final", help="Path to base pre-trained model")
    parser.add_argument("--output_dir", type=str, default="./outputs/post_trained", help="Output directory")
    parser.add_argument("--sft_dataset", type=str, default="HuggingFaceH4/ultrachat_200k", help="SFT Dataset ID")
    parser.add_argument("--dpo_dataset", type=str, default="HuggingFaceH4/ultrafeedback_binarized", help="DPO Dataset ID")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA Rank")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Pre-trained model directory {args.model} does not exist.")
        sys.exit(1)
        
    # Run SFT
    sft_adapter_path = run_sft(
        model_path=args.model,
        dataset_name=args.sft_dataset,
        output_dir=args.output_dir,
        lora_r=args.lora_r
    )
    
    # Run DPO
    run_dpo(
        model_path=args.model,
        adapter_path=sft_adapter_path,
        dataset_name=args.dpo_dataset,
        output_dir=args.output_dir
    )
    print("✅ Complete post-training pipeline (SFT + DPO) finished successfully!")
