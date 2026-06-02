import os
import sys

# DDP Rank Isolation for Cache and Datasets
# Kaggle's dual-T4 runs in the same container. Separating HF_HOME prevents rank collisions
# and 'Bad file descriptor' network socket deadlock issues.
_rank = os.environ.get("RANK", "0")
os.environ["HF_HOME"] = f"/tmp/hf_cache_{_rank}"
os.environ["HF_DATASETS_CACHE"] = f"/tmp/hf_cache_{_rank}/datasets"
os.environ["FORGE_NO_SKIP"] = "1"  # Bypass the slow dataset skip operation over the network

# Force DDP to communicate over local loopback interface to prevent routing deadlocks
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"


import logging
import torch
from transformers import TrainingArguments, AutoTokenizer
from forge import ForgeTrainer

# Suppress harmless torch.distributed kernel version warning
logging.getLogger("torch.distributed").setLevel(logging.ERROR)

# Ensure current directory is in system path for importing model and data packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import EmberConfig, EmberForCausalLM
from data import get_pretraining_mixture, PackedDataCollator

def main():
    print("=== Starting Ember-275M Pre-training Pipeline ===")
    
    # 1. Detect device and hyperparameter support
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected training device: {device_name}")
    
    # Dynamic bf16 support check (A100 supports bf16, T4 only supports fp16)
    use_bf16 = False
    use_compile = False  # torch_compile: big speedup on A100+, marginal/slow on T4
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()
        # torch.compile requires compute capability >= 8.0 (A100+) to be worthwhile.
        # On T4 (compute 7.5): first compile takes 5-10 minutes per GPU, and many
        # Triton kernels fall back to eager mode. Not worth it for Kaggle T4 runs.
        compute_cap = torch.cuda.get_device_properties(0).major
        use_compile = (compute_cap >= 8)
    
    print(f"Hardware precision setting: {'bf16' if use_bf16 else 'fp16'}")
    print(f"torch_compile: {'enabled (compute >= 8.0)' if use_compile else 'disabled (compute < 8.0, e.g. T4)'}")
    
    # 2. Load tokenizer
    # Check if we have trained the tokenizer locally first, otherwise load from HF hub
    tokenizer_path = "./tokenizer_output"
    if os.path.exists(tokenizer_path):
        print(f"Loading local tokenizer from {tokenizer_path}")
        tokenizer_name = tokenizer_path
    else:
        tokenizer_name = "Kush26/ember-tokenizer"
        print(f"Local tokenizer not found, defaulting to HF Hub path: {tokenizer_name}")
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error: Tokenizer '{tokenizer_name}' not found. Please train it first by running:")
        print("  python tokenizer/train_tokenizer.py")
        sys.exit(1)
        
    # 3. Model Architecture Config
    print("Initializing model architecture...")
    config = EmberConfig(
        vocab_size=len(tokenizer),
        hidden_size=1024,
        num_hidden_layers=18,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2730,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    
    model = EmberForCausalLM(config)
    
    # Enable gradient checkpointing to fit within 16GB T4 memory, but disable on A100 for speed
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        model.gradient_checkpointing_enable()
        print("✅ Gradient Checkpointing enabled (Limited Memory).")
    else:
        print("✅ Gradient Checkpointing disabled (Abundant Memory, High Speed).")
    
    # Print total parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total tied model parameters: {total_params:,} (~{total_params // 1_000_000}M)")

    # 4. Stream packed and tokenized dataset
    print("Loading mixed streaming pre-training dataset...")
    
    # Dynamic seed based on resumption status to ensure Rank 0 and Rank 1 receive
    # a fresh permutation of training data without waiting hours to skip elements.
    data_seed = 42
    output_dir = "./outputs"
    if os.path.exists(output_dir):
        # Scan output_dir to see if checkpoint folders exist
        checkpoints = [d for d in os.listdir(output_dir) if "checkpoint-" in d]
        if checkpoints:
            data_seed = 42 + len(checkpoints) * 1000
            print(f"Resumption detected (cached checkpoints present). Shifting data shuffle seed to {data_seed}.")
    
    train_dataset = get_pretraining_mixture(
        tokenizer_name=tokenizer_name,
        max_seq_len=2048,
        buffer_size=1000,
        seed=data_seed,
    )
        
    # Data collator for block-diagonal masking
    data_collator = PackedDataCollator()

    # 5. Define Training Arguments
    # Note: ForgeTrainer will dynamically override per_device_train_batch_size,
    # gradient_accumulation_steps, and max_steps according to the provider profile in forge.yaml.
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=2,  # Placeholder, overridden by forge.yaml profile
        gradient_accumulation_steps=128,  # Placeholder, overridden by forge.yaml profile
        learning_rate=3e-4,
        weight_decay=0.1,
        max_steps=100000,               # Default max steps (overridden by active profile)
        logging_steps=10,
        save_steps=50,                  # Default; kaggle profile overrides to 200
        warmup_steps=2000,              # 2,000 steps warm-up
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.95,
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=False,   # Dynamically handled above by model.gradient_checkpointing_enable()
        torch_compile=use_compile,      # A100+ only: slow compile + limited Triton on T4 (compute 7.5)
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,    # Important! We use custom keys (position_ids)
        dataloader_num_workers=0,       # 0 workers per GPU runs data loading in the main process thread.
                                        # This completely eliminates PyTorch multiprocessing deadlocks
                                        # when a streamed dataset (like Hindi Wikipedia) has fewer shards
                                        # than workers (num_shards=1 vs num_workers=2).
    )

    # 6. Initialize the ForgeTrainer
    # Automatically manages stateful resumption of streaming data, checkpoint syncing,
    # and profile adjustments for Nomad Training (Modal A100 vs Kaggle DDP vs Colab T4)
    print("Initializing ForgeTrainer...")
    trainer = ForgeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        config_path="forge.yaml"
    )

    # 7. Start/Resume Training
    print("Launching training...")
    trainer.train()
    print("Pre-training completed successfully!")

if __name__ == "__main__":
    main()
