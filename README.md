# Ember-275M

A high-performance, parameter-efficient decoder-only language model engineered from scratch using the **lm_forge** Nomad Training framework. Designed to pre-train seamlessly across diverse and heterogeneous GPU resources (Modal A100, Kaggle 2×T4, Colab T4) while preserving strict training step consistency.

---

## Architecture & Model Specification

Ember utilizes state-of-the-art transformer architecture design choices derived from modern Llama-3 and Gemma conventions:

| Parameter | Value | Description |
|---|---|---|
| **Parameter Count** | ~275M (Total Tied) | Extremely agile, high-density parameter layout |
| **Attention** | Grouped-Query Attention (GQA) | 16 Query heads, 8 Key/Value heads (2:1 GQA ratio) |
| **Head Dimension** | 64 | Standard high-efficiency projection size |
| **MLP Convention** | SwiGLU FFN | Swish-Gated Linear Units, intermediate dim = 2730 ($\approx \frac{8}{3} d_{model}$) |
| **Normalization** | RMSNorm | Pre-activation RMSNorm with standard epsilon $1e-6$ |
| **Positional Embedding** | RoPE (Rotary) | Base theta=10000, applied to Q and K projections only |
| **Word Embeddings** | Tied | Shared input and output projection weights (saves ~67M parameters) |
| **Vocabulary Size** | 65,536 (64K) | Byte-level BPE, optimized for multilingual and programming code coverage |
| **Context Length** | 2048 | Dynamic document boundary masking and causal packing |
| **Bias Terms** | None | No linear layer biases (Llama style) |

### Document Boundary Masking & Packing

Ember supports highly efficient **Sequence Packing** with **Document Boundary Masking** to eliminate cross-document attention leakage (where tokens in Document B attend to tokens in Document A).

- **Position IDs Reset:** Position IDs reset to `0` at document boundaries.
- **Attention Mask:** Custom 4D block-diagonal boolean mask ensures tokens only attend to other tokens belonging to the *same* document causal segment, even when packed within the same 2048-token sequence block.

---

## Repository Structure

```
ember/
├── model/
│   ├── __init__.py
│   ├── config.py         # HuggingFace PretrainedConfig subclass
│   └── transformer.py    # PyTorch implementation (RMSNorm, RoPE, GQA, SwiGLU)
├── tokenizer/
│   └── train_tokenizer.py # Byte-Level BPE multilingual & code tokenizer training
├── data/
│   ├── __init__.py
│   └── pipeline.py       # Sequence packing, source interleaving, 4D collator
├── evaluation/
│   ├── __init__.py
│   └── benchmarks.py     # Zero-shot validation harness (HellaSwag & ARC-Easy)
├── tests/
│   ├── test_model.py     # Unit tests (gradients, forward passes, masking, tied weights)
│   └── test_packing.py   # Unit tests (packer, position resetting, 4D mask collation)
├── train.py              # Main pre-training entrypoint with ForgeTrainer
├── sft_dpo.py            # Alignment script: Supervised Fine-Tuning -> DPO
├── forge.yaml            # Forge environment profiles and state sync configs
├── requirements.txt      # PyPI dependencies
└── README.md             # This documentation
```

---

## Step-by-Step Pre-training Guide

### 1. Installation

First, install the model dependencies:

```bash
uv pip install -r requirements.txt
```

Ensure `lm_forge` is installed and authenticated to your Hugging Face Hub account with write access.

### 2. Custom Tokenizer Training

Train the custom 64K Byte-Level BPE tokenizer. The script streams English text (WikiText), Hindi text (Wikipedia), and Python source code (CodeSearchNet) to construct a balanced vocabulary:

```bash
python tokenizer/train_tokenizer.py --vocab_size 65536 --sample_limit 30000 --push --repo_id "Kush26/ember-tokenizer"
```

### 3. Verify Tokenizer and Model Unit Tests

Run the complete test suite locally to verify the correctness of the architecture, sequence packing, and attention mask:

```bash
uv run pytest -v tests/
```

### 4. Running Pre-training (Nomad Training)

Pre-training dynamically scales batch sizes and gradient accumulation steps to match a global batch token count of **524,288 tokens per step** across any hardware profile:

- **Modal (1× A100 80GB):** `batch=8, accum=32`
- **Kaggle (2× T4 16GB DDP):** `batch=2, accum=64`
- **Colab (1× T4 16GB):** `batch=2, accum=128`

To start or resume pre-training in any environment, run:

```bash
# 1. Pull the latest checkpoint state from Hugging Face Hub
forge pull

# 2. Launch training (runs local Python, DDP accelerate launch, or Modal job automatically)
forge run local --script train.py
```

Checkpoints are automatically saved, verified, and uploaded to `Kush26/ember-checkpoints` on the `checkpoints` branch.

---

## Zero-Shot Evaluation

Benchmark the model zero-shot performance on HellaSwag (common-sense reasoning) and ARC-Easy (science QA) using our fast, lightweight likelihood-based evaluation harness:

```bash
python evaluation/benchmarks.py --model "./outputs/final" --samples 100
```

---

## Conversational Alignment: SFT & DPO

After pre-training, fine-tune the model to follow instructions and align with preferences using parameter-efficient fine-tuning (LoRA):

```bash
python sft_dpo.py --model "./outputs/final" --output_dir "./outputs/aligned" --sft_dataset "HuggingFaceH4/ultrachat_200k" --dpo_dataset "HuggingFaceH4/ultrafeedback_binarized"
```

1. **SFT Phase:** Uses `trl.SFTTrainer` with ChatML formatting and a low rank LoRA adapter to teach conversation structure.
2. **DPO Phase:** Uses `trl.DPOTrainer` to optimize preferred responses and suppress rejected answers directly.
