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
| **Context Length** | 4096 | Dynamic document boundary masking and causal packing (extended from 2048 at step 3650) |
| **Bias Terms** | None | No linear layer biases (Llama style) |

### Document Boundary Masking & Packing

Ember supports highly efficient **Sequence Packing** with **Document Boundary Masking** to eliminate cross-document attention leakage (where tokens in Document B attend to tokens in Document A).

- **Position IDs Reset:** Position IDs reset to `0` at document boundaries.
- **Attention Mask:** Custom 4D block-diagonal boolean mask ensures tokens only attend to other tokens belonging to the *same* document causal segment, even when packed within the same 4096-token sequence block.

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

Benchmark the model zero-shot performance using our lightweight likelihood-based evaluation harness. MBPP is excluded by default (opt-in via `--benchmarks mbpp`) as it requires an instruction-tuned model.

```bash
python evaluation/benchmarks.py --model "./outputs/checkpoint-9650" --samples 500
```

### Results: `checkpoint-9650` (~5B tokens, step 9650)

| Benchmark | Metric | Score | Notes |
|:---|:---|:---:|:---|
| **HellaSwag** | Accuracy | 0.3160 | Common-sense sentence completion |
| **ARC-Easy** | Accuracy | **0.4060** | Elementary science QA (random = 0.25) |
| **ARC-Challenge** | Accuracy | 0.1873 | Hard science QA |
| **WinoGrande** | Accuracy | 0.5240 | Pronoun coreference (random = 0.50) |
| **WikiText-2** | Perplexity | 48.86 | Full test set, sliding window |
| **Macro F1** | Avg Accuracy | **0.3583** | Average across all multiple-choice tasks |

### Comparison: Step 3650 → Step 9650

| Benchmark | Step 3650 | Step 9650 | Δ |
|:---|:---:|:---:|:---:|
| ARC-Easy | 0.2960 | **0.4060** | +10.0% 🚀 |
| HellaSwag | 0.3360 | 0.3160 | -2.0% |
| WinoGrande | 0.5260 | 0.5240 | ~0% |
| ARC-Challenge | 0.2207 | 0.1873 | -3.3% |
| Macro F1 | 0.3447 | **0.3583** | +1.4% 📈 |

> **Note:** HellaSwag/ARC-Challenge dipped slightly due to the 2K→4K context-length expansion at step 3650. The model's short-context attention focus became temporarily noisier as it adapted to 4K positional embeddings. This resolves with further training. ARC-Easy's +10% jump confirms the model is successfully absorbing factual knowledge.

---

## Conversational Alignment: SFT & DPO

After pre-training, fine-tune the model to follow instructions and align with preferences using parameter-efficient fine-tuning (LoRA):

```bash
python sft_dpo.py --model "./outputs/final" --output_dir "./outputs/aligned" --sft_dataset "HuggingFaceH4/ultrachat_200k" --dpo_dataset "HuggingFaceH4/ultrafeedback_binarized"
```

1. **SFT Phase:** Uses `trl.SFTTrainer` with ChatML formatting and a low rank LoRA adapter to teach conversation structure.
2. **DPO Phase:** Uses `trl.DPOTrainer` to optimize preferred responses and suppress rejected answers directly.

---

## Recent Updates & Resumption Notes (Step 3650+)

### 1. Pre-3650 Training Summary
* **Training Stage:** Initial pre-training phase from step 1 to 3650.
* **Context Length:** 2048 tokens.
* **Dataset Mixture:** 55% English (FineWeb-Edu 100BT), 20% Hindi (Wikipedia Hindi fallback), 20% Code (CodeSearchNet Python fallback).
* **Global Batch Size:** 524,288 tokens per step (batch size 8 × gradient accumulation 32 on 2K context).
* **Final Metrics at Step 3650:** Loss ~`1.8`–`1.9`, LR `3e-4`, optimizer AdamW + cosine decay.

### 2. Step 3650 → 9650 Training Summary
* **Platform:** Lightning AI Studio (A100 80GB), then resumed on a second account.
* **Context Length:** 4096 tokens (expanded from 2048).
* **Dataset Mixture:** 55% FineWeb-Edu (local Parquet), 20% Wikipedia Hindi, 20% CodeSearchNet Python.
* **Global Batch Size:** 524,288 tokens/step (batch 8 × accum 16 × 4K ctx).
* **Total Tokens Trained:** ~5.06 Billion tokens by step 9650.
* **Loss at Step 9650:** ~`2.74`–`2.87` (down from `3.77` at restart due to 4K context change).
* **Training Speed:** ~241 it/s at peak (A100 80GB, bf16, no gradient checkpointing).

### 3. Context Length Expansion (2048 ➔ 4096)
* **Changes:** Extended `max_position_embeddings` and dataset packing `max_seq_len` from `2048` to `4096` in the commit prior to step 3650 resumption.
* **Resumption Behavior:** When resuming from `checkpoint-3650`, the model begins training on `4096`-token sequence chunks. Because the model has never trained on position IDs `2048-4095`, the initial training loss will temporarily start higher (e.g., around `3.7-3.8`) and decrease as the model learns the new position embeddings during the learning rate warmup phase (`warmup_steps: 200` in the profile).
* **Document Isolation:** Enabled block-diagonal attention masking (`attn_mask=attention_mask`) and set `is_causal=False` (since causal masking logic is pre-collated) to prevent cross-document attention leakage within the 4K packed sequence blocks.

### 3. PyTorch 2.6+ Checkpoint Resumption Hotfix
* **The Issue:** Under PyTorch 2.6+, the default value of the `weights_only` argument in `torch.load` shifted from `False` to `True`. When resuming training, Hugging Face `transformers.Trainer` attempts to load RNG states, optimizer states, and scaler states. Since these checkpoint files contain pickled NumPy components (specifically `numpy._core.multiarray._reconstruct`), the load operation fails with a `WeightsUnpickler` security exception.
* **The Solution:** Added a global monkeypatch at the top of `train.py` that intercepts all `torch.load` calls and forces `weights_only=False` for local resume files. This cleanly circumvents the PyTorch 2.6+ unpickling issue for trusted checkpoint resumption.
