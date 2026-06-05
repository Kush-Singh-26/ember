# Critical Examination of IMPROVEMENT_PLAN.md vs. Actual Codebase

> Verified against: `data/pipeline.py`, `model/transformer.py`, `model/config.py`, `train.py`, `forge.yaml`, `eval_results.json`

---

## ✅ Claims That Are Accurate

| Plan Claim | Reality |
|---|---|
| SequencePacking pre-truncates every doc to `max_seq_len - 2` | **True.** `pipeline.py` L57-58 does exactly this |
| `.map()` also truncates via `truncation=True, max_length=max_seq_len` | **True.** L184, L204, L226 all do this — double truncation confirmed |
| `PackedDataCollator` never uses `document_ids` to build attention masks | **True.** L297-312: collator drops `document_ids`, never builds a mask |
| Model uses `is_causal=True` with no document mask | **True.** `transformer.py` L146: `F.scaled_dot_product_attention(..., is_causal=True)` |
| No `max_grad_norm` is set | **True.** `TrainingArguments` in `train.py` L214-240 has no `max_grad_norm` |
| `warmup_steps=2000` | **True.** L223 |

---

## ⚠️ Claims That Are Imprecise or Misleading

### 1. "50-75% waste per chunk" — The real number is worse *on average*, but the framing is off

The plan says a 500-token document wastes 75% of a 2048-slot chunk. But the actual effect is more nuanced:

- After truncation to `max_seq_len - 2 = 2046`, a 500-token doc becomes **502 tokens** (BOS + 500 + EOS).
- These 502 tokens **do** get added to the `buffer_input_ids` list and **are packed together** with subsequent documents.
- The chunk is only yielded when the buffer reaches ≥ 2048 tokens, so **short documents will pack together**. A 500-token doc is not a "2048-slot chunk" on its own.
- The **real bug** is different: a document with 3000 raw tokens gets truncated to 2048 tokens, wasting the remaining 952. The waste only materialises for **long documents**, not short ones.
- The plan's "500-token doc wastes 75%" example is factually wrong. What actually happens: a 3000-token document gets capped at 2048, losing 952 tokens of real content. FineWeb-Edu and Wikipedia articles often exceed 2048 tokens, so this is still a real and costly bug — just described incorrectly.

### 2. Phase 1.2 "Cross-Document Leakage" — Correct diagnosis, but the fix breaks checkpoint compatibility

The plan says "All changes are compatible with existing checkpoint-3650." This is only partially true for Phase 1.2:

- Changing `EmberAttention.forward()` signature and passing `attention_mask` is **additive** — checkpoint weights load fine since no new parameters are introduced.
- However, `is_causal=True` → `is_causal=False` with an explicit mask is a **training distribution shift**. The model has been trained for 3650 steps with every token attending to all previous tokens across document boundaries. Suddenly introducing block-diagonal masks changes the gradient signal dramatically. This could cause **loss spikes** on resumption.
- The plan does not mention any LR warmup after this change, which would be advisable.

### 3. Phase 2.2 "WSD Learning Rate Schedule" — Self-contradictory

The section is titled "Switch to WSD" but the code in the fix section still says `lr_scheduler_type="cosine"` with a note that WSD isn't built into HF Trainer. This makes the section actively confusing — it proposes a change but then doesn't make it. It should be removed or rewritten as a future note only.

### 4. Phase 3.1 "Effective data mixture ratios broken by `stopping_strategy='all_exhausted'`" — Correct, but incomplete

The plan is right that small datasets (Hindi Wikipedia, CodeSearchNet) will cycle many times while FineWeb-Edu dominates. However:

- `interleave_datasets` with explicit `probabilities` already controls the **sampling frequency**, not just which dataset is exhausted. A 0.60/0.20/0.20 split means ~60% of items drawn are from FineWeb-Edu regardless of corpus size. The effective ratio **is** approximately 60/20/20 tokens-wise in practice.
- The real problem with small datasets is **overfitting through repetition**. Hindi Wikipedia at 500M tokens and 20% weight means the model sees it ~40× over a 100B-token run. This is the correct concern, but the plan's causal description ("stopping_strategy causes 99% English") is wrong — `probabilities` directly controls this regardless of dataset size.

### 5. Phase 4.1 "Extend Context to 4K — No code change needed for RoPE" — Incorrect

The plan states: *"The RoPE cache will automatically extend to 4096 positions since it's computed from `max_position_embeddings`. No code change needed — just the config value."*

This is **false for checkpoint resumption**. The RoPE cache in `EmberRotaryEmbedding.__init__()` calls `_set_cos_sin_cache(max_len=max_position_embeddings)`. When resuming from checkpoint-3650 which was trained with `max_position_embeddings=2048`, the saved model config will have `max_position_embeddings=2048`. You must **explicitly pass** `max_position_embeddings=4096` when loading or override the config before `from_pretrained`. Failing to do this means the RoPE cache only covers 2048 positions and any token beyond position 2047 will be clamped (L66: `position_ids.clamp(max=self.max_seq_len_cached - 1)`), producing garbage positional encodings silently.

### 6. Phase 5.1 "Remove dead `from_pretrained` Override" — Incorrect, this is not dead code

The plan says to delete the `from_pretrained` classmethod because "it just calls `super()` and adds no value." This misses the companion method `_initialize_missing_keys()` (L337-351), which has a detailed docstring explaining a real HF 5.x bug with tied weights. The `from_pretrained` override exists specifically as a hook to make this documented override clear. Deleting it is harmless, but calling it "dead" ignores the intentional design. More importantly, `_initialize_missing_keys` (the `pass` method) is the **actual critical fix** and must not be removed.

---

## ❌ Claims That Are Wrong or Risky

### 7. The forge.yaml colab profile change is inconsistent

The plan proposes `gradient_accumulation_steps: 64` for colab with 4K context (vs 128 with 2K), reasoning "context doubled, halve accum to keep memory." But it also keeps `per_device_train_batch_size: 2`. With 4K context and batch_size=2, a T4 (16GB) will likely OOM before hitting the accumulation step. The plan should specify `per_device_train_batch_size: 1` for colab at 4K.

### 8. The modal-continue profile has an arithmetic comment error

```yaml
gradient_accumulation_steps: 16  # 8 * 16 * 1 GPU = 128 seq = 262K tokens/step (at 4K ctx: 524K)
```

At 4K context: 128 sequences × 4096 tokens = **524,288 tokens/step** ✅ — the 4K numbers are right.  
But the base calculation `128 seq = 262K tokens/step` assumes 2K context: 128 × 2048 = 262,144. This is internally inconsistent. If the profile targets 4K (which is the stated goal), the comment should reflect 4K math only.

---

## 🔍 Missed Issues Not in the Plan

### A. The `document_ids` remapping in the collator is silently dropped

`SequencePacker` computes `relative_doc_ids` (0-indexed within each chunk, L84-90) and puts them in the yielded dict. `PackedDataCollator` (L297-312) **never reads `document_ids`** from the feature dict. This is fine currently (no mask is built), but once Phase 1.2 is implemented, the mask construction must use the already-remapped `document_ids` from the packer — the plan's collator code correctly accesses `feature["document_ids"]`, so this will work. But it's worth noting the collator silently discards it today.

### B. `ignore_data_skip=True` means every run sees a different data order

The plan doesn't mention this, but `train.py` L234 sets `ignore_data_skip=True` plus a seed shift on resumption (`data_seed = 42 + len(checkpoints) * 1000`). This means resumed runs see fresh data permutations. The plan's data mixture changes are compatible with this, but adding new datasets (FineMath, The Stack) will change the shuffled interleave in ways that make ablation comparisons between runs unreliable.

### C. Current eval numbers reveal where the model actually hurts

From `eval_results.json`:
- **WinoGrande: 0.526** — near random (0.5 baseline). Commonsense reasoning is nearly absent.
- **ARC-Easy: 0.296** — below random (0.25 for 4-way MC). The model is anti-learning on factual QA.
- **ARC-Challenge: 0.221** — also below the ~0.25 random baseline.
- **HellaSwag: 0.336** — below random (0.25) but closer.
- **MBPP pass@1: 0.0** — zero correct code generation. CodeSearchNet at 20% weight and 300M total tokens hasn't helped at all.
- **Wikitext-2 PPL: 38.0** — high. For reference, GPT-2 124M achieves ~29.4 at this scale.

The ARC scores being **below random** strongly suggests either: (a) the model has a systematic bias from cross-document attention leakage (Phase 1.2 bug) confusing multi-choice answer distribution, or (b) there's a label-shift bug in the loss calculation. The plan's Phase 1 bug fixes are therefore likely more impactful than estimated.

The MBPP pass@1 = 0.0 supports Phase 3.2's push to The Stack — CodeSearchNet functions are too short and syntactically simple to teach real code generation.

---

## Summary Verdict

| Phase | Diagnosis Accuracy | Fix Correctness | Priority Assessment |
|---|---|---|---|
| 1.1 SequencePacking | ⚠️ Right bug, wrong example | ✅ Fix is correct | ✅ Correct: MUST DO |
| 1.2 Attention Masking | ✅ Correct | ⚠️ Safe but needs LR warmup on resume | ✅ Correct: MUST DO |
| 2.1 Gradient Clipping | ✅ Correct | ✅ Correct | ✅ Correct |
| 2.2 WSD Schedule | ❌ Self-contradictory | ❌ Fix doesn't match title | ⚠️ Misleading section |
| 2.3 Reduce Warmup | ✅ Correct | ✅ Correct | ✅ Correct |
| 3.1 Data Mixture Ratios | ⚠️ Wrong causal explanation | ✅ Fix is still valid | ✅ Correct |
| 3.2 Add FineMath | ✅ Correct | ✅ Correct | ✅ Correct |
| 4.1 Extend to 4K | ✅ Correct | ❌ RoPE claim is wrong for resumption | ⚠️ Needs explicit config override |
| 5.1 Remove from_pretrained | ❌ Misdiagnosed as dead code | ❌ Dangerous if `_initialize_missing_keys` also removed | ❌ Do not do this |
| 5.3 RoPE optimization | ✅ Correct | ⚠️ Implementation is non-standard | 🔵 Low priority |
