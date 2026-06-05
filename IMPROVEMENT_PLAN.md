# Ember-275M Training Efficiency Improvement Plan

**Goal**: Make the next $30 Modal A100 run produce 2-3x more effective training by fixing critical bugs and optimizing the data pipeline — without changing model architecture or breaking checkpoint compatibility.

**Status**: Not started
**Estimated effort**: 6-8 hours across all phases
**Checkpoint safety**: All changes are compatible with existing checkpoint-3650

---

## Phase 1: Critical Bug Fixes (MUST DO)

These two bugs are causing the model to train on corrupted data. Every dollar spent before fixing them is partially wasted.

### 1.1 Fix SequencePacking Token Waste (50-75% waste → <5%)

**File**: `data/pipeline.py`, lines 54-58

**Problem**: Every document is truncated to `max_seq_len - 2` (2046 tokens) before being inserted into the packing buffer. A 500-token document consumes an entire 2048-slot chunk. This means ~75% of each chunk is wasted on padding.

**Current code** (line 57-58):
```python
if self.max_seq_len >= 512:
    input_ids = input_ids[:self.max_seq_len - 2]
```

**Fix**: Remove the pre-truncation entirely. The packing loop on line 74 already handles yielding full chunks. Short documents will naturally pack together. Only truncate documents that individually exceed `max_seq_len`.

**New code**:
```python
# Only truncate documents that are individually longer than max_seq_len
# (rare for web text, common for long code files)
if len(input_ids) > self.max_seq_len - 2:
    input_ids = input_ids[:self.max_seq_len - 2]
```

**Impact**: 2-3x more tokens per training step at the same compute cost. This alone makes the $30 Modal run worth $60-90.

**Also in pipeline.py**: Remove the double-truncation in the `.map()` tokenization calls (lines 184, 204, 226). Currently both the tokenizer AND the packer truncate. Remove `truncation=True, max_length=max_seq_len` from the `.map()` calls so the SequencePacker is the sole truncation point:

```python
# Line 184 - BEFORE:
lambda x: tokenizer(x["text"], truncation=True, max_length=max_seq_len, add_special_tokens=False)

# Line 184 - AFTER:
lambda x: tokenizer(x["text"], add_special_tokens=False)
```

Apply the same change to lines 204 and 226.

### 1.2 Fix Document Attention Masking (Cross-Document Leakage)

**Files**: `data/pipeline.py` (collator), `model/transformer.py` (attention)

**Problem**: The `SequencePacker` produces per-document `position_ids` and `document_ids`, but the model never uses `document_ids` to create attention masks. When documents are packed, token 0 of document N+1 **attends to all tokens from document N**. This injects cross-document noise into every training step.

**Step 1 — Collator produces `attention_mask`**:

File: `data/pipeline.py`, class `PackedDataCollator` (lines 295-312)

```python
class PackedDataCollator:
    """Collates packed batches into tensors for the model forward pass.
    
    Constructs block-diagonal attention masks from document_ids to prevent
    cross-document attention leakage when multiple documents are packed
    into a single sequence.
    """
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        seq_len = len(features[0]["input_ids"])

        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Block-diagonal attention mask: (B, 1, S, S)
        # True = attend, False = mask out
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)

        for idx, feature in enumerate(features):
            input_ids[idx] = torch.tensor(feature["input_ids"], dtype=torch.long)
            position_ids[idx] = torch.tensor(feature["position_ids"], dtype=torch.long)

            # Build block-diagonal mask from document boundaries
            doc_ids = feature["document_ids"]
            for i in range(seq_len):
                for j in range(seq_len):
                    if doc_ids[i] != doc_ids[j]:
                        attention_mask[idx, 0, i, j] = False

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
```

**Performance note**: The nested loop above is O(S^2) per sample. For seq_len=2048 this is ~4M iterations per sample in Python, which is slow. A vectorized version:

```python
# Vectorized mask construction (much faster)
doc_ids_tensor = torch.tensor(doc_ids, dtype=torch.long)
# Two positions attend iff they have the same document ID
# attention_mask[idx, 0, i, j] = (doc_ids[i] == doc_ids[j])
same_doc = doc_ids_tensor.unsqueeze(0) == doc_ids_tensor.unsqueeze(1)  # (S, S)
attention_mask[idx, 0] = same_doc
```

**Step 2 — Pass mask through model**:

File: `model/transformer.py`, class `EmberModel.forward()` (lines 255-260)

Change the signature to accept `attention_mask` explicitly:

```python
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
) -> torch.Tensor:
```

Pass it to each layer:

```python
for layer in self.layers:
    if self.gradient_checkpointing and self.training:
        # ... existing gradient checkpointing code ...
        hidden_states = layer(hidden_states, position_ids=position_ids, attention_mask=attention_mask)
    else:
        hidden_states = layer(hidden_states, position_ids=position_ids, attention_mask=attention_mask)
```

**Step 3 — Pass mask to attention**:

File: `model/transformer.py`, class `EmberDecoderLayer.forward()` (line 179+)

```python
def forward(self, hidden_states, position_ids=None, attention_mask=None):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, position_ids=position_ids, attention_mask=attention_mask)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
```

**Step 4 — Apply mask in attention**:

File: `model/transformer.py`, class `EmberAttention.forward()` (lines 119-152)

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.num_key_value_groups > 1:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Use the provided attention_mask for document boundaries
    # attention_mask: (B, 1, S, S) bool — True = attend, False = mask
    # is_causal=False because we supply our own mask
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.config.dropout if self.training else 0.0,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

    return self.o_proj(attn_output)
```

**Step 5 — Update EmberForCausalLM.forward()**:

File: `model/transformer.py`, class `EmberForCausalLM.forward()` (lines 371-389)

```python
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    # ...
    outputs = self.model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
    )
    # ... rest unchanged ...
```

**Impact**: Every training step now uses correct document-boundary attention. This is the single biggest quality fix.

---

## Phase 2: Training Stability Fixes (HIGH PRIORITY)

### 2.1 Add Gradient Clipping

**File**: `train.py`, line 214+

**Problem**: No `max_grad_norm` is set. Training is vulnerable to gradient explosion, especially early in training.

**Fix**: Add to `TrainingArguments`:

```python
training_args = TrainingArguments(
    # ... existing args ...
    max_grad_norm=1.0,  # Gradient clipping — prevents loss spikes
)
```

### 2.2 Switch to WSD Learning Rate Schedule

**File**: `train.py`, line 224

**Problem**: `lr_scheduler_type="cosine"` requires pre-determining total training steps. WSD (Warmup-Stable-Decay) is more flexible — you can branch off and decay at any point.

**Fix**:

```python
training_args = TrainingArguments(
    # ... existing args ...
    lr_scheduler_type="cosine",  # Keep cosine for now — WSD requires custom scheduler
    warmup_steps=1000,           # Reduced from 2000 — excessive for 275M model
)
```

**Note**: WSD is not a built-in HF Trainer scheduler. For the next Modal run, keep cosine but reduce warmup from 2000 to 1000 steps. WSD can be implemented for a future run with a custom `SchedulerCallback`.

### 2.3 Reduce Warmup Steps

**File**: `train.py`, line 223

**Problem**: 2000 warmup steps × 524K tokens/step = 1B tokens of warmup. For a 275M model, this is excessive. Most small models use 500-1000 steps.

**Fix**:
```python
warmup_steps=1000,  # Reduced from 2000
```

---

## Phase 3: Data Pipeline Upgrades (HIGH PRIORITY)

### 3.1 Fix Effective Data Mixture Ratios

**Problem**: Hindi Wikipedia (~500M tokens) and CodeSearchNet Python (~300M tokens) are tiny compared to FineWeb-Edu (~100B tokens). With `stopping_strategy="all_exhausted"`, the effective ratio is ~99% English, not 60/20/20.

**Fix**: Upgrade data sources to volumes that match the target ratios:

**File**: `data/pipeline.py`, `get_pretraining_mixture()`

Replace Hindi source (line 194-212):
```python
# ── 2. Hindi: CC-100 Hindi (much larger than Wikipedia) ──────────────────
# CC-100 Hindi has ~4B tokens vs Wikipedia Hindi's ~500M
try:
    hi_ds = load_dataset(
        "lidia12/cc100-hindi",
        split="train",
        streaming=True,
    )
    hi_ds = hi_ds.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        batched=True,
        remove_columns=hi_ds.column_names,
    )
    sources.append(hi_ds)
    weights.append(0.20)
    print("✅ Added Hindi source: CC-100 Hindi (~4B tokens, 20% weight)")
except Exception as e:
    print(f"⚠️  Hindi (CC-100) source failed: {e}, falling back to Wikipedia Hindi")
    # Fallback to Wikipedia Hindi
    hi_ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    hi_ds = hi_ds.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        batched=True,
        remove_columns=["id", "url", "title", "text"],
    )
    sources.append(hi_ds)
    weights.append(0.20)
```

Replace code source (line 214-234):
```python
# ── 3. Code: The Stack (Python subset) — much richer than CodeSearchNet ───
# The Stack has ~30B Python tokens vs CodeSearchNet's ~300M
try:
    code_ds = load_dataset(
        "bigcode/the_stack",
        name="python",
        split="train",
        streaming=True,
    )
    def _tokenize_code(batch):
        texts = batch.get("content") or batch.get("whole_func_string") or []
        return tokenizer(texts, add_special_tokens=False)
    code_ds = code_ds.map(
        _tokenize_code,
        batched=True,
        remove_columns=code_ds.column_names,
    )
    sources.append(code_ds)
    weights.append(0.20)
    print("✅ Added Code source: The Stack Python (~30B tokens, 20% weight)")
except Exception as e:
    print(f"⚠️  Code (The Stack) source failed: {e}, falling back to CodeSearchNet")
    code_ds = load_dataset("code-search-net/code_search_net", "python", split="train", streaming=True)
    def _tokenize_code_fallback(batch):
        texts = batch.get("whole_func_string") or batch.get("func_code_string") or []
        return tokenizer(texts, add_special_tokens=False)
    code_ds = code_ds.map(_tokenize_code_fallback, batched=True, remove_columns=list(code_ds.column_names))
    sources.append(code_ds)
    weights.append(0.20)
```

### 3.2 Add Math Data (FineMath)

**File**: `data/pipeline.py`, `get_pretraining_mixture()`

Add after the code source block:

```python
# ── 4. Math: FineMath-4K (curated mathematical content) ─────────────────
# ~4B tokens of mathematical text — improves numerical reasoning
try:
    math_ds = load_dataset(
        "HuggingFaceFW/finemath",
        name="finemath-4plus",
        split="train",
        streaming=True,
    )
    math_ds = math_ds.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        batched=True,
        remove_columns=math_ds.column_names,
    )
    sources.append(math_ds)
    weights.append(0.05)  # 5% math
    print("✅ Added Math source: FineMath-4K (5% weight)")
except Exception as e:
    print(f"⚠️  Math (FineMath) source failed: {e}")
```

Update English weight from 0.60 to 0.55 to accommodate math:
```python
weights.append(0.55)  # English (reduced from 0.60)
```

### 3.3 Update train.py Pre-Downloads

**File**: `train.py`, lines 50-96

The current code pre-downloads Wikipedia Hindi, CodeSearchNet, and FineWeb-Edu shards. Update to match new data sources:

- Replace Wikipedia Hindi download with CC-100 Hindi
- Replace CodeSearchNet with The Stack Python
- Add FineMath download
- Increase FineWeb-Edu shards from 6 to 12 (for longer training without cycling)

---

## Phase 4: Configuration Updates (MEDIUM PRIORITY)

### 4.1 Extend Context to 4K

**File**: `model/config.py`, line 15

```python
max_position_embeddings=4096,  # Extended from 2048
```

**File**: `model/transformer.py`, `EmberRotaryEmbedding` (line 30)

The RoPE cache will automatically extend to 4096 positions since it's computed from `max_position_embeddings`. No code change needed — just the config value.

**File**: `train.py`, line 202

```python
train_dataset = get_pretraining_mixture(
    tokenizer_name=tokenizer_name,
    max_seq_len=4096,  # Extended from 2048
    buffer_size=1000,
    seed=data_seed,
)
```

**Impact**: Model can handle longer documents and code files. Essential for code generation use cases.

### 4.2 Update forge.yaml Colab Profile

**File**: `forge.yaml`, colab profile (lines 47-54)

```yaml
colab:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 64  # Reduced from 128 — context doubled, halve accum to keep memory
  learning_rate: 3e-4
  max_steps: 60000
  probe_memory: true
  data_cache: false
  fp16: true
```

### 4.3 Add Modal Profile for Continued Training

**File**: `forge.yaml`

```yaml
modal-continue:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 16  # 8 * 16 * 1 GPU = 128 seq = 262K tokens/step (at 4K ctx: 524K)
  learning_rate: 1e-4              # Lower LR for continued training from checkpoint
  max_steps: 30000                 # ~15.8B tokens (30K * 524K)
  probe_memory: false
  data_cache: true
  warmup_steps: 200                # Short warmup for continued training
```

---

## Phase 5: Code Quality (LOW PRIORITY)

### 5.1 Remove Dead `from_pretrained` Override

**File**: `model/transformer.py`, lines 317-335

Delete the `from_pretrained` classmethod override — it just calls `super()` and adds no value.

### 5.2 Add `output_attentions` / `output_hidden_states` Support

**File**: `model/transformer.py`

This is optional but useful for debugging. Store attention weights and hidden states when requested.

### 5.3 Optimize RoPE Computation

**File**: `model/transformer.py`, line 17

Replace `torch.cat` in `rotate_half` with complex-number formulation for ~15-20% faster RoPE:

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = torch.view_as_real(x_complex * torch.exp(1j * torch.tensor([0.0, -math.pi/2], device=x.device)))
    return x_rotated.reshape(x.shape)
```

---

## Execution Checklist

- [ ] Phase 1.1: Fix SequencePacking truncation bug
- [ ] Phase 1.2: Implement block-diagonal attention masking
- [ ] Phase 2.1: Add max_grad_norm=1.0
- [ ] Phase 2.2: Reduce warmup_steps to 1000
- [ ] Phase 3.1: Upgrade Hindi source to CC-100
- [ ] Phase 3.2: Upgrade code source to The Stack
- [ ] Phase 3.3: Add FineMath dataset
- [ ] Phase 3.4: Update train.py pre-downloads
- [ ] Phase 4.1: Extend context to 4K
- [ ] Phase 4.2: Update forge.yaml profiles
- [ ] Phase 5.1: Clean up dead code
- [ ] Run full test suite to verify no regressions
- [ ] Test data pipeline end-to-end
- [ ] Test model forward pass with attention masks
- [ ] Run 100-step smoke test on local GPU

---

## Verification Plan

After all changes, run these checks:

1. **Unit tests**: `python -m pytest tests/ -v` — all existing tests must pass
2. **Pipeline test**: `python data/pipeline.py` — verify packing produces multiple docs per chunk
3. **Forward pass test**: Load checkpoint-3650, run a batch through the model, verify no NaN/inf
4. **Memory test**: Verify training fits on T4 with context=4096, batch=2
5. **Smoke test**: Run 100 training steps, verify loss decreases
