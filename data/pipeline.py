import os
import itertools
import argparse
import queue
import threading
from typing import Dict, Iterator, List, Union
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoTokenizer

class SequencePacker(TorchIterableDataset):
    """
    Packs variable-length tokenized sequences into fixed-length chunks (default 2048),
    and creates custom position_ids (resetting at document boundaries).

    Supports proper multi-process sharding for both DDP (rank-based) and
    DataLoader workers (worker_id-based). Without rank sharding, both GPUs
    in DDP would train on identical batches, wasting half the compute.
    """
    def __init__(
        self,
        dataset: IterableDataset,
        max_seq_len: int = 2048,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 3,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        buffer_input_ids = []
        buffer_position_ids = []
        buffer_document_ids = []
        doc_counter = 0

        # --- Sharding: dataloader workers (worker_id-based) ---
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            iterator = itertools.islice(self.dataset, worker_info.id, None, worker_info.num_workers)
        else:
            iterator = iter(self.dataset)

        for item in iterator:
            input_ids = item.get("input_ids", [])
            if not input_ids:
                continue

            # Only truncate documents that individually exceed max_seq_len (leaving room for BOS + EOS).
            # Short documents are packed together naturally by the buffer loop below — no pre-truncation
            # needed. This fixes the bug where a 500-token doc was capped at 2046 tokens, wasting ~75%
            # of a 2048-token chunk. Now multiple short docs fill each chunk efficiently.
            if len(input_ids) > self.max_seq_len - 2:
                input_ids = input_ids[:self.max_seq_len - 2]

            # Pack doc with BOS and EOS tags
            doc_tokens = [self.bos_token_id] + list(input_ids) + [self.eos_token_id]
            doc_len = len(doc_tokens)

            # Position IDs restart from 0 for each document
            doc_pos = list(range(doc_len))
            doc_ids = [doc_counter] * doc_len
            doc_counter += 1

            buffer_input_ids.extend(doc_tokens)
            buffer_position_ids.extend(doc_pos)
            buffer_document_ids.extend(doc_ids)

            # Yield full chunks
            while len(buffer_input_ids) >= self.max_seq_len:
                chunk_input_ids = buffer_input_ids[:self.max_seq_len]
                chunk_position_ids = buffer_position_ids[:self.max_seq_len]
                chunk_doc_ids = buffer_document_ids[:self.max_seq_len]

                buffer_input_ids = buffer_input_ids[self.max_seq_len:]
                buffer_position_ids = buffer_position_ids[self.max_seq_len:]
                buffer_document_ids = buffer_document_ids[self.max_seq_len:]

                # Map the document IDs to be 0-indexed within this chunk
                unique_ids = []
                id_map = {}
                for d_id in chunk_doc_ids:
                    if d_id not in id_map:
                        id_map[d_id] = len(unique_ids)
                        unique_ids.append(d_id)
                relative_doc_ids = [id_map[d_id] for d_id in chunk_doc_ids]

                yield {
                    "input_ids": chunk_input_ids,
                    "position_ids": chunk_position_ids,
                    "document_ids": relative_doc_ids,
                }

def gen_sharded(dataset, num_shards, index):
    for i, item in enumerate(dataset):
        if i % num_shards == index:
            yield item

class ThreadedPrefetcher(TorchIterableDataset):
    """
    Wraps an IterableDataset to pre-fetch items asynchronously in a background thread.
    This avoids PyTorch dataloader worker deadlocks while hiding all network/I/O latency.
    """
    def __init__(self, dataset, buffer_size=16):
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        q = queue.Queue(maxsize=self.buffer_size)
        sentinel = object()

        def producer():
            try:
                for item in self.dataset:
                    q.put(item)
            except Exception as e:
                q.put(e)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

def get_pretraining_mixture(
    tokenizer_name: str,
    max_seq_len: int = 2048,
    buffer_size: int = 10_000,
    seed: int = 42,
) -> SequencePacker:
    """
    Loads, interleaves, tokenizes, and packs the pre-training data mixture.

    Data mixture (55% English / 20% Hindi / 20% Code / 5% Math):
      - English: FineWeb-Edu sample-100BT (~100B tokens, Parquet, streaming or local)
      - Hindi:   CC-100 Hindi (~4B tokens, streaming) with Wikipedia Hindi fallback
      - Code:    The Stack Python (~30B tokens, streaming) with CodeSearchNet fallback
      - Math:    FineMath-4+ (~4B tokens, streaming)

    All sources use streaming=True to avoid large disk downloads (critical for Kaggle
    which has ~100GB disk). FineWeb-Edu is read from pre-downloaded local Parquet shards
    when available, otherwise streamed from HF Hub.
    """
    print(f"Initializing data mixture with tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    sources = []
    weights = []

    # ── 1. English: FineWeb-Edu (Parquet, local or streaming) ───────────────
    # Higher quality than C4: educational content filtered by Llama3-70B classifier.
    # Outperforms C4 on MMLU, ARC, OpenBookQA benchmarks. Already in Parquet format.
    # If local Parquet shards exist (pre-downloaded by train.py), read from disk
    # to eliminate HF Hub network latency. Otherwise fall back to streaming.
    fineweb_dir = os.environ.get("FORGE_DATA_DIR", "/kaggle/working/fineweb-edu-parquet")
    try:
        if os.path.isdir(fineweb_dir) and os.listdir(fineweb_dir):
            print(f"Loading FineWeb-Edu from local Parquet: {fineweb_dir}")
            en_ds = load_dataset(
                "parquet",
                data_dir=fineweb_dir,
                split="train",
                streaming=True,
            )
        else:
            print("Loading FineWeb-Edu from HF Hub (streaming)...")
            en_ds = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-100BT",
                split="train",
                streaming=True,
            )
        en_ds = en_ds.map(
            lambda x: tokenizer(x["text"], add_special_tokens=False),
            batched=True,
            remove_columns=en_ds.column_names,
        )
        sources.append(en_ds)
        weights.append(0.55)  # Reduced from 0.60 to accommodate FineMath
        print("✅ Added English source: FineWeb-Edu sample-100BT (55% weight)")
    except Exception as e:
        print(f"⚠️  English (FineWeb-Edu) source failed: {e}")

    # ── 2. Hindi: CC-100 Hindi (~4B tokens) ──────────────────────────────────
    # Upgraded from Wikipedia Hindi (~500M tokens) — 8x more data reduces repetition overfitting.
    # Falls back to Wikipedia Hindi if CC-100 is unavailable.
    try:
        hi_ds = load_dataset(
            "statmt/cc100",
            "hi",
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
        try:
            wiki_dir = "/kaggle/working/wikipedia-hi-parquet"
            if os.path.isdir(wiki_dir) and os.listdir(wiki_dir):
                print(f"Loading Wikipedia Hindi from local Parquet: {wiki_dir}")
                hi_ds = load_dataset("parquet", data_dir=wiki_dir, split="train", streaming=True)
            else:
                print("Loading Wikipedia Hindi from HF Hub (streaming)...")
                hi_ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
            hi_ds = hi_ds.map(
                lambda x: tokenizer(x["text"], add_special_tokens=False),
                batched=True,
                remove_columns=hi_ds.column_names,
            )
            sources.append(hi_ds)
            weights.append(0.20)
            print("✅ Added Hindi source: Wikipedia Hindi (fallback, 20% weight)")
        except Exception as e2:
            print(f"⚠️  Hindi fallback also failed: {e2}")

    # ── 3. Code: The Stack Python (~30B tokens) ───────────────────────────────
    # Upgraded from CodeSearchNet (~300M tokens, ~180k functions) — 100x more data.
    # The Stack requires HF authentication for gated access. Falls back to CodeSearchNet.
    try:
        code_ds = load_dataset(
            "bigcode/the-stack",
            name="data/python",
            split="train",
            streaming=True,
        )
        code_ds = code_ds.map(
            lambda x: tokenizer(x["content"], add_special_tokens=False),
            batched=True,
            remove_columns=code_ds.column_names,
        )
        sources.append(code_ds)
        weights.append(0.20)
        print("✅ Added Code source: The Stack Python (~30B tokens, 20% weight)")
    except Exception as e:
        print(f"⚠️  Code (The Stack) source failed: {e}, falling back to CodeSearchNet")
        try:
            code_dir = "/kaggle/working/codesearchnet-parquet"
            if os.path.isdir(code_dir) and os.listdir(code_dir):
                print(f"Loading CodeSearchNet Python from local Parquet: {code_dir}")
                code_ds = load_dataset("parquet", data_dir=code_dir, split="train", streaming=True)
            else:
                print("Loading CodeSearchNet Python from HF Hub (streaming)...")
                code_ds = load_dataset("code-search-net/code_search_net", "python", split="train", streaming=True)
            def _tokenize_code(batch):
                texts = batch.get("whole_func_string") or batch.get("func_code_string") or []
                return tokenizer(texts, add_special_tokens=False)
            code_ds = code_ds.map(
                _tokenize_code,
                batched=True,
                remove_columns=list(code_ds.column_names),
            )
            sources.append(code_ds)
            weights.append(0.20)
            print("✅ Added Code source: CodeSearchNet Python (fallback, 20% weight)")
        except Exception as e2:
            print(f"⚠️  Code fallback also failed: {e2}")

    # ── 4. Math: FineMath-4+ (~4B tokens of curated mathematical content) ──────
    # Improves numerical reasoning, which is currently near-random on benchmarks.
    # 5% weight keeps it as a useful supplement without overwhelming other sources.
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
        print("✅ Added Math source: FineMath-4+ (~4B tokens, 5% weight)")
    except Exception as e:
        print(f"⚠️  Math (FineMath) source failed: {e}")

    if not sources:
        raise RuntimeError("Failed to load any data source for training mixture.")

    # Normalize weights in case some sources failed to load
    total_w = sum(weights)
    normalized_weights = [w / total_w for w in weights]

    # ── Rank Sharding ─────────────────────────────────────────────────────────
    # Shard at the DDP Rank level first to avoid redundant downloads in multi-GPU training
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if world_size > 1:
        sharded_sources = []
        for src in sources:
            try:
                # Use HF's native server-side file-level sharding
                sharded_sources.append(src.shard(num_shards=world_size, index=rank))
            except Exception:
                # Fallback to sample-level sharding (for single-file datasets like CodeSearchNet)
                sharded = IterableDataset.from_generator(
                    gen_sharded,
                    gen_kwargs={"dataset": src, "num_shards": world_size, "index": rank},
                    features=src.features
                )
                sharded_sources.append(sharded)
        sources = sharded_sources

    # ── Interleave ────────────────────────────────────────────────────────────
    # stopping_strategy="all_exhausted": smaller datasets (code, hindi) cycle until
    # the largest (FineWeb-Edu) is exhausted. Since FineWeb-Edu sample-100BT has ~100B tokens
    # steps (~30B tokens), FineWeb-Edu will never exhaust — the Trainer's max_steps stops us.
    print(f"Interleaving datasets with weights: {[f'{w:.2f}' for w in normalized_weights]}")
    mixed_dataset = interleave_datasets(
        datasets=sources,
        probabilities=normalized_weights,
        seed=seed,
        stopping_strategy="all_exhausted",
    )

    # Shuffle buffer: randomizes the interleave pattern so batches aren't [en,hi,code,en,hi,code,...]
    # buffer_size=10_000 keeps 10k examples in RAM and randomly samples from them.
    mixed_dataset = mixed_dataset.shuffle(seed=seed, buffer_size=buffer_size)

    # ── Pack sequences ────────────────────────────────────────────────────────
    packed_dataset = SequencePacker(
        dataset=mixed_dataset,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 3,
    )

    # Wrap the packed dataset in a ThreadedPrefetcher to fetch and pack batches
    # asynchronously in a background thread, preventing training deadlocks/network waits.
    return ThreadedPrefetcher(packed_dataset, buffer_size=256)

class PackedDataCollator:
    """Collates packed batches into tensors for the model forward pass.

    Constructs a block-diagonal causal attention mask from document_ids to prevent
    cross-document attention leakage when multiple documents are packed into a single
    sequence. Each token only attends to tokens in the same document that come at or
    before its position (causal within-document attention).
    """
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        seq_len = len(features[0]["input_ids"])

        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Block-diagonal causal mask: (B, 1, S, S), bool — True = attend, False = mask.
        # Shape (B, 1, S, S) broadcasts correctly to (B, num_heads, S, S) in SDPA.
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)

        # Base causal mask: token i can only attend to token j where j <= i.
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        for idx, feature in enumerate(features):
            input_ids[idx] = torch.tensor(feature["input_ids"], dtype=torch.long)
            position_ids[idx] = torch.tensor(feature["position_ids"], dtype=torch.long)

            # Vectorized block-diagonal mask: attend iff same document AND causal.
            # doc_ids_tensor[i] == doc_ids_tensor[j] → same_doc[i, j] = True
            doc_ids_tensor = torch.tensor(feature["document_ids"], dtype=torch.long)
            same_doc = doc_ids_tensor.unsqueeze(1) == doc_ids_tensor.unsqueeze(0)  # (S, S)
            attention_mask[idx, 0] = same_doc & causal_mask

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Verify the Data Pipeline")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_output", help="Path or HF ID of trained tokenizer")
    args = parser.parse_args()

    try:
        pipeline = get_pretraining_mixture(tokenizer_name=args.tokenizer)
        collator = PackedDataCollator()

        print("\nStreaming 3 sample packed batches...")
        pipeline_iter = iter(pipeline)
        samples = [next(pipeline_iter) for _ in range(3)]

        batch = collator(samples)

        print("\nCollated Batch Verification:")
        print(f"  input_ids shape:   {batch['input_ids'].shape}")
        print(f"  position_ids shape: {batch['position_ids'].shape}")
        print(f"  labels shape:       {batch['labels'].shape}")
        print("✅ Data pipeline built successfully!")

    except Exception as e:
        print(f"Pipeline verification failed: {e}")
        import traceback
        traceback.print_exc()
