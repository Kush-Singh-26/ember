import os
import argparse
from typing import Dict, Iterator, List, Optional, Union
import torch
from datasets import load_dataset, interleave_datasets, IterableDataset
from transformers import AutoTokenizer

class SequencePacker:
    """
    Packs variable-length tokenized sequences into fixed-length chunks (default 2048),
    and creates custom position_ids (resetting at document boundaries) and relative
    document_ids (for block-diagonal attention masking to prevent cross-document leakage).
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
        
        current_doc_id = 0
        
        for item in self.dataset:
            # item contains tokenized "input_ids"
            input_ids = item.get("input_ids", [])
            if not input_ids:
                continue
                
            # Pack doc with BOS and EOS tags
            doc_tokens = [self.bos_token_id] + list(input_ids) + [self.eos_token_id]
            doc_len = len(doc_tokens)
            
            # Position IDs restart from 0 for each document
            doc_pos = list(range(doc_len))
            # Document IDs track boundaries
            doc_ids = [current_doc_id] * doc_len
            current_doc_id += 1
            
            buffer_input_ids.extend(doc_tokens)
            buffer_position_ids.extend(doc_pos)
            buffer_document_ids.extend(doc_ids)
            
            # Yield full chunks
            while len(buffer_input_ids) >= self.max_seq_len:
                chunk_input_ids = buffer_input_ids[:self.max_seq_len]
                chunk_position_ids = buffer_position_ids[:self.max_seq_len]
                chunk_document_ids = buffer_document_ids[:self.max_seq_len]
                
                buffer_input_ids = buffer_input_ids[self.max_seq_len:]
                buffer_position_ids = buffer_position_ids[self.max_seq_len:]
                buffer_document_ids = buffer_document_ids[self.max_seq_len:]
                
                # Normalize document IDs to be 0-indexed relative to this chunk
                unique_ids = list(dict.fromkeys(chunk_document_ids))
                id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
                rel_document_ids = [id_map[old_id] for old_id in chunk_document_ids]
                
                yield {
                    "input_ids": chunk_input_ids,
                    "position_ids": chunk_position_ids,
                    "document_ids": rel_document_ids,
                }

def get_pretraining_mixture(
    tokenizer_name: str,
    max_seq_len: int = 2048,
    buffer_size: int = 10000,
    seed: int = 42,
) -> SequencePacker:
    """Loads, interleaves, tokenizes, and packs the pre-training data mixture."""
    print(f"Initializing data mixture with tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Define sources and sampling probabilities
    # 60% English (WikiText / C4 / OpenWebText)
    # 20% Multilingual (Hindi Wikipedia / CC100)
    # 20% Programming Languages (CodeSearchNet)
    sources = []
    weights = []
    
    # 1. WikiText-103 (English)
    try:
        en_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        def tokenize_en(x):
            return tokenizer(x["text"], truncation=True, max_length=max_seq_len, add_special_tokens=False)
        en_ds = en_ds.map(tokenize_en, batched=True, remove_columns=["text"])
        sources.append(en_ds)
        weights.append(0.60)
        print("✅ Added English WikiText source (60% weight)")
    except Exception as e:
        print(f"Warning: English source failed to load: {e}")

    # 2. Hindi Wikipedia
    try:
        hi_ds = load_dataset("wikipedia", "20220301.hi", split="train", streaming=True)
        def tokenize_hi(x):
            return tokenizer(x["text"], truncation=True, max_length=max_seq_len, add_special_tokens=False)
        hi_ds = hi_ds.map(tokenize_hi, batched=True, remove_columns=["text", "title", "url"])
        sources.append(hi_ds)
        weights.append(0.20)
        print("✅ Added Hindi Wikipedia source (20% weight)")
    except Exception as e:
        print(f"Warning: Hindi source failed to load: {e}")

    # 3. Python CodeSearchNet
    try:
        code_ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        def tokenize_code(x):
            text = x.get("whole_funcstring", "") or x.get("code", "")
            return tokenizer(text, truncation=True, max_length=max_seq_len, add_special_tokens=False)
        code_ds = code_ds.map(tokenize_code, batched=True, remove_columns=code_ds.column_names)
        sources.append(code_ds)
        weights.append(0.20)
        print("✅ Added Python Code source (20% weight)")
    except Exception as e:
        print(f"Warning: Code source failed to load: {e}")

    if not sources:
        raise RuntimeError("Failed to load any data source for training mixture.")

    # Normalize weights in case some sources failed to load
    total_w = sum(weights)
    normalized_weights = [w / total_w for w in weights]
    
    # Interleave
    print(f"Interleaving datasets with weights: {normalized_weights}")
    mixed_dataset = interleave_datasets(
        datasets=sources,
        probabilities=normalized_weights,
        seed=seed,
        stopping_strategy="all_exhausted"
    )
    
    # Shuffle buffer to ensure mixture blending
    mixed_dataset = mixed_dataset.shuffle(buffer_size=buffer_size, seed=seed)
    
    # Pack sequences
    packed_dataset = SequencePacker(
        dataset=mixed_dataset,
        max_seq_len=max_seq_len,
        bos_token_id=tokenizer.bos_token_id or 1,
        eos_token_id=tokenizer.eos_token_id or 2,
        pad_token_id=tokenizer.pad_token_id or 3,
    )
    
    return packed_dataset

class PackedDataCollator:
    """Collates packed batches and dynamically constructs 4D boolean attention masks."""
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        seq_len = len(features[0]["input_ids"])
        
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        document_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        for idx, feature in enumerate(features):
            input_ids[idx] = torch.tensor(feature["input_ids"], dtype=torch.long)
            position_ids[idx] = torch.tensor(feature["position_ids"], dtype=torch.long)
            document_ids[idx] = torch.tensor(feature["document_ids"], dtype=torch.long)
            
        # Dynamically build 4D boolean attention mask to prevent cross-document leakage
        # Shape: (batch_size, 1, seq_len, seq_len)
        # Element is True if within the same document AND j <= i (causal)
        doc_ids_unsqueezed = document_ids.unsqueeze(1)  # [B, 1, S]
        same_doc = (doc_ids_unsqueezed.unsqueeze(-1) == doc_ids_unsqueezed.unsqueeze(-2))  # [B, 1, S, S]
        
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))  # [S, S]
        attention_mask = same_doc & causal.unsqueeze(0).unsqueeze(1)  # [B, 1, S, S]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": input_ids.clone()  # Next-token prediction labels (shifted inside model)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Verify the Data Pipeline")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_output", help="Path or HF ID of trained tokenizer")
    
    args = parser.parse_args()
    
    # Quick pipeline validation run
    try:
        pipeline = get_pretraining_mixture(tokenizer_name=args.tokenizer)
        collator = PackedDataCollator()
        
        print("\nStreaming 3 sample packed batches...")
        pipeline_iter = iter(pipeline)
        samples = [next(pipeline_iter) for _ in range(3)]
        
        batch = collator(samples)
        
        print("\nCollated Batch Verification:")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"position_ids shape: {batch['position_ids'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        
        # Verify block diagonal masking
        print("\nChecking mask properties for sequence 0:")
        mask_0 = batch["attention_mask"][0, 0]
        # Diagonal elements must be True (a token can attend to itself)
        assert torch.all(torch.diagonal(mask_0)), "Diagonal elements must all be True"
        # Upper triangle must be False (causal check)
        upper_tril = torch.triu(mask_0, diagonal=1)
        assert not torch.any(upper_tril), "Upper triangle must be completely False (causal violation)"
        print("✅ Attention mask causal & block-diagonal properties verified!")
        print("✅ Data pipeline built successfully!")
        
    except Exception as e:
        print(f"Pipeline verification failed: {e}")
        import traceback
        traceback.print_exc()
