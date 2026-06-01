import pytest
import torch
from datasets import IterableDataset
from data.pipeline import SequencePacker, PackedDataCollator

def create_mock_iterable_dataset(sequences):
    def gen():
        for seq in sequences:
            yield {"input_ids": seq}
    # Wrap in simple generator
    class MockIterableDataset(IterableDataset):
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            for x in self.data:
                yield {"input_ids": x}
    return MockIterableDataset(sequences)

def test_sequence_packer_packing():
    # 2 documents of different lengths: length 3 and length 5
    # Tokenizer outputs:
    # doc 1: [10, 11, 12]
    # doc 2: [20, 21, 22, 23, 24]
    doc1 = [10, 11, 12]
    doc2 = [20, 21, 22, 23, 24]
    
    mock_ds = create_mock_iterable_dataset([doc1, doc2])
    
    # We pack with max_seq_len=6
    # doc1 is wrapped with BOS=1 and EOS=2 -> [1, 10, 11, 12, 2] (len 5)
    # doc2 is wrapped with BOS=1 and EOS=2 -> [1, 20, 21, 22, 23, 24, 2] (len 7)
    # Total tokens in buffer: 5 + 7 = 12 tokens
    # With max_seq_len=6, we should get exactly 2 chunks of length 6
    packer = SequencePacker(mock_ds, max_seq_len=6, bos_token_id=1, eos_token_id=2, pad_token_id=3)
    
    chunks = list(packer)
    assert len(chunks) == 2
    
    # Verify chunk 1
    # First doc: [1, 10, 11, 12, 2] (indices 0 to 4)
    # Second doc starts: [1] (index 5)
    # Total = 6 tokens
    chunk1 = chunks[0]
    assert chunk1["input_ids"] == [1, 10, 11, 12, 2, 1]
    # Position IDs:
    # First doc: [0, 1, 2, 3, 4]
    # Second doc: [0]
    assert chunk1["position_ids"] == [0, 1, 2, 3, 4, 0]
    # Relative Document IDs:
    # First doc tokens belong to doc 0
    # Second doc tokens belong to doc 1
    assert chunk1["document_ids"] == [0, 0, 0, 0, 0, 1]

    # Verify chunk 2
    # Second doc continues: [20, 21, 22, 23, 24, 2] (len 6)
    chunk2 = chunks[1]
    assert chunk2["input_ids"] == [20, 21, 22, 23, 24, 2]
    # Position IDs:
    # Starts at 1 since 0 was consumed in chunk1
    assert chunk2["position_ids"] == [1, 2, 3, 4, 5, 6]
    # Relative Document IDs:
    # All tokens belong to the same document in this chunk, so document_ids should be [0, 0, 0, 0, 0, 0]
    assert chunk2["document_ids"] == [0, 0, 0, 0, 0, 0]

def test_packed_data_collator_masking():
    # Construct a sample feature chunk
    # input_ids: [10, 11, 12, 20, 21, 22] (len 6)
    # position_ids: [0, 1, 2, 0, 1, 2] (resets at index 3)
    # document_ids: [0, 0, 0, 1, 1, 1] (doc boundary at index 3)
    feature = {
        "input_ids": [10, 11, 12, 20, 21, 22],
        "position_ids": [0, 1, 2, 0, 1, 2],
        "document_ids": [0, 0, 0, 1, 1, 1]
    }
    
    collator = PackedDataCollator()
    batch = collator([feature])
    
    # Check shape
    assert batch["input_ids"].shape == (1, 6)
    assert batch["attention_mask"].shape == (1, 1, 6, 6)
    
    mask = batch["attention_mask"][0, 0]
    
    # Let's inspect the causal block-diagonal properties:
    # Tokens 0, 1, 2 belong to Doc 0
    # Tokens 3, 4, 5 belong to Doc 1
    
    # 1. Causal properties
    # Cannot attend to future tokens (upper triangle must be False)
    assert not mask[0, 1]
    assert not mask[1, 2]
    assert not mask[3, 4]
    
    # 2. Block-diagonal properties (cross-document leakage check)
    # Token 3 (Doc 1) cannot attend to Token 0 (Doc 0), Token 1 (Doc 0), Token 2 (Doc 0)
    assert not mask[3, 0]
    assert not mask[3, 1]
    assert not mask[3, 2]
    
    # Token 4 (Doc 1) cannot attend to Token 0, 1, 2
    assert not mask[4, 0]
    assert not mask[4, 1]
    assert not mask[4, 2]
    
    # 3. Causal within same document check
    # Token 1 (Doc 0) can attend to Token 0 (Doc 0) and Token 1 (Doc 0)
    assert mask[1, 0]
    assert mask[1, 1]
    
    # Token 4 (Doc 1) can attend to Token 3 (Doc 1) and Token 4 (Doc 1)
    assert mask[4, 3]
    assert mask[4, 4]
    
    print("✅ PackedDataCollator mask construction test passed!")
