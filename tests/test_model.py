import pytest
import torch
from model.config import EmberConfig
from model.transformer import EmberForCausalLM, EmberModel

def test_model_config():
    config = EmberConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256
    )
    assert config.vocab_size == 1000
    assert config.hidden_size == 128
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.head_dim == 32
    assert config.intermediate_size == 256

def test_model_parameter_count():
    # Test that tied embeddings correctly reduce parameter count
    config = EmberConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        tie_word_embeddings=True
    )
    model = EmberForCausalLM(config)
    
    # Check that embedding weight and lm_head weight are the same object
    assert model.model.embed_tokens.weight is model.lm_head.weight
    
    # Let's count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")

def test_model_forward():
    config = EmberConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256
    )
    model = EmberForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

def test_model_forward_with_labels():
    config = EmberConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256
    )
    model = EmberForCausalLM(config)
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    assert outputs.loss is not None
    assert outputs.loss > 0.0
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

def test_document_masking():
    # Test pre-training with explicit block-diagonal attention mask (document boundary masking)
    config = EmberConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256
    )
    model = EmberForCausalLM(config)
    model.eval()

    batch_size = 1
    seq_len = 8
    
    # We pack two documents: doc1 of length 5, doc2 of length 3
    # position_ids: [0, 1, 2, 3, 4, 0, 1, 2]
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]], dtype=torch.long)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Create block-diagonal boolean attention mask: shape (batch_size, 1, seq_len, seq_len)
    # True = allowed, False = masked
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    # Doc 1 (indices 0 to 4): causal masking within doc1
    for i in range(5):
        for j in range(i + 1):
            mask[0, 0, i, j] = True
            
    # Doc 2 (indices 5 to 7): causal masking within doc2
    for i in range(3):
        for j in range(i + 1):
            mask[0, 0, 5 + i, 5 + j] = True
            
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=mask, position_ids=position_ids)
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

def test_gradient_checkpointing():
    config = EmberConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256
    )
    model = EmberForCausalLM(config)
    model.model.gradient_checkpointing = True
    model.train()
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    assert loss is not None
    loss.backward()
    
    # Check that gradients are computed for parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
