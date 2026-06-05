import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os
import sys

print("Downloading checkpoint...")
checkpoint_dir = snapshot_download(
    repo_id="Kush26/ember-checkpoints",
    revision="checkpoints",
    allow_patterns="checkpoints/ember-275m/checkpoint-3650/*"
)
model_path = os.path.join(checkpoint_dir, "checkpoints/ember-275m/checkpoint-3650")

print("Importing model architecture...")
from model.transformer import EmberForCausalLM

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EmberForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("Kush26/ember-tokenizer")

prompt = "def binary_search(arr, target):"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Running forward pass to trace NaN...")
with torch.no_grad():
    hidden_states = model.model.embed_tokens(inputs["input_ids"])
    print("Embeddings NaN?", torch.isnan(hidden_states).any().item())
    
    position_ids = torch.arange(inputs["input_ids"].shape[1], dtype=torch.long, device=device).unsqueeze(0)
    
    def check(name, tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        max_val = tensor.abs().max().item() if not has_nan else 'NaN'
        print(f"{name}: NaN={has_nan}, Inf={has_inf}, Max={max_val}")

    hidden_states = model.model.embed_tokens(inputs["input_ids"])
    check("Embeddings", hidden_states)
    
    position_ids = torch.arange(inputs["input_ids"].shape[1], dtype=torch.long, device=device).unsqueeze(0)
    
    print("\nWeight Statistics:")
    print("embed_tokens std:", model.model.embed_tokens.weight.std().item())
    print("layer 0 q_proj std:", model.model.layers[0].self_attn.q_proj.weight.std().item())
    print("layer 17 o_proj std:", model.model.layers[-1].self_attn.o_proj.weight.std().item())
    
    # check if they look exactly like initialization
    import math
    bos_token = torch.tensor([[tokenizer.bos_token_id]], device=device)
    current_input_ids = torch.cat([bos_token, inputs["input_ids"]], dim=-1)
    
    # Check if weights are completely untrained
    print("\nWeight check against fresh init:")
    torch.manual_seed(42)
    fresh_model = EmberForCausalLM(model.config)
    
    # Are they literally exactly the same as seed 42?
    diff_embed = (model.model.embed_tokens.weight - fresh_model.model.embed_tokens.weight).abs().max().item()
    diff_q = (model.model.layers[0].self_attn.q_proj.weight - fresh_model.model.layers[0].self_attn.q_proj.weight).abs().max().item()
    
    print("Max diff embed_tokens (vs seed 42):", diff_embed)
    print("Max diff q_proj layer 0 (vs seed 42):", diff_q)
    
    # What about seed 0?
    torch.manual_seed(0)
    fresh_model_0 = EmberForCausalLM(model.config)
    print("Max diff embed_tokens (vs seed 0):", (model.model.embed_tokens.weight - fresh_model_0.model.embed_tokens.weight).abs().max().item())


