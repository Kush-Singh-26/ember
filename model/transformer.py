from __future__ import annotations
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.config import EmberConfig

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key tensors."""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [batch, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class EmberRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(
            max_len=max_position_embeddings, device=inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, max_len: int, device, dtype):
        self.max_seq_len_cached = max_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # outer product: shape (max_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # concatenate to get shape (max_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, num_heads, seq_len, head_dim]
        # position_ids: [batch, seq_len]
        if position_ids is None:
            seq_len = x.shape[2]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        # Safety clamp: position_ids must stay within the pre-computed cache range.
        # SequencePacker already guarantees this, but clamp is torch.compile-friendly
        # (no .item() call) and protects against any edge-case overflow.
        position_ids = position_ids.clamp(max=self.max_seq_len_cached - 1)

        cos = self.cos_cached[position_ids]  # [batch, seq_len, dim]
        sin = self.sin_cached[position_ids]  # [batch, seq_len, dim]
        
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, dim]
        sin = sin.unsqueeze(1)  # [batch, 1, seq_len, dim]
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class EmberRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key or value states n_rep times along head dimension."""
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

class EmberAttention(nn.Module):
    def __init__(self, config: EmberConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)
        self.o_proj._forge_name = 'o_proj'
        
        self.rotary_emb = EmberRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
            
        if attention_mask is not None:
            # explicit block-diagonal or padding attention mask (boolean mask where True = keep, False = mask)
            # PyTorch SDPA attn_mask can be float or boolean.
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            # standard causal self-attention
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True
            )
            
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        
        return self.o_proj(attn_output)

class EmberMLP(nn.Module):
    def __init__(self, config: EmberConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.down_proj._forge_name = 'down_proj'
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class EmberDecoderLayer(nn.Module):
    def __init__(self, config: EmberConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = EmberAttention(config)
        self.mlp = EmberMLP(config)
        self.input_layernorm = EmberRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = EmberRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Pre-Norm Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Pre-Norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class EmberPreTrainedModel(PreTrainedModel):
    config_class = EmberConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EmberDecoderLayer"]

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        # GPT-2 scaled init: residual projections that write into the residual stream
        # (o_proj in attention, down_proj in MLP) use reduced std to prevent
        # variance from compounding over N layers. Without this, hidden state
        # RMS grows to ~100 over 18 layers, causing initial loss ~295 instead of ~11.
        residual_std = std / math.sqrt(2 * self.config.num_hidden_layers)
        
        if isinstance(module, nn.Linear):
            # Apply scaled init to the layers that project INTO the residual stream
            name = getattr(module, '_forge_name', '')
            if name in ('o_proj', 'down_proj'):
                module.weight.data.normal_(mean=0.0, std=residual_std)
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class EmberModel(EmberPreTrainedModel):
    def __init__(self, config: EmberConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([EmberDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = EmberRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Position IDs default
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states

class EmberForCausalLM(EmberPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: EmberConfig):
        super().__init__(config)
        self.model = EmberModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: EmberModel):
        self.model = decoder

    def get_decoder(self) -> EmberModel:
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs
        # Cast to float32 BEFORE lm_head to prevent bf16 overflow in the
        # 1024 x 65536 projection (bf16 max ~65504, large matmul can overflow)
        logits = self.lm_head(hidden_states.float())

        loss = None
        if labels is not None:
            num_items_in_batch = kwargs.pop("num_items_in_batch", None)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            if num_items_in_batch is not None:
                # For transformers >= 4.46 with gradient accumulation, trainer expects us
                # to return the sum of losses divided by the total tokens across ALL micro-batches.
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss / num_items_in_batch
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
                loss = loss_fct(shift_logits, shift_labels)


        if not return_dict:
            output = (logits,) + (outputs,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ) -> dict:
        model_inputs = {"input_ids": input_ids}
        # If attention mask is all 1s (no padding), we don't need it. 
        # Removing it allows SDPA to use fast `is_causal=True` mode without crashing.
        if attention_mask is not None and not attention_mask.all():
            model_inputs["attention_mask"] = attention_mask
        if kwargs.get("position_ids", None) is not None:
            model_inputs["position_ids"] = kwargs["position_ids"]
        return model_inputs
