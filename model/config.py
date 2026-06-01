from transformers import PretrainedConfig

class EmberConfig(PretrainedConfig):
    model_type = "ember"

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=1024,
        num_hidden_layers=18,
        num_attention_heads=16,      # Q heads
        num_key_value_heads=8,       # KV heads (GQA 2:1 ratio)
        head_dim=64,
        intermediate_size=2730,      # SwiGLU: ~8/3 * hidden_size
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        use_bias=False,
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.dropout = dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
