DEFAULT_VIT_CONFIG = {
    "input_channel": 19,
    "seq_length": 9250,
    "embed_dim": 32,
    "num_heads": 4,
    "num_blocks": 1,
    "block_hidden_dim": 64,
    "fc_hidden_dim": 16,
    "num_classes": 2,
    "dropout_p": 0.4,
    "use_channel_attn": True,
    "channel_attn_type": "eca",
    "channel_attn_reduction": 4,
    "eca_kernel_size": 3,
    "temporal_kernel_size": 7,
    "temporal_stride": 2,
    "shallow_hidden_dim": 64,
    "shallow_dropout_p": 0.2,
    "tf_dropout_p": 0.5,
    "stft_n_fft": 32,
    "stft_hop_length": 16,
    "ablation_mode": "full_fusion",
}


def get_default_vit_config() -> dict:
    return DEFAULT_VIT_CONFIG.copy()
