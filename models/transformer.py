from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for Transformer model.

    :param embed_dim: embedding dimension (if Transformer: same as input channel)
    :param num_heads: number of attention heads
    :param num_blocks: number of attention blocks
    :param block_hidden_dim: dimension of attention blocks
    :param fc_hidden_dim: dimension of feed forward layers
    :param dropout: dropout probability
    """

    embed_dim: int
    num_heads: int
    num_blocks: int
    block_hidden_dim: int
    fc_hidden_dim: int
    dropout: float


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.shape[2] == self.embed_dim,
            f"Input shape must be (batch_size, seq_len, {self.embed_dim}) in AttentionBlock.",
        )

        # Multi-head Attention
        x, _ = self.attention(input, input, input, need_weights=False)

        # Add & Norm
        x = self.norm1(x + input)

        # Feed Forward
        ff_input = x
        x = self.feedforward(x)

        # Add & Norm
        x = self.norm2(x + ff_input)
        
        # #GPT说这版更合理，原版在上面
        # attn_out, _ = self.attention(input, input, input, need_weights=False)
        # x = self.norm1(attn_out + input)

        # ff_out = self.feedforward(x)
        # x = self.norm2(ff_out + x)

        
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attention(query, context, context, need_weights=False)
        x = self.norm1(query + attn_out)
        ff_out = self.feedforward(x)
        return self.norm2(x + ff_out)


class Transformer(nn.Module):
    """Transformer model for EEG signals."""

    def __init__(
        self,
        input_channel: int,
        seq_length: int,
        num_heads: int,
        num_blocks: int,
        block_hidden_dim: int,
        fc_hidden_dim: int,
        num_classes: int,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        super(Transformer, self).__init__()
        # signal channel == embedding dimension
        self.signal_channel = input_channel
        self.seq_length = seq_length

        # Embedding
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, self.signal_channel).normal_(std=0.02)
        )

        # Attention Blocks
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(self.signal_channel, num_heads, block_hidden_dim)
                for _ in range(num_blocks)
            ]
        )

        # Decoding layers
        self.global_max_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1), nn.Dropout(p=dropout_p)
        )
        self.fc = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(self.signal_channel, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, input):
        torch._assert(
            input.shape[1:] == (self.seq_length, self.signal_channel),
            f"Expected shape of (batch, {self.seq_length}, {self.signal_channel})",
        )
        x = input + self.pos_embedding

        for layer in self.encoder:
            x = layer(x)

        x = x.permute(0, 2, 1)
        # x: (-1, embed_dim, seq_len)
        x = self.global_max_pool(x)
        x = self.fc(x)
        return x

class ChannelAttention1D(nn.Module):
    """
    输入: x [B, C, T]
    输出: x' [B, C, T]，对每个通道乘以一个权重 (0~1)
    """
    def __init__(self, channels: int, reduction: int = 4, dropout_p: float = 0.0):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        w = x.mean(dim=-1)          # [B, C]  全局平均池化(时间维)
        w = self.mlp(w)             # 返回权重，方便可视化
        w = w.unsqueeze(-1)         # [B, C, 1]
        return x * w, w             # 返回权重，方便可视化
class StaticChannelGating(nn.Module):
    """
    全局可学习的通道门控：每个电极一个标量权重(0~1)。
    输入: x [B, C, T]
    输出: x' [B, C, T], w [C]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(channels))  # 初始为0 => sigmoid=0.5

    def forward(self, x):
        w = torch.sigmoid(self.logits)          # [C]
        return x * w.view(1, -1, 1), w

class ECALayer1D(nn.Module):
    """
    ECA for EEG 1D signals.

    输入: x [B, C, T]
    输出: x' [B, C, T], w [B, C, 1]

    思路:
    1) 对时间维做全局平均池化 -> [B, C]
    2) 用 1D conv 在通道维上做局部交互
    3) sigmoid 得到每个通道的权重
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "ECA kernel size must be odd."

        self.avg_pool = nn.AdaptiveAvgPool1d(1)   # [B, C, T] -> [B, C, 1]
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T]
        y = self.avg_pool(x)                      # [B, C, 1]
        y = y.squeeze(-1)                         # [B, C]
        y = y.unsqueeze(1)                        # [B, 1, C]
        y = self.conv(y)                          # [B, 1, C]
        y = self.sigmoid(y)                       # [B, 1, C]
        y = y.squeeze(1).unsqueeze(-1)            # [B, C, 1]

        # 和你当前实现保持一致：返回加权后的特征 + 权重
        return x * y, y


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(1, max_length, embed_dim).normal_(std=0.02)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pos_embedding[:, :seq_len, :]


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(self.dropout(x)), dim=1)
        return torch.sum(weights * x, dim=1)


class TemporalConvEmbedding(nn.Module):
    def __init__(self, input_channel: int, embed_dim: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv1d(
            input_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ShallowTemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout_p: float = 0.0):
        super().__init__()
        hidden_dim = max(embed_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderStack(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_blocks: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class STFTSpectrogram(nn.Module):
    def __init__(self, n_fft: int = 128, hop_length: int = 32, normalized: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps = x.shape
        n_fft = max(2, min(self.n_fft, time_steps))
        if n_fft % 2 == 1:
            n_fft -= 1
        n_fft = max(n_fft, 2)
        hop_length = max(1, min(self.hop_length, n_fft // 2 if n_fft > 2 else 1))
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)

        spec = torch.stft(
            x.reshape(batch_size * channels, time_steps),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
            normalized=self.normalized,
        )
        spec = spec.abs()
        return spec.view(batch_size, channels, spec.shape[-2], spec.shape[-1])


class TimeFrequencyEncoder(nn.Module):
    def __init__(self, input_channel: int, embed_dim: int, dropout_p: float = 0.0):
        super().__init__()
        hidden_dim = max(1, embed_dim // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.mean(dim=2)
        return x.transpose(1, 2)


class SemanticGatedFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)
        self.frequency_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, temporal_feat: torch.Tensor, frequency_feat: torch.Tensor) -> torch.Tensor:
        z_t = self.temporal_proj(temporal_feat)
        z_f = self.frequency_proj(frequency_feat)
        gate = torch.sigmoid(self.gate(torch.cat([z_t, z_f], dim=-1)))
        return gate * z_t + (1.0 - gate) * z_f

class ViTransformer(nn.Module):
    """Vision Transformer model for EEG signals."""

    def __init__(
        self,
        input_channel: int,
        seq_length: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        block_hidden_dim: int,
        fc_hidden_dim: int,
        num_classes: int,
        dropout_p: float = 0.0,
        use_channel_attn: bool = False,
        channel_attn_type: str = "static",
        channel_attn_reduction: int = 4,
        eca_kernel_size: int = 3,
        temporal_kernel_size: int = 7,
        temporal_stride: int = 2,
        shallow_hidden_dim: int | None = None,
        shallow_dropout_p: float | None = None,
        tf_dropout_p: float | None = None,
        stft_n_fft: int = 128,
        stft_hop_length: int = 32,
    ) -> torch.Tensor:
        """'input_channel' will be converted to 'embed_dim' through 1D convolution."""
        super(ViTransformer, self).__init__()
        self.signal_channel = input_channel
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        # ===== Channel attention (optional) =====
        self.use_channel_attn = use_channel_attn
        self.channel_attn_type = channel_attn_type
        self.last_channel_attn = None  # 保存权重，方便可视化

        if self.use_channel_attn:
            if self.channel_attn_type == "static":
                self.chan_attn = StaticChannelGating(self.signal_channel)
            elif self.channel_attn_type == "dynamic":
                self.chan_attn = ChannelAttention1D(
                    channels=self.signal_channel,
                    reduction=channel_attn_reduction,
                    dropout_p=0.0,
                )
            elif self.channel_attn_type == "eca":
                self.chan_attn = ECALayer1D(
                    channels=self.signal_channel,
                    k_size=eca_kernel_size,
                )

            else:
                raise ValueError(f"Unknown channel_attn_type: {self.channel_attn_type}")

        shallow_hidden_dim = shallow_hidden_dim or block_hidden_dim
        shallow_dropout_p = dropout_p if shallow_dropout_p is None else shallow_dropout_p
        tf_dropout_p = dropout_p if tf_dropout_p is None else tf_dropout_p

        reduced_seq_length = self._conv_output_length(
            seq_length,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2,
        )

        # ===== Temporal branch =====
        self.proj = TemporalConvEmbedding(
            self.signal_channel,
            embed_dim,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
        )
        self.temporal_pos_embedding = TemporalPositionalEncoding(embed_dim, reduced_seq_length)
        self.temporal_shallow_encoder = ShallowTemporalEncoder(
            embed_dim,
            shallow_hidden_dim,
            dropout_p=shallow_dropout_p,
        )
        self.temporal_transformer = TransformerEncoderStack(
            embed_dim,
            num_heads,
            num_blocks,
            block_hidden_dim,
        )
        self.temporal_pool = AttentionPooling(embed_dim, dropout_p=dropout_p)

        # ===== Time-frequency branch =====
        self.stft = STFTSpectrogram(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            normalized=True,
        )
        self.tf_encoder = TimeFrequencyEncoder(
            input_channel=self.signal_channel,
            embed_dim=embed_dim,
            dropout_p=tf_dropout_p,
        )
        self.cross_attn_1 = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=block_hidden_dim,
            dropout_p=dropout_p,
        )
        self.cross_attn_2 = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=block_hidden_dim,
            dropout_p=dropout_p,
        )
        self.tf_pool = AttentionPooling(embed_dim, dropout_p=dropout_p)

        # ===== Fusion + classification =====
        self.gated_fusion = SemanticGatedFusion(embed_dim)
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    @staticmethod
    def _conv_output_length(length: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
        return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def forward(self, input):
        torch._assert(
            input.shape[1:] == (self.signal_channel, self.seq_length),
            f"Expected shape of (batch, {self.signal_channel}, {self.seq_length})",
        )

        x = input  # [B, C, T]

        # ===== apply channel attention BEFORE proj =====
        if self.use_channel_attn:
            x, w = self.chan_attn(x)
            # dynamic: w [B, C, 1] -> 保存成 [B, C]
            # static : w [C]      -> 保存成 [C]
            if w.dim() == 3:
                self.last_channel_attn = w.squeeze(-1)  # [B, C]
            else:
                self.last_channel_attn = w              # [C]
        else:
            self.last_channel_attn = None

        # ===== temporal branch =====
        x_t = self.proj(x)                      # [B, D, T']
        x_t = self.temporal_shallow_encoder(x_t)
        z_t1 = x_t.transpose(1, 2)              # [B, T', D]
        z_t1 = self.temporal_pos_embedding(z_t1)
        z_t2 = self.temporal_transformer(z_t1)  # [B, T', D]
        z_t = self.temporal_pool(z_t2)          # [B, D]

        # ===== time-frequency branch =====
        spec = self.stft(x)                     # [B, C, F, T_f]
        z_f1 = self.tf_encoder(spec)            # [B, T_f, D]
        z_f2 = self.cross_attn_1(z_f1, z_t1)    # [B, T_f, D]
        z_f3 = z_f2 + self.cross_attn_2(z_f2, z_t2)
        z_f = self.tf_pool(z_f3)                # [B, D]

        # ===== fusion + classification =====
        z = self.gated_fusion(z_t, z_f)
        return self.mlp(z)


if __name__ == "__main__":
    from torchinfo import summary

    # Original EEG-Transformer:
    # Classification of attention deficit/hyperactivity disorder
    # based on EEG signals using a EEG-Transformer model, 2023.
    model = Transformer(
        input_channel=56,
        seq_length=385,
        num_heads=4,
        num_blocks=6,
        num_classes=3,
        block_hidden_dim=56,
        fc_hidden_dim=64,
        dropout_p=0.5,
    )
    summary(model, input_size=(1, 385, 56))
