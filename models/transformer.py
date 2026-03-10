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
        x = self.feedforward(x)

        # Add & Norm
        x = self.norm2(x + input)
        
        # #GPT说这版更合理，原版在上面
        # attn_out, _ = self.attention(input, input, input, need_weights=False)
        # x = self.norm1(attn_out + input)

        # ff_out = self.feedforward(x)
        # x = self.norm2(ff_out + x)

        
        return x


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


import torch
import torch.nn as nn

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
        w = self.mlp(w)             # [B, C]
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
    ) -> torch.Tensor:
        """'input_channel' will be converted to 'embed_dim' through 1D convolution."""
        super(ViTransformer, self).__init__()
        self.signal_channel = input_channel
        self.seq_length = seq_length

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
            else:
                raise ValueError(f"Unknown channel_attn_type: {self.channel_attn_type}")

        # ===== Embedding =====
        self.proj = nn.Conv1d(self.signal_channel, embed_dim, kernel_size=3, padding=1)

        self.transformer = Transformer(
            embed_dim,
            seq_length,
            num_heads,
            num_blocks,
            block_hidden_dim,
            fc_hidden_dim,
            num_classes,
            dropout_p,
        )        
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

        # ===== original pipeline =====
        x = self.proj(x)            # [B, embed_dim, T]
        x = x.permute(0, 2, 1)      # [B, T, embed_dim]
        x = self.transformer(x)     # [B, num_classes]
        return x


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
