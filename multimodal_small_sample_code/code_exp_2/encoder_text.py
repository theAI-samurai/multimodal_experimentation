import torch
import torch.nn as nn

class SimpleTextEncoder(nn.Module):
    """Your original text encoder"""
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        if mask is not None:
            # Convert to bool for Transformer (True = attend)
            mask = mask.bool()
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)  # mean pooling
        return self.proj(x)