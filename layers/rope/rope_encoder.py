import torch.nn as nn
from .rope_attn import MHAWithRoPE

class RoPEEncoderLayer(nn.Module):
   
    def __init__(self, d_model, n_heads, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MHAWithRoPE(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        act = nn.ReLU() if activation.lower()=="relu" else nn.GELU()
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), act, nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        x = x + self.drop1(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x, None
