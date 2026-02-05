import torch
import torch.nn as nn
import torch.nn.functional as F

def _rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even."
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, L, device=None, dtype=None):
        t = torch.arange(L, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", t, self.inv_freq)  # [L, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [L, hd]
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]  # [1,1,L,hd]
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin

def apply_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

class MHAWithRoPE(nn.Module):
    """时间分支专用；输入 [B,L,E]，输出 [B,L,E]"""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.hd = d_model // n_heads
        assert self.hd % 2 == 0, "RoPE需要偶数 head_dim"
        self.qkv = nn.Linear(d_model, 3*d_model, bias=True)
        self.o   = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.hd)

    def forward(self, x, attn_mask=None):
        B, L, E = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,L,self.h,self.hd).permute(0,2,1,3)  # [B,h,L,hd]
        k = k.view(B,L,self.h,self.hd).permute(0,2,1,3)
        v = v.view(B,L,self.h,self.hd).permute(0,2,1,3)
        cos, sin = self.rope(L, device=x.device, dtype=x.dtype)
        q, k = apply_rope(q, k, cos, sin)  # ★RoPE对Q/K的旋转注入相对位置信息

        # PyTorch 2.x 高效注意力（SDPA，会自动选择最佳内核）
        q_ = q.transpose(1,2).reshape(B*self.h, L, self.hd)
        k_ = k.transpose(1,2).reshape(B*self.h, L, self.hd)
        v_ = v.transpose(1,2).reshape(B*self.h, L, self.hd)
        out = F.scaled_dot_product_attention(
            q_, k_, v_, None,
            dropout_p=self.drop.p if self.training else 0.0, is_causal=False
        )  # [B*h,L,hd]
        out = out.view(B, self.h, L, self.hd).transpose(1,2).reshape(B, L, E)
        return self.o(out)

#RoPE 的定义即“在注意力里对 Q/K 旋转”，不是在 embedding 相加位置向量，这也是我们专门做一份注意力实现的原因。
#SDPA 是 PyTorch 官方提供的高效注意力入口，会自动切换 Flash/高效/数学后端。