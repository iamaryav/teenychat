"""
- decoder only transformer
- Dense model
- no bias in linear layers
- RoPE
- Pre Norm
- RMSNorm without learnable parameters
- ReLU or other?
- GQA, MHA
- kv cache
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    hidden_size: int = 768
    intermediate_size: int = 768 * 4
    num_hidden_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 2
    theta_base: int = 100000

# torch has default implementation
def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class RotaryEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        pass

def apply_rotary_embd(x, cos, sin):
    # you have a multidimensional vector
    # apply the rotary embd - roatate it by some angle
    # x, y pair from weights first half x and second half y
    # then apply the rotation formula
    assert x.ndim == 4 # Multi Head Attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # spliting the hidden dim in two half (x, y)
    # rotate in clockwise rotation
    y1 = x1 * cos - x2 * sin # 
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], 3)

def _precompute_rotary_embedding(seq_len, head_size, base=100000):
    channel_range = torch.arange(0, head_size, 2)
    t = torch.arange(0, seq_len)
    inv_freq = 1.0 / (base ** (channel_range / head_size))
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin



class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        assert config.hidden_size % config.num_heads == 0
        assert config.num_kv_heads <= config.num_heads and config.num_heads % config.num_kv_heads == 0 
        self.head_size = config.hidden_size // config.num_heads
        self.query = nn.Linear(config.hidden_size, config.num_heads * self.head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos_sin):
        # self causal attention implementation
        B, T, C = x.size()
        # Multi head
        # (B, T, C) -> (B, T, nh, h) -> (B, nh, T, h)
        q = self.query(x).view(B, T, self.num_heads, self.head_size) # (B, T, num_heads, head_size)
        k = self.key(x).view(B, T, self.num_kv_heads, self.head_size) # (B, T, num_kv_heads, head_size)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # (B, num_kv_heads, T, head_size)
        # apply rotary embd
        cos, sin = cos_sin
        q, k = apply_rotary_embd(q, cos, sin), apply_rotary_embd(k, cos, sin)
        q, k = norm(q), norm(k) # Qk Norm
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        enable_gqa = q.size(1) != k.size(1) # num_heads in key and query are not same
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa) # (B, num_heads, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, -1) # (B, T, C)
        y = self.out_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = nn.ReLU()
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        pass



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CausalAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        pass



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embd = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

        self.rotary_embd = RotaryEmbedding(config=config)
        # last projection depends on the vocab size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)



    def forward(self, x, y):
        pass
        


if __name__ == "__main__":
    torch.manual_seed(0)
    # test the causal attention method
    # (num_heads, num_kv_heads): MHA and GQA
    for num_heads, num_kv_heads in [(2, 2), (4, 2)]:
        config = GPTConfig(max_seq_len=4, hidden_size=16, num_heads=num_heads, num_kv_heads=num_kv_heads)
        head_size = config.hidden_size // config.num_heads
        B, T, C = 2, config.max_seq_len, config.hidden_size
        cos_sin = _precompute_rotary_embedding(T, head_size, config.theta_base) # (1, T, 1, head_size // 2)
        sa = CausalAttention(config)

        x = torch.randn((B, T, C))
        out = sa(x, cos_sin)
        assert out.shape == (B, T, C)

        # causal check: a change in the last token must not change earlier positions
        x2 = x.clone()
        x2[:, -1] += 1.0
        out2 = sa(x2, cos_sin)
        assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-6)
        assert not torch.allclose(out[:, -1], out2[:, -1])
        print(f"ok num_heads={num_heads} num_kv_heads={num_kv_heads} out={tuple(out.shape)}")
