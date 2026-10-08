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
import torch.nn.functional as f

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    hidden_size: int = 768
    intermediate_size: int = 768 * 4
    num_hidden_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 2
    rms_norm_eps: float = 1e-6

# torch has default implementation
def rms_norm(x):
    pass

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
    x1, x2 = x[..., d:], x[..., :d] # spliting the hidden dim in two half (x, y)
    # rotate in clockwise rotation
    y1 = x1 * cos - x2 * sin # 
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], 3)




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

    def forward(self, x, cos, sin):
        # self causal attention implementation
        B, T, C = x.size()
        # Multi head
        # (B, T, C) -> (B, T, nh, h) -> (B, nh, T, h)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        k = self.key(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # (B, num_kv_heads, T, head_size)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2) # (B, num_kv_heads, T, head_size)
        # apply rotary embd
        q, k = apply_rotary_embd(q), apply_rotary_embd(k)
        
        # repeat k and v nh // n_kv_h
        # calculate attention 
        # reshape and apply the out_proj
        # return



        pass

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
    config = GPTConfig()
    gpt = GPT(config)
    print("main")
