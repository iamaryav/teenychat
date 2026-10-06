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
    num_hidden_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 2
    rms_norm_eps: float = 1e-8

# torch has default implementation
# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size, rms_norm_eps):
#         pass

class RotaryEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        pass


class CausalAttention(nn.Module):
    pass

class MLP(nn.Module):
    def __init__(self):
        self.
        self.act_fn = nn.ReLU()



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

        # last projection depends on the vocab size
        self.rotary_embd = RotaryEmbedding(config=config)
        # torch has inbuilt version for rms norm
        # self.rms_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)



    def forward(self, x, y):
        pass
        


if __name__ == "__main__":
    config = GPTConfig()
    gpt = GPT(config)
    print("main")
