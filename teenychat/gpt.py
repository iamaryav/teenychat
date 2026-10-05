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

import torch
import torch.nn as nn

class GPTConfig:
    def __init__(self):
        self.vocab_size: int = 50257
        self.num_layers: int = 12
        self.hidden_dim: int = 768
        self.max_seq_len: int = 1024
        self.head_dim: int = 0
        self.num_kv_heads: int = 2


class Block(nn.Module):
    def __init__(self, config, x):
        super().__init__()



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embd = nn.Embeding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        


if __name__ == "__main__":
    config = GPTConfig()
    gpt = GPT(config)
    print("main")
