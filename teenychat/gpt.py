"""
- decoder only transformer
- Dense model
- RoPE
- Pre LN
- RMSNorm
- ReLU or other
- GQA, MHA
- kv cache
"""

import torch
import torch.nn as nn

class GPTConfig:
    def __init__(self):
        self.num_layers: int = 12,
        self.hidden_dim: int = 768,
        self.max_seq_len: int = 1024,
        self.head_dim: int = 0,
        self.num_kv_heads = 2,
        # self.intermediate_size = ?,
        # self.
        pass

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        print("hi")


if __name__ == "__main__":
    config = GPTConfig()
    gpt = GPT(config)
    print("main")
