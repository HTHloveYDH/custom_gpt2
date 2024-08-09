import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # super(CausalSelfAttention, self).__init__()
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.IS_INIT_SCALE = True
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            'bias', 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # # naive attention
        # y = self.naive_scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
        # flash attention (more efficient)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh * hs)
        # output projection
        y = self.c_proj(y)  # (B, T, nh * hs)
        return y
    
    def naive_scaled_dot_product_attention(self, q, k, v, is_causal=False):
        T = q.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if is_causal:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return y

class TanhGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):
    def __init__(self, config):
        # super(MLP, self).__init__()
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # self.gelu = TanhGELU()
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.IS_INIT_SCALE = True

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        # super(Block, self).__init__()
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def foward(self, x):
        # [note]: pre-normalization version (layer normalization first)
        # [note]: clean residual pathway is desirable that gradient from top can flow straight to the x
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x