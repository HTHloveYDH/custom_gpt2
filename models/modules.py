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
        ratio = 1 + 2 / config.n_group_head
        self.c_attn = nn.Linear(config.n_embd, int(ratio * config.n_embd))
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            'bias', 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head
        self.n_group_head = config.n_group_head
        self.n_embd = config.n_embd
        self.c_proj.IS_INIT_SCALE = True
    
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads"
        # hs is "head size", and C (number of channels) = nh * hs
        # ngh is "number of query heads that are divided into one attention group"
        # nkvh is "number of key & value heads", nkvh = nh // ngh, C (number of channels) = ngh * nkvh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh * hs = C = 768 channels in the Transformer
        qkv = self.c_attn(x)  # (B, T, ratio * C), ratio = 1 + 2 / ngh
        q, k, v = qkv.split(
            [self.n_embd, self.n_embd // self.n_group_head, self.n_embd // self.n_group_head], dim=2
        )  # (B, T, C), (B, T, C // ngh), (B, T, C // ngh)
        # q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3)  # (B, nh, T, hs)
        # k = k.view(B, T, self.n_head // self.n_group_head, C // self.n_head).permute(0, 2, 1, 3)  # (B, nh // ngh, T, hs)
        # v = v.view(B, T, self.n_head // self.n_group_head, C // self.n_head).permute(0, 2, 1, 3)  # (B, nh // ngh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head // self.n_group_head, C // self.n_head).transpose(1, 2)  # (B, nh // ngh, T, hs)
        v = v.view(B, T, self.n_head // self.n_group_head, C // self.n_head).transpose(1, 2)  # (B, nh // ngh, T, hs)
        # repeat key, value if necessary
        k = CausalSelfAttention.repeat_kv(k, self.n_group_head)  # (B, nh, T, hs)
        v = CausalSelfAttention.repeat_kv(v, self.n_group_head)  # (B, nh, T, hs)
        # normal attention
        # y = self._normal_scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
        # flash attention (more efficient)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh * hs)
        # output projection
        y = self.c_proj(y)  # (B, T, nh * hs)
        return y

    def _normal_scaled_dot_product_attention(self, q, k, v, is_causal=False):
        T = q.size(-2)  # T can be 1
        # suppose T_gen is length of generated sentence by far
        # T_gen = k.size(-2) = v.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T_gen)
        if is_causal:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # (B, nh, T, T_gen)
        att = F.softmax(att, dim=-1)  # (B, nh, T, T_gen)
        y = att @ v  # (B, nh, T, T_gen) @ (B, nh, T_gen, hs) -> (B, nh, T, hs), nh * hs = C = n_embd
        return y
    
    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        B, n_kv_head, T, head_size = x.shape  # n_kv_head = n_head // n_group_head, head_size = n_embd // self.n_head
        if n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]  # (B, n_kv_head, new_dim, T, head_size)
            .expand(B, n_kv_head, n_rep, T, head_size)  # (B, n_kv_head, n_rep, T, head_size), n_rep = n_group_head
            .reshape(B, n_kv_head * n_rep, T, head_size)  # (B, n_kv_head * n_rep, T, head_size), n_rep = n_group_head
        )

class KVCacheCausalSelfAttention(CausalSelfAttention):
    def __init__(self, config):
        # super(KVCacheCausalSelfAttention, self).__init__(config)
        super().__init__(config)
        self.block_size = config.block_size
        self.register_buffer(
            'k_cache', 
            torch.zeros(
                config.num_return_sequences, 
                config.n_head // config.n_group_head, 
                config.block_size, 
                config.n_embd // config.n_head
            )
        )  # 12M
        self.register_buffer(
            'v_cache', 
            torch.zeros(
                config.num_return_sequences, 
                config.n_head // config.n_group_head, 
                config.block_size, 
                config.n_embd // config.n_head
            )
        )  # 12M
        self.next_token_idx = 0  # start from 0

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads"
        # hs is "head size", and C (number of channels) = nh * hs
        # ngh is "number of query heads that are divided into one attention group"
        # nkvh is "number of key & value heads", nkvh = nh // ngh, C (number of channels) = ngh * nkvh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)  # (B, T, ratio * C), ratio = 1 + 2 / ngh
        q, k, v = qkv.split(
            [self.n_embd, self.n_embd // self.n_group_head, self.n_embd // self.n_group_head], dim=2
        )  # (B, T, C), (B, T, C // ngh), (B, T, C // ngh)
        # q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3)  # (B, nh, T, hs)
        # k = k.view(B, T, self.n_head // self.n_group_head, C // self.n_head).permute(0, 2, 1, 3)  # (B, nh // ngh, T, hs)
        # v = v.view(B, T, self.n_head // self.n_group_head, C // self.n_head).permute(0, 2, 1, 3)  # (B, nh // ngh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head // self.n_group_head, C // self.n_head).transpose(1, 2)  # (B, nh // ngh, T, hs)
        v = v.view(B, T, self.n_head // self.n_group_head, C // self.n_head).transpose(1, 2)  # (B, nh // ngh, T, hs)
        # input x is prompts of shape (B, T, C)
        if self.next_token_idx == 0:
            self.k_cache[:, :, :T, :] = k  # (B, nh, block_size, hs), k is of shape (B, nh // ngh, T, hs)
            self.v_cache[:, :, :T, :] = v  # (B, nh, block_size, hs), v is of shape (B, nh // ngh, T, hs)
            # repeat key, value if necessary
            k = CausalSelfAttention.repeat_kv(self.k_cache[:, :, :T, :], self.n_group_head)
            v = CausalSelfAttention.repeat_kv(self.v_cache[:, :, :T, :], self.n_group_head)
            # y = self._normal_scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
            # re-assemble all head outputs side by side
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T_gen, hs) -> (B, T_gen, nh, hs) -> (B, T_gen, nh * hs)
        # input x is newly generated token of shape (B, T = 1, C) in last inference
        else:
            if self.next_token_idx == self.block_size:   
                self._shift_cache()  # shift k_cache, v_cache
            # suppose T_gen is length of generated sentence in current slide window (length = 1024) by far
            T_gen = self.next_token_idx + 1
            self.k_cache[:, :, self.next_token_idx:T_gen, :] = k  # (B, nh, block_size, hs), k is of shape (B, nh, T = 1, hs)
            self.v_cache[:, :, self.next_token_idx:T_gen, :] = v  # (B, nh, block_size, hs), v is of shape (B, nh, T = 1, hs)
            # repeat key, value if necessary
            k = CausalSelfAttention.repeat_kv(self.k_cache[:, :, :T, :], self.n_group_head)
            v = CausalSelfAttention.repeat_kv(self.v_cache[:, :, :T, :], self.n_group_head)
            # y = self._kvcache_scaled_dot_product_attention(q, k, v)  # (B, nh, T = 1, hs)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B, nh, T = 1, hs)
            # re-assemble all head outputs side by side
            y = y.transpose(1, 2).contiguous().view(B, 1, C)  # (B, nh, T = 1, hs) -> (B, T = 1, nh, hs) -> (B, T = 1, nh * hs)
        # output projection
        y = self.c_proj(y)  # (B, T = 1, nh * hs)
        # the next token idx
        self.next_token_idx += T  # (mostly, + 1)
        return y
    
    def _kvcache_scaled_dot_product_attention(self, q, k, v):
        # T = q.size(2) = 1
        # suppose T_gen is length of generated sentence in current slide window (length = 1024) by far
        # T_gen = k.size(-2) = v.size(-2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T = 1, T_gen)
        att = F.softmax(att, dim=-1)  # (B, nh, T = 1, T_gen)
        y = att @ v  # (B, nh, T = 1, T_gen) @ (B, nh, T_gen, hs) -> (B, nh, T = 1, hs), nh * hs = C = n_embd
        return y

    def _shift_cache(self):
        # self.k_cache[:, :, :-1, :] = self.k_cache[:, :, 1:, :]  # (B, nh, self.block_size, hs), O(block_size), not efficient
        # self.v_cache[:, :, :-1, :] = self.v_cache[:, :, 1:, :]  # (B, nh, self.block_size, hs), O(block_size), not efficient
        self.k_cache = torch.cat(
            [
                self.k_cache[:, :, 1:, :],
                torch.zeros(
                    (self.k_cache.shape[0], self.k_cache.shape[1], 1, self.k_cache.shape[3]),
                    device=self.k_cache.device
                )
            ],
            dim=2
        )
        self.v_cache = torch.cat(
            [
                self.v_cache[:, :, 1:, :],
                torch.zeros(
                    (self.v_cache.shape[0], self.v_cache.shape[1], 1, self.v_cache.shape[3]),
                    device=self.v_cache.device
                )
            ],
            dim=2
        )
        self.next_token_idx = self.block_size - 1

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
        if config.kv_cache:
            self.attn = KVCacheCausalSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # [note]: pre-normalization version (layer normalization first)
        # [note]: clean residual pathway is desirable that gradient from top can flow straight to the x
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
