from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers (defaults to 12 gpt2)
    n_head: int = 12  # number of heads (defaults to 12 gpt2)
    n_embd: int = 768  # embedding dimension (defaults to 768 of gpt2)
    n_group_head: int = 1  # number of query heads that are divided into one attention group
    
    num_return_sequences: int = 4  # number of sentences that generate simultaneously
    kv_cache: bool = False  # enable kv cace or not?