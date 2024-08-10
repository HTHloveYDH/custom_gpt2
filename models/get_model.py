import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from models.GPT import GPT
from config.GPTConfig import GPTConfig


def get_model(gpt_config, device, dp:bool, device_ids:list):
    # create model
    config_args = {
        'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[gpt_config['model_type']]
    config_args['vocab_size'] = gpt_config['vocab_size']
    config_args['block_size'] = gpt_config['block_size']
    model = GPT(GPTConfig(**config_args))
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    if dp:
        model = DDP(model, device_ids=device_ids)
    raw_model = model.module if dp else model # always contains the "raw" unwrapped model
    enc = tiktoken.get_encoding('gpt2')
    return model, raw_model, enc