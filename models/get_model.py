import os
import functools

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import tiktoken

from models.GPT import GPT
from config.GPTConfig import GPTConfig


def get_model(gpt_config:dict, device, dist_strategy:str, device_ids:list):
    assert gpt_config['load_weights'] in ['official', 'local', None], f"load weights: {gpt_config['load_weights']}  is not supported"
    # create model
    if gpt_config['load_weights'] == 'official':
        model = GPT.from_official_pretrained("gpt2")  # or init from OpenAI GPT-2
    elif gpt_config['load_weights'] == 'local':
        assert os.path.exists(gpt_config['ckpt_path'])
        model = GPT.from_local_pretrained("gpt2", gpt_config['ckpt_path'])  # or init from OpenAI GPT-2
    else:
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[gpt_config['model_type']]
        assert config_args['n_head'] % gpt_config['n_group_head'] == 0, f'make sure n_head is divisible by n_group_head'
        config_args['n_group_head'] = gpt_config['n_group_head']
        config_args['vocab_size'] = gpt_config['vocab_size']
        config_args['block_size'] = gpt_config['block_size']
        config_args['num_return_sequences'] = gpt_config['num_return_sequences']
        config_args['kv_cache'] = gpt_config['kv_cache']
        model = GPT(GPTConfig(**config_args))
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    if dist_strategy == 'ddp':
        model = DDP(model, device_ids=device_ids)
    elif dist_strategy == 'fsdp':
        # reference: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp
        model = FSDP(model)
        # my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
        # model = FSDP(
        #     model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True)
        # )
    print(f'distribute strategy is set to {dist_strategy}')
    raw_model = model.module if dist_strategy in ['ddp', 'fsdp'] else model # always contains the "raw" unwrapped model
    enc = tiktoken.get_encoding('gpt2')
    return model, raw_model, enc
