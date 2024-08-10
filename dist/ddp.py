import os

import torch
from torch.distributed import init_process_group, destroy_process_group


def init_ddp(torch_mp_launch:bool, ddp_local_rank:int, ddp_world_size:int):
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), 'for now i think we need CUDA for DDP'
    # launch by torch.multiprocessing
    if torch_mp_launch:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        init_process_group(backend='nccl', rank=ddp_local_rank, world_size=ddp_world_size)
        ddp_global_rank = ddp_local_rank
    # lanuch by torchrun
    else:
        init_process_group(backend='nccl')
        ddp_global_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = ddp_global_rank == 0 # this process will do logging, checkpointing etc.
    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    return ddp_global_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type

def ternimate_ddp():
    destroy_process_group()