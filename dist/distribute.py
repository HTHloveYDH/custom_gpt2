import torch

from dist.ddp import init_ddp, ternimate_ddp
from dist.fsdp import init_fsdp, ternimate_fsdp


def init_dist(dist_strategy:str, torch_mp_launch:bool, dp_local_rank:int, dp_world_size:int):
    if dist_strategy == 'ddp':
        dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type = init_ddp(
            torch_mp_launch, dp_local_rank, dp_world_size
        )
    elif dist_strategy == 'fsdp':
        dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type = init_fsdp(
            torch_mp_launch, dp_local_rank, dp_world_size
        )
    else:
        # vanilla, non-DDP run
        dp_global_rank = 0
        dp_local_rank = 0
        dp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    return dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type

def ternimate_dist(dist_strategy:str):
    if dist_strategy == 'ddp':
        ternimate_ddp()
    elif dist_strategy == 'fsdp':
        ternimate_fsdp()
    else:
        pass