## lanuch task 
[reference]: https://www.youtube.com/watch?v=KaAJtI1T2x4&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj
### simple launch on one node:
```bash
python train/main.py --ddp_world_size 1
```

### DDP (FSDP) launch on one node by torch.multiprocessing (e.g. 8 GPUs):
```bash
python train/main.py --ddp_world_size 8
```

### DDP (FSDP) launch on one node by torchrun (e.g. 8 GPUs):
```bash
torchrun --standalone --nproc_per_node=8 train/main.py --ddp_world_size 8
```

### DDP (FSDP) launch on multi node by torchrun (e.g. 2 * 8 GPUs, two nodes):
```bash
# on node 0#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=0 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py --ddp_world_size 8
```

```bash
# on node 1#
torchrun --nproc_per_node=8 --nnodes=2 -node_rank=1 --rdzv_backend=c10d --rdzv_endpoint=xxx.xxx.xxx.xxx:xxxx train/main.py --ddp_world_size 8
```

## GPT configs:
### gpt2
n_layer=12, n_head=12, n_embd=768 # 124M params

### gpt2-medium
n_layer=24, n_head=16, n_embd=1024 # 350M params

### gpt2-large
gpt2-largen_layer=36, n_head=20, n_embd=1280  # 774M params

### gpt2-xl
n_layer=48, n_head=25, n_embd=1600  # 1558M params