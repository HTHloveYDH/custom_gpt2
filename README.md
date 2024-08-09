### simple launch:
```bash
python train/main.py
```


### DDP launch for e.g. 8 GPUs:
```bash
torchrun --standalone --nproc_per_node=8 train/main.py
```

### GPT configs:
#### gpt2
n_layer=12, n_head=12, n_embd=768 # 124M params

#### gpt2-medium
n_layer=24, n_head=16, n_embd=1024 # 350M params

#### gpt2-large
gpt2-largen_layer=36, n_head=20, n_embd=1280  # 774M params

#### gpt2-xl
n_layer=48, n_head=25, n_embd=1600  # 1558M params