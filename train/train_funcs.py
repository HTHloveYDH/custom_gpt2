import os
import math
import time

import torch
import torch.distributed as dist


def _get_lr(it, max_lr = 6e-4, warmup_steps = 715, max_steps = 19073):
    min_lr = max_lr * 0.1
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def get_optimizer(raw_model, weight_decay:float, learning_rate:float, device_type:str, master_process:bool):
    # get optimizer!
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay, learning_rate=learning_rate, device_type=device_type, 
        master_process=master_process
    )
    return optimizer

def train_grad_accum_steps(model, train_loader, optimizer, grad_accum_steps:int, max_lr:int, \
                           warmup_steps:int, max_steps:int, device, device_type:str, master_process:bool, \
                           dp:bool, dp_world_size:int, global_grad_accum_stage:int, log_file_path:str):
    t0 = time.time()
    # do grad_accum_steps steps of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if dp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()  # loss in a single gradient accumulation stage
        loss.backward()
    if dp:
        # loss_accum = (Sigma 0 -> (process_num - 1): (local_rank_loss / grad_accum_steps)) / process_num
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    global_step = (global_grad_accum_stage + 1) * grad_accum_steps - 1
    lr = _get_lr(global_step, max_lr, warmup_steps, max_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == 'cuda':
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * dp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f'global_step {global_step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}')
        with open(log_file_path, 'a') as f:
            f.write(f'{global_step} train {loss_accum.item():.6f}\n')


def valid_epoch_wise(model, raw_model, val_loader, device, device_type:str, master_process:bool, \
                     dp:bool, val_steps:int, global_grad_accum_stage:int, last_grad_accum_stage:int, \
                     log_file_path:str, ckpt_dir:str):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_steps
            val_loss_accum += loss.detach()
    if dp:
        # val_loss_accum = (Sigma 0 -> (process_num - 1): (local_rank_loss / val_steps)) / process_num
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)  # all_reduce (mean)
    if master_process:
        print(f'validation loss: {val_loss_accum.item():.4f}')
        with open(log_file_path, 'a') as f:
            f.write(f'{global_grad_accum_stage} val {val_loss_accum.item():.4f}\n')
        if global_grad_accum_stage > 0 and (global_grad_accum_stage % 5000 == 0 or last_grad_accum_stage):
            # optionally write model checkpoints
            log_dir = os.path.dirname(log_file_path)
            save_curr_model_path = os.path.join(log_dir, f'model_{global_grad_accum_stage:05d}.pt')
            _save_ckpt(raw_model, global_grad_accum_stage, val_loss_accum.item(), save_curr_model_path)
            checkpoint_path = os.path.join(ckpt_dir, f'model.pt')
            _save_ckpt(raw_model, global_grad_accum_stage, val_loss_accum.item(), checkpoint_path)

def _save_ckpt(model, global_grad_accum_stage:int, val_loss_accum:float, checkpoint_path:str):
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'global_grad_accum_stage': global_grad_accum_stage,
        'val_loss': val_loss_accum
    }
    # you might also want to add optimizer.state_dict() and
    # rng seeds etc., if you wanted to more exactly resume training
    torch.save(checkpoint, checkpoint_path)

def resume_from_ckpt(model, ckpt_dir:str):
    checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.exists(checkpoint_path):
        print('Loading checkpoint directory')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
    return model
