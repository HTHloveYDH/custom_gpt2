import os
import sys
import time

import torch

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from data_pipeline.DataLoaderLiteFactory import DataLoaderLiteFactory
from models.get_model import get_model
from train.train_funcs import train_grad_accum_steps, valid_epoch_wise
from gen.gen_funcs import gen_sentences
from utils.load_config import load_config_from_json as load_configs


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join('.', log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    # load configs
    gpt_config, train_config, data_config, cloud_config, dist_config = load_configs('test')
    # train configs
    learning_rate = train_config['learning_rate']  # defaults to 6e-4
    weight_decay = train_config['weight_decay']  # defaults to 0.1
    micro_batch_size = train_config['micro_batch_size']  # defaults to 64
    total_token_size = train_config['total_token_size']  # 2**19, ~0.5M, in number of tokens
    sequence_length = train_config['sequence_length']  # defaults to 1024
    steps_per_epoch = train_config['steps_per_epoch']
    grad_accum_steps = train_config['grad_accum_steps']
    val_steps = train_config['val_steps']
    epochs = train_config['epochs']
    max_lr = train_config['max_lr']
    warmup_steps = train_config['warmup_steps']
    seed = train_config['seed']  # defaults to 1337
    max_length = train_config['max_length']
    num_return_sequences = train_config['num_return_sequences']
    # data configs
    data_root = data_config['data_root']
    data_format = data_config['data_format']
    # gpt configs
    use_compile = gpt_config['use_compile']
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type = init_dist(ddp)
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    assert total_token_size % (micro_batch_size * sequence_length * ddp_world_size) == 0, "make sure total_token_size is divisible by B * T * ddp_world_size"
    steps_per_epoch = total_token_size // (micro_batch_size * sequence_length * ddp_world_size)
    assert steps_per_epoch % grad_accum_steps == 0, "make sure steps_per_epoch is divisible by grad_accum_steps"
    grad_accum_stages_per_epoch = steps_per_epoch // grad_accum_steps
    if master_process:
        print(f"total desired batch size: {total_token_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    torch.set_float32_matmul_precision('high')

    ''' ________________________________________ load dataset ________________________________________ '''
    DataLoaderLite_factory = DataLoaderLiteFactory()
    train_loader = DataLoaderLite_factory.create(
        data_format, 
        **{
            'B': micro_batch_size, 'T': sequence_length, 'process_rank': ddp_rank, 
            'num_processes': ddp_world_size, 'data_root': data_root, 'master_process': master_process, 
            'split': 'train'
        }
    )
    val_loader = DataLoaderLite_factory.create(
        data_format, 
        **{
            'B': micro_batch_size, 'T': sequence_length, 'process_rank': ddp_rank, 
            'num_processes': ddp_world_size, 'data_root': data_root, 'master_process': master_process, 
            'split': 'val'
        }
    )

    ''' ____________________________________ build & compile model ___________________________________ '''
    model, raw_model = get_model(gpt_config, device, ddp, ddp_local_rank)

    ''' ____________________________________________ train ___________________________________________ '''
    # get optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay, learning_rate=learning_rate, device_type=device_type
    )
    # start train loop
    max_grad_accum_stages = epochs * grad_accum_stages_per_epoch
    max_steps = max_grad_accum_stages * grad_accum_steps
    for epoch in range(epochs):
        print(f'epoch: {epoch} / {epochs}:')
        for grad_accum_stage in range(grad_accum_stages_per_epoch):
            global_grad_accum_stage = epoch * grad_accum_stages_per_epoch + grad_accum_stage
            last_grad_accum_stage = (global_grad_accum_stage == max_grad_accum_stages - 1)
            # once in a while evaluate our validation loss
            if global_grad_accum_stage % 250 == 0 or last_grad_accum_stage:
                valid_epoch_wise(
                    model, raw_model, val_loader, device, device_type, master_process, ddp, val_steps, 
                    global_grad_accum_stage, last_grad_accum_stage, log_file, log_dir
                )
            # once in a while generate from the model (except step 0, which is noise)
            if ((global_grad_accum_stage > 0 and global_grad_accum_stage % 250 == 0) or last_grad_accum_stage) \
                and (not use_compile):
                tokens = enc.encode("Hello, I'm a language model,")
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                x = tokens.to(device)
                gen_sentences(
                    model, enc, x, device, device_type, num_return_sequences, max_length, ddp_rank
                )
            # train model on several batches
            train_grad_accum_steps(
                model, train_loader, optimizer, grad_accum_steps, max_lr, warmup_steps, max_steps, device, 
                device_type, master_process, ddp, ddp_world_size, global_grad_accum_stage, log_file
            )
    if ddp:
        ternimate_dist()