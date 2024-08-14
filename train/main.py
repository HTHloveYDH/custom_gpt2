import os
import sys
import argparse

import torch
import torch.multiprocessing as mp

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from data_pipeline.DataLoaderLiteFactory import DataLoaderLiteFactory
from models.get_model import get_model
from train.train_funcs import train_grad_accum_steps, valid_epoch_wise, get_optimizer, resume_from_ckpt
from gen.gen_funcs import gen_sentences_v1 as gen_sentences
from utils.load_config import load_config_from_json as load_configs


def main(dp_local_rank=0, dp_world_size=1, torch_mp_launch=False):
    ''' __________________________________________ setup _____________________________________________ '''
    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join('.', log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    # load configs
    gpt_config, train_config, data_config, cloud_config, dist_config = load_configs('train')
    # distribute configs
    dist_strategy = dist_config['dist_strategy']
    assert dist_strategy in ['ddp', 'fssdp', 'default'], f'distribute strategy: {dist_strategy} is not supported'
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
    save_interval = train_config['save_interval']
    ckpt_dir = train_config['ckpt_dir']
    # data configs
    data_root = data_config['data_root']
    data_format = data_config['data_format']
    # gpt configs
    use_compile = gpt_config['use_compile']
    num_return_sequences = 4 if gpt_config['load_weights'] == 'official' else num_return_sequences
    assert num_return_sequences == gpt_config['num_return_sequences']
    # set up DP (distributed data parallel or fully sharded data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    dp = dist_strategy in ['ddp', 'fsdp']
    dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type = init_dist(
        dist_strategy, torch_mp_launch, dp_local_rank, dp_world_size
    )
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    assert total_token_size % (micro_batch_size * sequence_length * dp_world_size) == 0, "make sure total_token_size is divisible by B * T * dp_world_size"
    steps_per_epoch = total_token_size // (micro_batch_size * sequence_length * dp_world_size)
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
            'B': micro_batch_size, 'T': sequence_length, 'process_rank': dp_global_rank, 
            'num_processes': dp_world_size, 'data_root': data_root, 'master_process': master_process, 
            'split': 'train'
        }
    )
    val_loader = DataLoaderLite_factory.create(
        data_format, 
        **{
            'B': micro_batch_size, 'T': sequence_length, 'process_rank': dp_global_rank, 
            'num_processes': dp_world_size, 'data_root': data_root, 'master_process': master_process, 
            'split': 'val'
        }
    )

    ''' ____________________________________ build & compile model ___________________________________ '''
    device_ids = [dp_local_rank]
    model, raw_model, enc = get_model(gpt_config, device, dist_strategy, device_ids)

    ''' ____________________________________________ train ___________________________________________ '''
    # get optimizer
    optimizer = get_optimizer(raw_model, weight_decay, learning_rate, device_type, master_process)
    # start train loop
    max_grad_accum_stages = epochs * grad_accum_stages_per_epoch
    max_steps = max_grad_accum_stages * grad_accum_steps
    resume_from_ckpt(raw_model, ckpt_dir)
    for epoch in range(epochs):
        print(f'epoch: {epoch} / {epochs}:')
        for grad_accum_stage in range(grad_accum_stages_per_epoch):
            global_grad_accum_stage = epoch * grad_accum_stages_per_epoch + grad_accum_stage
            last_grad_accum_stage = (global_grad_accum_stage == max_grad_accum_stages - 1)
            # train model on several batches
            train_grad_accum_steps(
                model, train_loader, optimizer, grad_accum_steps, max_lr, warmup_steps, max_steps, device, 
                device_type, master_process, dp, dp_world_size, global_grad_accum_stage, log_file
            )
            # once in a while evaluate our validation loss
            if global_grad_accum_stage % save_interval == 0 or last_grad_accum_stage:
                valid_epoch_wise(
                    model, raw_model, val_loader, device, device_type, master_process, dp, val_steps, 
                    global_grad_accum_stage, last_grad_accum_stage, ckpt_dir, log_file, log_dir
                )
            # once in a while generate from the model (except step 0, which is noise)
            if (global_grad_accum_stage % save_interval == 0 or last_grad_accum_stage) and (not use_compile):
                tokens = enc.encode("Hello, I'm a language model,")
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                x = tokens.to(device)
                gen_sentences(
                    model, enc, x, device, device_type, num_return_sequences, max_length, dp_global_rank
                )
    ternimate_dist(dist_strategy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--dp_world_size', type=int, help='Total processes to train the model')
    parser.add_argument('--torch_mp_launch', action='store_true')
    args = parser.parse_args()
    # launch by torch.multiprocessing
    if args.torch_mp_launch:
        mp.spawn(main, args=(args.dp_world_size, args.torch_mp_launch), nprocs=args.dp_world_size)
    # launch by torchrun or python
    else:
        main()
