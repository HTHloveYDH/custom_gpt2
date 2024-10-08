import os
import sys
import time

import torch
import tiktoken

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from models.get_model import get_model
from gen.gen_funcs import gen_sentences_v2, gen_sentences_v1
from utils.load_config import load_config_from_json as load_configs


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    gpt_config, gen_config, cloud_config, dist_config = load_configs('gen')
    # distribute configs
    dist_type = dist_config['dist_type']
    assert dist_type in ['ddp', 'fsdp', 'default'], f'distribute strategy: {dist_type} is not supported'
    # generation configs
    seed = gen_config['seed']  # defaults to 1337
    max_length = gen_config['max_length']
    num_return_sequences = gen_config['num_return_sequences']
    gen_func = gen_config['gen_func']
    assert gen_func in ['v1', 'v2'], f'generation function version: {gen_func} is not supported'
    # gpt configs
    use_compile = gpt_config['use_compile']
    num_return_sequences = 4 if gpt_config['load_weights'] == 'official' else num_return_sequences
    assert num_return_sequences == gpt_config['num_return_sequences']
    gpt_config['n_group_head'] = 1 if gpt_config['load_weights'] == 'official' else gpt_config['n_group_head']
    # set up DP (distributed data parallel or fully sharded data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    dp = dist_type in ['ddp', 'fsdp']
    dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type = init_dist(
        dist_type, False, 0, 1
    )
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    ''' ____________________________________ build & compile model ___________________________________ '''
    model, raw_model = get_model(gpt_config, device, dist_type)

    ''' ____________________________________________ test ___________________________________________ '''
    gen_sentences = {'v1': gen_sentences_v1, 'v2': gen_sentences_v2}[gen_func]
    # get tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    tokens = tokenizer.encode(gen_config['prompt'])
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    t0 = time.time()
    gen_sentences(
        model, tokenizer, x, device, device_type, num_return_sequences, max_length, dp_global_rank
    )
    print('time cost for generating sentences: ', time.time() - t0, ' seconds')
    ternimate_dist(dist_type)

if __name__ == '__main__':
    main()
