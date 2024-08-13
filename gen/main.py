import os
import sys

import torch

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from models.get_model import get_model
from gen.gen_funcs import gen_sentences
from utils.load_config import load_config_from_json as load_configs


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    gpt_config, test_config, cloud_config, dist_config = load_configs('gen')
    # distribute configs
    dist_strategy = dist_config['dist_strategy']
    assert dist_strategy == 'default'
    # train configs
    seed = test_config['seed']  # defaults to 1337
    max_length = test_config['max_length']
    num_return_sequences = test_config['num_return_sequences']
    # gpt configs
    use_compile = gpt_config['use_compile']
    assert num_return_sequences == gpt_config['num_return_sequences']
    # set up DP (distributed data parallel or fully sharded data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    dp = dist_strategy in ['ddp', 'fsdp']
    dp_global_rank, dp_local_rank, dp_world_size, master_process, device, device_type = init_dist(
        dist_strategy, False, 0, 1
    )
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    ''' ____________________________________ build & compile model ___________________________________ '''
    device_ids = [dp_local_rank]
    model, raw_model, enc = get_model(gpt_config, device, dist_strategy, device_ids)

    ''' ____________________________________________ test ___________________________________________ '''
    tokens = enc.encode(test_config['prompt'])
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    gen_sentences(
        model, enc, x, device, device_type, num_return_sequences, max_length, dp_global_rank
    )
    ternimate_dist(dist_strategy)

if __name__ == '__main__':
    main()
