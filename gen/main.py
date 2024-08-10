import os
import sys

import tiktoken
import torch

sys.path.append(os.getcwd())
from dist.distribute import init_dist, ternimate_dist
from models.get_model import get_model
from gen.gen_funcs import gen_sentences
from utils.load_config import load_config_from_json as load_configs


def main():
    ''' __________________________________________ setup _____________________________________________ '''
    gpt_config, test_config, cloud_config, dist_config = load_configs('gen')
    # train configs
    seed = test_config['seed']  # defaults to 1337
    max_length = test_config['max_length']
    num_return_sequences = test_config['num_return_sequences']
    ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type = init_dist(False)
    # set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    ''' ____________________________________ build & compile model ___________________________________ '''
    enc, model, _ = get_model(gpt_config, device, False, ddp_local_rank)

    ''' ____________________________________________ test ___________________________________________ '''
    tokens = enc.encode(test_config['prompt'])
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to('cuda')
    gen_sentences(
        model, enc, x, device, device_type, num_return_sequences, max_length, ddp_rank
    )

if __name__ == '__main__':
    main()