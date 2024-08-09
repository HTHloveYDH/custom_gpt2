import os
import json


def load_json(filename:str):
    with open(filename, 'r') as f:
        content = json.load(f)
    return content

def load_config_from_json(mode:str):
    gpt_config = load_json(os.path.join('.', 'config', 'gpt_config.json'))
    cloud_config = load_json(os.path.join('.', 'config', 'cloud_config.json'))
    dist_config = load_json(os.path.join('.', 'config', 'dist_config.json'))
    if mode == 'train':
        train_config = load_json(os.path.join('.', 'config', 'train_config.json'))
        data_config = load_json(os.path.join('.', 'config', 'data_config.json'))
        return gpt_config, train_config, data_config, cloud_config, dist_config
    elif mode == 'test':
        test_config = load_json(os.path.join('.', 'config', 'test_config.json'))
        return gpt_config, test_config, cloud_config, dist_config
    else:
        raise ValueError(f'configuration mode: {mode} is not supported!')