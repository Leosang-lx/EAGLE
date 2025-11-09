from dataclasses import dataclass, field
from typing import List

@dataclass
class ASDConfig:
    platform = 'server'  # 'server' or 'client'
    mode = 'demo'  # 'demo' or 'eval'
    model_name: str = 'llama3.1-8b'
    spec_method: str = 'ea3'  # 'ea' or 'ea3'

    total_token: int = 64
    top_k: int = 10
    depth: int = 6
    expand_size: int = 64

    warmup: int = 10
    test: int = 10

    max_new_tokens: int = 256

    log: bool = True
    prof: bool = True
    
    if mode == 'demo':
        your_message: str ='Hello'
        temperature: float = 0.0
    
    # model path
    if platform == 'server':
        weights_dir: str = '/home/liux/big_file/'
    else:
        weights_dir: str = 'C:/model_file/'
    model_dir = {
        'llama3-8b': {
            'base_dir': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'ea_dir': 'yuhuili/EAGLE-LLaMA3-Instruct-8B'
        },
        'llama3.1-8b': {
            'base_dir': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'ea3_dir': 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B'
        },
        'vicuna-13b': {
            'base_dir': 'lmsys/vicuna-13b-v1.3',
            'ea3_dir': 'yuhuili/EAGLE3-Vicuna1.3-13B'
        }
    }

    base_model_path: str = weights_dir + model_dir[model_name]['base_dir']
    ea_model_path: str = weights_dir + model_dir[model_name][f'{spec_method}_dir']

config = ASDConfig()