import torch
import numpy as np
import random
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model):
    print("Model Summary:")
    print(f"Total Parameters: {count_parameters(model):,}")
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")