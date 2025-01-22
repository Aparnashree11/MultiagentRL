# env_setup/utils.py

import numpy as np
import random
import torch

def set_global_seed(seed):
    """
    Set seed for reproducibility across different libraries.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_message(message):
    """
    Simple logger function to print messages or save them to a file.
    """
    print(f"[LOG]: {message}")