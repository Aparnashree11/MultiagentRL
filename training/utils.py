# training/utils.py
import os
import torch

def save_checkpoint(model_state_dict, filename):
    """
    Save a model checkpoint to the specified filename.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the checkpoint
    torch.save(model_state_dict, filename)

    
def load_checkpoint(filename):
    """
    Load model checkpoint on the appropriate device.
    """
    return torch.load(filename, map_location=torch.device('cpu'))

