import torch
import src.utils as log

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        log.info(f"Using device: {device} (CUDA)")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        log.info(f"Using device: {device} (MPS)")
        return device
    else:
        device = torch.device('cpu')
        log.info(f"Using device: {device} (CPU)")
        return device
