
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    print("WARNING: using CPU fallback, things will be really slow :(")
    return torch.device("cpu")

DEVICE: torch.device = get_device()
