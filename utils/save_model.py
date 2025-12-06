import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               path,
               full_model = True):
    if full_model:
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), path)


