class MinMaxNormalize:
    def __init__(self, min_vals, max_vals):
        self.min = min_vals
        self.max = max_vals

    def __call__(self, tensor):
        return (tensor - self.min[:, None, None]) / (self.max[:, None, None] - self.min[:, None, None])


class MinMaxUnnormalize:
    def __init__(self, min_vals, max_vals):
        self.min = min_vals
        self.max = max_vals

    def __call__(self, tensor):
        return tensor * (self.max[:, None, None] - self.min[:, None, None]) + self.min[:, None, None]

from torch.utils.data import DataLoader
import torch

def compute_min_max(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    min_val = torch.tensor([float('inf')] * 3)
    max_val = torch.tensor([float('-inf')] * 3)

    for images, _ in loader:
        # images shape: [B, C, H, W]
        batch_min = images.amin(dim=[0, 2, 3])
        batch_max = images.amax(dim=[0, 2, 3])
        min_val = torch.min(min_val, batch_min)
        max_val = torch.max(max_val, batch_max)
    
    return min_val, max_val

def unnormalize(dataset, min_value, max_value):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    min_val = torch.tensor([float('inf')] * 3)
    max_val = torch.tensor([float('-inf')] * 3)

    for images, _ in loader:
        # images shape: [B, C, H, W]
        batch_min = images.amin(dim=[0, 2, 3])
        batch_max = images.amax(dim=[0, 2, 3])
        min_val = torch.min(min_val, batch_min)
        max_val = torch.max(max_val, batch_max)
    
    return min_val, max_val