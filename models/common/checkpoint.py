from dataclasses import dataclass
from typing import Any, Dict, Optional, OrderedDict
import torch

class WelfordNormalizer:
    def __init__(self):
        self.k = 0
        self.M = torch.zeros(3)
        self.S = torch.zeros(3)
    
    def update(self, x):
        self.k += 1
        oldM = self.M.clone()
        self.M += (x - self.M) / self.k
        self.S += (x - oldM) * (x - self.M)
    
    def mean(self):
        return self.M

    def variance(self):
        if self.k > 1:
            return self.S / (self.k - 1)
        else:
            return torch.full((3,), float('nan'))

    def std_dev(self):
        return torch.sqrt(self.variance())

    def normalize(self, x):
        return (x - self.mean()) / self.std_dev()

    def inverse(self, x):
        return x * self.std_dev() + self.mean()
    

@dataclass
class Checkpoint:
    model_state_dict: OrderedDict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    normalizer: Optional[WelfordNormalizer]

