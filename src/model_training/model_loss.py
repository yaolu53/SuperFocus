import torch
import torch.nn as nn
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        return (self.weights * (pred - target) ** 2).mean()