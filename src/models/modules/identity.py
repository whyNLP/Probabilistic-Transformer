import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return x

    def _get_hyperparams(self):
        model_hps = {}
        return model_hps