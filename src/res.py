# pytorch implementation of residual normalization layer

import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, module, d_model, half=True):
        super(Residual, self).__init__()
        self.net = module
        self.half = half
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, **kwargs):
        x = self.net(inputs, **kwargs)

        if self.half:
            inputs = self.layer_norm(inputs)
            return (x * 0.5) + inputs
        else:
            inputs = self.layer_norm(inputs)
            return x + inputs

