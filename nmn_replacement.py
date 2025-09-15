"""
Simple replacement for YatNMN module.
This provides a basic neural network module that can be used as a drop-in replacement.
"""
import torch
import torch.nn as nn


class YatNMN(nn.Module):
    """A simple replacement for the YatNMN module used in the original script."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = nn.GELU()  # Using GELU for better performance
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


# Create a mock nmn.torch.nmn module structure
class nmn:
    class torch:
        class nmn:
            YatNMN = YatNMN