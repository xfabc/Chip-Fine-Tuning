'''
融合层
'''


import torch
import torch.nn as nn

class KnowledgeFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h, knowledge):
        g = self.gate(torch.cat([h, knowledge], dim=-1))
        return h + g * knowledge