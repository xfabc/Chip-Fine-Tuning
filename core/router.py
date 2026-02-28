'''
路由器
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeRouter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @torch.no_grad()
    def score(self, h, chips):
        query = self.q_proj(h)
        scores = []
        for chip in chips:
            k_vec = chip(h)
            sim = F.cosine_similarity(query, k_vec, dim=-1).mean(dim=-1)
            scores.append(sim)
        return torch.stack(scores, dim=-1)