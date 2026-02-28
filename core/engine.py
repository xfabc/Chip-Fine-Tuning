'''
知识芯片总引擎
'''
import torch
from core.knowledge_chip import KnowledgeChip
from core.router import KnowledgeRouter
from core.fusion import KnowledgeFusion
from config.chip_config import ChipConfig

class KnowledgeEngine:
    def __init__(self):
        self.chips = []
        self.router = None
        self.fusion = None
        self.hidden_dim = None

    def init(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.router = KnowledgeRouter(hidden_dim).eval()
        self.fusion = KnowledgeFusion(hidden_dim).eval()

    def add_chip(self, chip: KnowledgeChip):
        chip.eval()
        self.chips.append(chip)

    @torch.no_grad()
    def enhance(self, h):
        if not self.chips:
            return h

        scores = self.router.score(h, self.chips)
        top_vals, top_idx = torch.topk(scores, k=ChipConfig.TOP_K)
        weights = top_vals.softmax(dim=-1)

        B, L, D = h.shape
        knowledge = torch.zeros_like(h)
        for b in range(B):
            ks = 0
            for i, w in zip(top_idx[b], weights[b]):
                ks += w * self.chips[i](h[b:b+1])
            knowledge[b] = ks
        return self.fusion(h, knowledge)