'''
知识芯片类
'''
import torch
import torch.nn as nn
import os
from config.chip_config import ChipConfig

os.makedirs(ChipConfig.CHIP_DIR, exist_ok=True)

class KnowledgeChip(nn.Module):
    def __init__(self, hidden_dim, chip_id, desc=""):
        super().__init__()
        self.chip_id = chip_id
        self.desc = desc
        self.hidden_dim = hidden_dim

        self.k_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h):
        return self.k_net(h)

    def save(self):
        path = os.path.join(ChipConfig.CHIP_DIR, f"{self.chip_id}.pt")
        torch.save({
            "state_dict": self.state_dict(),
            "chip_id": self.chip_id,
            "desc": self.desc,
            "hidden_dim": self.hidden_dim
        }, path)

    @staticmethod
    def load(chip_id):
        path = os.path.join(ChipConfig.CHIP_DIR, f"{self.chip_id}.pt")
        ckpt = torch.load(path)
        chip = KnowledgeChip(ckpt["hidden_dim"], ckpt["chip_id"], ckpt["desc"])
        chip.load_state_dict(ckpt["state_dict"])
        return chip