'''
知识芯片训练脚本
'''
import torch
import torch.nn.functional as F
from core.knowledge_chip import KnowledgeChip

def train_chip(chip: KnowledgeChip, dataloader, epochs=3, lr=1e-4):
    optim = torch.optim.Adam(chip.parameters(), lr=lr)
    for _ in range(epochs):
        for h, target in dataloader:
            loss = F.mse_loss(chip(h), target)
            optim.zero_grad()
            loss.backward()
            optim.step()
    print(f"训练完成：{chip.chip_id}")