import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Optional
from dataclasses import dataclass
import transformers

# ==============================================
# 全局配置（通用所有大模型）
# ==============================================
@dataclass
class KnowledgeChipConfig:
    hidden_dim: int = None    # 自动从模型获取
    top_k: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    chip_dir: str = "./knowledge_chips"
    use_gate_fusion: bool = True

os.makedirs(KnowledgeChipConfig.chip_dir, exist_ok=True)

# ==============================================
# 🧩 知识芯片（独立训练、独立保存）
# ==============================================
class KnowledgeChip(nn.Module):
    def __init__(self, hidden_dim, chip_id: str, desc: str = ""):
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
        path = os.path.join(KnowledgeChipConfig.chip_dir, f"{self.chip_id}.pt")
        torch.save({
            "state_dict": self.state_dict(),
            "chip_id": self.chip_id,
            "desc": self.desc,
            "hidden_dim": self.hidden_dim
        }, path)

    @staticmethod
    def load(chip_id):
        path = os.path.join(KnowledgeChipConfig.chip_dir, f"{ch chip_id}.pt")
        ckpt = torch.load(path, map_location=KnowledgeChipConfig.device)
        chip = KnowledgeChip(ckpt["hidden_dim"], ckpt["chip_id"], ckpt["desc"])
        chip.load_state_dict(ckpt["state_dict"])
        return chip

# ==============================================
# 🚦 路由器 + 融合层（通用结构）
# ==============================================
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

class KnowledgeFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())

    def forward(self, h, knowledge):
        g = self.gate(torch.cat([h, knowledge], dim=-1))
        return h + g * knowledge

# ==============================================
# 🏭 通用知识芯片引擎（核心！适配所有大模型）
# ==============================================
class UniversalKnowledgeEngine:
    def __init__(self, config: KnowledgeChipConfig):
        self.config = config
        self.device = config.device
        self.chips: List[KnowledgeChip] = []
        self.router = None
        self.fusion = None

    def init_with_model(self, model):
        """自动读取模型维度，适配所有大模型"""
        hidden_dim = model.config.hidden_size
        self.config.hidden_dim = hidden_dim
        self.router = KnowledgeRouter(hidden_dim).to(self.device).eval()
        self.fusion = KnowledgeFusion(hidden_dim).to(self.device).eval()
        print(f"✅ 自动适配模型: hidden_dim = {hidden_dim}")

    def add_chip(self, chip: KnowledgeChip):
        chip.eval().to(self.device)
        self.chips.append(chip)
        print(f"🔌 加载知识芯片: {chip.chip_id} | {chip.desc}")

    def remove_chip(self, chip_id):
        self.chips = [c for c in self.chips if c.chip_id != chip_id]

    @torch.no_grad()
    def enhance_hidden(self, hidden_states):
        if not self.chips:
            return hidden_states

        scores = self.router.score(hidden_states, self.chips)
        top_vals, top_idx = torch.topk(scores, k=self.config.top_k)
        weights = top_vals.softmax(dim=-1)

        B, L, D = hidden_states.shape
        knowledge = torch.zeros_like(hidden_states)
        for b in range(B):
            ks = 0
            for i, w in zip(top_idx[b], weights[b]):
                ks += w * self.chips[i](hidden_states[b:b+1])
            knowledge[b] = ks

        return self.fusion(hidden_states, knowledge)

# ==============================================
# 🔗 万能 Hook：自动注入 FFN 前（Llama/Qwen/ChatGLM 通用）
# ==============================================
def apply_knowledge_chip_hook(model, engine: UniversalKnowledgeEngine):
    """
    自动给所有层 FFN 前注入知识
    支持：Llama, Qwen, ChatGLM, Baichuan, DeepSeek
    """
    def ffnhook(module, input, output):
        hidden_states = input[0]
        enhanced = engine.enhance_hidden(hidden_states)
        return (enhanced,) + input[1:]

    # 自动找到所有 FFN 并注册 hook
    for name, module in model.named_modules():
        if any(kw in name.lower() for kw in ["mlp", "ffn", "feedforward"]):
            module.register_forward_hook(ffnhook)
            print(f"🎯 Hook 注入: {name}")

# ==============================================
# 🚀 部署演示：一键加载任意大模型 + 知识芯片
# ==============================================
if __name__ == "__main__":
    # -------------------
    # 1. 加载任意大模型
    # -------------------
    model_name = "Qwen/Qwen-1.5-0.5B-Chat"  # 可替换：
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "THUDM/chatglm3-6b"
    # model_name = "baichuan-inc/Baichuan2-7B-Chat"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # -------------------
    # 2. 初始化知识芯片引擎
    # -------------------
    cfg = KnowledgeChipConfig()
    engine = UniversalKnowledgeEngine(cfg)
    engine.init_with_model(model)

    # -------------------
    # 3. 加载/创建芯片
    # -------------------
    chip1 = KnowledgeChip(model.config.hidden_size, "math_v2", "数学知识")
    chip2 = KnowledgeChip(model.config.hidden_size, "law_v1", "法律知识")
    chip1.save()
    chip2.save()

    engine.add_chip(chip1)
    engine.add_chip(chip2)

    # -------------------
    # 4. 注入 Hook（万能适配）
    # -------------------
    apply_knowledge_chip_hook(model, engine)

    # -------------------
    # 5. 推理测试
    # -------------------
    text = "请解释一下三角形内角和"
    inputs = tokenizer([text], return_tensors="pt").to(cfg.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n✅ 通用大模型 + 知识芯片 部署成功！")