import torch
from fastapi import FastAPI
from models.model_loader import load_model, apply_knowledge_hook
from core.engine import KnowledgeEngine
from core.knowledge_chip import KnowledgeChip

app = FastAPI(title="知识芯片大模型服务")

# 加载模型
model, tokenizer = load_model()
hidden_dim = model.config.hidden_size

# 初始化引擎
engine = KnowledgeEngine()
engine.init(hidden_dim)

# 加载芯片
engine.add_chip(KnowledgeChip(hidden_dim, "math_v1", "数学"))
engine.add_chip(KnowledgeChip(hidden_dim, "law_v1", "法律"))

# 注入 Hook
apply_knowledge_hook(model, engine)
model.eval()

# API
@app.post("/chat")
def chat(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"prompt": prompt, "response": resp}