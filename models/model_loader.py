'''
统一加载 Llama / Qwen / ChatGLM
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.model_config import ModelConfig

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ModelConfig.MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def apply_knowledge_hook(model, engine):
    def hook_fn(module, input, output):
        enhanced = engine.enhance(input[0])
        return (enhanced,) + input[1:]
    for name, module in model.named_modules():
        if any(kw in name.lower() for kw in ["mlp", "ffn", "feedforward"]):
            module.register_forward_hook(hook_fn)