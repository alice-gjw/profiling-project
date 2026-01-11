import torch
from transformers import AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"
).eval()

attn_layer = model.model.layers[0].self_attn

print("=== attn_layer attributes ===")
print([x for x in dir(attn_layer) if not x.startswith('_')])

print("\n=== model.model attributes ===")
print([x for x in dir(model.model) if not x.startswith('_')])