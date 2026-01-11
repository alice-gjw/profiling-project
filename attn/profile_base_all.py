import torch
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer
from profile_utils import print_profile_results

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
).eval()

text = "Alice is falling down the"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

for _ in range(10):
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

print_profile_results(prof, "Base Attention Results")