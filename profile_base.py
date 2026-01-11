import torch
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

text = "Alice is falling down the"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# Warm-up runs
for _ in range(3):
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

print(f"\n{'='*60}")
print("Base Attention Results")
print(f"{'='*60}\n")

print(f"{'Operation':<35} {'Time (ms)':>12} {'Memory (MB)':>12} {'Calls':>8}")
print("-" * 70)

for event in prof.key_averages().sort(key=lambda x: -x.self_cuda_time_total)[:15]:
    print(f"{event.key[:35]:<35} {event.self_cuda_time_total / 1000:>12.3f} {event.cuda_memory_usage / 1e6:>12.2f} {event.count:>8}")

prof.export_chrome_trace("trace_standard.json")