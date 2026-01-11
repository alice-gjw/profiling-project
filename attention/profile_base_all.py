import torch
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

text = "Alice is falling down the"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

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

print()
print("Base Attention Results:")
print()

events = prof.key_averages().sort(key=lambda x: -x.self_cuda_time_total)[:15]

print("LATENCY (ms)")
for event in events:
    print(f"  {event.key[:40]:<40} {event.self_cuda_time_total / 1000:.3f}")

print("\nMEMORY (MB)")
for event in events:
    print(f"  {event.key[:40]:<40} {event.cuda_memory_usage / 1e6:.2f}")

print("\nCALLS")
for event in events:
    print(f"  {event.key[:40]:<40} {event.count}")

prof.export_chrome_trace("trace_standard.json")