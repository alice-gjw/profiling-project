import torch
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
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

print()
print("Flash Attn Results:")
print()

events = prof.key_averages().sort(key=lambda x: -x.self_cuda_time_total)[:15]

print("Latency(ms): GPU kernel execution time")
for event in events:
    print(f"  {event.key[:40]:<40} {event.self_cuda_time_total / 1000:.3f}")

print("\nMemory (MB)")
for event in events:
    print(f"  {event.key[:40]:<40} {event.cuda_memory_usage / 1e6:.2f}")

print("\nCalls")
for event in events:
    print(f"  {event.key[:40]:<40} {event.count}")