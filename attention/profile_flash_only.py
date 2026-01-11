import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2"
    ).cuda().eval()
except Exception as e:
    print(f"Flash Attention not available: {e}")
    exit(1)

batch_size = 1
seq_len = 64
hidden_size = model.config.hidden_size

attn_layer = model.transformer.h[0].attn
hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)

for _ in range(10):
    with torch.no_grad():
        attn_layer(hidden_states)

with profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True
) as prof:
    with torch.no_grad():
        for _ in range(100):
            attn_layer(hidden_states)

print()
print("Flash Attn Results - Only Attention Layer:")
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