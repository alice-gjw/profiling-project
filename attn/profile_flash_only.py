import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer
from profile_utils import print_profile_results

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    dtype=torch.float16,
    device_map="auto"
).eval()

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

print_profile_results(prof, "Flash Attn Results - Only Attention Layer")