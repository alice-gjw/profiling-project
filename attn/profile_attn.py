import argparse
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer
from profile_utils import print_profile_results

def load_model(model_name, attn_impl):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_impl
    ).eval()
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def profile_full_generation(model, tokenizer, title):
    text = "Alice is falling down the"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    for _ in range(10):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=50)
    
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=50)
        torch.cuda.synchronize()

    print_profile_results(prof, f"{title} - Full Generation:")

def profile_attention_layer(model, title, seq_len=64):
    batch_size = 1
    hidden_size = model.config.hidden_size

    attn_layer = model.model.layers[0].self_attn
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)

    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    cos, sin = rotary_emb(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    for _ in range(10):
        with torch.no_grad():
            attn_layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)

    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            for _ in range(100):
                attn_layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
        torch.cuda.synchronize()

    print_profile_results(prof, f"{title} - Attention Layer Only:")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true", help="Use base attention")
    parser.add_argument("--flash_attn_2", action="store_true", help="Use flash attention 2")
    parser.add_argument("--full", action="store_true", help="Profile full generation")
    parser.add_argument("--attn_layer", action="store_true", help="Profile attention layer only")
    args = parser.parse_args()

    model_name = "mistralai/Mistral-7B-v0.1"

    if args.eager:
        model, tokenizer = load_model(model_name, "eager")
        if args.full:
            profile_full_generation(model, tokenizer, "Base Attn")
        if args.attn_layer:
            profile_attention_layer(model, "Base Attn")

    if args.flash_attn_2:
        model, tokenizer = load_model(model_name, "flash_attention_2")
        if args.full:
            profile_full_generation(model, tokenizer, "Flash Attn")
        if args.attn_layer:
            profile_attention_layer(model, "Flash Attn")