def print_profile_results(prof, title, top_n=15):
    print()
    print(f"{title}:")
    print()

    events = prof.key_averages()
    attn_metrics_filter = [
        "aten::bmm",     # Q@K and scores@V (base attention)
        "aten::_softmax",     # Attention softmax
        "aten::_flash_attention_forward",     # If FlashAttn enabled
        "aten::mm",     # Linear layers
        "aten::silu"     # Activation
    ]
    events = [e for e in events if e.key in attn_metrics_filter]
    events = sorted(events, key=lambda x: -x.self_device_time_total)[:top_n]

    print("Latency (ms): GPU kernel execution time")
    for event in events:
        print(f"  {event.key[:40]:<40} {event.self_device_time_total / 1000:.3f}")

    print("\nMemory (MB)")
    for event in events:
        print(f"  {event.key[:40]:<40} {event.self_device_memory_usage / 1e6:.2f}")

    print("\nCalls")
    for event in events:
        print(f"  {event.key[:40]:<40} {event.count}")