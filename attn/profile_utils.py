def print_profile_results(prof, title, top_n=15):
    print()
    print(f"{title}:")
    print()

    events = prof.key_averages()
    attn_metrics_filter = [
        "aten::bmm",     # Q@K and scores@V (base attention)
        "aten::_softmax",     # Attention softmax
        "aten::mm",     # Linear layers
        "aten::silu"     # Activation
    ]
    flash_keywords = ["flash_fwd_kernel"]
    
    events = [e for e in events if 
              e.key in attn_metrics_filter or
              any(kw in e.key for kw in flash_keywords)]
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