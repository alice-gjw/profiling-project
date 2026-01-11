def print_profile_results(prof, title, top_n=15):
    print()
    print(f"{title}:")
    print()

    events = prof.key_averages()
    events = sorted(events, key=lambda x: -x.self_device_time_total)[:top_n]

    print("Latency (ms): GPU kernel execution time")
    for event in events:
        print(f"  {event.key[:40]:<40} {event.self_device_time_total / 1000:.3f}")

    print("\nMemory (MB)")
    for event in events:
        print(f"  {event.key[:40]:<40} {event.cuda_memory_usage / 1e6:.2f}")

    print("\nCalls")
    for event in events:
        print(f"  {event.key[:40]:<40} {event.count}")