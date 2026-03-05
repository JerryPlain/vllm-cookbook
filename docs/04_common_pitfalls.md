# Common Pitfalls

- `max_model_len` too large for available GPU memory can cause OOM.
- Invalid TP causes startup failure (heads not divisible by TP).
- Disable/enable features (`chunked_prefill`, `enforce_eager`, custom all-reduce) based on hardware behavior.
- Prefer fail-fast checks before launching expensive workloads.
