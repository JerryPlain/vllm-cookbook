# vLLM Cookbook

A reproducible set of recipes for using vLLM:
- Minimal generation
- Chat template prompts
- Token-ids API
- Tensor Parallel (TP) checks
- LoRA inference (`LoRARequest`)

## Install
```bash
pip install -e .
```

## Quickstart
```bash
bash scripts/run_minimal.sh
```

## Directory
- `docs/`: practical notes and pitfalls
- `examples/`: runnable scripts for common usage patterns
- `scripts/`: shell wrappers with sensible defaults
- `src/vllm_cookbook/`: reusable helpers
- `tests/`: lightweight checks
