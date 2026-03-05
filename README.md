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
- `docs/zh/`: Chinese deep-dive notes for architecture and operations
- `examples/`: runnable scripts for common usage patterns
- `scripts/`: shell wrappers with sensible defaults
- `src/vllm_cookbook/`: reusable helpers
- `tests/`: lightweight checks

## Chinese Documentation
- `docs/zh/00_仓库导读.md`
- `docs/zh/01_vllm核心原理.md`
- `docs/zh/02_lora注入与服务.md`
- `docs/zh/03_参数与调优清单.md`
