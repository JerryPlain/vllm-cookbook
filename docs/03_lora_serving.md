# LoRA Serving

This repo uses runtime LoRA attachment through `vllm.lora.request.LoRARequest`.

## When to use
- Multi-adapter serving
- Reusing one base model with optional adapters

## Caveats
- Confirm adapter compatibility with base model
- Keep `LORA_PATH` optional for fallback runs without adapter
