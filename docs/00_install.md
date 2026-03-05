# Install

## Recommended baseline
- Python 3.9+
- CUDA-compatible PyTorch build
- `pip install -e .`

## Environment variables
- `CUDA_VISIBLE_DEVICES`: GPU IDs
- `VLLM_DOWNLOAD_DIR`: model cache directory (default: `/assets/hub`)
- `MODEL`: model id/path
- `TP`: tensor parallel size
- `LORA_PATH`: optional LoRA checkpoint path

## Notes
Keep torch/cuda aligned with your server drivers. This repo intentionally avoids hard-pinning GPU stack versions.
