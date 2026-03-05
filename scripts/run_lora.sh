#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-/assets/hub}"
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export TP="${TP:-1}"
export LORA_PATH="${LORA_PATH:-/path/to/lora/checkpoint}"

python -u examples/03_lora_request.py
