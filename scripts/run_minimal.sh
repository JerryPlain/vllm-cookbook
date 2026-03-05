#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-/assets/hub}"
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export TP="${TP:-1}"

python -u examples/00_minimal_generate.py
