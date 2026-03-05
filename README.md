# vLLM Cookbook

A structured, reproducible repository for vLLM inference workflows, including baseline generation, prompt construction, tensor-parallel validation, and LoRA runtime injection.

## 1. Scope

This repository is designed for three engineering objectives:

1. **Reproducibility**: each knowledge point is mapped to a runnable script.
2. **Operational stability**: frequent failure modes are handled through fail-fast checks.
3. **Knowledge retention**: principles and troubleshooting guidance are documented in Chinese technical notes.

## 2. Repository Layout

- `examples/`: scenario-specific runnable scripts
- `scripts/`: shell entry points with consistent environment defaults
- `src/vllm_cookbook/`: reusable helper modules
- `docs/`: concise English notes
- `docs/zh/`: Chinese deep-dive documentation
- `tests/`: lightweight validation

## 3. Installation

```bash
pip install -e .
```

## 4. Quick Start

Run baseline generation:

```bash
bash scripts/run_minimal.sh
```

Run LoRA path:

```bash
LORA_PATH=/path/to/lora/checkpoint bash scripts/run_lora.sh
```

## 5. Example Matrix

| Script | Primary Purpose | Recommended Usage |
|---|---|---|
| `examples/00_minimal_generate.py` | Minimal end-to-end vLLM generation | First run on a new machine / new model |
| `examples/01_chat_template.py` | Chat-template-based prompt construction | Chat/instruct models with role formatting |
| `examples/02_token_ids_api.py` | Direct `prompt_token_ids` inference | Token-level control, cached token pipelines |
| `examples/03_lora_request.py` | Runtime LoRA injection via `LoRARequest` | Multi-adapter serving and A/B adapter checks |
| `examples/04_tp_check_heads.py` | Tensor parallel validity check | Before changing TP or launching long jobs |
| `examples/Qwen3PrivacyGuard_vllm.py` | Judge-style safety evaluation pipeline | Structured scoring for `(question, response)` pairs |

## 6. Recommended Execution Order

1. Validate baseline runtime with `00_minimal_generate.py`.
2. Select prompt interface:
   - chat workflow: `01_chat_template.py`
   - token-id workflow: `02_token_ids_api.py`
3. Add adapter evaluation with `03_lora_request.py` if LoRA is required.
4. Validate TP legality with `04_tp_check_heads.py` prior to larger runs.

## 7. Key Runtime Parameters

- `tensor_parallel_size`
  - Tensor-parallel degree.
  - Must satisfy: `num_attention_heads % tensor_parallel_size == 0`.

- `pipeline_parallel_size`
  - Pipeline-parallel degree.
  - Default `1` in this repository to prioritize debuggability and reproducibility.

- `gpu_memory_utilization`
  - Fraction of GPU memory allocated to vLLM.
  - Higher values may improve throughput but increase OOM risk.

- `max_model_len`
  - Maximum context length.
  - Directly affects KV cache memory footprint.

- `enable_chunked_prefill`
  - Improves robustness for long-context prefill on many workloads.

- `enforce_eager`
  - Generally easier to debug and often more stable across heterogeneous environments.

- `disable_custom_all_reduce`
  - Useful in environments where custom collective kernels are unstable.

## 8. LoRA Runtime Flow

In `examples/03_lora_request.py`, the workflow is:

1. Validate TP compatibility with `assert_tp_valid`.
2. Initialize `LLM(..., enable_lora=True)`.
3. Build `LoRARequest` when `LORA_PATH` is provided.
4. Execute generation with or without adapter on the same baseline path.

This design supports direct comparison between base and adapter behaviors under identical runtime settings.

## 9. Troubleshooting

- **TP configuration error**
  1. Execute `examples/04_tp_check_heads.py`.
  2. Adjust TP until divisibility is satisfied.

- **CUDA OOM**
  1. Reduce `max_model_len`.
  2. Lower `gpu_memory_utilization`.
  3. Reduce concurrency and re-validate with `TP=1`.

- **LoRA load/effect issue**
  1. Verify `LORA_PATH`.
  2. Confirm base-model inference first.
  3. Re-test adapter path with conservative settings (`TP=1`).

## 10. Environment Variables

Defined in `.env.example`:

- `CUDA_VISIBLE_DEVICES`
- `VLLM_DOWNLOAD_DIR`
- `MODEL`
- `TP`
- `LORA_PATH`

## 11. Qwen3 Privacy Guard Script

`examples/Qwen3PrivacyGuard_vllm.py` is a judge-model pipeline rather than a general chat demo.

### 11.1 Purpose

Given `(question, response)` pairs, the model produces structured XML labels under `<answer>...</answer>` for safety-oriented classification:

- `refuse`
- `disclose`
- `privacy`
- `guidance`

### 11.2 Processing Stages

1. Build judge prompts via `format_prompt`.
2. Optionally apply input-budget truncation.
3. Run batched generation with vLLM.
4. Parse XML tags via `extract_xml_output`.
5. Return structured records for downstream evaluation.

### 11.3 Truncation Policy in `format_prompt`

When `max_input_length` is set:

1. Estimate fixed template overhead.
2. Reserve a safety margin.
3. Allocate remaining budget to question/response.
4. Keep question tail and response head to retain high-signal spans.

This policy improves robustness under long-input constraints while preserving evaluation-relevant content.

## 12. Chinese Documentation

- `docs/zh/00_仓库导读.md`
- `docs/zh/01_vllm核心原理.md`
- `docs/zh/02_lora注入与服务.md`
- `docs/zh/03_参数与调优清单.md`
