# vLLM Cookbook: End-to-End Privacy Alignment Pipeline

本仓库围绕一个完整流程构建：

1. 使用 LoRA 做 SFT 微调（`train_sft.py`）
2. 使用 vLLM 生成模型回答（`response_generation.py`）
3. 使用 Qwen3 Privacy Guard 评估回答（`response_evaluation.py` + `evaluators/Qwen3PrivacyGuard_vllm.py`）
4. 产出结构化评估结果和汇总指标（`summary.json`）

README 以当前代码实现为准，不是泛化模板。

## 1. Repository Structure

- `train_sft.py`: LoRA SFT 训练入口（TRL `SFTTrainer`）
- `response_generation.py`: vLLM 批量生成回答（支持 LoRARequest）
- `response_evaluation.py`: 调用 Qwen3Guard evaluator 进行评测并汇总
- `evaluators/Qwen3PrivacyGuard_vllm.py`: 评测模型（judge）vLLM 推理实现
- `evaluators/system_prompts/system_prompt_response_evaluation_20260109.txt`: judge system prompt
- `examples/`: 轻量化参考脚本（最小生成、chat template、LoRA 注入、TP 检查等）
- `src/vllm_cookbook/`: 通用工具函数（如 TP 合法性检查）
- `docs/zh/`: 中文原理与调参文档

## 2. Pipeline Overview

### Stage A: SFT Training (LoRA)

入口：`train_sft.py`

目标：
- 在基础模型上训练 LoRA adapter
- 训练过程中按 epoch 导出 adapter checkpoint

核心机制：
- 数据输入格式：`[{"instruction": "...", "output": "..."}]`
- 训练标签构造：prompt 部分 label 为 `-100`，target 部分参与 loss
- 回调 `PeftSavingCallback` 在每次保存时落盘 adapter

输出目录：
- `./sft_results/<dataset_name>/<model_suffix>/checkpoint-epoch-<E>`

该目录可直接用于 vLLM `LoRARequest` 推理。

### Stage B: Response Generation (vLLM)

入口：`response_generation.py`

目标：
- 在评测数据集上批量生成目标模型回答
- 支持 base 模型和 LoRA adapter 两种推理路径

核心机制：
- `tensor_parallel_size = --num_gpus`
- `--adapter_path` 非空时，自动启用 `enable_lora=True` + `LoRARequest`
- 基于 adapter path 生成稳定 `run_id`，避免实验覆盖
- 如果 eval 结果已存在且未 `--overwrite`，支持 skip

输出目录：
- `model_responses/<dataset_name>/<run_id>/response_by_<model_alias>.json`

### Stage C: Response Evaluation (Qwen3Guard)

入口：`response_evaluation.py`

目标：
- 对 `(question, response)` 执行 judge 评测
- 输出逐样本标注和全局统计

核心机制：
- 调用 `Evaluator_Qwen3PrivacyGuard`（vLLM 推理）
- judge 输出 XML 标签并解析为结构化字段：
  - `refuse`
  - `disclose`
  - `privacy`
  - `guidance`
- 汇总统计到 `summary.json`（含 counts/rates）

输出目录：
- `eval_results/<dataset_name>/<run_id>/eval_results_target=<target_model_name>.json`
- `eval_results/<dataset_name>/<run_id>/summary.json`

## 3. End-to-End Commands

以下命令按真实 pipeline 顺序组织。

### 3.1 Train LoRA Adapter

```bash
python3 train_sft.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --data_path ./data/sft_train.json \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --lora_r 16 \
  --lora_alpha 32
```

训练完成后，选择一个 adapter checkpoint，例如：
`./sft_results/<dataset>/<model>/checkpoint-epoch-5`

### 3.2 Generate Responses with vLLM

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 response_generation.py \
  --dataset_name multi_opensource_v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path ./sft_results/<dataset>/<model>/checkpoint-epoch-5 \
  --num_gpus 4 \
  --max_tokens 2048 \
  --output_dir ./model_responses \
  --eval_root ./eval_results
```

不带 LoRA 时将 `--adapter_path` 留空。

### 3.3 Evaluate Responses with Qwen3Guard

```bash
python3 response_evaluation.py \
  --input_path ./model_responses/multi_opensource_v1/<run_id>/response_by_Qwen2.5-7B-Instruct.json \
  --output_dir ./eval_results \
  --dataset_name multi_opensource_v1 \
  --target_model_name Qwen2.5-7B-Instruct \
  --model_path ./Qwen3-4B-Instruct-2507_V1 \
  --system_prompt_path system_prompt_response_evaluation_20260109.txt \
  --batch_size 1 \
  --max_new_tokens 2560
```

## 4. Data Contracts

### 4.1 SFT Input Contract (`train_sft.py`)

每条样本至少包含：
- `instruction` (str)
- `output` (str)

### 4.2 Generation Dataset Contract (`response_generation.py`)

`DATASET_REGISTRY` 当前内置：
- `multi_opensource_v1` -> `./eval_datasets/alignment_data_v2_privacy_leakage.json`

每条样本至少应包含：
- `question` (str)

### 4.3 Evaluation Input Contract (`response_evaluation.py`)

评估输入 JSON 每条样本至少应包含：
- `question` (str)
- `response` (str)

## 5. Qwen3PrivacyGuard Evaluator Design

文件：`evaluators/Qwen3PrivacyGuard_vllm.py`

### 5.1 What It Does

- 使用 vLLM 启动 judge 模型
- 构造评测 prompt（system + user template）
- 生成评测文本
- 从 `<answer>...</answer>` 中抽取四个标签

### 5.2 Prompt Construction (`format_prompt`)

`format_prompt` 的核心是长度预算控制：

1. 先计算模板开销 token（system + template wrapper）
2. 在 `max_input_length` 下分配 question/response 预算
3. question 左截断保留尾部，response 右截断保留头部

这是为了在长文本下保留判别信号。

### 5.3 Output Parsing (`extract_xml_output`)

解析目标格式：

```xml
<answer>
  <refuse>0|1</refuse>
  <disclose>0|1</disclose>
  <privacy>0|1</privacy>
  <guidance>0|1</guidance>
</answer>
```

## 6. Key Paths and Artifacts

- 训练拆分缓存：`./sft_training_data/<dataset_name>/train_split_*.json`
- LoRA 训练产物：`./sft_results/<dataset_name>/<model_suffix>/checkpoint-epoch-*`
- 生成结果：`./model_responses/<dataset>/<run_id>/response_by_<model>.json`
- 评测结果：`./eval_results/<dataset>/<run_id>/eval_results_target=<model>.json`
- 汇总指标：`./eval_results/<dataset>/<run_id>/summary.json`

## 7. Operational Notes

- TP 合法性：`num_attention_heads % tensor_parallel_size == 0`
- `response_generation.py` 与 evaluator 都采用 `pipeline_parallel_size=1`
- `response_generation.py` 的 `run_id` 与 adapter path 绑定，便于追踪实验
- 已有评测结果时默认 skip，重复实验请加 `--overwrite`
- 评估流程与生成流程目前为两个独立 CLI 步骤（推荐显式分步执行）

## 8. Quick Diagnostic Checklist

1. 生成前检查 TP 是否可整除（可参考 `examples/04_tp_check_heads.py`）
2. `response_generation.py` 的 `dataset_name` 是否在 `DATASET_REGISTRY` 中
3. `response_evaluation.py` 的 `input_path` 是否与生成输出路径一致
4. `target_model_name` 是否与生成文件命名匹配（用于输出文件名与追踪）
5. judge `model_path` 与 `system_prompt_path` 是否可访问

## 9. Minimal Example Scripts (Reference Only)

`examples/` 提供的是最小学习路径，不直接替代上述生产 pipeline：

- `examples/00_minimal_generate.py`: 最小 vLLM 推理
- `examples/01_chat_template.py`: chat template 输入
- `examples/02_token_ids_api.py`: `prompt_token_ids` 输入
- `examples/03_lora_request.py`: LoRARequest 注入
- `examples/04_tp_check_heads.py`: TP 合法性校验

## 10. Chinese Deep-Dive Docs

- `docs/zh/00_仓库导读.md`
- `docs/zh/01_vllm核心原理.md`
- `docs/zh/02_lora注入与服务.md`
- `docs/zh/03_参数与调优清单.md`
