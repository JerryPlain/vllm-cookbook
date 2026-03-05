# vLLM Cookbook

一个面向实战的 vLLM 参考仓库：
- 每个核心知识点都提供可运行脚本
- 每个关键坑点都有 fail-fast 检查
- 文档解释“为什么这样配”，而不只是给命令

## 1. 这个仓库是干什么的

当你以后再用 vLLM，通常会遇到三类问题：
1. 环境换了，之前能跑的参数突然不能跑。
2. TP / 显存 / 上下文参数组合不稳定，报错难定位。
3. LoRA 注入流程分散在聊天记录里，无法复用。

这个仓库就是把这些问题收敛成：
- `examples/`：最小可运行模板
- `src/vllm_cookbook/`：复用工具函数（如 TP 合法性检查）
- `docs/zh/`：中文原理与排障知识库

## 2. 快速开始

### 安装
```bash
pip install -e .
```

### 跑最小示例
```bash
bash scripts/run_minimal.sh
```

### 跑 LoRA 示例
```bash
LORA_PATH=/path/to/lora/checkpoint bash scripts/run_lora.sh
```

## 3. examples 为啥分成这几个

核心原则：**一个文件只解决一个场景**，方便你按需复制和排错。

### [examples/00_minimal_generate.py](/Users/jerryplain/projects/vllm-cookbook/examples/00_minimal_generate.py)
用途：最小通电自检（模型能否启动并生成）。

你该在什么时候用：
- 新机器/新环境第一跑
- 换模型后先验证基础链路

### [examples/01_chat_template.py](/Users/jerryplain/projects/vllm-cookbook/examples/01_chat_template.py)
用途：标准聊天格式（system/user/assistant）构造。

你该在什么时候用：
- 任务是对话模型
- 不想手写 prompt 格式，避免模板错误

### [examples/02_token_ids_api.py](/Users/jerryplain/projects/vllm-cookbook/examples/02_token_ids_api.py)
用途：直接传 `prompt_token_ids`，精确控制分词输入。

你该在什么时候用：
- 你已有上游 tokenizer 流水线
- 你要做 token 级实验/缓存/对齐

### [examples/03_lora_request.py](/Users/jerryplain/projects/vllm-cookbook/examples/03_lora_request.py)
用途：运行时 LoRA 注入（`LoRARequest`）。

你该在什么时候用：
- 一个基座模型挂多个 adapter
- 需要“有 LoRA / 无 LoRA”快速切换对比

### [examples/04_tp_check_heads.py](/Users/jerryplain/projects/vllm-cookbook/examples/04_tp_check_heads.py)
用途：启动前 TP 合法性检查（heads 能否被 TP 整除）。

你该在什么时候用：
- 每次改 `TP` 前
- 启动大任务前的 fail-fast 检查

## 4. 推荐使用顺序（固定流程）

1. 先跑 `00_minimal_generate.py`，确认环境正常。
2. 根据任务选择：
   - 对话任务 -> `01_chat_template.py`
   - token 级控制 -> `02_token_ids_api.py`
3. 需要 LoRA -> `03_lora_request.py`
4. 改 TP 前先跑 `04_tp_check_heads.py`

## 5. 关键参数解释（重点）

以下参数在 `00_minimal_generate.py` 和 `03_lora_request.py` 最常见：

- `tensor_parallel_size`
  - 张量并行度（TP）。
  - 必须满足：`num_attention_heads % TP == 0`。
  - TP 越大，单卡权重压力可能降低，但通信开销更大。

- `pipeline_parallel_size`
  - 流水并行度（PP）。
  - 本仓库默认设为 `1`，原因是先保证最小可复现与排错简单。
  - 只有当模型很大、单机资源和调度都准备好时，再考虑 PP > 1。

- `gpu_memory_utilization`
  - 允许 vLLM 使用的显存比例。
  - 过高可能更快，但更容易 OOM。

- `max_model_len`
  - 最大上下文长度，直接影响 KV Cache 显存占用。

- `enable_chunked_prefill`
  - 常用于长上下文 prefill 稳定性优化。

- `enforce_eager`
  - 更利于调试与兼容；部分环境下更稳定。

- `disable_custom_all_reduce`
  - 某些硬件/驱动组合下可提升稳定性（性能可能有变化）。

## 6. LoRA 注入机制（你最常用）

在 [examples/03_lora_request.py](/Users/jerryplain/projects/vllm-cookbook/examples/03_lora_request.py) 里：
1. 先做 `assert_tp_valid(model, tp)`。
2. 初始化 `LLM(..., enable_lora=True)`。
3. 如果有 `LORA_PATH`，构造 `LoRARequest` 传给 `generate()`。
4. 如果没有 `LORA_PATH`，退化为基座模型推理。

这样一个脚本覆盖了“有 LoRA / 无 LoRA”两条路径，便于对比和排错。

## 7. 常见报错和排查顺序

- 报 TP 不合法
  1. 先跑 `examples/04_tp_check_heads.py`
  2. 调整 `TP` 到 heads 可整除值

- 报 OOM
  1. 降低 `max_model_len`
  2. 降低 `gpu_memory_utilization`
  3. 降低并发或先用 `TP=1` 验证

- LoRA 不生效或报错
  1. 检查 `LORA_PATH` 是否正确
  2. 先不带 LoRA 跑通基座模型
  3. 再带 LoRA + `TP=1` 跑最小闭环

## 8. 环境变量速查

参考 [.env.example](/Users/jerryplain/projects/vllm-cookbook/.env.example)：

- `CUDA_VISIBLE_DEVICES`: 使用哪些 GPU
- `VLLM_DOWNLOAD_DIR`: 模型缓存目录
- `MODEL`: 模型 ID 或本地路径
- `TP`: 张量并行度
- `LORA_PATH`: LoRA 目录（可选）

## 9. 仓库结构

- `docs/`: 英文简版文档
- `docs/zh/`: 中文深度文档（原理、LoRA、调优）
- `examples/`: 可运行示例
- `scripts/`: Shell 启动脚本
- `src/vllm_cookbook/`: 复用工具函数
- `tests/`: 轻量测试

## 10. 中文深度文档入口

- [docs/zh/00_仓库导读.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/00_仓库导读.md)
- [docs/zh/01_vllm核心原理.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/01_vllm核心原理.md)
- [docs/zh/02_lora注入与服务.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/02_lora注入与服务.md)
- [docs/zh/03_参数与调优清单.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/03_参数与调优清单.md)

## 11. `Qwen3PrivacyGuard_vllm.py` 解析

文件： [examples/Qwen3PrivacyGuard_vllm.py](/Users/jerryplain/projects/vllm-cookbook/examples/Qwen3PrivacyGuard_vllm.py)

这个脚本是“裁判模型（judge）”推理器，不是普通聊天脚本。输入是 `(question, response)` 对，输出是 XML 标签评分。

### 11.1 目标与输出协议

- 目标：判断回答是否有隐私风险、是否拒答等。
- 输出协议：模型应生成 `<answer>...</answer>`，内部有四个标签：
  - `refuse`
  - `disclose`
  - `privacy`
  - `guidance`

### 11.2 关键流程（从输入到结果）

1. 把 `(question, response)` 组装成 judge prompt（`format_prompt`）。
2. 按 `max_input_length` 做可选截断（先算模板开销，再分配 token 预算）。
3. 用 vLLM 批量生成文本结果。
4. 用 `extract_xml_output` 从文本中提取标签。
5. 返回结构化结果字典，便于后处理/打分/落盘。

### 11.3 为什么 `format_prompt` 要“分预算截断”

`format_prompt` 里做了三件关键事：

1. 先估算固定模板开销（system + chat template 包装）。
2. 把剩余 token 预算按比例分给 question/response。
3. 采用“question 保尾部、response 保头部”的策略：
   - question 尾部通常包含最新约束和意图
   - response 头部通常最先暴露是否合规

这样可以在超长输入下，最大化保留对评测最有信息量的片段。

### 11.4 你应该关注的参数

- `tensor_parallel_size`: TP 并行度，必须满足 heads 可整除。
- `pipeline_parallel_size`: 本脚本固定为 1，优先保证稳定和可复现。
- `max_model_len`: 影响上下文上限与 KV cache 显存压力。
- `max_new_tokens`: 当前默认较大，若只要标签可适当减小以提速。

### 11.5 两种调用方式

- `generate_pairs`：轻量路径，直接对 Python 列表做推理。
- `test_dataset`：HF Dataset 管线路径，适合批量处理和 map 流程。

### 11.6 使用建议

1. 先用 `generate_pairs` 验证功能，再上 `test_dataset` 做大批量。
2. 如果你只要 XML 标签，优先收紧 `max_new_tokens` 控制成本。
3. 保证 `questions` 与 `responses` 长度一致，避免 `zip` 静默截断。
