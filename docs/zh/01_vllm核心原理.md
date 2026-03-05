# vLLM 核心原理（面向实践）

本文不追求论文级形式化，而是解释“为什么这些参数会影响可用性与性能”。

## 1. vLLM 是什么

vLLM 是一个高吞吐、低延迟的 LLM 推理引擎。它不是训练框架，而是服务与推理优化框架。

它的价值在于：
- 对 KV Cache 的高效管理。
- 动态批处理（continuous batching）提升 GPU 利用率。
- 为在线服务场景提供更稳定吞吐。

## 2. 推理过程的关键阶段

一个请求通常经历两段：

1. Prefill：把输入 prompt 编码并建立初始 KV Cache。
2. Decode：逐 token 生成输出，持续读写 KV Cache。

实践里常见现象：
- 长 prompt 时 prefill 压力更大。
- 高并发时 decode 调度策略对吞吐影响更明显。

## 3. 为什么 TP 会有“heads 必须整除”

在注意力层里，多头会分摊到不同并行分片上。若 `num_attention_heads` 不能被 `tp` 整除，就无法均匀切分。

因此你会看到规则：

`num_attention_heads % tp == 0`

这就是本仓库 `assert_tp_valid()` 的意义：
- 在启动引擎前快速报错。
- 避免启动后才因并行配置失败。

## 4. 常见运行参数背后逻辑

- `gpu_memory_utilization`
  - 越大越“激进”利用显存，但更容易 OOM。
- `max_model_len`
  - 上下文上限直接影响 KV Cache 规模，显存压力随之上升。
- `enable_chunked_prefill`
  - 常用于改善长上下文 prefill 的稳定性与吞吐。
- `enforce_eager`
  - 更利于调试；在某些环境可能比图模式更稳。
- `disable_custom_all_reduce`
  - 在部分硬件/驱动组合上可提升稳定性（代价是性能可能变化）。

## 5. 从“能跑”到“稳定跑”的路径

1. 先固定模型与 TP=1 验证功能正确。
2. 再增大 TP，先做 heads 整除检查。
3. 最后调 `max_model_len` 与内存参数，观察 OOM 边界。

不要同时改太多参数，否则无法判断问题来源。
