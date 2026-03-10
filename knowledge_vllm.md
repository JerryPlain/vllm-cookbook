# vLLM Knowledge Base

这份文档的目标不是简单介绍“vLLM 是什么”，而是结合当前仓库的代码，把你真正会遇到的问题讲清楚：

- vLLM 到底解决什么问题，不解决什么问题
- 这个仓库里 vLLM 被怎样使用
- 参数为什么这样配
- 常见坑为什么会发生，怎么排查，怎么修
- LoRA、TP、多卡、评测、长上下文、部署分别该怎么思考

如果你是第一次接触 vLLM，建议按本文顺序读；如果你已经在跑仓库代码，可以直接跳到对应章节查问题。

## 1. 先建立正确心智模型

### 1.1 vLLM 是什么

vLLM 是一个 LLM 推理引擎。核心价值是：

- 高吞吐推理
- 更高效的 KV Cache 管理
- 更适合在线/批量请求调度
- 更适合多请求长度不一致的真实场景

它不是训练框架。这个仓库里的训练仍然是 `train_sft.py` 基于 Hugging Face / TRL / PEFT 完成，vLLM 只负责推理和服务。

### 1.2 为什么大家会用 vLLM

因为真实业务里，推理通常不是“单条 prompt + 固定长度输出”。

实际情况往往是：

- 每条 prompt 长度不同
- 每条输出长度不同
- 有些请求很快结束，有些请求很慢
- 同时还想尽量把 GPU 利用起来

普通 `transformers.generate()` 当然能跑，但在高吞吐场景下，经常会被以下问题拖垮：

- padding 浪费
- 批内最慢样本拖住其他样本
- KV Cache 管理不够高效
- 显存利用和吞吐调度不够“服务化”

vLLM 的优势正是在这里。

### 1.3 vLLM 不是“更快的 transformers”，而是“更像推理系统”

理解这一点很重要。

`transformers` 更像模型库。
vLLM 更像面向推理场景优化过的执行引擎。

所以你在用 vLLM 时，不应该只想着：

- “我怎么把 `generate()` 换个库调用”

更应该想：

- “我的请求如何进入引擎”
- “显存主要消耗在哪里”
- “吞吐瓶颈在 prefill 还是 decode”
- “为什么这个参数会影响稳定性”

## 2. 推理到底发生了什么

### 2.1 两个阶段：Prefill 和 Decode

一个生成请求通常分两步：

1. Prefill：把输入 prompt 编码，建立初始 KV Cache
2. Decode：逐 token 生成输出，持续读写 KV Cache

这两个阶段的瓶颈不同：

- prompt 很长时，prefill 压力大
- 并发很高时，decode 调度更关键
- 上下文很大时，KV Cache 会直接吃掉大量显存

### 2.2 KV Cache 为什么是 vLLM 的核心

模型在生成下一个 token 时，不想每次都把前面的上下文重新算一遍，于是会缓存 attention 的历史状态，这就是 KV Cache。

问题在于：

- 上下文越长，KV Cache 越大
- 并发请求越多，总 KV Cache 越大
- `max_model_len` 不是“只是个逻辑长度”，而是直接影响显存预算

所以很多 OOM 的根因并不是“模型参数太大”，而是：

- 模型权重 + KV Cache + 并发请求 + 输出长度上限

叠加之后爆了。

### 2.3 continuous batching 为什么重要

你可以把 vLLM 理解成“统一请求队列 + 动态调度”。

普通固定 batch 更像：

- 一批请求一起进
- 一起 pad
- 一起跑
- 谁慢谁拖住全组

vLLM 更像：

- 每条请求独立进入引擎
- 引擎动态决定谁先继续 decode
- 已完成的请求释放资源后，其余请求继续使用资源

因此它特别适合：

- 批量评测
- API 服务
- prompt / output 长度差异很大
- GPU 希望尽量持续满载

## 3. 这个仓库里 vLLM 在哪里

仓库主要有四类 vLLM 用法。

### 3.1 最小推理

文件：

- [examples/00_minimal_generate.py](/Users/jerryplain/projects/vllm-cookbook/examples/00_minimal_generate.py)
- [examples/01_chat_template.py](/Users/jerryplain/projects/vllm-cookbook/examples/01_chat_template.py)
- [examples/02_token_ids_api.py](/Users/jerryplain/projects/vllm-cookbook/examples/02_token_ids_api.py)

分别演示：

- 最小 `LLM(...) + SamplingParams(...) + generate(...)`
- 用 tokenizer 的 chat template 先拼 prompt 再生成
- 直接传 `prompt_token_ids`

### 3.2 LoRA 推理

文件：

- [examples/03_lora_request.py](/Users/jerryplain/projects/vllm-cookbook/examples/03_lora_request.py)
- [response_generation.py](/Users/jerryplain/projects/vllm-cookbook/response_generation.py)

这里用的是运行时 LoRA 注入：

- `LLM(..., enable_lora=True)`
- `LoRARequest(...)`
- `generate(..., lora_request=...)`

这意味着仓库走的是“基座模型 + 运行时适配器”的服务路线，而不是“先 merge 再单模型部署”。

### 3.3 Judge 模型推理

文件：

- [evaluators/Qwen3PrivacyGuard_vllm.py](/Users/jerryplain/projects/vllm-cookbook/evaluators/Qwen3PrivacyGuard_vllm.py)
- [response_evaluation.py](/Users/jerryplain/projects/vllm-cookbook/response_evaluation.py)

这是仓库里最像真实生产推理的部分：

- 构造大量 judge prompts
- 用 vLLM 批量推理
- 输出结构化 XML
- 做评测汇总

### 3.4 TP 合法性检查

文件：

- [src/vllm_cookbook/tp.py](/Users/jerryplain/projects/vllm-cookbook/src/vllm_cookbook/tp.py)
- [examples/04_tp_check_heads.py](/Users/jerryplain/projects/vllm-cookbook/examples/04_tp_check_heads.py)

这里做了一件很对的事：在引擎启动前 fail-fast。

如果 `num_attention_heads % tp != 0`，vLLM 根本起不来。越早报错越省时间。

## 4. 你应该怎么安装和准备环境

### 4.1 最低思路

你至少需要：

- Python 3.9+
- NVIDIA GPU
- 驱动、CUDA、PyTorch 版本相互兼容
- 可安装 vLLM 的环境

仓库本身的最小安装方式是：

```bash
pip install -e .
```

参考：

- [docs/00_install.md](/Users/jerryplain/projects/vllm-cookbook/docs/00_install.md)
- [docs/01_quickstart.md](/Users/jerryplain/projects/vllm-cookbook/docs/01_quickstart.md)

### 4.2 最容易踩坑的是“版本兼容”，不是代码

最常见的环境问题不是你的 Python 脚本写错了，而是：

- PyTorch 编译 CUDA 版本和系统驱动不匹配
- vLLM 版本和 torch / CUDA 组合不兼容
- 某些后端库缺失或 ABI 不一致

实操建议：

1. 先确认服务器驱动和 CUDA 大版本
2. 选与之兼容的 PyTorch
3. 再安装对应能工作的 vLLM
4. 先跑最小样例，再跑复杂链路

不要一上来就跑多卡 + LoRA + 长上下文 + 大评测。

### 4.3 仓库里实际依赖的环境变量

常见变量包括：

- `CUDA_VISIBLE_DEVICES`
- `VLLM_DOWNLOAD_DIR`
- `MODEL`
- `TP`
- `LORA_PATH`

其中最重要的是：

- `CUDA_VISIBLE_DEVICES` 决定 vLLM 实际能看到哪些卡
- `TP` 决定 tensor parallel 切分数
- `VLLM_DOWNLOAD_DIR` 决定模型缓存位置

### 4.4 关于缓存目录

仓库默认多处使用：

```bash
/assets/hub
```

这在多用户服务器上是合理的，因为：

- 模型不用每个人重复下载
- 缓存位置统一

但你必须确认：

- 目录存在
- 有读权限
- 最好也有写权限

否则会出现：

- 模型拉取失败
- tokenizer / config 能读，weights 不能写
- 不同脚本行为不一致

## 5. 最小可运行心智闭环

如果你要真正搞懂 vLLM，不要先读最复杂脚本，而是按下面顺序跑。

### 5.1 第一步：最小文本生成

看：

- [examples/00_minimal_generate.py](/Users/jerryplain/projects/vllm-cookbook/examples/00_minimal_generate.py)

这一步你只需要理解五件事：

1. `LLM(...)` 是引擎入口
2. `SamplingParams(...)` 控制生成策略
3. `generate(...)` 返回 `RequestOutput` 列表，不是纯文本
4. `outputs[0].outputs[0].text` 才是生成文本
5. 很多“稳定性参数”是在引擎初始化时给的，不是在生成时给的

### 5.2 第二步：chat template

看：

- [examples/01_chat_template.py](/Users/jerryplain/projects/vllm-cookbook/examples/01_chat_template.py)

这一步要理解：

- 聊天模型不是把用户文本直接塞进去就行
- 你最好使用模型自己的 tokenizer/chat template 来构造 prompt
- 否则角色标记、assistant 前缀、特殊 token 可能不对

这是很多“模型明明能跑，但回答怪异”的根因。

### 5.3 第三步：token ids 接口

看：

- [examples/02_token_ids_api.py](/Users/jerryplain/projects/vllm-cookbook/examples/02_token_ids_api.py)

这个接口很重要，因为它能解决“输入到底被怎样 tokenize”的不确定性。

在评测、离线缓存、严格复现实验时，直接传 `prompt_token_ids` 很有价值。

### 5.4 第四步：LoRA 注入

看：

- [examples/03_lora_request.py](/Users/jerryplain/projects/vllm-cookbook/examples/03_lora_request.py)

这一步要理解：

- LoRA 是运行时注入，不是自动 merge
- 引擎初始化时必须 `enable_lora=True`
- 真正使用适配器是在 `generate()` 时给 `lora_request`

### 5.5 第五步：批量评测

看：

- [response_generation.py](/Users/jerryplain/projects/vllm-cookbook/response_generation.py)
- [evaluators/Qwen3PrivacyGuard_vllm.py](/Users/jerryplain/projects/vllm-cookbook/evaluators/Qwen3PrivacyGuard_vllm.py)

这一步是你从“能跑”走向“工程化”的关键。

## 6. vLLM 的核心 API，在这个仓库里分别怎么用

### 6.1 `LLM(...)`

这是引擎初始化入口。

仓库里常见配置包括：

- `model`
- `download_dir`
- `tensor_parallel_size`
- `pipeline_parallel_size=1`
- `gpu_memory_utilization`
- `trust_remote_code=True`
- `enable_chunked_prefill=True`
- `enforce_eager=True`
- `max_model_len`
- `disable_custom_all_reduce=True`
- `seed`
- `enable_lora`

这些参数不是随便写的。

### 6.2 `SamplingParams(...)`

这是“每次生成请求”的采样配置。

仓库里的用法分两类：

确定性生成场景，例如：

```python
SamplingParams(temperature=0.0, max_tokens=256)
```

Judge 生成场景，例如：

```python
SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=self.max_new_tokens,
    n=self.n,
)
```

要理解一点：

- 引擎参数控制“怎么运行”
- sampling 参数控制“生成什么样”

两者不是一层东西。

### 6.3 `generate(...)`

仓库里有三种输入方式：

1. 直接传 prompt 字符串列表
2. 传 `prompt_token_ids`
3. 传 `lora_request`

你要特别注意：

- vLLM 返回的是结构化对象，不是纯字符串
- 代码里要自己提取 `out.outputs[0].text`

这是很多从 `transformers` 切过来的人最先踩的坑。

## 7. 参数怎么理解，不要死记

### 7.1 `tensor_parallel_size`

这是张量并行切分数。

直观理解：

- 模型按张量维度切到多张卡上
- 让单卡放不下的模型能跑
- 或让多卡共同承担推理

但不是“卡越多越好”。

它会带来：

- 通信开销
- 配置复杂度
- heads 整除约束

最关键约束是：

```text
num_attention_heads % tensor_parallel_size == 0
```

仓库在 [src/vllm_cookbook/tp.py](/Users/jerryplain/projects/vllm-cookbook/src/vllm_cookbook/tp.py) 里专门做了检查。

### 7.2 `pipeline_parallel_size`

这个仓库基本固定为 `1`。

这意味着当前仓库的主要多卡策略是 TP，不是 PP。

原因很简单：

- TP 是当前场景最直接的需求
- PP 会让部署与调试复杂度进一步上升
- cookbook 优先展示最实用、最稳定的一条路线

如果你现在还没完全理解 TP，不要急着上 PP。

### 7.3 `gpu_memory_utilization`

这表示 vLLM 允许自己使用 GPU 显存的大致比例。

理解方式：

- 值大：更激进，更可能提高吞吐
- 值小：更保守，更不容易 OOM

仓库多处使用 `0.90`，属于偏积极但不极端的默认值。

如果你有 OOM，不要第一反应只看 batch size，也要看这个值。

### 7.4 `max_model_len`

这是最常被低估的参数。

它不是“只是允许更长输入”，而是会直接影响：

- KV Cache 预算
- 可承载并发
- OOM 风险

仓库里不同场景用得不一样：

- `response_generation.py` 里是 `8192`
- `Qwen3PrivacyGuard_vllm.py` 里是 `32768`

为什么 judge 设得更大？

因为评测 prompt 会包含：

- system prompt
- 用户问题
- 目标模型回答

而且回答可能很长。

但你要知道，设大不代表“免费”。

### 7.5 `enable_chunked_prefill`

这是长上下文和大 prompt 场景里很有价值的稳定性参数。

可以理解为：prefill 阶段不要一次把所有长输入硬顶进去，而是更分块、更可控。

它常见收益是：

- 更稳
- 长上下文更容易跑通
- 某些场景吞吐更好

### 7.6 `enforce_eager`

它通常意味着更偏调试友好、兼容性友好。

代价是：

- 某些环境下性能未必最优

但在 cookbook、实验验证、复杂组合环境里，先稳往往更重要。

### 7.7 `disable_custom_all_reduce`

这个参数的本质不是“性能开关”，而是“兼容性 / 稳定性开关”。

在部分机器、驱动、通信环境组合上，关闭 custom all-reduce 反而更稳。

如果你遇到：

- 多卡初始化异常
- 通信相关奇怪报错
- 某些环境下无规律 hang 住

这个参数值得优先尝试。

## 8. LoRA 在 vLLM 里到底怎么工作

### 8.1 仓库采用的是运行时注入路线

文档和代码都很明确：

- [docs/03_lora_serving.md](/Users/jerryplain/projects/vllm-cookbook/docs/03_lora_serving.md)
- [docs/zh/02_lora注入与服务.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/02_lora注入与服务.md)
- [examples/03_lora_request.py](/Users/jerryplain/projects/vllm-cookbook/examples/03_lora_request.py)

核心方式：

```python
llm = LLM(..., enable_lora=True)
lora_req = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path=...)
outputs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_req)
```

### 8.2 为什么这种方式适合这个仓库

因为这个仓库的流程是：

1. 训练 LoRA adapter
2. 保留基座模型不动
3. 运行时切换是否挂 adapter
4. 继续做 response generation / evaluation

这样适合：

- 实验比较
- 一个 base model 对多个 adapter
- 不想每次 merge 一套完整模型目录

### 8.3 LoRA 常见坑

#### 坑 1：路径对了，但还是加载失败

原因通常是：

- 指向的不是 adapter 根目录
- checkpoint 不完整
- 目录权限不足

做法：

1. 先确认目录结构
2. 先不带 LoRA 跑通 base model
3. 再 `TP=1` 带 LoRA 跑最小闭环

#### 坑 2：LoRA 和基座模型不匹配

例如：

- 基座模型版本不同
- 训练时用的 tokenizer / architecture 与推理时不一致

这类问题不会总是报得很清楚，但输出质量会明显异常，甚至直接加载失败。

#### 坑 3：忘了在引擎初始化时开 `enable_lora=True`

只传 `LoRARequest` 不够。
引擎本身必须允许 LoRA。

#### 坑 4：把 LoRA 问题误判成 vLLM 问题

正确顺序应该是：

1. 先 base model 跑通
2. 再 LoRA + TP=1
3. 再 LoRA + 多卡

否则你会把：

- 权重兼容问题
- 显存问题
- TP 配置问题

混在一起看。

## 9. 这个仓库里的 judge 推理，为什么很值得学

### 9.1 它展示了真实业务里 vLLM 的典型用法

文件：

- [evaluators/Qwen3PrivacyGuard_vllm.py](/Users/jerryplain/projects/vllm-cookbook/evaluators/Qwen3PrivacyGuard_vllm.py)

它不是简单“给一句话，回一句话”，而是：

- 对 `(question, response)` 对做评估
- 先做 prompt 组装
- 控制输入 token 预算
- 再做批量生成
- 再解析结构化标签

这比最小 demo 更接近实际工程。

### 9.2 它展示了一个非常重要的实践：token budget 管理

`format_prompt()` 里做了这些事：

1. 先估算 system prompt 和模板开销
2. 预留 safety margin
3. 剩余 budget 在 question / response 之间分配
4. question 保尾部
5. response 保头部

这不是随便写的，而是非常实战。

因为评测时真正有价值的信息常常是：

- question 的最后意图
- response 的前几段行为

如果你无脑截断，很可能把最有判断价值的部分截掉。

### 9.3 它也展示了一个工程事实：vLLM 输入不一定非得是文本

这里先 tokenizer，再传：

```python
inputs = [{"prompt_token_ids": ids} for ids in tokenized["input_ids"]]
```

这样做的好处是：

- tokenizer 行为可控
- 输入更可复现
- 更适合复杂 prompt 构建链路

## 10. 仓库里的部署思路，本质上有哪几种

### 10.1 Python 内嵌引擎

这是仓库的主路线。

即：

- Python 脚本里直接 `from vllm import LLM`
- 初始化引擎
- 调 `generate()`

适合：

- 离线批处理
- 评测
- 研究实验
- 内部 pipeline

优点：

- 简单直接
- 容易把数据处理逻辑和推理逻辑放一起
- 更容易调试

缺点：

- 不天然是服务接口
- 生命周期管理要自己做

### 10.2 运行时 LoRA 服务

这个仓库虽然没有单独写 OpenAI-compatible server 的完整部署脚本，但 LoRA 注入的思想已经很清楚：

- 基座模型常驻
- 按请求附加 adapter

这类路线适合多 adapter 服务。

### 10.3 OpenAI-compatible server

vLLM 本身也支持服务化部署，但当前仓库核心代码不是围绕这一模式写的。

所以你要分清：

- 仓库重点：嵌入式 Python 推理
- vLLM 能力边界：也可以独立起服务

如果你后续要走 API 服务，可以把本文中的参数理解迁移过去，但不要误以为仓库已经把整套 server 运维层做好了。

## 11. 如何思考“参数该怎么调”

### 11.1 不要一上来就调吞吐，先调通闭环

推荐顺序：

1. 单卡
2. TP=1
3. base model
4. 短 prompt
5. 小 `max_model_len`
6. 最小 demo 跑通

然后再逐步加：

1. 长 prompt
2. 批量请求
3. LoRA
4. 多卡 TP
5. 更高 `gpu_memory_utilization`

### 11.2 每次只动一个变量

这是仓库中文档里反复强调但很多人做不到的事。

如果你同时改：

- TP
- LoRA
- `max_model_len`
- prompt 模板
- attention backend

你根本不知道谁导致了失败。

### 11.3 从“是否能跑”到“是否稳定”要分开看

一个配置能跑一次，不代表它能稳定跑。

稳定性要看：

- 是否频繁 OOM
- 是否偶发 hang
- 是否多轮运行后资源释放正常
- 不同数据长度下是否表现一致

### 11.4 从“显存”角度而不是“batch size”角度思考

很多人只会问：

- “batch 开多大”

但对 vLLM 更好的问题是：

- 模型权重占多少
- KV Cache 占多少
- `max_model_len` 设多大
- 并发请求量多大
- 输出上限多大
- TP 后每张卡实际负担多少

这才是接近根因的思考方式。

## 12. 常见坑，总结版

### 12.1 OOM

表现：

- 启动时报 OOM
- 生成时报 OOM
- 长输入时更容易炸

常见原因：

- `max_model_len` 太大
- `gpu_memory_utilization` 太激进
- 请求过长
- 并发太高
- LoRA / 多卡叠加后内存边界更紧

解决顺序：

1. 降低 `max_model_len`
2. 降低 `gpu_memory_utilization`
3. 缩短 prompt / 减少 `max_tokens`
4. 先去掉 LoRA
5. 退回 TP=1 验证

### 12.2 TP 非法

表现：

- 引擎启动失败
- attention head sharding 相关报错

原因：

- `num_attention_heads % tp != 0`

解决：

1. 用 [src/vllm_cookbook/tp.py](/Users/jerryplain/projects/vllm-cookbook/src/vllm_cookbook/tp.py) 先查
2. 用 [examples/04_tp_check_heads.py](/Users/jerryplain/projects/vllm-cookbook/examples/04_tp_check_heads.py) 先跑
3. 不要拍脑袋设置 TP

### 12.3 tokenizer / chat template 不一致

表现：

- 模型回答风格异常
- 明明是 instruct/chat model，输出却像 continuation
- 特殊 token 或角色边界混乱

原因：

- 没用模型原生 template
- 手搓 prompt 但格式不匹配

解决：

优先走：

- `tokenizer.apply_chat_template(...)`

参考：

- [examples/01_chat_template.py](/Users/jerryplain/projects/vllm-cookbook/examples/01_chat_template.py)

### 12.4 把 vLLM 输出结构理解错

表现：

- 代码把返回值当字符串
- 取不到文本

原因：

- `generate()` 返回的是对象列表

解决：

用：

```python
out.outputs[0].text
```

`Qwen3PrivacyGuard_vllm.py` 里已经明确修正了这一点。

### 12.5 评测长输入被截断，但你自己没意识到

表现：

- judge 结果不稳定
- 某些长样本判得很怪

原因：

- 总输入超了 token budget
- 但你没检查 system prompt 开销
- 或截断策略不合理

仓库里的做法是先估模板开销，再分配 question / response 预算，这是一种正确思路。

### 12.6 多卡通信问题

表现：

- 初始化 hang
- all-reduce 相关异常
- 某些机器能跑，某些不能

处理思路：

1. 先单卡验证
2. 再 TP
3. 保留 `disable_custom_all_reduce=True`
4. 检查 `CUDA_VISIBLE_DEVICES` 和 TP 是否一致

### 12.7 资源没释放，后续任务异常

表现：

- 第一轮能跑，第二轮异常
- 显存像没清干净

仓库里在多个地方显式做了清理：

```python
destroy_model_parallel()
destroy_distributed_environment()
gc.collect()
torch.cuda.empty_cache()
```

这不是多余代码，尤其适合一个 Python 进程里连续起多个 engine 的情况。

### 12.8 attention backend / worker 启动方式导致的环境差异

`response_generation.py` 里设置了：

```python
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "4"
```

这说明当前仓库作者已经在用“环境固定”来换稳定性。

你要理解这背后的意思：

- 某些 attention backend 在特定机器上更稳或更快
- 多进程启动方式会影响兼容性
- 线程数会影响 CPU 侧开销和系统扰动

如果你换环境后行为变了，不要只盯 Python 代码，环境变量也要纳入排查。

### 12.9 evaluator 没有通用 HF 回退

这是仓库里的一个重要事实，不是 bug。

[evaluators/Qwen3PrivacyGuard_vllm.py](/Users/jerryplain/projects/vllm-cookbook/evaluators/Qwen3PrivacyGuard_vllm.py) 默认就是 vLLM 路线，没有像 `judge_vllm_inference/JailJudge_guard_vllm.py` 那样自动回退到 HF。

所以如果：

- 你没有 GPU
- 你没装 vLLM
- 你的 vLLM 环境没配好

这条 evaluator 链路不会自然降级。

### 12.10 数据过滤后索引对齐问题

`response_generation.py` 里有一段逻辑：

1. 先把 dataset map 成 prompt
2. 再 filter 掉空 prompt
3. 最后按 `enumerate(outputs)` 去取 `raw_data[i]`

如果真的发生了过滤，`formatted` 和 `raw_data` 的原始索引可能不再严格一致。

这意味着：

- 一旦存在被过滤样本，结果写回时可能错位

在当前数据契约下如果 `question` 普遍非空，这个问题未必触发；但你扩展数据集时要记住它。

这是非常典型的工程坑：

- vLLM 没错
- 数据管道索引错了
- 最后你以为是模型输出乱了

## 13. 你应该如何部署

### 13.1 离线批量生成部署

最适合这个仓库现状的方式是：

1. 固定模型与环境
2. 固定缓存目录
3. 先跑生成
4. 再跑评测
5. 输出结果 JSON

这对应：

- [response_generation.py](/Users/jerryplain/projects/vllm-cookbook/response_generation.py)
- [response_evaluation.py](/Users/jerryplain/projects/vllm-cookbook/response_evaluation.py)

适合：

- benchmark
- 研究
- 大规模离线实验

### 13.2 多卡部署思路

如果你要多卡，推荐路径：

1. 同模型先单卡跑通
2. 确认 attention heads 和 TP 可整除
3. 再让 `tensor_parallel_size` 对应到实际 GPU 数
4. 最后观察吞吐是否真的变好

因为多卡不总是更快。

如果模型不够大、请求不够多，通信开销会抵消收益。

### 13.3 多 LoRA 部署思路

如果你的目标是：

- 一个基础模型
- 多个垂直 adapter
- 动态切换

那就坚持运行时注入，而不是反复 merge。

更合理的思路是：

- base model 常驻
- adapter 路径规范化管理
- 请求级选择 LoRA

### 13.4 Judge 部署思路

Judge 模型和目标模型最好分开思考：

- 目标模型负责生成回答
- judge 模型负责评估回答

不要把它们混在一个 engine 生命周期里。

仓库当前也是这样分的：

- 生成一条链
- 评测一条链

这是合理的，便于单独扩缩容和调试。

## 14. 一个靠谱的排查顺序

当 vLLM 出问题时，建议按下面顺序定位。

### 14.1 先分清是哪一层的问题

四层最常见：

1. 环境层：驱动 / CUDA / torch / vLLM
2. 模型层：模型路径、tokenizer、LoRA 兼容性
3. 并行层：TP、可见 GPU、通信
4. 业务层：prompt 构造、数据过滤、结果解析

### 14.2 最小化复现

永远先缩小到：

- 单卡
- TP=1
- base model
- 一个 prompt
- 短 `max_tokens`

这一步都跑不通，就不要继续猜多卡问题。

### 14.3 再逐层加回复杂度

建议顺序：

1. 一个 prompt
2. 一批 prompt
3. token ids 输入
4. LoRA
5. 多卡 TP
6. 长上下文
7. 大规模评测

### 14.4 记录你改了什么

每次实验最好固定并记录：

- 模型
- TP
- `max_model_len`
- `gpu_memory_utilization`
- backend 环境变量
- 是否 LoRA
- 输入长度分布

否则你后面根本无法复盘“为什么这次能跑，上次不行”。

## 15. 推荐学习路径

如果你的目标是“完全搞懂 vLLM 在这个仓库里的用法”，建议按这个顺序读：

1. [docs/zh/00_仓库导读.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/00_仓库导读.md)
2. [docs/zh/01_vllm核心原理.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/01_vllm核心原理.md)
3. [examples/00_minimal_generate.py](/Users/jerryplain/projects/vllm-cookbook/examples/00_minimal_generate.py)
4. [examples/01_chat_template.py](/Users/jerryplain/projects/vllm-cookbook/examples/01_chat_template.py)
5. [examples/02_token_ids_api.py](/Users/jerryplain/projects/vllm-cookbook/examples/02_token_ids_api.py)
6. [examples/03_lora_request.py](/Users/jerryplain/projects/vllm-cookbook/examples/03_lora_request.py)
7. [response_generation.py](/Users/jerryplain/projects/vllm-cookbook/response_generation.py)
8. [evaluators/Qwen3PrivacyGuard_vllm.py](/Users/jerryplain/projects/vllm-cookbook/evaluators/Qwen3PrivacyGuard_vllm.py)
9. [docs/04_common_pitfalls.md](/Users/jerryplain/projects/vllm-cookbook/docs/04_common_pitfalls.md)
10. [docs/zh/03_参数与调优清单.md](/Users/jerryplain/projects/vllm-cookbook/docs/zh/03_参数与调优清单.md)

## 16. 最后给你的结论

如果只用一句话概括 vLLM：

它不是“让模型能生成”的库，而是“让模型在真实推理场景中更高效、更可部署、更可调度地生成”的引擎。

如果只用一句话概括这个仓库里的 vLLM：

它把 vLLM 用在了三件最有实战价值的事上：

- 离线批量生成
- 运行时 LoRA 注入
- judge 模型高吞吐评测

如果只用一句话概括最重要的经验：

不要把 vLLM 当成黑盒；从环境、显存、TP、tokenizer、prompt 构造、结果解析六个层面同时理解它，你才真的会用。
