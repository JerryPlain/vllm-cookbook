# TP / PP

## TP validity rule
`num_attention_heads % tensor_parallel_size == 0`

If not divisible, vLLM launch fails for tensor-parallel attention sharding.

Use `examples/04_tp_check_heads.py` before starting long jobs.
