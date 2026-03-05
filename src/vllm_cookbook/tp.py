from transformers import AutoConfig


def get_num_attention_heads(model: str) -> int:
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    n = getattr(cfg, "num_attention_heads", None)
    if n is None:
        raise ValueError("num_attention_heads not found in model config.")
    return int(n)


def assert_tp_valid(model: str, tp: int) -> None:
    n = get_num_attention_heads(model)
    if tp <= 0:
        raise ValueError(f"Invalid TP: tp must be > 0, got {tp}")
    if n % tp != 0:
        raise ValueError(f"Invalid TP: num_attention_heads={n} not divisible by tp={tp}")
