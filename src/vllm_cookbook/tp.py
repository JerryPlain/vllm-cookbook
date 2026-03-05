"""Tensor parallel helper utilities.

The main goal is to fail fast with clear errors before vLLM engine startup.
"""

from transformers import AutoConfig


def get_num_attention_heads(model: str) -> int:
    """Load model config and return total attention head count.

    Args:
        model: Hugging Face model ID or local model path.

    Returns:
        Integer number of attention heads.

    Raises:
        ValueError: if the config does not expose `num_attention_heads`.
    """
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    num_heads = getattr(cfg, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError("num_attention_heads not found in model config.")
    return int(num_heads)


def assert_tp_valid(model: str, tp: int) -> None:
    """Validate TP divisibility rule required by attention sharding.

    Rule: `num_attention_heads % tp == 0`

    Args:
        model: Hugging Face model ID or local model path.
        tp: Tensor parallel size.

    Raises:
        ValueError: if tp <= 0 or divisibility rule is violated.
    """
    num_heads = get_num_attention_heads(model)

    if tp <= 0:
        raise ValueError(f"Invalid TP: tp must be > 0, got {tp}")

    if num_heads % tp != 0:
        raise ValueError(
            f"Invalid TP: num_attention_heads={num_heads} not divisible by tp={tp}"
        )
