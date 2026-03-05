from vllm_cookbook.tp import assert_tp_valid


def test_assert_tp_valid_divisible() -> None:
    # Lightweight unit-style sanity check with a mocked divisible case.
    class Dummy:
        num_attention_heads = 32

    def fake_get_num_attention_heads(_: str) -> int:
        return Dummy.num_attention_heads

    import vllm_cookbook.tp as tp

    original = tp.get_num_attention_heads
    tp.get_num_attention_heads = fake_get_num_attention_heads
    try:
        assert_tp_valid("dummy", 8)
    finally:
        tp.get_num_attention_heads = original
