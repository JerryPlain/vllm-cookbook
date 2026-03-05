"""Standalone TP validator script.

Run this before expensive jobs to verify TP is compatible with the model.
"""

import os

from vllm_cookbook.tp import assert_tp_valid


def main() -> None:
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tp = int(os.getenv("TP", "4"))

    assert_tp_valid(model, tp)
    print("OK: TP is valid.")


if __name__ == "__main__":
    main()
