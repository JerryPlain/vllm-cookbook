import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    text = "Write a one-line summary of tensor parallelism."
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    llm = LLM(model=model, trust_remote_code=True)
    out = llm.generate(prompt_token_ids=[token_ids], sampling_params=SamplingParams(max_tokens=80))
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
