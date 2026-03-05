import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Give me three tips to debug CUDA OOM in vLLM."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=model, trust_remote_code=True)
    out = llm.generate([prompt], SamplingParams(temperature=0.2, max_tokens=200))
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
