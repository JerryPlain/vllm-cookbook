"""Build a chat-style prompt with the tokenizer template and run vLLM."""

import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # Use the model's native chat template so role formatting stays correct.
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Give me three tips to debug CUDA OOM in vLLM."},
    ]

    # Convert structured chat messages into one model-ready text prompt.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # Add assistant prefix so generation starts in assistant role.
        add_generation_prompt=True,
    )

    llm = LLM(model=model, trust_remote_code=True)
    out = llm.generate([prompt], SamplingParams(temperature=0.2, max_tokens=200))
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
