import os

from vllm import LLM, SamplingParams


def main() -> None:
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tp = int(os.getenv("TP", "1"))

    llm = LLM(
        model=model,
        download_dir=os.getenv("VLLM_DOWNLOAD_DIR", "/assets/hub"),
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        max_model_len=8192,
        disable_custom_all_reduce=True,
        seed=42,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=256)
    prompts = ["Explain what tensor parallelism is in one paragraph."]

    outputs = llm.generate(prompts, sampling_params=sp)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
