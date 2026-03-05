import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from vllm_cookbook.tp import assert_tp_valid


def main() -> None:
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tp = int(os.getenv("TP", "1"))
    lora_path = os.getenv("LORA_PATH", "")

    assert_tp_valid(model, tp)

    llm = LLM(
        model=model,
        download_dir=os.getenv("VLLM_DOWNLOAD_DIR", "/assets/hub"),
        tensor_parallel_size=tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enable_lora=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        max_model_len=8192,
        disable_custom_all_reduce=True,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=256)
    prompts = ["Tell me a compliant way to refuse sharing SSN."]

    lora_req = None
    if lora_path:
        lora_req = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path=lora_path)

    out = llm.generate(prompts, sampling_params=sp, lora_request=lora_req)
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
