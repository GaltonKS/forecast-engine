"""The /query model: vLLM on a Modal GPU, scaling to zero when idle.

Untested here (needs a Modal account). Structure follows Modal's vLLM example
(modal.com/docs/examples/vllm_inference, checked Nov 2026); if `modal deploy`
rejects a parameter, compare against that page.

    pip install modal && modal setup
    modal secret create forecast-llm VLLM_API_KEY=<a long random string>
    modal deploy deploy/modal_vllm.py

Then set on BOTH forecast-api and forecast-worker in Render:
    LLM_BASE_URL = <the URL modal deploy prints>/v1
    LLM_MODEL    = forecast-llm
    LLM_API_KEY  = the same string as VLLM_API_KEY

Change the model or GPU with environment variables at deploy time, e.g.
    LLM_HF_MODEL=Qwen/Qwen2.5-14B-Instruct LLM_GPU=L40S modal deploy deploy/modal_vllm.py
and re-run scripts/eval_llm.py to compare.
"""
import os
import subprocess

import modal

MODEL = os.environ.get("LLM_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("LLM_GPU", "L4")  # 24 GB is enough for a 7B model at 8k context; L40S for 14B
SERVED_NAME = "forecast-llm"  # what LLM_MODEL must be set to
PORT = 8000
MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "LLM_HF_MODEL": MODEL})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("forecast-llm")


@app.server(
    image=vllm_image,
    gpu=GPU,
    port=PORT,
    scaledown_window=10 * MINUTES,  # idle time before scaling to zero (no GPU cost while idle)
    startup_timeout=15 * MINUTES,  # first start downloads the model weights
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("forecast-llm")],
    # Modal's proxy auth is off; vLLM's --api-key below protects the endpoint instead,
    # because the forecast worker sends the key as a standard Bearer token.
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm", "serve", os.environ["LLM_HF_MODEL"],
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0", "--port", str(PORT),
            "--api-key", os.environ["VLLM_API_KEY"],
            "--max-model-len", "8192",  # prompts are ~3k tokens; replies under 1.5k
            "--gpu-memory-utilization", "0.90",
        ]
        self.proc = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.proc.terminate()
