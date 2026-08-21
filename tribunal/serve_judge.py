"""Launch the vLLM server that hosts the judge model (Qwen2.5-32B-Instruct, 4-bit quantized).

Run this in its own terminal and leave it running while evaluations execute:

    python tribunal/serve_judge.py

The default judge (Qwen2.5-32B-Instruct, 4-bit) requires roughly 20-24GB of VRAM.
"""

import subprocess
import sys

from tribunal.config import CONFIG

JUDGE_MODEL = CONFIG.get("judge_model", "Qwen/Qwen2.5-32B-Instruct-AWQ")

command = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", JUDGE_MODEL,
    "--dtype", "half",
    "--gpu-memory-utilization", "0.90",
    "--port", "8000",
    "--api-key", "EMPTY",
]

if "awq" in JUDGE_MODEL.lower():
    command.extend(["--quantization", "awq"])
elif "bitsandbytes" in JUDGE_MODEL.lower() or "32b" in JUDGE_MODEL.lower():
    command.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])

print(f"Starting 4-bit vLLM judge server ({JUDGE_MODEL})...")
print("Leave this running while evaluations run.")

process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
try:
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
except KeyboardInterrupt:
    print("\nStopping server.")
    process.terminate()