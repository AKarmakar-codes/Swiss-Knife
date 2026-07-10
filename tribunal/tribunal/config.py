"""Default settings. Edit values here, or override the common ones from the
command line (see `python -m tribunal.run_eval --help`)."""

CONFIG = {
    # Judge model, served via vLLM (see serve_judge.py).
    "judge_model": "Qwen/Qwen2.5-32B-Instruct",
    "vllm_url": "http://localhost:8000/v1",
    "vllm_api_key": "EMPTY",

    # Input: a folder with one .jsonl file per model. The filename (without
    # .jsonl) is used as the model's label. One file = one model.
    # Each line: {"id": ..., "prompt": "...", "response": "..."}
    "input_path": "inputs/",

    # Where results are written.
    "output_folder": "eval_results/",

    # Number of records to score per file (None = all).
    "sample_size": None,

    # Run the judge-free Detoxify toxicity check alongside the judge.
    "use_detoxify": True,

    # Validity gate: skip outputs that are empty, too short, or degenerate
    # (these get reported as invalid rather than judged).
    "min_output_length": 10,
    "role_tokens": ["assistant", "assistant:", "human:", "human"],

    # Judge generation.
    "judge_max_tokens": 1024,
    "judge_timeout": 180,

    # Flush partial results to disk every N records.
    "save_every": 5,
}

# The six judged metrics, grouped.
QUALITY_METRICS = ["response_quality", "relevance", "helpfulness"]
SAFETY_METRICS = ["toxicity", "harmfulness", "refusal"]
ALL_METRICS = QUALITY_METRICS + SAFETY_METRICS
