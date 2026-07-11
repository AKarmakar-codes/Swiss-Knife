# Tribunal

Tribunal scores LLM responses for safety and quality using a local open-weight
model as the judge, and compares any number of models side by side. You drop one
file of responses per model into a folder, and it returns per-response scores,
per-model summary tables, and an interactive HTML report. The judge runs on your
own GPU through vLLM, so there are no API keys and nothing leaves the machine.

It is meant for evaluating responses you already have, for example the outputs
of several models or several variants of one model after some change you made.

## How it works

Put one `.jsonl` file per model into `inputs/`. The filename is the model's
label: `qwen_ablated.jsonl` becomes the model `qwen_ablated` in the report.
One file is one model. If a run was split across several files, concatenate them
into a single file first.

Each line of a file is one response:

```json
{"id": 1, "prompt": "the user's request", "response": "the model's answer"}
```

The response is read from `response`, falling back to `output_steered` or
`output`, so files from other generation pipelines work without reformatting.

## Metrics

Each response is scored from 0 to 1. Three rubrics describe response quality:
`response_quality` (coherence, fluency, completeness, which also catches
repetition and degeneration), `relevance` (on the prompt's topic), and
`helpfulness` (useful for the request).

Three describe safety: `toxicity` (how offensive the language is),
`harmfulness` (whether the content could enable real-world harm, which is
separate from toxicity), and `refusal` (how much the model declined).

A judge-free Detoxify score is also reported as a deterministic toxicity
cross-check.

## Scope

Tribunal scores responses that already exist. It does not generate adversarial
prompts or run attacks, it does not check factual correctness, and it does not
audit demographic bias. It needs a GPU to host the judge.

## Requirements

A CUDA GPU. The default judge, Qwen2.5-32B-Instruct in 4-bit, needs about 24GB
of VRAM. For a smaller GPU, set a smaller judge model in `tribunal/config.py`.

```bash
pip install -r requirements.txt
```

## Usage

The judge runs as a separate server, so use two terminals. tmux is convenient
when working over SSH, since the run survives a dropped connection.

Start the judge and wait for it to finish loading the model:

```bash
python serve_judge.py
```

Drop your model files into `inputs/`, then in a second terminal:

```bash
python -m tribunal.run_eval
```

With no arguments it scores everything in `inputs/`. Two sample files are
included so it runs out of the box. Options: `--input` for a different folder,
`--output` for a different results folder, `--sample-size N` to cap records per
file, `--judge-url` if the server is not on the default port, and
`--no-detoxify`. Anything not passed falls back to `tribunal/config.py`.

## Output

Results are written to `eval_results/`:

- `report/index.html` and four linked views: the safety vs response-quality
  plane, a 3D version with refusal as the third axis, per-rubric bars, and the
  refusal diagnostic. Each is a standalone HTML file that opens offline.
- `model_summary.csv`: one row per model with mean scores and the two axes
- `summary.csv`: per model and per metric, with mean, median, std, and
  abstention counts
- `<model>_eval.csv`: per-response scores and the judge's reasoning
- `combined_results.csv`: every response across all models
- `eval_*.log`: the run log

## Reading the report

The trade-off plane places each model by two axes. Response quality is the mean
of the three quality rubrics. Safety is `1 - mean(toxicity, harmfulness)`, so
higher is safer. Refusal is kept off these axes because it has no fixed good
direction: refusing a harmful request and refusing a benign one both raise it.
Read refusal against harmfulness instead. A model whose harmfulness fell while
refusal stayed low got safer while still answering; one whose refusal jumped may
be over-refusing rather than getting safer.

## Layout

```
tribunal/
  config.py     settings and metric groupings
  data.py       reading input, validation
  judge.py      vLLM judge backend
  metrics.py    the six rubrics and the Detoxify scorer
  pipeline.py   scoring loop, aggregation
  report.py     interactive HTML report
  run_eval.py   command-line entrypoint
serve_judge.py  launches the vLLM judge server
inputs/         drop one .jsonl per model here
```

## Notes

Scoring resumes per record: results flush to disk as they are produced, and a
rerun skips responses already scored. Adding a new model file scores only that
file and rebuilds the report from all of them.

Obviously broken outputs (empty, too short, stuck in a repetition loop) are
filtered before judging and reported as invalid.

If the judge fails or declines to score a response, that row is marked as
abstained and left out of the aggregate rather than counted as zero. The
abstention counts are in `summary.csv`. This matters when scoring deliberately
weakened models, whose worst outputs are the ones a judge is most likely to
decline.

## License

MIT. See `LICENSE`.
