# CaKE+ HalluEditBench reproduction

This directory contains the two selected 1000-edit settings and their original
results. The implementation is an addition to official
[zjunlp/CaKE](https://github.com/zjunlp/CaKE), based on main commit
`df528e13fffbd2117e6041c129285e1e146933d2`.

## Run

The recorded runs used Python 3.10 with these packages:

```bash
pip install torch==2.8.0 transformers==4.57.1 peft==0.17.1 \
  accelerate==1.11.0 datasets==3.6.0 PyYAML==6.0.3
python run_cake_plus.py --list
python run_cake_plus.py llama_batch_1000 --dry-run
python run_cake_plus.py llama_batch_1000 --gpu 0
python run_cake_plus.py qwen_current_1000 --gpu 1
```

Model weights must be available through Hugging Face or a local cache. A local
model path can be supplied with `--set model_name=/path/to/model`. Each command
uses one GPU. Run commands sequentially on a given GPU.

The launcher writes `config.yaml`, `run.log`, `metrics.json`, `summary.json`,
a final model and a checkpoint after each completed edit batch to
`output/cake_plus/<experiment>/`. Rerunning the same configuration/output resumes
from the last completed batch; final evaluation restarts in full. Use `--output`
for a different output folder. Completed results are protected against overwrite.

## Final settings

| Setting | Llama-3-8B-Instruct | Qwen2.5-7B-Instruct |
| --- | --- | --- |
| YAML entry | `llama_batch_1000` | `qwen_current_1000` |
| Learning rate / epochs / training batch | 5e-5 / 30 / 8 | 2e-5 / 25 / 8 |
| KL reference | `batch` | `current` |
| Edit KL / general KL | 0.001 / 0.01 | 0.1 / 0.1 |
| General sample pool / samples per training batch | 16 / 1 | 16 / 2 |
| Edit frequency / seed | 100 / 42 | 100 / 42 |

Both settings train and apply the full test3 target set: up/down/gate projections
and q/v projections in layers 0, 1, 27, 2, 3, 26 and 24. LoRA uses rank 8, alpha
32, training dropout 0.05 and apply-stage dropout 0.1. `epochs` counts full passes
over expanded training examples, not a fixed optimizer-step limit.

`batch` uses a separate frozen reference refreshed before each 100-edit batch.
`current` retains the recorded run's in-place reference behavior; it does not
freeze a separate model. General samples contribute KL without edit
cross-entropy. KL uses KL(reference || edited) with `batchmean` reduction.
The general task file contains the exact first 16 examples used by these runs.

## Results and logs

Values are percentages over 1000 cases. Gen. is rephrase accuracy; Por. is the
mean of 2-, 3-, 4-, 5- and 6-hop accuracy.

| Model | Source | Eff. | Gen. | Por. | Loc. |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-3-8B-Instruct | Reproduction | 69.60 | 69.50 | 19.14 | 41.70 |
| Llama-3-8B-Instruct | Paper | 75.00 | 70.50 | 19.20 | 33.30 |
| Qwen2.5-7B-Instruct | Reproduction | 52.80 | 50.80 | 13.76 | 1.30 |
| Qwen2.5-7B-Instruct | Paper | 51.80 | 47.20 | 17.88 | 2.30 |

`results/llama_batch/` and `results/qwen_current/` each contain the original
`run.log.gz`, per-case `metrics.json.gz`, and aggregate `summary.json`.
`artifacts.sha256.json` records checksums for these artifacts and the input data.
The compressed HalluEditBench datasets preserve the source dataset bytes,
including case order and pre-edit locality answers, and are read directly by
the launcher. Scoring uses the recorded prompts, substring matching and
11-response robustness evaluation.

These results predate the YAML launcher. Qwen also predates checkpoint support.
The packaged launcher has not been used for a fresh full GPU rerun.
