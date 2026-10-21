# CaKE+ reproduction

Code and configurations for HalluEditBench, MQuAKE-T and WikiBigEdit, based on
official [zjunlp/CaKE](https://github.com/zjunlp/CaKE) main commit
`df528e13fffbd2117e6041c129285e1e146933d2`. The original entrypoints are unchanged.
Datasets, general-task examples, weights, logs and results are supplied or
generated locally; this contribution contains only code, configuration and docs.

## Setup and inputs

The packaging environment uses Python 3.10 and these packages:

```bash
pip install torch==2.8.0 transformers==4.57.1 peft==0.17.1 \
  accelerate==1.11.0 datasets==3.6.0 PyYAML==6.0.3
```

Model weights must be available through Hugging Face or a local cache. Supply a
local model with `--set model_name=/path/to/model`.

Provide the prepared JSON datasets at the `data_file` paths in
[experiments.yaml](experiments.yaml), or override `data_file` with an external
path. JSON gzip files are also accepted when the path ends in `.gz`. Use the
same prepared data and case order as the original runs: the first 1000 records
are selected without shuffling the dataset. Training examples are shuffled.

Each record needs `case_id` and a `requested_rewrite` list. Rewrites contain
`prompt` (formatted with `subject`), `subject` and `target_new.str`; optional
`rephrase_prompt` and `learning_prompt` lists contain `question`/`answer` pairs.
Evaluation additionally requires the dataset's original fields:

- HalluEditBench: original question, object, subject, paraphrase, yes/no,
  multiple-choice, reversed-relation and locality fields, plus 2–6-hop questions.
  Preserve the model-specific prepared data, including `locality_pre_ans`.
- MQuAKE-T: `new_single_hops` with answers and aliases, `questions`, `new_answer`
  and `new_answer_alias`.
- WikiBigEdit: `update`, `rephrase`, `personas`, `ans`, `mhop`, `mhop_ans`, `loc`
  and `loc_ans`.

The general-task pool is an external JSON list, configured by
`general_task_file`. Entries support either `text`/`target` or
`instruction`/`input`/`output`. The first 16 supported examples form the pool;
reuse the original pool to compare with historical runs. A missing pool causes
an error when `general_per_batch` is positive. Set `general_per_batch=0` to
explicitly disable mixing.

## Run

```bash
python run_cake_plus.py --list
python run_cake_plus.py llama_batch_1000 --dry-run
python run_cake_plus.py llama_batch_1000 --gpu 0
python run_cake_plus.py qwen_current_1000 --gpu 1
python run_cake_plus.py llama_mquake_t_1000 --gpu 0
python run_cake_plus.py qwen_mquake_t_1000 --gpu 1
python run_cake_plus.py llama_wikibigedit_1000 --gpu 0
python run_cake_plus.py qwen_wikibigedit_1000 --gpu 1
```

Each invocation uses one GPU. Run sequentially on each GPU; two GPUs may run
one experiment each. Change external paths without editing the YAML:

```bash
python run_cake_plus.py llama_mquake_t_1000 --gpu 0 \
  --set data_file=/path/to/MQuAKE-T-cake.json \
  --set general_task_file=/path/to/general_task_pool.json
```

## Configurations

| YAML entry | Dataset | Model | LR | Epochs | Edit training batch | KL reference | Edit KL | General KL | General samples per batch |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `llama_batch_1000` | HalluEditBench | Llama-3-8B-Instruct | 5e-5 | 30 | 8 | batch | 0.001 | 0.01 | 1 |
| `qwen_current_1000` | HalluEditBench | Qwen2.5-7B-Instruct | 2e-5 | 25 | 8 | current | 0.1 | 0.1 | 2 |
| `llama_mquake_t_1000` | MQuAKE-T | Llama-3-8B-Instruct | 3e-5 | 30 | 8 | current | 0.1 | 0.1 | 2 |
| `qwen_mquake_t_1000` | MQuAKE-T | Qwen2.5-7B-Instruct | 5e-5 | 30 | 8 | current | 0.1 | 0.1 | 2 |
| `llama_wikibigedit_1000` | WikiBigEdit | Llama-3-8B-Instruct | 5e-5 | 30 | 8 | current | 0.1 | 0.1 | 2 |
| `qwen_wikibigedit_1000` | WikiBigEdit | Qwen2.5-7B-Instruct | 5e-5 | 30 | 8 | current | 0.1 | 0.1 | 2 |

HalluEditBench entries retain the two selected full-run configurations.
MQuAKE-T and WikiBigEdit entries transcribe the original completed runs from
`run_llama3_8b_mquake_t.sh`, `run_qwen2_5_7b_mquake_t.sh`,
`run_llama3_8b_wikibigedit.sh` and `run_qwen2_5_7b_wikibigedit.sh` (September 28,
2026). The later no-general-task comparisons used different settings and are
not the source of these four entries. All entries set seed 42; the original
September shell scripts did not explicitly seed global RNGs, so their exact
training order cannot be recovered from the scripts alone.

All configurations use 1000 edits, groups of 100 edits, LoRA rank 8, alpha 32,
training dropout 0.05, apply-stage dropout 0.1 and a 16-example general pool.
Training and application both use the full test3 target set: up/down/gate
projections and q/v projections in layers 0, 1, 27, 2, 3, 26 and 24. `epochs`
counts passes over expanded training examples; there is no fixed step limit.

`batch` loads a separate frozen reference and refreshes it before each group of
100 edits. `current` retains the original implementation's in-place reference
behavior, with no separately frozen model. General examples contribute KL only,
without edit cross-entropy. KL uses KL(reference || edited) with `batchmean`
reduction; a null `kl_lambda_general` falls back to `kl_lambda`.

Evaluation retains the recorded prompts, generation lengths and substring
matching. MQuAKE-T reports hop-wise efficacy and multi-hop accuracy; WikiBigEdit
reports update, rephrase, persona, multi-hop and locality accuracy. Historical
metric keys, including `generalizaton`, are preserved. HalluEditBench retains
the 11-response robustness evaluation and model-specific locality targets.

## Outputs and recovery

The launcher writes resolved `config.yaml`, `run.log`, `metrics.json`,
`summary.json`, a final model (when `save_model=true`) and a checkpoint after
each completed group of edits under `output/cake_plus/<experiment>/`.
These generated files are ignored by Git. Use `--output` for another directory.

Rerun the same command and output directory to resume the last completed edit
group, including RNG state. An interrupted group restarts; final evaluation
restarts in full. A different configuration or edit dataset is rejected by the
checkpoint signature. Completed output is protected against overwrite.

The historical experiments predate this unified YAML launcher. Packaging these
settings does not constitute a fresh full GPU rerun.
