"""Run one named CaKE+ experiment from YAML; each invocation uses one GPU."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import gzip
import os
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments/cake_plus/experiments.yaml"


def load_experiment(path, name, overrides=()):
    with open(path) as handle:
        catalog = yaml.safe_load(handle)
    experiments = catalog["experiments"]
    config = dict(catalog["defaults"])
    values = experiments[name]
    unknown = values.keys() - config.keys()
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config.update(values)
    for override in overrides:
        key, value = override.split("=", 1)
        if key not in config:
            raise ValueError(f"Unknown config key: {key}")
        config[key] = yaml.safe_load(value)
    for key in ("num_edits", "edit_freq", "batch_size", "epochs", "max_memory_gb", "general_task_num"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("general_per_batch", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("lr", "kl_lambda", "kl_lambda_general", "train_lora_dropout", "lora_dropout"):
        value = config[key]
        if key == "kl_lambda_general" and value is None:
            continue
        if type(value) not in (int, float) or value < 0:
            raise ValueError(f"{key} must be a nonnegative number")
    if config["lr"] == 0 or any(config[k] >= 1 for k in ("train_lora_dropout", "lora_dropout")):
        raise ValueError("lr must be positive; dropout must be less than one")
    if config["kl_ref"] not in ("current", "batch"):
        raise ValueError("kl_ref must be current or batch")
    if type(config["save_model"]) is not bool:
        raise ValueError("save_model must be a boolean")
    return config


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile

    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


def run(config, output):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from CaKE_method.cake_plus import cake_plus_continual_edit
    from CaKE_method.halluedit_eval import calculate_averages

    if torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU using --gpu or CUDA_VISIBLE_DEVICES")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    dataset = ROOT / "datasets" / f'{config["datatype"]}-cake.json'
    if dataset.exists():
        handle = dataset.open()
    else:
        handle = gzip.open(str(dataset) + ".gz", "rt")
    with handle:
        data = json.load(handle)
    if config["num_edits"] > len(data):
        raise ValueError("num_edits exceeds the available dataset")
    if config["general_task_file"] and not (ROOT / config["general_task_file"]).is_file():
        raise FileNotFoundError(config["general_task_file"])
    hparams = SimpleNamespace(**config)
    hparams.device = 0
    hparams.rank = 8
    hparams.lora_alpha = 32
    hparams.num_train_epochs = config["epochs"]
    hparams.per_device_train_batch_size = config["batch_size"]
    hparams.learning_rate = config["lr"]
    hparams.checkpoint_dir = str(output / "checkpoint")
    hparams.max_memory = {0: f'{config["max_memory_gb"]}GiB', "cpu": "120GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name, device_map="auto", max_memory=hparams.max_memory,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model, metrics = cake_plus_continual_edit(
        model, tokenizer, data[:config["num_edits"]], hparams,
        config["edit_freq"], config["datatype"], test_generation=False,
    )
    for name, value in (("metrics", metrics), ("summary", calculate_averages(metrics))):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    if config["save_model"]:
        model.save_pretrained(output / "model")
        tokenizer.save_pretrained(output / "model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--gpu", help="One physical GPU index")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved YAML without loading a model")
    args = parser.parse_args()
    if args.list:
        with args.config.open() as handle:
            print("\n".join(yaml.safe_load(handle)["experiments"]))
        return
    if not args.experiment:
        parser.error("provide an experiment name or --list")
    try:
        config = load_experiment(args.config, args.experiment, args.set)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if args.gpu is not None:
        if not args.gpu.isdigit():
            parser.error("--gpu must be one GPU index")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    output = (args.output or ROOT / "output/cake_plus" / args.experiment).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        parser.error("completed output already exists; select a new --output")
    resolved = output / "config.yaml"
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != config:
        parser.error("output belongs to a different configuration; select a new --output")
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    os.chdir(ROOT)
    with (output / "run.log").open("a") as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print(yaml.safe_dump(config, sort_keys=False), flush=True)
            try:
                run(config, output)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
