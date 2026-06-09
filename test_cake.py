import argparse
import os

from transformers import AutoTokenizer

from eval_utils import calculate_averages
from test_cake_helpers import (
    build_method_context,
    build_output_paths,
    get_method_config,
    load_hparams_bundle,
    load_json,
    load_model_for_method,
    maybe_build_apply_algo,
    run_edit_mode,
    save_json,
)


def build_kl_suffix(editing_method, kl_lambda):
    if editing_method not in ["CAKE_KL", "CAKE_Batch_KL"]:
        return ""
    return f"_kl_{str(kl_lambda).replace('.', '_')}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", required=True, type=str)
    parser.add_argument("--metrics_save_dir", default="./output", type=str)
    parser.add_argument("--datatype", default=None, type=str)
    parser.add_argument("--model_type", default=None, type=str)
    parser.add_argument(
        "--edit_mode",
        default="single_edit",
        choices=["single_edit", "multi_edit", "sequential_edit", "continual_edit"],
        type=str,
    )
    parser.add_argument("--edit_freq", default=10, type=int)
    parser.add_argument("--kl_lambda", default=0.05, type=float)
    parser.add_argument(
        "--save_model",
        default="true",
        type=str,
        choices=["true", "false"],
    )
    parser.add_argument("--model_save_dir", default="./saved_models", type=str)
    args = parser.parse_args()

    method_config = get_method_config(args.editing_method)
    hparams_bundle = load_hparams_bundle(args, method_config)
    hparams = hparams_bundle["hparams"]

    if args.editing_method in ["CAKE_KL", "CAKE_Batch_KL"]:
        hparams.kl_lambda = args.kl_lambda

    alg_name, apply_algo = maybe_build_apply_algo(hparams, method_config)
    model_path = hparams.model_name
    model = load_model_for_method(model_path, method_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    data = load_json(f"./datasets/{args.datatype}-cake.json")
    context = build_method_context(args, hparams_bundle, alg_name, apply_algo, data)
    kl_suffix = build_kl_suffix(args.editing_method, args.kl_lambda)

    os.makedirs(args.metrics_save_dir, exist_ok=True)

    print(f"Evaluating {args.edit_mode.replace('_', '-')} retention...")
    model, metrics = run_edit_mode(
        args.edit_mode, args.editing_method, model, tokenizer, data, context
    )
    results = calculate_averages(metrics)
    metrics_path, results_path = build_output_paths(args, kl_suffix, hparams)
    save_json(metrics, metrics_path)
    save_json(results, results_path)

    if args.edit_mode == "continual_edit" and args.save_model.lower() == "true":
        model_cache_dir = (
            f"{args.model_save_dir}/{args.editing_method}_{args.model_type}_"
            f"{args.datatype}_{args.edit_mode}_freq_{args.edit_freq}{kl_suffix}"
        )
        os.makedirs(model_cache_dir, exist_ok=True)
        model.save_pretrained(model_cache_dir)
        tokenizer.save_pretrained(model_cache_dir)
        print(f"Model saved to {model_cache_dir}")
