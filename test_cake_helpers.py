import json

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from EasyEdit.easyeditor import (
    AlphaEditHyperParams,
    CAKEWISEHyperParams,
    LoRAHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    UltraEditHyperParams,
    WISEHyperParams,
)
from EasyEdit.easyeditor.util.alg_dict import ALG_DICT
from CaKE_method.cake_batch import (
    cake_batch_continual_edit,
    cake_batch_multi_edit,
    cake_batch_sequential_edit,
)
from CaKE_method.cake_batch_kl import (
    cake_batch_kl_continual_edit,
    cake_batch_kl_multi_edit,
    cake_batch_kl_sequential_edit,
)
from CaKE_method.cake_kl import (
    cake_kl_continual_edit,
    cake_kl_multi_edit,
    cake_kl_sequential_edit,
    cake_kl_single_edit,
)
from CaKE_method.cake_Overtone import cake_overtone
from CaKE_method.cake_wise import (
    cake_wise_continual_edit,
    cake_wise_multi_edit,
    cake_wise_sequential_edit,
)
from Edit_mode.continual_edit import (
    cake_continual_edit,
    continual_edit,
    continual_edit_lora,
    continual_edit_rome,
    continual_edit_ultraedit,
    continual_edit_wise,
    continual_eval,
)
from Edit_mode.multi_edit import (
    cake_multi_edit,
    multi_edit,
    multi_edit_rome,
    multi_edit_wise,
)
from Edit_mode.sequential_edit import (
    cake_sequential_edit,
    sequential_edit,
    sequential_edit_rome,
    sequential_edit_wise,
)
from edit_utils import (
    cake,
    edit,
    edit_ifmet,
    edit_mello,
    edit_rome,
    edit_wise,
    get_sent_embeddings,
    resolve_lora_training_config,
)

METHOD_CONFIGS = {
    "MEMIT": {
        "hparams_class": MEMITHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.float32,
        "needs_apply_algo": True,
    },
    "IFMET": {
        "hparams_class": MEMITHyperParams,
        "hparams_source": "ifmet",
        "torch_dtype": torch.float32,
        "needs_apply_algo": True,
    },
    "LoRA": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.float32,
        "needs_apply_algo": True,
    },
    "WISE": {
        "hparams_class": WISEHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.float32,
        "needs_apply_algo": True,
    },
    "ROME": {
        "hparams_class": ROMEHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.float32,
        "needs_apply_algo": True,
    },
    "CAKE": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "lora",
        "torch_dtype": torch.bfloat16,
    },
    "Mello": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "lora",
        "torch_dtype": torch.float32,
    },
    "CAKE_WISE": {
        "hparams_class": CAKEWISEHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.bfloat16,
    },
    "CAKE_KL": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "lora",
        "torch_dtype": torch.bfloat16,
    },
    "CAKE_Batch": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "lora",
        "torch_dtype": torch.bfloat16,
    },
    "CAKE_OverTone": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "lora",
        "torch_dtype": torch.bfloat16,
    },
    "CAKE_Batch_KL": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "lora",
        "torch_dtype": torch.bfloat16,
    },
    "AlphaEdit": {
        "hparams_class": AlphaEditHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.float32,
        "needs_apply_algo": True,
    },
    "UltraEdit": {
        "hparams_class": UltraEditHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.bfloat16,
    },
    "Original": {
        "hparams_class": LoRAHyperParams,
        "hparams_source": "method",
        "torch_dtype": torch.float32,
    },
}


def get_method_config(editing_method):
    return METHOD_CONFIGS.get(
        editing_method,
        {
            "hparams_class": LoRAHyperParams,
            "hparams_source": "method",
            "torch_dtype": torch.float32,
            "needs_apply_algo": True,
        },
    )


def load_hparams_bundle(args, method_config):
    hparams_class = method_config["hparams_class"]
    hparams_source = method_config["hparams_source"]

    if hparams_source == "lora":
        hparams = hparams_class.from_hparams(
            f"./EasyEdit/hparams/LoRA/{args.model_type}.yaml"
        )
        return {"hparams": hparams}

    if hparams_source == "ifmet":
        hparams_s = hparams_class.from_hparams(
            f"./EasyEdit/hparams/{args.editing_method}/{args.model_type}-shallow.yaml"
        )
        hparams_d = hparams_class.from_hparams(
            f"./EasyEdit/hparams/{args.editing_method}/{args.model_type}-deeper.yaml"
        )
        return {"hparams": hparams_s, "hparams_s": hparams_s, "hparams_d": hparams_d}

    hparams = hparams_class.from_hparams(
        f"./EasyEdit/hparams/{args.editing_method}/{args.model_type}.yaml"
    )
    return {"hparams": hparams}


def maybe_build_apply_algo(hparams, method_config):
    if not method_config.get("needs_apply_algo", False):
        return None, None

    alg_name = hparams.alg_name
    return alg_name, ALG_DICT[alg_name]


def load_model_for_method(model_path, method_config):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=method_config["torch_dtype"],
    )


def build_method_context(args, hparams_bundle, alg_name, apply_algo, data):
    hparams = hparams_bundle["hparams"]
    context = {
        "hparams": hparams,
        "hparams_s": hparams_bundle.get("hparams_s"),
        "hparams_d": hparams_bundle.get("hparams_d"),
        "datatype": args.datatype,
        "model_path": hparams.model_name,
        "alg_name": alg_name,
        "apply_algo": apply_algo,
        "edit_freq": args.edit_freq,
        "loc_data": None,
        "loc_index": None,
        "task_prompt": None,
        "stop": None,
        "contriever": None,
        "contriever_tokenizer": None,
        "embs": None,
        "new_facts": None,
    }

    if args.editing_method == "WISE":
        context["loc_data"] = load_json("./datasets/ZsRE/zsre_mend_train.json")[:7000]
        context["loc_index"] = 0
        return context

    if args.editing_method == "Mello":
        contriever = AutoModel.from_pretrained(
            "/data1/xubuqiang/contriever-msmarco",
            device_map=f"cuda:{hparams.device}",
        )
        contriever_tokenizer = AutoTokenizer.from_pretrained(
            "/data1/xubuqiang/contriever-msmarco"
        )
        with open("./prompts/MeLLo-prompt.txt", "r") as prompt_file:
            task_prompt = prompt_file.read()

        new_facts = set()
        for item in data:
            for rewrite in item["requested_rewrite"]:
                new_facts.add(
                    f'{rewrite["prompt"].format(rewrite["subject"])} '
                    f'{rewrite["target_new"]["str"]}'
                )
        fact_list = list(new_facts)

        context.update(
            {
                "task_prompt": task_prompt,
                "stop": ["Retrieved fact:"],
                "contriever": contriever,
                "contriever_tokenizer": contriever_tokenizer,
                "embs": get_sent_embeddings(
                    fact_list,
                    contriever,
                    contriever_tokenizer,
                    hparams.device,
                ),
                "new_facts": fact_list,
            }
        )

    return context


def save_json(payload, output_path):
    with open(output_path, "w") as output_file:
        json.dump(payload, output_file, indent=4)


def load_json(input_path):
    with open(input_path, "r") as input_file:
        return json.load(input_file)


def build_output_paths(args, kl_suffix, hparams=None):
    base_name = (
        f"{args.editing_method}_{args.model_type}_{args.datatype}_{args.edit_mode}"
    )

    if args.edit_mode in {"sequential_edit", "continual_edit", "multi_edit"}:
        base_name = f"{base_name}_freq_{args.edit_freq}"

    if args.edit_mode == "continual_edit":
        suffix = f"{kl_suffix}_test3_lr_1e-5_epochs_30"
    else:
        suffix = kl_suffix

    metrics_path = f"{args.metrics_save_dir}/{base_name}_metrics{suffix}.json"
    result_path = f"{args.metrics_save_dir}/{base_name}_res{suffix}.json"
    return metrics_path, result_path


def get_method_training_config(args, hparams=None):
    method_name_map = {
        "CAKE": "cake",
        "CAKE_KL": "cake_kl",
        "CAKE_Batch": "cake_batch",
        "CAKE_Batch_KL": "cake_batch_kl",
        "CAKE_WISE": "cake_wise",
        "CAKE_OverTone": "overtone",
        "LoRA": "lora",
    }
    method_name = method_name_map.get(args.editing_method)
    if method_name is None:
        return None

    if hparams is None:
        method_config = get_method_config(args.editing_method)
        hparams = load_hparams_bundle(args, method_config)["hparams"]

    return resolve_lora_training_config(
        hparams,
        method_name=method_name,
        allow_hparams_batch_size=(method_name == "lora"),
    )


def build_continual_hparam_suffix(args, hparams=None):
    training_config = get_method_training_config(args, hparams)
    if training_config is not None:
        lr = training_config["lr"]
        num_steps = training_config["num_steps"]
        return f"_lr_{lr}_epochs_{num_steps}"

    return ""


def run_single_cake(model, tokenizer, item, context):
    return cake(
        model,
        tokenizer,
        item,
        context["hparams"],
        context["datatype"],
        test_generation=False,
    )


def run_single_cake_kl(model, tokenizer, item, context):
    return cake_kl_single_edit(
        model,
        tokenizer,
        item,
        context["hparams"],
        context["model_path"],
        context["datatype"],
        test_generation=False,
    )


def run_single_cake_overtone(model, tokenizer, item, context):
    return cake_overtone(
        model,
        tokenizer,
        item,
        context["hparams"],
        context["datatype"],
        test_generation=False,
    )


def run_single_wise(model, tokenizer, item, context):
    metrics, updated_loc_index = edit_wise(
        model,
        tokenizer,
        item,
        context["hparams"],
        context["loc_data"],
        context["loc_index"],
        context["apply_algo"],
        context["datatype"],
        test_generation=False,
    )
    context["loc_index"] = updated_loc_index
    return model, metrics


def run_single_mello(model, tokenizer, item, context):
    metrics = edit_mello(
        model,
        context["task_prompt"],
        context["stop"],
        tokenizer,
        item,
        context["hparams"],
        context["contriever"],
        context["contriever_tokenizer"],
        context["embs"],
        context["new_facts"],
        context["datatype"],
        test_generation=False,
    )
    return model, metrics


def run_single_ifmet(model, tokenizer, item, context):
    return edit_ifmet(
        model,
        tokenizer,
        item,
        context["hparams_s"],
        context["hparams_d"],
        context["apply_algo"],
        context["datatype"],
        test_generation=False,
    )


def run_single_rome(model, tokenizer, item, context):
    return edit_rome(
        model,
        tokenizer,
        item,
        context["hparams"],
        context["apply_algo"],
        context["datatype"],
        test_generation=False,
    )


def run_single_default(model, tokenizer, item, context):
    return edit(
        model,
        tokenizer,
        item,
        context["hparams"],
        context["alg_name"],
        context["apply_algo"],
        context["datatype"],
        test_generation=False,
    )


SINGLE_EDIT_METHOD_RUNNERS = {
    "CAKE": run_single_cake,
    "CAKE_KL": run_single_cake_kl,
    "CAKE_OverTone": run_single_cake_overtone,
    "WISE": run_single_wise,
    "Mello": run_single_mello,
    "IFMET": run_single_ifmet,
    "ROME": run_single_rome,
}


def run_single_edit(method, model, tokenizer, data, context):
    all_metrics = []
    runner = SINGLE_EDIT_METHOD_RUNNERS.get(method, run_single_default)

    for item in tqdm(data[:100]):
        model, metrics = runner(model, tokenizer, item, context)
        print(metrics)
        all_metrics.append(metrics)
    return model, all_metrics


def run_sequential_cake(model, tokenizer, items, context):
    return cake_sequential_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_sequential_rome(model, tokenizer, items, context):
    return sequential_edit_rome(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_sequential_cake_wise(model, tokenizer, items, context):
    return cake_wise_sequential_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_sequential_cake_kl(model, tokenizer, items, context):
    return cake_kl_sequential_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_sequential_cake_batch(model, tokenizer, items, context):
    return cake_batch_sequential_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_sequential_cake_batch_kl(model, tokenizer, items, context):
    return cake_batch_kl_sequential_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_sequential_wise(model, tokenizer, items, context):
    return sequential_edit_wise(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["loc_data"],
        context["loc_index"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_sequential_default(model, tokenizer, items, context):
    return sequential_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["alg_name"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


SEQUENTIAL_EDIT_METHOD_RUNNERS = {
    "CAKE": run_sequential_cake,
    "ROME": run_sequential_rome,
    "CAKE_WISE": run_sequential_cake_wise,
    "CAKE_KL": run_sequential_cake_kl,
    "CAKE_Batch": run_sequential_cake_batch,
    "CAKE_Batch_KL": run_sequential_cake_batch_kl,
    "WISE": run_sequential_wise,
}


def run_sequential_edit(method, model, tokenizer, data, context):
    runner = SEQUENTIAL_EDIT_METHOD_RUNNERS.get(method, run_sequential_default)
    return runner(model, tokenizer, data[:100], context)


def run_continual_cake(model, tokenizer, items, context):
    return cake_continual_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_rome(model, tokenizer, items, context):
    return continual_edit_rome(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_cake_wise(model, tokenizer, items, context):
    return cake_wise_continual_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_cake_kl(model, tokenizer, items, context):
    return cake_kl_continual_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_cake_batch(model, tokenizer, items, context):
    return cake_batch_continual_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_cake_batch_kl(model, tokenizer, items, context):
    return cake_batch_kl_continual_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_wise(model, tokenizer, items, context):
    return continual_edit_wise(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["loc_data"],
        context["loc_index"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_ultraedit(model, tokenizer, items, context):
    return continual_edit_ultraedit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_lora(model, tokenizer, items, context):
    return continual_edit_lora(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_original(model, tokenizer, items, context):
    return continual_eval(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_continual_default(model, tokenizer, items, context):
    return continual_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["alg_name"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


CONTINUAL_EDIT_METHOD_RUNNERS = {
    "CAKE": run_continual_cake,
    "ROME": run_continual_rome,
    "CAKE_WISE": run_continual_cake_wise,
    "CAKE_KL": run_continual_cake_kl,
    "CAKE_Batch": run_continual_cake_batch,
    "CAKE_Batch_KL": run_continual_cake_batch_kl,
    "WISE": run_continual_wise,
    "UltraEdit": run_continual_ultraedit,
    "LoRA": run_continual_lora,
    "Original": run_continual_original,
}


def run_continual_edit(method, model, tokenizer, data, context):
    runner = CONTINUAL_EDIT_METHOD_RUNNERS.get(method, run_continual_default)
    return runner(model, tokenizer, data[:1000], context)


def run_multi_rome(model, tokenizer, items, context):
    return multi_edit_rome(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_multi_cake(model, tokenizer, items, context):
    return cake_multi_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["model_path"],
        context["datatype"],
        test_generation=False,
    )


def run_multi_wise(model, tokenizer, items, context):
    return multi_edit_wise(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["loc_data"],
        context["loc_index"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


def run_multi_cake_wise(model, tokenizer, items, context):
    return cake_wise_multi_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["model_path"],
        context["datatype"],
        test_generation=False,
    )


def run_multi_cake_kl(model, tokenizer, items, context):
    return cake_kl_multi_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["model_path"],
        context["datatype"],
        test_generation=False,
    )


def run_multi_cake_batch_kl(model, tokenizer, items, context):
    return cake_batch_kl_multi_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["model_path"],
        context["datatype"],
        test_generation=False,
    )


def run_multi_cake_batch(model, tokenizer, items, context):
    return cake_batch_multi_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["edit_freq"],
        context["model_path"],
        context["datatype"],
        test_generation=False,
    )


def run_multi_default(model, tokenizer, items, context):
    return multi_edit(
        model,
        tokenizer,
        items,
        context["hparams"],
        context["alg_name"],
        context["apply_algo"],
        context["edit_freq"],
        context["datatype"],
        test_generation=False,
    )


MULTI_EDIT_METHOD_RUNNERS = {
    "ROME": run_multi_rome,
    "CAKE": run_multi_cake,
    "WISE": run_multi_wise,
    "CAKE_WISE": run_multi_cake_wise,
    "CAKE_KL": run_multi_cake_kl,
    "CAKE_Batch_KL": run_multi_cake_batch_kl,
    "CAKE_Batch": run_multi_cake_batch,
}


def run_multi_edit(method, model, tokenizer, data, context):
    runner = MULTI_EDIT_METHOD_RUNNERS.get(method, run_multi_default)
    return runner(model, tokenizer, data[:100], context)


MODE_RUNNERS = {
    "single_edit": run_single_edit,
    "sequential_edit": run_sequential_edit,
    "continual_edit": run_continual_edit,
    "multi_edit": run_multi_edit,
}


def run_edit_mode(edit_mode, editing_method, model, tokenizer, data, context):
    return MODE_RUNNERS[edit_mode](editing_method, model, tokenizer, data, context)
