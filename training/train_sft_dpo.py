import argparse
import contextlib
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tool_schemas import TOOL_SCHEMAS, schemas_for
from transformers import TrainerCallback

FAMILIES = ("qwen3.8-moe", "qwen3-32b")
_COMPUTED_ROOT = Path(os.environ.get("ADS_BENCH_COMPUTED_DIR") or (Path(__file__).resolve().parent.parent / "computed"))
DEFAULT_DATASET_DIR = str(_COMPUTED_ROOT / "datasets")
MODELS_ROOT = str(_COMPUTED_ROOT / "models")
SMOKE_ROOT = str(_COMPUTED_ROOT / "smoke")

ATTENTION_PROJ = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
MLP_PROJ = frozenset({"gate_proj", "up_proj", "down_proj"})
EXPERT_PROJ = MLP_PROJ | {"w1", "w2", "w3"}
FUSED_EXPERT_PARAMS = frozenset({"gate_up_proj", "down_proj"})
ROUTER_LEAVES = frozenset({"gate", "router", "shared_expert_gate"})
VISION_MARKERS = ("visual", "vision")

SMOKE_STEPS = 4


@dataclass(frozen=True)
class LoraTargets:
    modules: tuple
    parameters: tuple


@dataclass(frozen=True)
class DeviceInfo:
    kind: str
    index: int | None
    bf16: bool

    @property
    def dtype_name(self):
        return "bfloat16" if self.bf16 else "float32"


def is_router(name):
    leaf = name.rsplit(".", 1)[-1]
    return leaf in ROUTER_LEAVES or "router" in leaf


def is_vision(name):
    return any(marker in name for marker in VISION_MARKERS)


def select_lora_targets(module_names, family, lora_experts=False, parameter_names=()):
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}")
    moe = family == "qwen3.8-moe"
    modules = []
    for name in module_names:
        if is_router(name) or is_vision(name):
            continue
        parts = name.split(".")
        leaf, scope = parts[-1], parts[:-1]
        if leaf in ATTENTION_PROJ:
            modules.append(name)
        elif not moe and leaf in MLP_PROJ and "experts" not in scope:
            modules.append(name)
        elif moe and leaf in MLP_PROJ and any(p.startswith("shared_expert") for p in scope):
            modules.append(name)
        elif moe and lora_experts and leaf in EXPERT_PROJ and "experts" in scope:
            modules.append(name)
    parameters = []
    if moe and lora_experts:
        for name in parameter_names:
            parts = name.split(".")
            if parts[-1] in FUSED_EXPERT_PARAMS and "experts" in parts[:-1] and not is_vision(name):
                parameters.append(name)
    if not modules and not parameters:
        raise ValueError(f"no LoRA target modules found for family {family!r}")
    return LoraTargets(tuple(modules), tuple(parameters))


def parse_device(spec):
    spec = spec.strip().lower()
    if spec in ("auto", "cpu"):
        return spec, None
    match = re.fullmatch(r"cuda(?::(\d+))?", spec)
    if not match:
        raise ValueError(f"unsupported device {spec!r}; use auto, cpu, cuda or cuda:N")
    return "cuda", int(match.group(1)) if match.group(1) else None


def configure_device(spec, smoke):
    kind, index = parse_device(spec)
    if kind == "cuda" and index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    import torch

    if kind == "auto":
        kind = "cpu" if smoke else ("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {spec!r} requested but CUDA is not available")
        return DeviceInfo("cuda", index, bool(torch.cuda.is_bf16_supported()))
    return DeviceInfo("cpu", None, False)


def resolve_output_dir(output_dir, model_family, smoke):
    if output_dir:
        return Path(output_dir)
    return Path(SMOKE_ROOT if smoke else MODELS_ROOT) / model_family


def build_parser():
    p = argparse.ArgumentParser(prog="train_sft_dpo.py", description="LoRA SFT + DPO for the deposition agent")
    p.add_argument("--model-family", choices=FAMILIES, help="required except with --merge-adapter")
    p.add_argument("--model-path", help="HF id or local directory of the base model; required unless --smoke")
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--stage", choices=("sft", "dpo", "both"), default="both")
    p.add_argument("--smoke", action="store_true", help="tiny offline random model, 4 steps per stage")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    p.add_argument("--init-adapter", default=None, help="LoRA adapter directory to continue from in --stage dpo")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-experts", action="store_true", help="also adapt routed experts (MoE only)")
    p.add_argument("--sft-epochs", type=float, default=2)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--dpo-epochs", type=float, default=1)
    p.add_argument("--dpo-lr", type=float, default=5e-6)
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--dpo-batch-size", type=int, default=None)
    p.add_argument("--dpo-grad-accum", type=int, default=None)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge-adapter", action="store_true", help="merge --adapter-dir into --model-path and save")
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--merged-dir", default=None)
    p.add_argument("--merge-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return p


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.merge_adapter:
        if not (args.model_path and args.adapter_dir and args.merged_dir):
            parser.error("--merge-adapter needs --model-path, --adapter-dir and --merged-dir")
        return args
    if not args.model_family:
        parser.error("--model-family is required")
    if not args.smoke and not args.model_path:
        parser.error("--model-path is required unless --smoke is given")
    args.output_dir = str(resolve_output_dir(args.output_dir, args.model_family, args.smoke))
    return args


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset file missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_tool_names(record):
    unknown = [t for t in record["tools"] if t not in TOOL_SCHEMAS]
    if unknown:
        raise ValueError(f"record {record.get('id')}: unknown tools {unknown}")


def check_arguments_are_dicts(record, messages):
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call["function"]["arguments"], dict):
                raise ValueError(f"record {record.get('id')}: tool_call arguments must be a dict")


def render_chat(tokenizer, messages, tools, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def assistant_spans(tokenizer, messages, tools):
    full = render_chat(tokenizer, messages, tools)
    end_token = tokenizer.eos_token
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = render_chat(tokenizer, messages[:index], tools, add_generation_prompt=True)
        if not full.startswith(prefix):
            raise ValueError(
                f"chat template does not render messages[:{index}] + generation prompt as a prefix of the full "
                "conversation; assistant spans cannot be located"
            )
        stop = full.find(end_token, len(prefix))
        if stop < 0:
            raise ValueError(f"assistant turn {index} is not terminated by {end_token!r}")
        spans.append((len(prefix), stop + len(end_token)))
    return full, spans


def label_ids_from_spans(tokenizer, text, spans):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer is required for offset-based loss masking")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    labels = []
    for token_id, (start, end) in zip(ids, encoded["offset_mapping"]):
        inside = any(start < span_end and end > span_start for span_start, span_end in spans)
        labels.append(token_id if inside else -100)
    return ids, labels


def encode_with_labels(tokenizer, messages, tools):
    text, spans = assistant_spans(tokenizer, messages, tools)
    return label_ids_from_spans(tokenizer, text, spans)


def sft_features(tokenizer, record):
    check_tool_names(record)
    check_arguments_are_dicts(record, record["messages"])
    ids, labels = encode_with_labels(tokenizer, record["messages"], schemas_for(record["tools"]))
    return {"input_ids": ids, "labels": labels}


def build_sft_rows(tokenizer, records, max_length):
    rows, dropped = [], []
    for record in records:
        features = sft_features(tokenizer, record)
        if len(features["input_ids"]) > max_length:
            dropped.append(record["id"])
        else:
            rows.append(features)
    if not rows:
        raise ValueError("no SFT records fit into --max-length")
    return rows, dropped


def dpo_record_to_preference(record):
    check_tool_names(record)
    for key in ("messages_context", "chosen", "rejected"):
        check_arguments_are_dicts(record, record[key])
    return {
        "prompt": record["messages_context"],
        "chosen": record["chosen"],
        "rejected": record["rejected"],
        "tools": json.dumps(schemas_for(record["tools"])),
    }


def common_prefix_length(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def preference_features(tokenizer, preference):
    tools = json.loads(preference["tools"])
    prompt_text = render_chat(tokenizer, preference["prompt"], tools, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    chosen_full, chosen_labels = encode_with_labels(tokenizer, preference["prompt"] + preference["chosen"], tools)
    rejected_full, rejected_labels = encode_with_labels(tokenizer, preference["prompt"] + preference["rejected"], tools)
    prompt_len = min(common_prefix_length(prompt_ids, chosen_full), common_prefix_length(prompt_ids, rejected_full))
    features = {
        "prompt_ids": prompt_ids[:prompt_len],
        "chosen_ids": chosen_full[prompt_len:],
        "rejected_ids": rejected_full[prompt_len:],
        "chosen_mask": [int(label != -100) for label in chosen_labels[prompt_len:]],
        "rejected_mask": [int(label != -100) for label in rejected_labels[prompt_len:]],
    }
    if not any(features["chosen_mask"]) or not any(features["rejected_mask"]):
        raise ValueError("a preference continuation has no assistant tokens")
    return features


def build_dpo_rows(tokenizer, records, max_length):
    rows, dropped = [], []
    for record in records:
        features = preference_features(tokenizer, dpo_record_to_preference(record))
        longest = max(len(features["chosen_ids"]), len(features["rejected_ids"]))
        if len(features["prompt_ids"]) + longest > max_length:
            dropped.append(record["id"])
        else:
            rows.append(features)
    if not rows:
        raise ValueError("no DPO records fit into --max-length")
    return rows, dropped


def lora_config_for(model, args):
    import torch
    from peft import LoraConfig

    module_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    parameter_names = [n for n, p in model.named_parameters() if p.ndim == 3]
    targets = select_lora_targets(module_names, args.model_family, args.lora_experts, parameter_names)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0 if targets.parameters else args.lora_dropout,
        target_modules=list(targets.modules),
        target_parameters=list(targets.parameters) or None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return config, targets


class MetricsLog(TrainerCallback):
    def __init__(self, path, stage):
        self.path = path
        self.stage = stage
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"stage": self.stage, "step": state.global_step, "epoch": state.epoch}
        row.update({k: v for k, v in logs.items() if isinstance(v, (int, float, str))})
        self.records.append(row)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def common_trainer_kwargs(args, device, out_dir, smoke, batch_size, grad_accum):
    kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=device.bf16,
        fp16=False,
        use_cpu=device.kind == "cpu",
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        max_length=args.max_length,
        logging_steps=1 if smoke else args.logging_steps,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=True,
        save_total_limit=2,
    )
    if smoke:
        kwargs.update(max_steps=SMOKE_STEPS, eval_strategy="steps", eval_steps=2, save_strategy="no")
    else:
        kwargs.update(max_steps=args.max_steps, eval_strategy="epoch", save_strategy="epoch")
    return kwargs


def sft_config(args, device, out_dir, smoke):
    from trl import SFTConfig

    kwargs = common_trainer_kwargs(args, device, out_dir / "sft_checkpoints", smoke, args.batch_size, args.grad_accum)
    return SFTConfig(num_train_epochs=args.sft_epochs, learning_rate=args.sft_lr, packing=False, **kwargs)


def dpo_config(args, device, out_dir, smoke):
    from trl import DPOConfig

    kwargs = common_trainer_kwargs(
        args,
        device,
        out_dir / "dpo_checkpoints",
        smoke,
        args.dpo_batch_size or args.batch_size,
        args.dpo_grad_accum or args.grad_accum,
    )
    return DPOConfig(
        num_train_epochs=args.dpo_epochs,
        learning_rate=args.dpo_lr,
        beta=args.dpo_beta,
        precompute_ref_log_probs=False,
        **kwargs,
    )


def pretokenized_dpo_trainer_class():
    import torch
    from trl import DPOTrainer
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    class AssistantOnlyPreferenceCollator(DataCollatorForPreference):
        def torch_call(self, examples):
            output = super().torch_call(examples)
            masks = [[0] * len(e["prompt_ids"]) + list(e["chosen_mask"]) for e in examples]
            masks += [[0] * len(e["prompt_ids"]) + list(e["rejected_mask"]) for e in examples]
            completion_mask = torch.zeros_like(output["completion_mask"])
            for row, mask in enumerate(masks):
                completion_mask[row, : len(mask)] = torch.tensor(mask, dtype=completion_mask.dtype)
            output["completion_mask"] = completion_mask
            return output

    class PretokenizedDPOTrainer(DPOTrainer):
        def __init__(self, *args, **kwargs):
            tokenizer = kwargs["processing_class"]
            kwargs["data_collator"] = AssistantOnlyPreferenceCollator(
                pad_token_id=tokenizer.pad_token_id, max_length=kwargs["args"].max_length
            )
            super().__init__(*args, **kwargs)

        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            return dataset

        def _set_signature_columns_if_needed(self):
            super()._set_signature_columns_if_needed()
            self._signature_columns += ["chosen_mask", "rejected_mask"]

    return PretokenizedDPOTrainer


def save_adapter(trainer, tokenizer, directory):
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.save_pretrained(str(directory), selected_adapters=["default"])
    tokenizer.save_pretrained(str(directory))


def release_memory():
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_batch_shapes(trainer):
    batch = next(iter(trainer.get_train_dataloader()))
    return {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")}


def summarise_logs(records, keys):
    out = {}
    for key in keys:
        values = [r[key] for r in records if key in r]
        out[key] = values
    return out


def run_sft(args, ctx, out_dir):
    from datasets import Dataset
    from trl import SFTTrainer

    model = ctx.load_model()
    peft_config, targets = lora_config_for(model, args)
    train_rows, train_dropped = build_sft_rows(ctx.tokenizer, ctx.records["sft_train"], args.max_length)
    val_rows, val_dropped = build_sft_rows(ctx.tokenizer, ctx.records["sft_val"], args.max_length)
    ctx.tokenizer.padding_side = "right"
    metrics = MetricsLog(out_dir / "metrics.jsonl", "sft")
    trainer = SFTTrainer(
        model=model,
        args=sft_config(args, ctx.device, out_dir, ctx.smoke),
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows),
        processing_class=ctx.tokenizer,
        peft_config=peft_config,
        callbacks=[metrics],
    )
    shapes = first_batch_shapes(trainer)
    trainer.model.config.use_cache = False
    result = trainer.train()
    eval_result = trainer.evaluate()
    adapter_dir = out_dir / "sft_adapter"
    save_adapter(trainer, ctx.tokenizer, adapter_dir)
    summary = {
        "adapter_dir": str(adapter_dir),
        "train_loss": result.training_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "global_step": trainer.state.global_step,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "dropped_train": train_dropped,
        "dropped_val": val_dropped,
        "lora_modules": len(targets.modules),
        "lora_parameters": list(targets.parameters),
        "lora_dropout": peft_config.lora_dropout,
        "batch_shapes": shapes,
        "logs": summarise_logs(metrics.records, ("loss", "eval_loss", "grad_norm")),
        "label_fraction": sum(1 for r in train_rows for t in r["labels"] if t != -100)
        / sum(len(r["labels"]) for r in train_rows),
    }
    del trainer, model
    release_memory()
    return summary


def run_dpo(args, ctx, out_dir, init_adapter):
    from datasets import Dataset

    model = ctx.load_model()
    peft_config, targets = lora_config_for(model, args)
    if init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        peft_config = None
    train_rows, train_dropped = build_dpo_rows(ctx.tokenizer, ctx.records["dpo_train"], args.max_length)
    val_rows, val_dropped = build_dpo_rows(ctx.tokenizer, ctx.records["dpo_val"], args.max_length)
    ctx.tokenizer.padding_side = "left"
    metrics = MetricsLog(out_dir / "metrics.jsonl", "dpo")
    trainer = pretokenized_dpo_trainer_class()(
        model=model,
        args=dpo_config(args, ctx.device, out_dir, ctx.smoke),
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows),
        processing_class=ctx.tokenizer,
        peft_config=peft_config,
        callbacks=[metrics],
    )
    shapes = first_batch_shapes(trainer)
    trainer.model.config.use_cache = False
    result = trainer.train()
    eval_result = trainer.evaluate()
    adapter_dir = out_dir / "dpo_adapter"
    save_adapter(trainer, ctx.tokenizer, adapter_dir)
    summary = {
        "adapter_dir": str(adapter_dir),
        "train_loss": result.training_loss,
        "eval_loss": eval_result.get("eval_loss"),
        "global_step": trainer.state.global_step,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "dropped_train": train_dropped,
        "dropped_val": val_dropped,
        "init_adapter": init_adapter,
        "lora_modules": len(targets.modules),
        "batch_shapes": shapes,
        "logs": summarise_logs(
            metrics.records, ("loss", "eval_loss", "grad_norm", "rewards/margins", "rewards/accuracies")
        ),
    }
    del trainer, model
    release_memory()
    return summary


def merge_adapter(model_path, adapter_dir, out_dir, dtype_name):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype_name))
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(str(out_dir))
    return str(out_dir)


@dataclass
class RunContext:
    tokenizer: object
    load_model: object
    records: dict
    device: DeviceInfo
    smoke: bool


def real_context(args, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not tokenizer.chat_template:
        raise ValueError(f"tokenizer at {args.model_path} has no chat template")
    dtype = torch.bfloat16 if device.bf16 else torch.float32

    def load_model():
        kwargs = {"device_map": {"": 0}} if device.kind == "cuda" else {}
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype, **kwargs)
        model.config.use_cache = False
        return model

    dataset_dir = Path(args.dataset_dir)
    records = {}
    if args.stage in ("sft", "both"):
        records["sft_train"] = load_jsonl(dataset_dir / "sft_train.jsonl")
        records["sft_val"] = load_jsonl(dataset_dir / "sft_val.jsonl")
    if args.stage in ("dpo", "both"):
        records["dpo_train"] = load_jsonl(dataset_dir / "dpo_train.jsonl")
        records["dpo_val"] = load_jsonl(dataset_dir / "dpo_val.jsonl")
    return RunContext(tokenizer, load_model, records, device, False)


def smoke_context(args, device, out_dir):
    import smoke_assets
    import torch

    tokenizer = smoke_assets.build_smoke_tokenizer(args.model_family)
    base_dir = out_dir / "smoke_base"
    base = smoke_assets.build_smoke_model(args.model_family, tokenizer, args.seed)
    base.save_pretrained(str(base_dir))
    tokenizer.save_pretrained(str(base_dir))
    args.model_path = str(base_dir)

    def load_model():
        model = smoke_assets.build_smoke_model(args.model_family, tokenizer, args.seed)
        model.config.use_cache = False
        return model.to("cuda") if device.kind == "cuda" else model

    records = {
        "sft_train": smoke_assets.synthetic_sft_records(8, args.seed),
        "sft_val": smoke_assets.synthetic_sft_records(4, args.seed + 1),
        "dpo_train": smoke_assets.synthetic_dpo_records(8, args.seed),
        "dpo_val": smoke_assets.synthetic_dpo_records(4, args.seed + 1),
    }
    torch.manual_seed(args.seed)
    return RunContext(tokenizer, load_model, records, device, True)


def library_versions():
    import importlib.metadata as md

    names = ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
    return {name: md.version(name) for name in names}


def write_train_config(out_dir, args, device, extra):
    payload = {"args": vars(args), "device": {"kind": device.kind, "index": device.index, "dtype": device.dtype_name}}
    payload["versions"] = library_versions()
    payload.update(extra)
    (out_dir / "train_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_stages(args, ctx, out_dir):
    results = {}
    init_adapter = args.init_adapter
    if args.stage in ("sft", "both"):
        results["sft"] = run_sft(args, ctx, out_dir)
        init_adapter = results["sft"]["adapter_dir"]
    if args.stage in ("dpo", "both"):
        results["dpo"] = run_dpo(args, ctx, out_dir, init_adapter)
    return results


def smoke_checks(args, ctx, out_dir, results):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    problems = []
    for stage, summary in results.items():
        losses = summary["logs"]["loss"]
        if len(losses) != SMOKE_STEPS or not all(finite(v) for v in losses):
            problems.append(f"{stage}: non-finite or missing step losses {losses}")
        eval_loss = summary["eval_loss"]
        if not finite(eval_loss):
            problems.append(f"{stage}: non-finite eval loss {eval_loss}")
    if "dpo" in results and not results["dpo"]["logs"]["rewards/margins"]:
        problems.append("dpo: no reward margins logged")

    final = results.get("dpo") or results["sft"]
    probe = ctx.tokenizer("<|im_start|>user\nstatus<|im_end|>\n", add_special_tokens=False, return_tensors="pt")
    base = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32)
    peft_logits = PeftModel.from_pretrained(base, final["adapter_dir"]).eval()(**probe).logits
    merged_dir = merge_adapter(args.model_path, final["adapter_dir"], out_dir / "smoke_merged", "float32")
    merged_logits = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32).eval()(**probe).logits
    with torch.no_grad():
        diff = float((peft_logits - merged_logits).abs().max())
    if not math.isfinite(diff) or diff > 1e-3:
        problems.append(f"merged model diverges from adapter model: max abs logit diff {diff}")

    sample = ctx.records["sft_train"][1]
    tools = schemas_for(sample["tools"])
    text, spans = assistant_spans(ctx.tokenizer, sample["messages"], tools)
    ids, labels = label_ids_from_spans(ctx.tokenizer, text, spans)
    return problems, {
        "chat_template_sample": text,
        "trained_text_sample": ctx.tokenizer.decode([t for t, label in zip(ids, labels) if label != -100]),
        "merge_max_abs_logit_diff": diff,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.merge_adapter:
        print(merge_adapter(args.model_path, args.adapter_dir, args.merged_dir, args.merge_dtype))
        return 0
    device = configure_device(args.device, args.smoke)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    import transformers

    transformers.set_seed(args.seed)
    started = time.time()
    ctx = smoke_context(args, device, out_dir) if args.smoke else real_context(args, device)
    write_train_config(out_dir, args, device, {"status": "started", "smoke": args.smoke})

    chatter = sys.stderr if args.smoke else sys.stdout
    with contextlib.redirect_stdout(chatter):
        results = run_stages(args, ctx, out_dir)

    final_stage = "dpo" if "dpo" in results else "sft"
    extra = {
        "status": "completed",
        "smoke": args.smoke,
        "seconds": round(time.time() - started, 1),
        "final_adapter": results[final_stage]["adapter_dir"],
        "results": results,
    }
    problems, smoke_report = [], {}
    if args.smoke:
        problems, smoke_report = smoke_checks(args, ctx, out_dir, results)
        extra["smoke_report"] = smoke_report
        extra["problems"] = problems
    write_train_config(out_dir, args, device, extra)

    if args.smoke:
        print(
            json.dumps(
                {
                    "ok": not problems,
                    "problems": problems,
                    "family": args.model_family,
                    "device": device.kind,
                    "versions": library_versions(),
                    "output_dir": str(out_dir),
                    "results": results,
                    **smoke_report,
                },
                indent=2,
            )
        )
        return 1 if problems else 0
    print(json.dumps({"final_adapter": extra["final_adapter"], "output_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
