import json
import os
from pathlib import Path

import pytest

import smoke_assets
import train_sft_dpo as t
from tool_schemas import schemas_for

TEMPLATE_DIR = Path(os.environ.get("ADS_BENCH_COMPUTED_DIR") or (Path(__file__).resolve().parent.parent / "computed")) / "templates"
REAL_TEMPLATES = {
    "real-qwen3-32b": ("qwen3-32b", TEMPLATE_DIR / "qwen3-32b.chat_template.jinja"),
    "real-qwen3.5-moe": ("qwen3.8-moe", TEMPLATE_DIR / "Qwen_Qwen3.5-35B-A3B.chat_template.jinja"),
}


def make_tokenizer(kind):
    if kind in smoke_assets.TEMPLATE_STYLE:
        return smoke_assets.build_smoke_tokenizer(kind)
    family, path = REAL_TEMPLATES[kind]
    if not path.is_file():
        pytest.skip(f"template not downloaded: {path}")
    tokenizer = smoke_assets.build_smoke_tokenizer(family)
    tokenizer.chat_template = path.read_text(encoding="utf-8")
    return tokenizer


@pytest.fixture(scope="module", params=["qwen3-32b", "qwen3.8-moe", "real-qwen3-32b", "real-qwen3.5-moe"])
def tokenizer(request):
    return make_tokenizer(request.param)


def labelled_text(tokenizer, features):
    return tokenizer.decode([i for i, label in zip(features["input_ids"], features["labels"]) if label != -100])


def masked_text(tokenizer, features):
    return tokenizer.decode([i for i, label in zip(features["input_ids"], features["labels"]) if label == -100])


DENSE_MODULES = [
    "model.embed_tokens",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.self_attn.q_norm",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.visual.blocks.0.attn.q_proj",
    "lm_head",
]

MOE_MODULES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate",
    "model.layers.0.mlp.router",
    "model.layers.0.mlp.experts.0.gate_proj",
    "model.layers.0.mlp.experts.0.up_proj",
    "model.layers.0.mlp.experts.0.down_proj",
    "model.layers.0.mlp.shared_expert.gate_proj",
    "model.layers.0.mlp.shared_expert.up_proj",
    "model.layers.0.mlp.shared_expert.down_proj",
    "model.layers.0.mlp.shared_expert_gate",
    "model.layers.1.linear_attn.in_proj_qkv",
    "lm_head",
]

FUSED_PARAMS = [
    "model.layers.0.mlp.experts.gate_up_proj",
    "model.layers.0.mlp.experts.down_proj",
    "model.layers.0.mlp.shared_expert.down_proj.weight",
]


def test_dense_targets_attention_and_mlp_only():
    targets = t.select_lora_targets(DENSE_MODULES, "qwen3-32b")
    assert set(targets.modules) == {
        f"model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")
    } | {f"model.layers.0.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")}
    assert targets.parameters == ()


def test_moe_default_targets_attention_and_shared_expert_only():
    targets = t.select_lora_targets(MOE_MODULES, "qwen3.8-moe", parameter_names=FUSED_PARAMS)
    assert set(targets.modules) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.shared_expert.gate_proj",
        "model.layers.0.mlp.shared_expert.up_proj",
        "model.layers.0.mlp.shared_expert.down_proj",
    }
    assert targets.parameters == ()


def test_moe_lora_experts_adds_routed_experts_but_never_routers():
    targets = t.select_lora_targets(MOE_MODULES, "qwen3.8-moe", lora_experts=True, parameter_names=FUSED_PARAMS)
    assert "model.layers.0.mlp.experts.0.up_proj" in targets.modules
    assert targets.parameters == (
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    )
    for name in targets.modules:
        leaf = name.rsplit(".", 1)[-1]
        assert leaf not in ("gate", "router", "shared_expert_gate")


def test_dense_family_never_selects_routed_experts_or_routers():
    targets = t.select_lora_targets(MOE_MODULES, "qwen3-32b")
    assert "model.layers.0.mlp.shared_expert.gate_proj" in targets.modules
    assert not any(".experts." in name for name in targets.modules)
    assert "model.layers.0.mlp.gate" not in targets.modules


def test_no_matching_modules_raises():
    with pytest.raises(ValueError):
        t.select_lora_targets(["lm_head", "model.embed_tokens"], "qwen3-32b")
    with pytest.raises(ValueError):
        t.select_lora_targets(DENSE_MODULES, "gpt-oss")


@pytest.mark.parametrize("family", ["qwen3-32b", "qwen3.8-moe"])
def test_targets_on_real_model_class_never_hit_router(family):
    import torch

    tokenizer = smoke_assets.build_smoke_tokenizer(family)
    model = smoke_assets.build_smoke_model(family, tokenizer, 0)
    linear = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    fused = [n for n, p in model.named_parameters() if p.ndim == 3]
    targets = t.select_lora_targets(linear, family, lora_experts=True, parameter_names=fused)
    assert targets.modules
    assert not any(n.endswith((".mlp.gate", ".shared_expert_gate")) for n in targets.modules)
    assert not any("lm_head" in n or "embed" in n for n in targets.modules)
    if family == "qwen3.8-moe":
        assert all(".experts." in p for p in targets.parameters) and targets.parameters


def test_sft_masks_tool_results_and_keeps_assistant_tokens(tokenizer):
    for record in smoke_assets.synthetic_sft_records(4, 3):
        features = t.sft_features(tokenizer, record)
        assert len(features["input_ids"]) == len(features["labels"])
        trained = labelled_text(tokenizer, features)
        masked = masked_text(tokenizer, features)
        assistants = [m for m in record["messages"] if m["role"] == "assistant"]
        for message in assistants:
            for call in message.get("tool_calls", []):
                assert call["function"]["name"] in trained
            if message["content"]:
                assert message["content"] in trained
        for message in record["messages"]:
            if message["role"] == "tool":
                assert message["content"] not in trained
                assert message["content"] in masked
            if message["role"] in ("system", "user"):
                assert message["content"] not in trained
        assert trained.count("<|im_end|>") == len(assistants)
        assert "<tool_response>" not in trained


def test_sft_labels_equal_input_ids_where_unmasked(tokenizer):
    record = smoke_assets.synthetic_sft_records(2, 1)[1]
    features = t.sft_features(tokenizer, record)
    for token_id, label in zip(features["input_ids"], features["labels"]):
        assert label in (-100, token_id)
    assert any(label != -100 for label in features["labels"])


def test_sft_parallel_tool_calls_render_both_calls_and_both_results(tokenizer):
    record = smoke_assets.synthetic_sft_records(2, 1)[1]
    text, spans = t.assistant_spans(tokenizer, record["messages"], schemas_for(record["tools"]))
    assert text.count("describe_machine") >= 2 and text.count("system_status") >= 2
    assert text.count("<tool_response>") == 2
    assert len(spans) == 2


def test_sft_arguments_must_be_dicts(tokenizer):
    record = smoke_assets.synthetic_sft_records(2, 1)[0]
    record["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"count": 3}'
    with pytest.raises(ValueError, match="arguments must be a dict"):
        t.sft_features(tokenizer, record)


def test_sft_unknown_tool_rejected(tokenizer):
    record = smoke_assets.synthetic_sft_records(1, 1)[0]
    record["tools"] = ["no_such_tool"]
    with pytest.raises(ValueError, match="unknown tools"):
        t.sft_features(tokenizer, record)


def test_assistant_spans_reject_template_with_unstable_prefix():
    tokenizer = smoke_assets.build_smoke_tokenizer("qwen3-32b")
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{{- '<|im_start|>assistant\\n' }}\n{%- endif %}", "{{- '<|im_start|>bot\\n' }}\n{%- endif %}"
    )
    record = smoke_assets.synthetic_sft_records(1, 1)[0]
    with pytest.raises(ValueError, match="prefix"):
        t.sft_features(tokenizer, record)


def test_build_sft_rows_drops_overlong_records(tokenizer):
    records = smoke_assets.synthetic_sft_records(4, 2)
    lengths = [len(t.sft_features(tokenizer, r)["input_ids"]) for r in records]
    limit = sorted(lengths)[1]
    rows, dropped = t.build_sft_rows(tokenizer, records, limit)
    assert len(rows) + len(dropped) == 4
    assert all(len(r["input_ids"]) <= limit for r in rows)
    assert len(dropped) == sum(1 for n in lengths if n > limit)
    with pytest.raises(ValueError):
        t.build_sft_rows(tokenizer, records, 10)


def test_dpo_record_to_preference_keeps_dict_arguments_and_tools():
    record = smoke_assets.synthetic_dpo_records(1, 5)[0]
    preference = t.dpo_record_to_preference(record)
    assert preference["prompt"] == record["messages_context"]
    call = preference["chosen"][0]["tool_calls"][0]["function"]
    assert isinstance(call["arguments"], dict) and call["name"] == "tail_log"
    assert [tool["function"]["name"] for tool in json.loads(preference["tools"])] == record["tools"]


def test_dpo_features_split_prompt_and_mask_tool_results(tokenizer):
    record = smoke_assets.synthetic_dpo_records(1, 5)[0]
    preference = t.dpo_record_to_preference(record)
    features = t.preference_features(tokenizer, preference)
    tools = json.loads(preference["tools"])
    chosen_text = t.render_chat(tokenizer, preference["prompt"] + preference["chosen"], tools)
    rejected_text = t.render_chat(tokenizer, preference["prompt"] + preference["rejected"], tools)
    assert tokenizer.decode(features["prompt_ids"] + features["chosen_ids"]) == tokenizer.decode(
        tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
    )
    assert tokenizer.decode(features["prompt_ids"] + features["rejected_ids"]) == tokenizer.decode(
        tokenizer(rejected_text, add_special_tokens=False)["input_ids"]
    )
    assert len(features["chosen_mask"]) == len(features["chosen_ids"])
    assert len(features["rejected_mask"]) == len(features["rejected_ids"])
    chosen_trained = tokenizer.decode([i for i, m in zip(features["chosen_ids"], features["chosen_mask"]) if m])
    chosen_env = tokenizer.decode([i for i, m in zip(features["chosen_ids"], features["chosen_mask"]) if not m])
    tool_result = preference["chosen"][1]["content"]
    assert "tail_log" in chosen_trained and preference["chosen"][2]["content"] in chosen_trained
    assert tool_result in chosen_env and tool_result not in chosen_trained
    rejected_trained = tokenizer.decode([i for i, m in zip(features["rejected_ids"], features["rejected_mask"]) if m])
    assert preference["rejected"][0]["content"] in rejected_trained
    assert features["chosen_ids"] != features["rejected_ids"]


def test_dpo_context_with_earlier_tool_turns_stays_in_prompt(tokenizer):
    sft = smoke_assets.synthetic_sft_records(3, 1)[2]
    record = smoke_assets.synthetic_dpo_records(1, 1)[0]
    record["messages_context"] = sft["messages"][:6]
    record["chosen"] = [{"role": "assistant", "content": "Log shows the alarm."}]
    record["rejected"] = [{"role": "assistant", "content": "All clear."}]
    features = t.preference_features(tokenizer, t.dpo_record_to_preference(record))
    prompt_text = tokenizer.decode(features["prompt_ids"])
    assert "pump_fault" in prompt_text and "<tool_response>" in prompt_text
    assert sum(features["chosen_mask"]) > 0 and sum(features["rejected_mask"]) > 0


def test_dpo_arguments_must_be_dicts():
    record = smoke_assets.synthetic_dpo_records(1, 5)[0]
    record["chosen"][0]["tool_calls"][0]["function"]["arguments"] = "{}"
    with pytest.raises(ValueError, match="arguments must be a dict"):
        t.dpo_record_to_preference(record)


def test_build_dpo_rows_drops_overlong_records(tokenizer):
    records = smoke_assets.synthetic_dpo_records(4, 2)
    rows, dropped = t.build_dpo_rows(tokenizer, records, 100000)
    assert len(rows) == 4 and not dropped
    with pytest.raises(ValueError):
        t.build_dpo_rows(tokenizer, records, 10)


def test_common_prefix_length():
    assert t.common_prefix_length([1, 2, 3], [1, 2, 9, 4]) == 2
    assert t.common_prefix_length([], [1]) == 0


@pytest.mark.parametrize("family", ["qwen3-32b", "qwen3.8-moe"])
def test_default_output_dir_per_family(family):
    args = t.parse_args(["--model-family", family, "--model-path", "some/model"])
    assert Path(args.output_dir) == Path(t.MODELS_ROOT) / family


def test_smoke_default_output_dir_is_separate_from_models():
    args = t.parse_args(["--model-family", "qwen3-32b", "--smoke"])
    assert Path(args.output_dir) == Path(t.SMOKE_ROOT) / "qwen3-32b"


def test_explicit_output_dir_wins(tmp_path):
    args = t.parse_args(["--model-family", "qwen3-32b", "--model-path", "m", "--output-dir", str(tmp_path)])
    assert Path(args.output_dir) == tmp_path


def test_model_path_required_without_smoke():
    with pytest.raises(SystemExit):
        t.parse_args(["--model-family", "qwen3-32b"])


def test_merge_mode_requires_adapter_and_target():
    with pytest.raises(SystemExit):
        t.parse_args(["--model-family", "qwen3-32b", "--merge-adapter", "--model-path", "m"])
    args = t.parse_args(["--merge-adapter", "--model-path", "m", "--adapter-dir", "a", "--merged-dir", "o"])
    assert args.merge_adapter and args.model_family is None


def test_model_family_required_outside_merge_mode():
    with pytest.raises(SystemExit):
        t.parse_args(["--model-path", "m"])


def test_unknown_family_and_stage_rejected():
    with pytest.raises(SystemExit):
        t.parse_args(["--model-family", "llama", "--model-path", "m"])
    with pytest.raises(SystemExit):
        t.parse_args(["--model-family", "qwen3-32b", "--model-path", "m", "--stage", "ppo"])


@pytest.mark.parametrize(
    "spec,expected",
    [("auto", ("auto", None)), ("CPU", ("cpu", None)), ("cuda", ("cuda", None)), ("cuda:1", ("cuda", 1))],
)
def test_parse_device(spec, expected):
    assert t.parse_device(spec) == expected


def test_parse_device_rejects_garbage():
    with pytest.raises(ValueError):
        t.parse_device("tpu:0")


def test_load_jsonl_roundtrip_and_missing(tmp_path):
    path = tmp_path / "x.jsonl"
    records = smoke_assets.synthetic_sft_records(2, 0)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n\n", encoding="utf-8")
    assert t.load_jsonl(path) == records
    with pytest.raises(FileNotFoundError):
        t.load_jsonl(tmp_path / "missing.jsonl")
