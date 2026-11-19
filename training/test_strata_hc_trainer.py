import json
from pathlib import Path

import numpy as np
import pytest
import torch

from strata_hc_trainer import (
    HC_DOWN_BYTES,
    HC_DOWN_OFFSET,
    HC_NORM_BYTES,
    HC_NORM_OFFSET,
    HC_UP_BYTES,
    HC_UP_OFFSET,
    IM_END_ID,
    IM_START_ID,
    ASSISTANT_ID,
    NEWLINE_ID,
    THINK_START_ID,
    THINK_END_ID,
    DOUBLE_NEWLINE_ID,
    Qwen4ExpHcHead,
    bf16_bytes_to_f32,
    blend_head_weights,
    export_strata_pack,
    f32_to_bf16_bytes,
    find_assistant_target_positions,
    load_strata_install_config,
    normalize_dpo_record,
    normalize_sft_record,
    patch_dense_bin,
    read_hc_layout_from_index,
    resolve_record_tools,
    select_stratified_records,
    split_train_val,
    train_dpo_epoch,
    train_dpo_with_tool_guard,
    train_sft_epoch,
)


def numpy_reference_head(r, w_norm, w_down, w_up, w_out):
    t_len = r.shape[0]
    rms = np.sqrt(np.mean(r ** 2, axis=-1, keepdims=True) + 1e-6)
    xn = (r / rms).reshape(t_len, -1) * w_norm.reshape(1, -1)
    lo = (xn @ w_down.T) * (1.0 / r.shape[1])
    lo = lo / (1.0 + np.exp(-lo))
    gate = lo @ w_up.T
    sig_gate = 1.0 / (1.0 + np.exp(-gate))
    gated = (xn * sig_gate).reshape(t_len, r.shape[1], r.shape[2])
    mixed = gated.mean(axis=1)
    return mixed @ w_out.T


def test_qwen4exp_hc_head_matches_cpp_reference():
    rng = np.random.default_rng(42)
    t_len, hc, n_embd, hc_lr, vocab = 5, 4, 64, 16, 128
    hc_dim = hc * n_embd
    r = rng.standard_normal((t_len, hc, n_embd)).astype(np.float32)
    w_norm = (1.0 + 0.1 * rng.standard_normal(hc_dim)).astype(np.float32)
    w_down = (0.05 * rng.standard_normal((hc_lr, hc_dim))).astype(np.float32)
    w_up = (0.05 * rng.standard_normal((hc_dim, hc_lr))).astype(np.float32)
    w_out = (0.05 * rng.standard_normal((vocab, n_embd))).astype(np.float32)

    expected = numpy_reference_head(r, w_norm, w_down, w_up, w_out)
    head = Qwen4ExpHcHead(
        torch.from_numpy(w_norm),
        torch.from_numpy(w_down),
        torch.from_numpy(w_up),
        torch.from_numpy(w_out),
        hc=hc,
        n_embd=n_embd,
    )
    with torch.no_grad():
        actual = head(torch.from_numpy(r)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_find_assistant_target_positions_skips_think_header_and_user_turns():
    ids = [
        IM_START_ID, 846, NEWLINE_ID, 100, 101, IM_END_ID, NEWLINE_ID,
        IM_START_ID, ASSISTANT_ID, NEWLINE_ID, THINK_START_ID, DOUBLE_NEWLINE_ID, THINK_END_ID, DOUBLE_NEWLINE_ID,
        501, 502, 503, IM_END_ID, NEWLINE_ID,
    ]
    positions, targets = find_assistant_target_positions(ids)
    assert targets == [501, 502, 503, IM_END_ID]
    assert [ids[p + 1] for p in positions] == targets


def test_sft_and_dpo_training_improve_objective():
    torch.manual_seed(7)
    hc, n_embd, hc_lr, vocab = 4, 32, 8, 64
    hc_dim = hc * n_embd
    head = Qwen4ExpHcHead(
        torch.ones(hc_dim),
        0.02 * torch.randn(hc_lr, hc_dim),
        0.02 * torch.randn(hc_dim, hc_lr),
        0.1 * torch.randn(vocab, n_embd),
        hc=hc,
        n_embd=n_embd,
    )
    base_head = head.clone_frozen()

    r_sft = torch.randn(24, hc, n_embd)
    y_sft = torch.randint(0, vocab, (24,))
    sft_metrics = train_sft_epoch(head, base_head, [(r_sft, y_sft)], lr=5e-3, steps=6, batch_size=12)
    assert sft_metrics["final_loss"] < sft_metrics["initial_loss"]

    r_chosen = torch.randn(10, hc, n_embd)
    y_chosen = torch.randint(0, vocab, (10,))
    r_rejected = r_chosen.clone()
    y_rejected = (y_chosen + 1) % vocab
    dpo_pairs = [{"r_chosen": r_chosen, "y_chosen": y_chosen, "r_rejected": r_rejected, "y_rejected": y_rejected}]
    dpo_metrics = train_dpo_epoch(head, base_head, dpo_pairs, lr=5e-3, beta=0.2, steps=6)
    assert dpo_metrics["final_margin"] > dpo_metrics["initial_margin"]
    assert dpo_metrics["final_loss"] < dpo_metrics["initial_loss"]


def test_patch_dense_bin_preserves_offsets_and_trailing_bytes(tmp_path):
    total_size = HC_UP_OFFSET + HC_UP_BYTES + 1024
    src = tmp_path / "dense_src.bin"
    dst = tmp_path / "dense_dst.bin"
    rng = np.random.default_rng(9)
    raw = bytearray(rng.integers(0, 256, size=total_size, dtype=np.uint8).tobytes())
    src.write_bytes(raw)

    w_down = np.full((320, 10240), 0.25, dtype=np.float32)
    w_norm = np.full((10240,), 1.5, dtype=np.float32)
    w_up = np.full((10240, 320), -0.125, dtype=np.float32)

    patch_dense_bin(src, dst, w_norm, w_down, w_up)
    patched = dst.read_bytes()
    assert len(patched) == total_size
    assert patched[HC_UP_OFFSET + HC_UP_BYTES :] == bytes(raw[HC_UP_OFFSET + HC_UP_BYTES :])

    read_down = bf16_bytes_to_f32(patched[HC_DOWN_OFFSET : HC_DOWN_OFFSET + HC_DOWN_BYTES]).reshape(320, 10240)
    read_norm = np.frombuffer(patched[HC_NORM_OFFSET : HC_NORM_OFFSET + HC_NORM_BYTES], dtype=np.float32)
    read_up = bf16_bytes_to_f32(patched[HC_UP_OFFSET : HC_UP_OFFSET + HC_UP_BYTES]).reshape(10240, 320)
    np.testing.assert_allclose(read_down, w_down, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(read_norm, w_norm, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(read_up, w_up, rtol=1e-2, atol=1e-3)


def test_blend_head_weights_interpolates_exact_midpoint():
    w_norm_base = torch.full((64,), 1.0)
    w_down_base = torch.full((8, 64), 0.2)
    w_up_base = torch.full((64, 8), -0.4)
    w_out = torch.zeros((32, 16), dtype=torch.bfloat16)
    base_head = Qwen4ExpHcHead(w_norm_base, w_down_base, w_up_base, w_out, hc=4, n_embd=16)
    ckpt = {
        "output_hc_norm.weight": torch.full((64,), 2.0),
        "output_hc_down.weight": torch.full((8, 64), 0.6),
        "output_hc_up.weight": torch.full((64, 8), 0.4),
    }
    n_50, d_50, u_50 = blend_head_weights(base_head, ckpt, alpha=0.5)
    np.testing.assert_allclose(n_50, 1.5, rtol=1e-6)
    np.testing.assert_allclose(d_50, 0.4, rtol=1e-6)
    np.testing.assert_allclose(u_50, 0.0, atol=1e-6)


def test_resolve_record_tools_supports_missing_dicts_and_names():
    assert resolve_record_tools({}) == []
    assert resolve_record_tools({"tools": None}) == []
    inline_schema = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]
    assert resolve_record_tools({"tools": inline_schema}) == inline_schema
    named = resolve_record_tools({"tools": ["describe_machine"]})
    assert len(named) == 1
    assert named[0]["function"]["name"] == "describe_machine"


def test_normalize_sft_and_dpo_records_handle_generic_and_legacy():
    sft_msgs, sft_tools = normalize_sft_record({
        "system": "Be brief.",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
        ],
    })
    assert sft_msgs[0] == {"role": "system", "content": "Be brief."}
    assert sft_msgs[2]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert sft_tools == []

    ctx1, ch1, rj1, t1 = normalize_dpo_record({
        "system": "Sys",
        "prompt": "Q?",
        "chosen": "Good",
        "rejected": "Bad",
    })
    assert ctx1 == [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Q?"}]
    assert ch1 == [{"role": "assistant", "content": "Good"}]
    assert rj1 == [{"role": "assistant", "content": "Bad"}]
    assert t1 == []

    ctx2, ch2, rj2, t2 = normalize_dpo_record({
        "tools": ["describe_machine"],
        "messages_context": [{"role": "user", "content": "check"}],
        "chosen": [{"role": "assistant", "content": "ok"}],
        "rejected": [{"role": "assistant", "content": "nope"}],
    })
    assert ctx2 == [{"role": "user", "content": "check"}]
    assert ch2 == [{"role": "assistant", "content": "ok"}]
    assert rj2 == [{"role": "assistant", "content": "nope"}]
    assert t2[0]["function"]["name"] == "describe_machine"


def test_select_stratified_records_handles_dict_tools_and_ungrouped():
    rows = [
        {"prompt": "p1", "tools": [{"type": "function", "function": {"name": "a"}}]},
        {"prompt": "p2"},
        {"prompt": [{"role": "user", "content": "p3"}], "tools": [{"type": "function", "function": {"name": "b"}}]},
        {"messages": [{"role": "user", "content": "p4"}]},
        {"messages": [{"role": "user", "content": "p5"}]},
        {"messages": [{"role": "user", "content": "p6"}]},
    ]
    kept_all = select_stratified_records(rows, "task_family", None, default_per_group=4)
    assert len(kept_all) == 6
    kept_zero = select_stratified_records(rows, "task_family", 0, default_per_group=4)
    assert len(kept_zero) == 6
    capped = select_stratified_records(rows, "task_family", 2, default_per_group=4)
    assert len(capped) == 2


def test_split_train_val_is_deterministic_and_non_empty():
    rows = [{"id": i} for i in range(10)]
    tr1, va1 = split_train_val(rows, val_ratio=0.2, seed=123)
    tr2, va2 = split_train_val(rows, val_ratio=0.2, seed=123)
    assert tr1 == tr2 and va1 == va2
    assert len(tr1) == 8 and len(va1) == 2
    assert {r["id"] for r in tr1}.isdisjoint({r["id"] for r in va1})

    tr_one, va_one = split_train_val([{"id": 0}], val_ratio=0.2, seed=123)
    assert len(tr_one) == 1 and va_one == []


def test_load_strata_install_config_and_index_layout(tmp_path):
    cfg_path = tmp_path / "strata-custom.json"
    cfg_path.write_text(json.dumps({
        "args": [
            "--pack", "packs/q2_0",
            "--native", "models/shard1.gguf",
            "--ple-gguf", "models/shard2.gguf",
            "--expert-profile", "data/expert-profile.bin",
            "--mtp", "mtp/rt",
        ],
        "cwd": "terminal/current",
        "model_name": "qwen3.8-flash-next-q2_0",
    }), encoding="utf-8")
    parsed = load_strata_install_config(cfg_path)
    assert parsed["pack_src"] == Path("packs/q2_0")
    assert parsed["shard1_gguf"] == Path("models/shard1.gguf")
    assert parsed["shard2_gguf"] == Path("models/shard2.gguf")
    assert parsed["expert_profile"] == Path("data/expert-profile.bin")
    assert parsed["mtp_dir"] == Path("mtp/rt")
    assert parsed["strata_root"] == Path("terminal/current")
    assert parsed["model_name"] == "qwen3.8-flash-next-q2_0"

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "index.txt").write_text(
        "output_hc_down.weight 0 4 0 6553600 0 6553600 10240 320\n"
        "output_hc_norm.weight 0 2 6553600 40960 6553600 40960 10240 0\n"
        "output_hc_up.weight 0 4 6594560 6553600 6594560 6553600 320 10240\n",
        encoding="utf-8",
    )
    layout = read_hc_layout_from_index(pack_dir)
    assert layout == {
        "down": (0, HC_DOWN_BYTES),
        "norm": (6553600, HC_NORM_BYTES),
        "up": (6594560, HC_UP_BYTES),
    }

    (pack_dir / "index.txt").write_text("output_hc_norm.weight 0 2 6553600 999\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_hc_layout_from_index(pack_dir)


def test_training_history_and_tool_guard_dpo():
    torch.manual_seed(11)
    hc, n_embd, hc_lr, vocab = 4, 32, 8, 64
    hc_dim = hc * n_embd
    head = Qwen4ExpHcHead(
        torch.ones(hc_dim),
        0.02 * torch.randn(hc_lr, hc_dim),
        0.02 * torch.randn(hc_dim, hc_lr),
        0.1 * torch.randn(vocab, n_embd),
        hc=hc,
        n_embd=n_embd,
    )
    base_head = head.clone_frozen()

    r_sft = torch.randn(16, hc, n_embd)
    y_sft = torch.randint(0, vocab, (16,))
    sft_res = train_sft_epoch(
        head, base_head, [(r_sft, y_sft)],
        lr=5e-3, steps=4, batch_size=8,
        val_batches=[(r_sft[:8], y_sft[:8])], log_every=1, val_every=2,
    )
    assert len(sft_res["history"]) == 5
    assert "val_sft_loss" in sft_res["history"][0]
    assert "val_sft_loss" in sft_res["history"][2]

    r_ch = torch.randn(8, hc, n_embd)
    y_ch = torch.randint(0, vocab, (8,))
    pairs = [{"r_chosen": r_ch, "y_chosen": y_ch, "r_rejected": r_ch.clone(), "y_rejected": (y_ch + 1) % vocab}]
    r_tool = torch.randn(6, hc, n_embd)
    dpo_res = train_dpo_with_tool_guard(
        head, base_head, pairs, r_tool_train=r_tool,
        lr=5e-3, beta=0.2, anchor_coef=0.1, tool_kl_coef=5.0, steps=4,
        val_pairs=pairs, val_sft_batches=[(r_sft[:8], y_sft[:8])], log_every=1, val_every=2,
    )
    assert dpo_res["final_margin"] > dpo_res["initial_margin"]
    assert len(dpo_res["history"]) == 5
    assert dpo_res["history"][-1]["train_kl_tool"] >= 0.0
    assert dpo_res["history"][-1]["val_dpo_acc"] == 1.0


def test_export_strata_pack_rewrites_custom_install_cfg(tmp_path):
    pack_src = tmp_path / "pack_src"
    pack_dst = tmp_path / "pack_dst"
    pack_src.mkdir()
    (pack_src / "tokenizer").mkdir()
    (pack_src / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
    (pack_src / "index.txt").write_text(
        "output_hc_down.weight 0 4 0 6553600 0 6553600 10240 320\n"
        "output_hc_norm.weight 0 2 6553600 40960 6553600 40960 10240 0\n"
        "output_hc_up.weight 0 4 6594560 6553600 6594560 6553600 320 10240\n",
        encoding="utf-8",
    )
    (pack_src / "experts.bin").write_bytes(b"EXPERTS")
    (pack_src / "dense.bin").write_bytes(bytes(HC_UP_OFFSET + HC_UP_BYTES + 64))

    cfg_src = tmp_path / "strata-q2_0.json"
    cfg_dst = tmp_path / "strata-q2_0-tuned.json"
    cfg_src.write_text(json.dumps({
        "args": ["--pack", str(pack_src)],
        "tokenizer": str(pack_src / "tokenizer"),
        "model_name": "qwen3.8-flash-next-q2_0",
    }), encoding="utf-8")

    head = Qwen4ExpHcHead(
        torch.ones(10240),
        torch.zeros(320, 10240),
        torch.zeros(10240, 320),
        torch.zeros(16, 2560, dtype=torch.bfloat16),
    )
    out = export_strata_pack(
        head,
        pack_dst=pack_dst,
        pack_src=pack_src,
        install_cfg_src=cfg_src,
        install_cfg_dst=cfg_dst,
        model_name="qwen3.8-flash-next-q2_0-custom",
    )
    assert out == pack_dst
    assert (pack_dst / "experts.bin").stat().st_ino == (pack_src / "experts.bin").stat().st_ino
    updated = json.loads(cfg_dst.read_text(encoding="utf-8"))
    assert updated["args"][1] == str(pack_dst)
    assert updated["tokenizer"] == str(pack_dst / "tokenizer")
    assert updated["model_name"] == "qwen3.8-flash-next-q2_0-custom"


def test_example_jsonl_files_normalize_cleanly():
    examples_dir = Path(__file__).resolve().parent / "examples"
    sft_rows = [json.loads(line) for line in (examples_dir / "sft_sample.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    dpo_rows = [json.loads(line) for line in (examples_dir / "dpo_sample.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(sft_rows) == 4 and len(dpo_rows) == 4
    for row in sft_rows:
        msgs, tools = normalize_sft_record(row)
        assert msgs and isinstance(tools, list)
    for row in dpo_rows:
        ctx, ch, rj, tools = normalize_dpo_record(row)
        assert ctx and ch and rj and isinstance(tools, list)
