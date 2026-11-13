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
    f32_to_bf16_bytes,
    find_assistant_target_positions,
    patch_dense_bin,
    train_dpo_epoch,
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
