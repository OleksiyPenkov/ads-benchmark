import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tool_schemas import schemas_for

_LOCAL_STRATA = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "Strata"
STRATA_ROOT = Path(os.environ.get("STRATA_ROOT") or (_LOCAL_STRATA / "terminal" / "current"))
STRATA_EXE = Path(os.environ.get("STRATA_EXE") or (_LOCAL_STRATA / "components" / "engine" / "current" / "strata.exe"))
STRATA_DATA = Path(os.environ.get("STRATA_DATA_DIR") or (HERE.parent / "computed" / "strata-data"))
PACK_SRC = STRATA_DATA / "packs" / "iq3_s"
PACK_DST = STRATA_DATA / "packs" / "iq3_s_ads"
SHARD1_GGUF = STRATA_DATA / "models" / "IQ3_S" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
SHARD2_GGUF = STRATA_DATA / "models" / "IQ3_S" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
EXPERT_PROFILE = STRATA_ROOT / "data" / "expert-profile.bin"
MTP_DIR = STRATA_DATA / "mtp" / "rt"
CHAT_TEMPLATE_PATH = STRATA_ROOT / "serve" / "chat_template.jinja"
INSTALL_CFG_SRC = _LOCAL_STRATA / "install" / "strata-iq3_s.json"
INSTALL_CFG_DST = _LOCAL_STRATA / "install" / "strata-iq3_s-ads.json"

COMPUTED_ROOT = Path(os.environ.get("ADS_BENCH_COMPUTED_DIR") or (HERE.parent / "computed"))
DATASET_DIR = COMPUTED_ROOT / "datasets"
RESIDUAL_DIR = COMPUTED_ROOT / "residuals"
MODEL_OUT_DIR = COMPUTED_ROOT / "models" / "qwen3.8-moe"

HC_DOWN_OFFSET = 0
HC_DOWN_BYTES = 6553600
HC_NORM_OFFSET = 6553600
HC_NORM_BYTES = 40960
HC_UP_OFFSET = 6594560
HC_UP_BYTES = 6553600
ROW_BYTES_BF16 = 4 * 2560 * 2

IM_START_ID = 248045
IM_END_ID = 248046
ASSISTANT_ID = 74455
USER_ID = 846
NEWLINE_ID = 198
DOUBLE_NEWLINE_ID = 271
THINK_START_ID = 248068
THINK_END_ID = 248069

TOOL_CALL_ID = 248058


def split_tool_and_text_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    tool_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    text_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for r, y in batches:
        turn_start = 0
        for i in range(len(y)):
            if int(y[i].item()) == IM_END_ID or i == len(y) - 1:
                r_turn = r[turn_start : i + 1]
                y_turn = y[turn_start : i + 1]
                if len(y_turn) > 0:
                    if int(y_turn[0].item()) == TOOL_CALL_ID or TOOL_CALL_ID in y_turn.tolist():
                        tool_batches.append((r_turn, y_turn))
                    else:
                        text_batches.append((r_turn, y_turn))
                turn_start = i + 1
    return tool_batches, text_batches



def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).copy()


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding = 0x7FFF + ((u32 >> 16) & 1)
    u16 = ((u32 + rounding) >> 16).astype(np.uint16)
    return u16.tobytes()


class Qwen4ExpHcHead(nn.Module):
    def __init__(
        self,
        w_norm: torch.Tensor,
        w_down: torch.Tensor,
        w_up: torch.Tensor,
        w_out: torch.Tensor,
        hc: int = 4,
        n_embd: int = 2560,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hc = hc
        self.n_embd = n_embd
        self.hc_dim = hc * n_embd
        self.eps = eps
        self.w_norm = nn.Parameter(w_norm.detach().clone().float())
        self.w_down = nn.Parameter(w_down.detach().clone().float())
        self.w_up = nn.Parameter(w_up.detach().clone().float())
        self.register_buffer("w_out", w_out.detach().clone(), persistent=False)

    def clone_frozen(self) -> "Qwen4ExpHcHead":
        copy = Qwen4ExpHcHead(
            self.w_norm.data,
            self.w_down.data,
            self.w_up.data,
            self.w_out,
            hc=self.hc,
            n_embd=self.n_embd,
            eps=self.eps,
        )
        for p in copy.parameters():
            p.requires_grad_(False)
        return copy

    def forward_mixed(self, r: torch.Tensor) -> torch.Tensor:
        rf = r.float().reshape(-1, self.hc, self.n_embd)
        rms = torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        xn = (rf * rms).reshape(-1, self.hc_dim) * self.w_norm
        lo = F.silu((xn @ self.w_down.t()) * (1.0 / float(self.hc)))
        gate = lo @ self.w_up.t()
        gated = (xn * torch.sigmoid(gate)).reshape(-1, self.hc, self.n_embd)
        return gated.mean(dim=1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        mixed = self.forward_mixed(r)
        return F.linear(mixed.to(self.w_out.dtype), self.w_out).float()

    def token_logprobs(self, r: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(r)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)

    def anchor_loss(self, ref: "Qwen4ExpHcHead") -> torch.Tensor:
        return (
            (self.w_norm - ref.w_norm).pow(2).mean()
            + (self.w_down - ref.w_down).pow(2).mean()
            + (self.w_up - ref.w_up).pow(2).mean()
        )


def find_assistant_target_positions(ids: list[int], min_tok_idx: int = 0) -> tuple[list[int], list[int]]:
    positions: list[int] = []
    targets: list[int] = []
    n = len(ids)
    i = max(0, min_tok_idx)
    while i < n - 3:
        if ids[i] == IM_START_ID and ids[i + 1] == ASSISTANT_ID and ids[i + 2] == NEWLINE_ID:
            start = i + 3
            end = start
            while end < n and ids[end] != IM_END_ID:
                end += 1
            if end < n and ids[end] == IM_END_ID:
                content_start = start
                for k in range(start, end):
                    if ids[k] == THINK_END_ID:
                        content_start = k + 1
                        if content_start < end and ids[content_start] in (NEWLINE_ID, DOUBLE_NEWLINE_ID):
                            content_start += 1
                        break
                for tok_idx in range(content_start, end + 1):
                    if tok_idx >= 1 and tok_idx >= min_tok_idx:
                        positions.append(tok_idx - 1)
                        targets.append(ids[tok_idx])
                i = end + 1
                continue
        i += 1
    return positions, targets


def eval_sft_batches(head: Qwen4ExpHcHead, batches: list[tuple[torch.Tensor, torch.Tensor]], max_tokens: int = 512) -> float:
    all_r = torch.cat([b[0].float() for b in batches], dim=0)[:max_tokens]
    all_y = torch.cat([b[1].long() for b in batches], dim=0)[:max_tokens]
    losses = []
    with torch.no_grad():
        for i in range(0, len(all_y), 128):
            logits = head(all_r[i : i + 128])
            losses.append(float(F.cross_entropy(logits, all_y[i : i + 128]).item()))
    return float(np.mean(losses)) if losses else 0.0


def train_sft_epoch(
    head: Qwen4ExpHcHead,
    base_head: Qwen4ExpHcHead,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    lr: float = 5e-5,
    anchor_coef: float = 5e-2,
    kl_coef: float = 0.1,
    steps: int = 20,
    batch_size: int = 128,
) -> dict:
    all_r = torch.cat([b[0].float() for b in batches], dim=0)
    all_y = torch.cat([b[1].long() for b in batches], dim=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    eval_n = min(len(all_y), max(batch_size * 2, 64))
    initial_loss = eval_sft_batches(head, [(all_r[:eval_n], all_y[:eval_n])], max_tokens=eval_n)
    n_total = len(all_y)

    for _ in range(steps):
        idx = torch.randint(0, n_total, (min(batch_size, n_total),))
        r_b = all_r[idx]
        y_b = all_y[idx]
        optimizer.zero_grad()
        logits = head(r_b)
        ce = F.cross_entropy(logits, y_b)
        with torch.no_grad():
            ref_logits = base_head(r_b)
            ref_probs = F.softmax(ref_logits, dim=-1)
        kl = F.kl_div(F.log_softmax(logits, dim=-1), ref_probs, reduction="batchmean")
        loss = ce + kl_coef * kl + anchor_coef * head.anchor_loss(base_head)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

    final_loss = eval_sft_batches(head, [(all_r[:eval_n], all_y[:eval_n])], max_tokens=eval_n)
    return {"initial_loss": initial_loss, "final_loss": final_loss, "steps": steps, "tokens": int(n_total)}


def eval_dpo_pairs(
    head: Qwen4ExpHcHead,
    base_head: Qwen4ExpHcHead,
    pairs: list[dict],
    beta: float = 0.1,
) -> dict:
    losses, margins, accs = [], [], []
    with torch.no_grad():
        for p in pairs:
            rc = p["r_chosen"].float()
            yc = p["y_chosen"].long()
            rr = p["r_rejected"].float()
            yr = p["y_rejected"].long()
            ref_c = base_head.token_logprobs(rc, yc).mean()
            ref_r = base_head.token_logprobs(rr, yr).mean()
            pi_c = head.token_logprobs(rc, yc).mean()
            pi_r = head.token_logprobs(rr, yr).mean()
            margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(margin)
            losses.append(float(loss.item()))
            margins.append(float(margin.item()))
            accs.append(1.0 if margin.item() > 0 else 0.0)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "margin": float(np.mean(margins)) if margins else 0.0,
        "accuracy": float(np.mean(accs)) if accs else 0.0,
    }


def train_dpo_epoch(
    head: Qwen4ExpHcHead,
    base_head: Qwen4ExpHcHead,
    pairs: list[dict],
    lr: float = 2e-5,
    beta: float = 0.1,
    anchor_coef: float = 5e-2,
    steps: int = 20,
) -> dict:
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    cached_pairs = []
    for p in pairs:
        rc = p["r_chosen"].float()
        yc = p["y_chosen"].long()
        rr = p["r_rejected"].float()
        yr = p["y_rejected"].long()
        with torch.no_grad():
            ref_c = base_head.token_logprobs(rc, yc).mean()
            ref_r = base_head.token_logprobs(rr, yr).mean()
        cached_pairs.append((rc, yc, rr, yr, ref_c, ref_r))

    def quick_eval() -> tuple[float, float, float]:
        losses, margins, accs = [], [], []
        with torch.no_grad():
            for rc, yc, rr, yr, ref_c, ref_r in cached_pairs[: min(len(cached_pairs), 24)]:
                pi_c = head.token_logprobs(rc, yc).mean()
                pi_r = head.token_logprobs(rr, yr).mean()
                margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
                loss = -F.logsigmoid(margin)
                losses.append(float(loss.item()))
                margins.append(float(margin.item()))
                accs.append(1.0 if margin.item() > 0 else 0.0)
        return float(np.mean(losses)), float(np.mean(margins)), float(np.mean(accs))

    init_loss, init_margin, init_acc = quick_eval()
    for step in range(steps):
        rc, yc, rr, yr, ref_c, ref_r = cached_pairs[step % len(cached_pairs)]
        optimizer.zero_grad()
        pi_c = head.token_logprobs(rc, yc).mean()
        pi_r = head.token_logprobs(rr, yr).mean()
        margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        sft_nll = -pi_c
        loss = -F.logsigmoid(margin) + 0.15 * sft_nll + anchor_coef * head.anchor_loss(base_head)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

    final_loss, final_margin, final_acc = quick_eval()
    return {
        "initial_loss": init_loss,
        "final_loss": final_loss,
        "initial_margin": init_margin,
        "final_margin": final_margin,
        "initial_accuracy": init_acc,
        "final_accuracy": final_acc,
        "steps": steps,
    }


def patch_dense_bin(
    src_dense: Path,
    dst_dense: Path,
    w_norm: np.ndarray,
    w_down: np.ndarray,
    w_up: np.ndarray,
) -> None:
    src_dense = Path(src_dense)
    dst_dense = Path(dst_dense)
    dst_dense.parent.mkdir(parents=True, exist_ok=True)
    if src_dense.resolve() != dst_dense.resolve():
        shutil.copyfile(src_dense, dst_dense)

    down_bytes = f32_to_bf16_bytes(w_down.reshape(320, 10240))
    norm_bytes = np.ascontiguousarray(w_norm.reshape(10240), dtype=np.float32).tobytes()
    up_bytes = f32_to_bf16_bytes(w_up.reshape(10240, 320))
    assert len(down_bytes) == HC_DOWN_BYTES
    assert len(norm_bytes) == HC_NORM_BYTES
    assert len(up_bytes) == HC_UP_BYTES

    with open(dst_dense, "r+b") as f:
        f.seek(HC_DOWN_OFFSET)
        f.write(down_bytes)
        f.seek(HC_NORM_OFFSET)
        f.write(norm_bytes)
        f.seek(HC_UP_OFFSET)
        f.write(up_bytes)


def load_strata_tokenizer_and_template():
    sys.path.insert(0, str(STRATA_ROOT / "tools"))
    import jinja2
    import strata_tokenizer as ST

    tpath = PACK_SRC / "tokenizer"
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
    tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
    tmpl = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"]).from_string(
        CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    return tok, tmpl


def load_head_from_strata(device: str = "cpu", pack_src: Path = PACK_SRC) -> Qwen4ExpHcHead:
    sys.path.insert(0, str(STRATA_ROOT / "third_party" / "llama.cpp" / "gguf-py"))
    sys.path.insert(0, str(STRATA_ROOT / "ref"))
    from load import Gguf

    pack_src = Path(pack_src)
    base_slice = pack_src / "output_hc_base.bin"
    dense_file = pack_src / "dense.bin"
    if not base_slice.is_file() and dense_file.is_file():
        with open(dense_file, "rb") as f:
            base_slice.write_bytes(f.read(HC_UP_OFFSET + HC_UP_BYTES))
    src_file = base_slice if base_slice.is_file() else dense_file
    with open(src_file, "rb") as f:
        raw_head = f.read(HC_UP_OFFSET + HC_UP_BYTES)
    w_down = bf16_bytes_to_f32(raw_head[HC_DOWN_OFFSET : HC_DOWN_OFFSET + HC_DOWN_BYTES]).reshape(320, 10240)
    w_norm = np.frombuffer(raw_head[HC_NORM_OFFSET : HC_NORM_OFFSET + HC_NORM_BYTES], dtype=np.float32).copy()
    w_up = bf16_bytes_to_f32(raw_head[HC_UP_OFFSET : HC_UP_OFFSET + HC_UP_BYTES]).reshape(10240, 320)

    gguf = Gguf(str(SHARD1_GGUF))
    w_out = gguf.tensor("output.weight")
    head = Qwen4ExpHcHead(
        torch.from_numpy(w_norm),
        torch.from_numpy(w_down),
        torch.from_numpy(w_up),
        torch.from_numpy(w_out).to(torch.bfloat16),
    )
    return head.to(device)
def blend_head_weights(
    base_head: Qwen4ExpHcHead,
    ckpt: dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = float(alpha)
    w_norm = (1.0 - a) * base_head.w_norm.detach().cpu().float() + a * ckpt["output_hc_norm.weight"].detach().cpu().float()
    w_down = (1.0 - a) * base_head.w_down.detach().cpu().float() + a * ckpt["output_hc_down.weight"].detach().cpu().float()
    w_up = (1.0 - a) * base_head.w_up.detach().cpu().float() + a * ckpt["output_hc_up.weight"].detach().cpu().float()
    return w_norm.numpy(), w_down.numpy(), w_up.numpy()


def export_strata_pack(
    head: Qwen4ExpHcHead,
    pack_dst: Path = PACK_DST,
    also_patch_src: bool = False,
    pack_src: Path = PACK_SRC,
) -> Path:
    pack_src = Path(pack_src)
    pack_dst = Path(pack_dst)
    pack_dst.mkdir(parents=True, exist_ok=True)
    for fn in ("index.txt", "native_experts.txt"):
        if (pack_src / fn).exists() and (pack_src / fn).resolve() != (pack_dst / fn).resolve():
            shutil.copy2(pack_src / fn, pack_dst / fn)
    tok_dst = pack_dst / "tokenizer"
    if not tok_dst.exists() and (pack_src / "tokenizer").exists():
        shutil.copytree(pack_src / "tokenizer", tok_dst)
    exp_dst = pack_dst / "experts.bin"
    if not exp_dst.exists() and (pack_src / "experts.bin").exists():
        os.link(pack_src / "experts.bin", exp_dst)

    w_norm = head.w_norm.detach().cpu().float().numpy()
    w_down = head.w_down.detach().cpu().float().numpy()
    w_up = head.w_up.detach().cpu().float().numpy()
    patch_dense_bin(pack_src / "dense.bin", pack_dst / "dense.bin", w_norm, w_down, w_up)
    if also_patch_src:
        patch_dense_bin(pack_src / "dense.bin", pack_src / "dense.bin", w_norm, w_down, w_up)

    if pack_src.resolve() == PACK_SRC.resolve() and INSTALL_CFG_SRC.exists():
        cfg = json.loads(INSTALL_CFG_SRC.read_text(encoding="utf-8"))
        for key in ("args", "args_written"):
            if key in cfg and "--pack" in cfg[key]:
                idx = cfg[key].index("--pack")
                cfg[key][idx + 1] = str(pack_dst)
        cfg["tokenizer"] = str(pack_dst / "tokenizer")
        cfg["model_name"] = "qwen3.8-flash-next-iq3_s-ads"
        INSTALL_CFG_DST.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return pack_dst


def restore_base_pack(pack_src: Path = PACK_SRC) -> None:
    pack_src = Path(pack_src)
    base_slice = pack_src / "output_hc_base.bin"
    if not base_slice.is_file():
        raise FileNotFoundError(f"Missing base slice backup: {base_slice}")
    raw = base_slice.read_bytes()
    targets = [pack_src / "dense.bin"]
    if pack_src.resolve() == PACK_SRC.resolve():
        targets.append(PACK_DST / "dense.bin")
    for target in targets:
        if target.is_file():
            with open(target, "r+b") as f:
                f.seek(0)
                f.write(raw)

def select_stratified_records(records: list[dict], key_field: str, per_group: int) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key_field, "default")].append(r)
    selected = []
    rng = random.Random(20261110)
    for k in sorted(groups.keys()):
        items = list(groups[k])
        rng.shuffle(items)
        selected.extend(items[:per_group])
    selected.sort(
        key=lambda r: (
            tuple(r.get("tools", ())),
            (r.get("messages") or r.get("messages_context"))[0].get("content", ""),
        )
    )
    return selected


def extract_residuals_with_strata(
    sft_train_per_family: int = 4,
    sft_val_per_family: int = 1,
    dpo_train_per_pattern: int = 5,
    dpo_val_per_pattern: int = 1,
    out_dir: Path = RESIDUAL_DIR,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_pt = out_dir / "strata_hc_residuals.pt"
    tok, tmpl = load_strata_tokenizer_and_template()

    def load_rows(fn: str) -> list[dict]:
        return [
            json.loads(line)
            for line in (DATASET_DIR / fn).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    sft_train_raw = select_stratified_records(load_rows("sft_train.jsonl"), "task_family", sft_train_per_family)
    sft_val_raw = select_stratified_records(load_rows("sft_val.jsonl"), "task_family", sft_val_per_family)
    dpo_train_raw = select_stratified_records(load_rows("dpo_train.jsonl"), "pattern", dpo_train_per_pattern)
    dpo_val_raw = select_stratified_records(load_rows("dpo_val.jsonl"), "pattern", dpo_val_per_pattern)

    dump_bin = out_dir / "temp_r_dump.bin"
    serve_log = out_dir / "strata_extract.log"
    if dump_bin.exists():
        dump_bin.unlink()

    env = dict(os.environ, STRATA_PREFILL_DUMP_R_ALL=str(dump_bin))
    cmd = [
        str(STRATA_EXE),
        "--serve",
        "--short-read", "0",
        "--prompt-cache", "12",
        "--conversation-cache-slots", "12",
        "--pack", str(PACK_SRC),
        "--native", str(SHARD1_GGUF),
        "--ple-gguf", str(SHARD2_GGUF),
        "--expert-profile", str(EXPERT_PROFILE),
        "--expert-cache", "2048",
        "--spec", "4",
        "--mtp", str(MTP_DIR),
        "--prefill", "4096",
        "--max-context", "8192",
        "--mmap-experts",
    ]

    print(f"[extract] launching strata.exe --serve for residual extraction...", flush=True)
    with open(serve_log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=lf, text=True, env=env)
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("strata.exe exited before READY")
                if line.startswith("READY"):
                    break

            prev_size = 0

            def run_gen(ids: list[int], extra_keys: str = "") -> tuple[int, bytes]:
                nonlocal prev_size
                key_part = (extra_keys.strip() + " ") if extra_keys.strip() else ""
                proc.stdin.write(f"GEN 1 {key_part}" + ",".join(map(str, ids)) + "\n")
                proc.stdin.flush()
                reused = 0
                while True:
                    line = proc.stdout.readline().strip()
                    if line.startswith("REUSED "):
                        reused = int(line.split()[1])
                    elif line.startswith("ERR"):
                        raise RuntimeError(f"strata GEN error: {line}")
                    elif line.startswith("DONE"):
                        cur_size = dump_bin.stat().st_size if dump_bin.exists() else 0
                        delta = cur_size - prev_size
                        with open(dump_bin, "rb") as rf:
                            rf.seek(prev_size)
                            chunk = rf.read(delta)
                        prev_size = cur_size
                        return reused, chunk

            pinned_headers = set()

            def ensure_pinned(sys_msg: dict, tool_names: list[str]) -> None:
                key = (tuple(tool_names), sys_msg.get("content", ""))
                if key in pinned_headers:
                    return
                tools = schemas_for(tool_names)
                sys_only = tmpl.render(messages=[sys_msg], tools=tools, add_generation_prompt=False)
                sys_ids = tok.encode(sys_only, parse_special=True)
                warm_msgs = [sys_msg, {"role": "user", "content": "ping"}]
                warm_ids = tok.encode(tmpl.render(messages=warm_msgs, tools=tools, add_generation_prompt=False), parse_special=True)
                run_gen(warm_ids, f"pin={len(sys_ids)}")
                pinned_headers.add(key)

            def extract_sft_split(records: list[dict], split_name: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
                out_batches = []
                t0 = time.perf_counter()
                for idx, rec in enumerate(records):
                    tools = schemas_for(rec["tools"])
                    rendered = tmpl.render(messages=rec["messages"], tools=tools, add_generation_prompt=False)
                    ids = tok.encode(rendered, parse_special=True)
                    if len(ids) > 4000:
                        continue
                    ensure_pinned(rec["messages"][0], rec["tools"])
                    positions, targets = find_assistant_target_positions(ids)
                    if not positions:
                        continue
                    reused, chunk = run_gen(ids, "ckpt=0")
                    delta_rows = len(chunk) // ROW_BYTES_BF16
                    if reused > positions[0] or reused + delta_rows <= positions[-1]:
                        continue
                    u16 = np.frombuffer(chunk, dtype=np.uint16).reshape(delta_rows, 4, 2560)
                    rel_pos = [p - reused for p in positions]
                    r_slice = bf16_bytes_to_f32(u16[rel_pos].tobytes()).reshape(len(positions), 4, 2560)
                    out_batches.append((
                        torch.from_numpy(r_slice).to(torch.bfloat16),
                        torch.tensor(targets, dtype=torch.int64),
                    ))
                    if (idx + 1) % 5 == 0 or idx + 1 == len(records):
                        print(f"[extract:{split_name}] {idx+1}/{len(records)} ({time.perf_counter()-t0:.1f}s)", flush=True)
                return out_batches

            def extract_dpo_split(records: list[dict], split_name: str) -> list[dict]:
                out_pairs = []
                t0 = time.perf_counter()
                for idx, rec in enumerate(records):
                    tools = schemas_for(rec["tools"])
                    ctx_msgs = rec["messages_context"]
                    prompt_txt = tmpl.render(messages=ctx_msgs, tools=tools, add_generation_prompt=False)
                    prompt_ids = tok.encode(prompt_txt, parse_special=True)
                    c_ids = tok.encode(tmpl.render(messages=ctx_msgs + rec["chosen"], tools=tools, add_generation_prompt=False), parse_special=True)
                    r_ids = tok.encode(tmpl.render(messages=ctx_msgs + rec["rejected"], tools=tools, add_generation_prompt=False), parse_special=True)
                    if max(len(c_ids), len(r_ids)) > 4000:
                        continue
                    ensure_pinned(ctx_msgs[0], rec["tools"])
                    c_pos, c_targ = find_assistant_target_positions(c_ids, min_tok_idx=len(prompt_ids))
                    r_pos, r_targ = find_assistant_target_positions(r_ids, min_tok_idx=len(prompt_ids))
                    if not c_pos or not r_pos:
                        continue
                    reused_c, chunk_c = run_gen(c_ids, "")
                    reused_r, chunk_r = run_gen(r_ids, "ckpt=0")
                    rows_c = len(chunk_c) // ROW_BYTES_BF16
                    rows_r = len(chunk_r) // ROW_BYTES_BF16
                    if reused_c > c_pos[0] or reused_r > r_pos[0]:
                        continue
                    u16_c = np.frombuffer(chunk_c, dtype=np.uint16).reshape(rows_c, 4, 2560)
                    u16_r = np.frombuffer(chunk_r, dtype=np.uint16).reshape(rows_r, 4, 2560)
                    rc_slice = bf16_bytes_to_f32(u16_c[[p - reused_c for p in c_pos]].tobytes()).reshape(len(c_pos), 4, 2560)
                    rr_slice = bf16_bytes_to_f32(u16_r[[p - reused_r for p in r_pos]].tobytes()).reshape(len(r_pos), 4, 2560)
                    out_pairs.append({
                        "id": rec["id"],
                        "pattern": rec.get("pattern"),
                        "r_chosen": torch.from_numpy(rc_slice).to(torch.bfloat16),
                        "y_chosen": torch.tensor(c_targ, dtype=torch.int64),
                        "r_rejected": torch.from_numpy(rr_slice).to(torch.bfloat16),
                        "y_rejected": torch.tensor(r_targ, dtype=torch.int64),
                    })
                    if (idx + 1) % 5 == 0 or idx + 1 == len(records):
                        print(f"[extract:{split_name}] {idx+1}/{len(records)} ({time.perf_counter()-t0:.1f}s)", flush=True)
                return out_pairs

            sft_train = extract_sft_split(sft_train_raw, "sft_train")
            sft_val = extract_sft_split(sft_val_raw, "sft_val")
            dpo_train = extract_dpo_split(dpo_train_raw, "dpo_train")
            dpo_val = extract_dpo_split(dpo_val_raw, "dpo_val")

            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()
            if dump_bin.exists():
                dump_bin.unlink()

    payload = {
        "sft_train": sft_train,
        "sft_val": sft_val,
        "dpo_train": dpo_train,
        "dpo_val": dpo_val,
    }
    torch.save(payload, cache_pt)
    print(
        f"[extract] saved {cache_pt}: sft_train={len(sft_train)}, sft_val={len(sft_val)}, "
        f"dpo_train={len(dpo_train)}, dpo_val={len(dpo_val)}",
        flush=True,
    )
    return cache_pt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and export local Strata HyperConnection SFT+DPO head")
    ap.add_argument("--extract", action="store_true", help="Extract pre-head residuals via strata.exe --serve")
    ap.add_argument("--train", action="store_true", help="Train output_hc_{norm,down,up} with SFT + DPO")
    ap.add_argument("--export-pack", action="store_true", help="Export patched dense.bin to packs/iq3_s_ads")
    ap.add_argument("--patch-active", action="store_true", help="Also patch active packs/iq3_s/dense.bin in-place")
    ap.add_argument("--restore-base", action="store_true", help="Restore stock output_hc_base.bin into dense.bin")
    ap.add_argument("--alpha", type=float, default=1.0, help="Interpolation weight between base and trained output_hc")
    ap.add_argument("--pack-src", type=Path, default=PACK_SRC, help="Source Strata pack directory containing dense.bin")
    ap.add_argument("--adapter-out", type=Path, default=MODEL_OUT_DIR / "output_hc_adapter.pt")
    ap.add_argument("--sft-per-family", type=int, default=4)
    ap.add_argument("--dpo-per-pattern", type=int, default=5)
    ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--dpo-steps", type=int, default=60)
    ap.add_argument("--sft-lr", type=float, default=5e-5)
    ap.add_argument("--dpo-lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)
    if args.restore_base:
        restore_base_pack(args.pack_src)
        print(f"[restore] restored stock output_hc_base.bin into {args.pack_src / 'dense.bin'}", flush=True)
        return 0

    torch.set_num_threads(args.threads)
    cache_pt = RESIDUAL_DIR / "strata_hc_residuals.pt"
    if args.extract or (args.train and not cache_pt.exists()):
        extract_residuals_with_strata(
            sft_train_per_family=args.sft_per_family,
            sft_val_per_family=max(1, args.sft_per_family // 2),
            dpo_train_per_pattern=args.dpo_per_pattern,
            dpo_val_per_pattern=max(1, args.dpo_per_pattern // 2),
            out_dir=RESIDUAL_DIR,
        )

    if args.train:
        data = torch.load(cache_pt, weights_only=True)
        head = load_head_from_strata(device="cpu")
        base_head = head.clone_frozen()

        train_tool, train_text = split_tool_and_text_batches(data["sft_train"])
        val_tool, val_text = split_tool_and_text_batches(data["sft_val"])
        if not train_text:
            train_text = data["sft_train"]
        if not val_text:
            val_text = data["sft_val"]

        val_sft_before = eval_sft_batches(head, val_text)
        val_dpo_before = eval_dpo_pairs(head, base_head, data["dpo_val"], beta=args.beta)
        print(f"[train] initial val_sft_loss={val_sft_before:.4f}, val_dpo={val_dpo_before}", flush=True)

        sft_res = train_sft_epoch(
            head, base_head, train_text, lr=args.sft_lr, anchor_coef=2.0, kl_coef=1.0, steps=args.sft_steps, batch_size=128
        )
        val_sft_after = eval_sft_batches(head, val_text)
        print(f"[train] SFT done: {sft_res}, val_sft_loss={val_sft_after:.4f}", flush=True)

        r_tool_train = torch.cat([b[0].float() for b in train_tool], dim=0) if train_tool else None
        ref_tool_probs = None
        if r_tool_train is not None:
            with torch.no_grad():
                ref_tool_probs = F.softmax(base_head(r_tool_train), dim=-1)

        optimizer = torch.optim.AdamW(head.parameters(), lr=args.dpo_lr, weight_decay=0.0)
        init_dpo = eval_dpo_pairs(head, base_head, data["dpo_train"][:8], beta=args.beta)
        for step in range(args.dpo_steps):
            p_item = data["dpo_train"][step % len(data["dpo_train"])]
            optimizer.zero_grad()
            rc, yc = p_item["r_chosen"].float(), p_item["y_chosen"].long()
            rr, yr = p_item["r_rejected"].float(), p_item["y_rejected"].long()
            with torch.no_grad():
                ref_c = base_head.token_logprobs(rc, yc).mean()
                ref_r = base_head.token_logprobs(rr, yr).mean()
            pi_c = head.token_logprobs(rc, yc).mean()
            pi_r = head.token_logprobs(rr, yr).mean()
            dpo_loss = -F.logsigmoid(args.beta * ((pi_c - ref_c) - (pi_r - ref_r)))
            loss = dpo_loss + 2.0 * head.anchor_loss(base_head)
            if r_tool_train is not None and ref_tool_probs is not None:
                t_idx = torch.randint(0, len(r_tool_train), (min(64, len(r_tool_train)),))
                kl_tool = F.kl_div(F.log_softmax(head(r_tool_train[t_idx]), dim=-1), ref_tool_probs[t_idx], reduction="batchmean")
                loss = loss + 5.0 * kl_tool
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        fin_dpo = eval_dpo_pairs(head, base_head, data["dpo_train"][:8], beta=args.beta)
        dpo_res = {
            "initial_loss": init_dpo["loss"],
            "final_loss": fin_dpo["loss"],
            "initial_margin": init_dpo["margin"],
            "final_margin": fin_dpo["margin"],
            "initial_accuracy": init_dpo["accuracy"],
            "final_accuracy": fin_dpo["accuracy"],
            "steps": args.dpo_steps,
        }
        val_dpo_after = eval_dpo_pairs(head, base_head, data["dpo_val"], beta=args.beta)
        val_sft_final = eval_sft_batches(head, val_text)
        print(f"[train] DPO done: {dpo_res}, val_dpo={val_dpo_after}, val_sft_final={val_sft_final:.4f}", flush=True)

        MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(args.adapter_out)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "output_hc_norm.weight": head.w_norm.detach().cpu(),
                "output_hc_down.weight": head.w_down.detach().cpu(),
                "output_hc_up.weight": head.w_up.detach().cpu(),
            },
            ckpt_path,
        )
        summary = {
            "model": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
            "adapter": str(ckpt_path),
            "sft_train": sft_res,
            "dpo_train": dpo_res,
            "val_sft_before": val_sft_before,
            "val_sft_after_sft": val_sft_after,
            "val_sft_final": val_sft_final,
            "val_dpo_before": val_dpo_before,
            "val_dpo_after": val_dpo_after,
        }
        (MODEL_OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.export_pack:
            dst = export_strata_pack(head, PACK_DST if args.pack_src == PACK_SRC else args.pack_src, also_patch_src=args.patch_active, pack_src=args.pack_src)
            print(f"[export] exported patched Strata pack to {dst}", flush=True)
    elif args.export_pack:
        head = load_head_from_strata(device="cpu", pack_src=args.pack_src)
        ckpt = torch.load(Path(args.adapter_out), weights_only=True)
        w_norm, w_down, w_up = blend_head_weights(head, ckpt, alpha=args.alpha)
        head.w_norm.data.copy_(torch.from_numpy(w_norm))
        head.w_down.data.copy_(torch.from_numpy(w_down))
        head.w_up.data.copy_(torch.from_numpy(w_up))
        dst = export_strata_pack(head, PACK_DST if args.pack_src == PACK_SRC else args.pack_src, also_patch_src=args.patch_active, pack_src=args.pack_src)
        print(f"[export] exported patched Strata pack (alpha={args.alpha}) to {dst}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
