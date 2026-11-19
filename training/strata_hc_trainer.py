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

try:
    from tool_schemas import schemas_for as _ads_schemas_for
except ImportError:
    _ads_schemas_for = None

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


def find_strata_exe(strata_root: Path | None = None) -> Path:
    env_exe = os.environ.get("STRATA_EXE")
    if env_exe:
        return Path(env_exe)
    candidates: list[Path] = []
    if strata_root is not None:
        sr = Path(strata_root)
        candidates.append(sr / "strata.exe")
        candidates.append(sr.parent.parent / "components" / "engine" / "current" / "strata.exe")
    candidates.append(_LOCAL_STRATA / "components" / "engine" / "current" / "strata.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    engine_dirs: list[Path] = []
    if strata_root is not None:
        engine_dirs.append(Path(strata_root).parent.parent / "components" / "engine")
    engine_dirs.append(_LOCAL_STRATA / "components" / "engine")
    for eng_dir in engine_dirs:
        if eng_dir.is_dir():
            matches = sorted(eng_dir.glob("*/strata.exe"))
            if matches:
                return matches[-1]
    which_exe = shutil.which("strata") or shutil.which("strata.exe")
    if which_exe:
        return Path(which_exe)
    return STRATA_EXE


def load_strata_install_config(cfg_path: Path) -> dict:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    args = cfg.get("args") or cfg.get("args_written") or []
    flag_map = {
        "--pack": "pack_src",
        "--native": "shard1_gguf",
        "--ple-gguf": "shard2_gguf",
        "--expert-profile": "expert_profile",
        "--mtp": "mtp_dir",
    }
    out: dict = {}
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in flag_map and i + 1 < len(args):
            out[flag_map[flag]] = Path(args[i + 1])
            i += 2
            continue
        i += 1
    if cfg.get("cwd"):
        out["strata_root"] = Path(cfg["cwd"])
    if cfg.get("model_name"):
        out["model_name"] = str(cfg["model_name"])
    return out


def read_hc_layout_from_index(pack_dir: Path) -> dict[str, tuple[int, int]]:
    layout = {
        "down": (HC_DOWN_OFFSET, HC_DOWN_BYTES),
        "norm": (HC_NORM_OFFSET, HC_NORM_BYTES),
        "up": (HC_UP_OFFSET, HC_UP_BYTES),
    }
    expected = {"down": HC_DOWN_BYTES, "norm": HC_NORM_BYTES, "up": HC_UP_BYTES}
    key_by_name = {
        "output_hc_down.weight": "down",
        "output_hc_norm.weight": "norm",
        "output_hc_up.weight": "up",
    }
    index_file = Path(pack_dir) / "index.txt"
    if not index_file.is_file():
        return layout
    for raw_line in index_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.split()
        if len(parts) < 5 or parts[0] not in key_by_name:
            continue
        key = key_by_name[parts[0]]
        offset = int(parts[3])
        nbytes = int(parts[4])
        if nbytes != expected[key]:
            raise ValueError(f"{parts[0]} byte size {nbytes} does not match expected {expected[key]}")
        layout[key] = (offset, nbytes)
    return layout


def resolve_record_tools(rec: dict) -> list[dict]:
    raw_tools = rec.get("tools")
    if not raw_tools:
        return []
    if all(isinstance(t, dict) for t in raw_tools):
        return list(raw_tools)
    if all(isinstance(t, str) for t in raw_tools):
        if _ads_schemas_for is None:
            raise ValueError("String tool names require tool_schemas.py; supply inline OpenAI tool schema dicts instead.")
        return _ads_schemas_for(raw_tools)
    raise ValueError("rec['tools'] must be a list of tool schema dicts or tool name strings.")


def _normalize_messages_for_template(msgs: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for msg in msgs:
        m = dict(msg)
        raw_tcs = m.get("tool_calls")
        if isinstance(raw_tcs, list):
            new_tcs = []
            for tc in raw_tcs:
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        fn_copy["arguments"] = json.loads(args) if args.strip() else {}
                    tc_copy["function"] = fn_copy
                else:
                    args = tc_copy.get("arguments")
                    if isinstance(args, str):
                        tc_copy["arguments"] = json.loads(args) if args.strip() else {}
                new_tcs.append(tc_copy)
            m["tool_calls"] = new_tcs
        normalized.append(m)
    return normalized


def normalize_sft_record(rec: dict) -> tuple[list[dict], list[dict]]:
    if "messages" not in rec or not isinstance(rec["messages"], list):
        raise ValueError("SFT record requires a 'messages' list")
    msgs = _normalize_messages_for_template(rec["messages"])
    sys_txt = rec.get("system")
    if sys_txt and (not msgs or msgs[0].get("role") != "system"):
        msgs.insert(0, {"role": "system", "content": sys_txt})
    return msgs, resolve_record_tools(rec)


def _normalize_completion(val, default_role: str = "assistant") -> list[dict]:
    if isinstance(val, str):
        return [{"role": default_role, "content": val}]
    if isinstance(val, dict):
        return [val]
    if isinstance(val, list):
        return list(val)
    raise ValueError(f"Unsupported completion type: {type(val).__name__}")


def normalize_dpo_record(rec: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if rec.get("messages_context") is not None:
        ctx_msgs = _normalize_messages_for_template(rec["messages_context"])
    elif rec.get("prompt") is not None:
        prompt = rec["prompt"]
        if isinstance(prompt, str):
            ctx_msgs = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            ctx_msgs = _normalize_messages_for_template(prompt)
        else:
            raise ValueError(f"Unsupported DPO prompt type: {type(prompt).__name__}")
    else:
        raise ValueError("DPO record requires 'messages_context' or 'prompt'")
    sys_txt = rec.get("system")
    if sys_txt and (not ctx_msgs or ctx_msgs[0].get("role") != "system"):
        ctx_msgs.insert(0, {"role": "system", "content": sys_txt})
    chosen_msgs = _normalize_messages_for_template(_normalize_completion(rec["chosen"], "assistant"))
    rejected_msgs = _normalize_messages_for_template(_normalize_completion(rec["rejected"], "assistant"))
    return ctx_msgs, chosen_msgs, rejected_msgs, resolve_record_tools(rec)


def split_train_val(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 20261110,
) -> tuple[list[dict], list[dict]]:
    if len(records) < 2 or val_ratio <= 0.0:
        return list(records), []
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(len(records) * val_ratio)))
    if n_val >= len(records):
        n_val = len(records) - 1
    val_idx = set(indices[:n_val])
    train = [records[i] for i in range(len(records)) if i not in val_idx]
    val = [records[i] for i in range(len(records)) if i in val_idx]
    return train, val


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
    if not batches:
        return 0.0
    all_r = torch.cat([b[0].float() for b in batches], dim=0)[:max_tokens]
    all_y = torch.cat([b[1].long() for b in batches], dim=0)[:max_tokens]
    if len(all_y) == 0:
        return 0.0
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
    val_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    log_every: int = 0,
    val_every: int = 0,
) -> dict:
    all_r = torch.cat([b[0].float() for b in batches], dim=0)
    all_y = torch.cat([b[1].long() for b in batches], dim=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    eval_n = min(len(all_y), max(batch_size * 2, 64))
    initial_loss = eval_sft_batches(head, [(all_r[:eval_n], all_y[:eval_n])], max_tokens=eval_n)
    n_total = len(all_y)

    history: list[dict] = []
    if log_every > 0:
        entry0: dict = {
            "step": 0,
            "train_ce": initial_loss,
            "train_kl": 0.0,
            "train_anchor": 0.0,
            "train_total": initial_loss,
        }
        if val_batches and val_every > 0:
            entry0["val_sft_loss"] = eval_sft_batches(head, val_batches)
        history.append(entry0)
        val_msg = f", val_sft={entry0['val_sft_loss']:.4f}" if "val_sft_loss" in entry0 else ""
        print(f"[sft] step=0/{steps} train_ce={initial_loss:.4f}{val_msg}", flush=True)

    for step in range(1, steps + 1):
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
        anchor_val = head.anchor_loss(base_head)
        loss = ce + kl_coef * kl + anchor_coef * anchor_val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        if log_every > 0 and (step % log_every == 0 or step == steps):
            entry = {
                "step": step,
                "train_ce": float(ce.item()),
                "train_kl": float(kl.item()),
                "train_anchor": float(anchor_val.item()),
                "train_total": float(loss.item()),
            }
            if val_batches and val_every > 0 and (step % val_every == 0 or step == steps):
                entry["val_sft_loss"] = eval_sft_batches(head, val_batches)
            history.append(entry)
            val_msg = f", val_sft={entry['val_sft_loss']:.4f}" if "val_sft_loss" in entry else ""
            print(
                f"[sft] step={step}/{steps} ce={entry['train_ce']:.4f} kl={entry['train_kl']:.4f}{val_msg}",
                flush=True,
            )

    final_loss = eval_sft_batches(head, [(all_r[:eval_n], all_y[:eval_n])], max_tokens=eval_n)
    res: dict = {"initial_loss": initial_loss, "final_loss": final_loss, "steps": steps, "tokens": int(n_total)}
    if history:
        res["history"] = history
    return res


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


def train_dpo_with_tool_guard(
    head: Qwen4ExpHcHead,
    base_head: Qwen4ExpHcHead,
    pairs: list[dict],
    r_tool_train: torch.Tensor | None = None,
    lr: float = 2e-5,
    beta: float = 0.1,
    anchor_coef: float = 2.0,
    tool_kl_coef: float = 5.0,
    steps: int = 60,
    val_pairs: list[dict] | None = None,
    val_sft_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    log_every: int = 0,
    val_every: int = 0,
) -> dict:
    ref_tool_probs = None
    if r_tool_train is not None and len(r_tool_train) > 0:
        with torch.no_grad():
            ref_tool_probs = F.softmax(base_head(r_tool_train), dim=-1)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    init_dpo = eval_dpo_pairs(head, base_head, pairs[:8], beta=beta)

    history: list[dict] = []
    if log_every > 0:
        entry0: dict = {
            "step": 0,
            "train_dpo_loss": init_dpo["loss"],
            "train_kl_tool": 0.0,
        }
        if val_pairs is not None and val_every > 0:
            vd0 = eval_dpo_pairs(head, base_head, val_pairs, beta=beta)
            entry0.update({
                "val_dpo_loss": vd0["loss"],
                "val_dpo_margin": vd0["margin"],
                "val_dpo_acc": vd0["accuracy"],
            })
        if val_sft_batches and val_every > 0:
            entry0["val_sft_loss"] = eval_sft_batches(head, val_sft_batches)
        history.append(entry0)
        val_msg = f", val_acc={entry0['val_dpo_acc']:.3f} val_margin={entry0['val_dpo_margin']:+.4f}" if "val_dpo_acc" in entry0 else ""
        print(f"[dpo] step=0/{steps} train_loss={init_dpo['loss']:.4f}{val_msg}", flush=True)

    for step in range(1, steps + 1):
        p_item = pairs[(step - 1) % len(pairs)]
        optimizer.zero_grad()
        rc, yc = p_item["r_chosen"].float(), p_item["y_chosen"].long()
        rr, yr = p_item["r_rejected"].float(), p_item["y_rejected"].long()
        with torch.no_grad():
            ref_c = base_head.token_logprobs(rc, yc).mean()
            ref_r = base_head.token_logprobs(rr, yr).mean()
        pi_c = head.token_logprobs(rc, yc).mean()
        pi_r = head.token_logprobs(rr, yr).mean()
        dpo_loss = -F.logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))
        loss = dpo_loss + anchor_coef * head.anchor_loss(base_head)
        kl_tool_val = 0.0
        if r_tool_train is not None and ref_tool_probs is not None:
            t_idx = torch.randint(0, len(r_tool_train), (min(64, len(r_tool_train)),))
            kl_tool = F.kl_div(F.log_softmax(head(r_tool_train[t_idx]), dim=-1), ref_tool_probs[t_idx], reduction="batchmean")
            kl_tool_val = float(kl_tool.item())
            loss = loss + tool_kl_coef * kl_tool
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        if log_every > 0 and (step % log_every == 0 or step == steps):
            entry: dict = {
                "step": step,
                "train_dpo_loss": float(dpo_loss.item()),
                "train_kl_tool": kl_tool_val,
            }
            if val_pairs is not None and val_every > 0 and (step % val_every == 0 or step == steps):
                vd = eval_dpo_pairs(head, base_head, val_pairs, beta=beta)
                entry.update({
                    "val_dpo_loss": vd["loss"],
                    "val_dpo_margin": vd["margin"],
                    "val_dpo_acc": vd["accuracy"],
                })
            if val_sft_batches and val_every > 0 and (step % val_every == 0 or step == steps):
                entry["val_sft_loss"] = eval_sft_batches(head, val_sft_batches)
            history.append(entry)
            val_msg = f", val_acc={entry['val_dpo_acc']:.3f} val_margin={entry['val_dpo_margin']:+.4f}" if "val_dpo_acc" in entry else ""
            print(
                f"[dpo] step={step}/{steps} loss={entry['train_dpo_loss']:.4f} kl_tool={kl_tool_val:.5f}{val_msg}",
                flush=True,
            )

    fin_dpo = eval_dpo_pairs(head, base_head, pairs[:8], beta=beta)
    res: dict = {
        "initial_loss": init_dpo["loss"],
        "final_loss": fin_dpo["loss"],
        "initial_margin": init_dpo["margin"],
        "final_margin": fin_dpo["margin"],
        "initial_accuracy": init_dpo["accuracy"],
        "final_accuracy": fin_dpo["accuracy"],
        "steps": steps,
    }
    if history:
        res["history"] = history
    return res


def patch_dense_bin(
    src_dense: Path,
    dst_dense: Path,
    w_norm: np.ndarray,
    w_down: np.ndarray,
    w_up: np.ndarray,
    layout: dict[str, tuple[int, int]] | None = None,
) -> None:
    src_dense = Path(src_dense)
    dst_dense = Path(dst_dense)
    if layout is None:
        layout = read_hc_layout_from_index(src_dense.parent)
    dst_dense.parent.mkdir(parents=True, exist_ok=True)
    if src_dense.resolve() != dst_dense.resolve():
        shutil.copyfile(src_dense, dst_dense)

    down_off, down_len = layout["down"]
    norm_off, norm_len = layout["norm"]
    up_off, up_len = layout["up"]

    down_bytes = f32_to_bf16_bytes(w_down.reshape(320, 10240))
    norm_bytes = np.ascontiguousarray(w_norm.reshape(10240), dtype=np.float32).tobytes()
    up_bytes = f32_to_bf16_bytes(w_up.reshape(10240, 320))
    assert len(down_bytes) == down_len
    assert len(norm_bytes) == norm_len
    assert len(up_bytes) == up_len

    with open(dst_dense, "r+b") as f:
        f.seek(down_off)
        f.write(down_bytes)
        f.seek(norm_off)
        f.write(norm_bytes)
        f.seek(up_off)
        f.write(up_bytes)


def load_strata_tokenizer_and_template(
    pack_src: Path = PACK_SRC,
    strata_root: Path | None = None,
):
    sr = Path(strata_root or STRATA_ROOT)
    tools_dir = str(sr / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import jinja2
    import strata_tokenizer as ST

    tpath = Path(pack_src) / "tokenizer"
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
    tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
    chat_tmpl_path = sr / "serve" / "chat_template.jinja"
    tmpl = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"]).from_string(
        chat_tmpl_path.read_text(encoding="utf-8")
    )
    return tok, tmpl


def load_head_from_strata(
    device: str = "cpu",
    pack_src: Path = PACK_SRC,
    shard1_gguf: Path | None = None,
    strata_root: Path | None = None,
    layout: dict[str, tuple[int, int]] | None = None,
) -> Qwen4ExpHcHead:
    sr = Path(strata_root or STRATA_ROOT)
    for p in (sr / "third_party" / "llama.cpp" / "gguf-py", sr / "ref"):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    from load import Gguf

    pack_src = Path(pack_src)
    layout = layout or read_hc_layout_from_index(pack_src)
    down_off, down_len = layout["down"]
    norm_off, norm_len = layout["norm"]
    up_off, up_len = layout["up"]
    header_end = max(down_off + down_len, norm_off + norm_len, up_off + up_len)

    base_slice = pack_src / "output_hc_base.bin"
    dense_file = pack_src / "dense.bin"
    if not base_slice.is_file() and dense_file.is_file():
        with open(dense_file, "rb") as f:
            base_slice.write_bytes(f.read(header_end))
    src_file = base_slice if base_slice.is_file() else dense_file
    with open(src_file, "rb") as f:
        raw_head = f.read(header_end)
    w_down = bf16_bytes_to_f32(raw_head[down_off : down_off + down_len]).reshape(320, 10240)
    w_norm = np.frombuffer(raw_head[norm_off : norm_off + norm_len], dtype=np.float32).copy()
    w_up = bf16_bytes_to_f32(raw_head[up_off : up_off + up_len]).reshape(10240, 320)

    gguf = Gguf(str(shard1_gguf or SHARD1_GGUF))
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
    install_cfg_src: Path | None = INSTALL_CFG_SRC,
    install_cfg_dst: Path | None = INSTALL_CFG_DST,
    model_name: str = "qwen3.8-flash-next-iq3_s-ads",
) -> Path:
    pack_src = Path(pack_src)
    pack_dst = Path(pack_dst)
    layout = read_hc_layout_from_index(pack_src)
    pack_dst.mkdir(parents=True, exist_ok=True)
    for fn in ("index.txt", "native_experts.txt"):
        if (pack_src / fn).exists() and (pack_src / fn).resolve() != (pack_dst / fn).resolve():
            shutil.copy2(pack_src / fn, pack_dst / fn)
    tok_dst = pack_dst / "tokenizer"
    if not tok_dst.exists() and (pack_src / "tokenizer").exists():
        shutil.copytree(pack_src / "tokenizer", tok_dst)
    exp_dst = pack_dst / "experts.bin"
    if not exp_dst.exists() and (pack_src / "experts.bin").exists():
        try:
            os.link(pack_src / "experts.bin", exp_dst)
        except OSError as exc:
            raise OSError(
                f"Cannot hardlink {pack_src / 'experts.bin'} to {exp_dst}: place --pack-dst on the same volume as --pack-src."
            ) from exc

    w_norm = head.w_norm.detach().cpu().float().numpy()
    w_down = head.w_down.detach().cpu().float().numpy()
    w_up = head.w_up.detach().cpu().float().numpy()
    patch_dense_bin(pack_src / "dense.bin", pack_dst / "dense.bin", w_norm, w_down, w_up, layout=layout)
    if also_patch_src:
        patch_dense_bin(pack_src / "dense.bin", pack_src / "dense.bin", w_norm, w_down, w_up, layout=layout)

    if install_cfg_src is not None and Path(install_cfg_src).is_file() and install_cfg_dst is not None:
        cfg = json.loads(Path(install_cfg_src).read_text(encoding="utf-8"))
        for key in ("args", "args_written"):
            if key in cfg and "--pack" in cfg[key]:
                idx = cfg[key].index("--pack")
                cfg[key][idx + 1] = str(pack_dst)
        cfg["tokenizer"] = str(pack_dst / "tokenizer")
        cfg["model_name"] = model_name
        Path(install_cfg_dst).parent.mkdir(parents=True, exist_ok=True)
        Path(install_cfg_dst).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return pack_dst


def restore_base_pack(pack_src: Path = PACK_SRC, pack_dst: Path | None = None) -> None:
    pack_src = Path(pack_src)
    base_slice = pack_src / "output_hc_base.bin"
    if not base_slice.is_file():
        raise FileNotFoundError(f"Missing base slice backup: {base_slice}")
    raw = base_slice.read_bytes()
    targets = [pack_src / "dense.bin"]
    if pack_dst is not None:
        targets.append(Path(pack_dst) / "dense.bin")
    elif pack_src.resolve() == PACK_SRC.resolve():
        targets.append(PACK_DST / "dense.bin")
    for target in targets:
        if target.is_file():
            with open(target, "r+b") as f:
                f.seek(0)
                f.write(raw)


def _record_sort_text(rec: dict) -> str:
    if rec.get("system"):
        return str(rec["system"])
    msgs = rec.get("messages") or rec.get("messages_context")
    if isinstance(msgs, list) and msgs:
        first = msgs[0]
        if isinstance(first, dict):
            return str(first.get("content") or "")
    prompt = rec.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        return str(prompt[0].get("content") or "")
    return ""


def select_stratified_records(
    records: list[dict],
    key_field: str,
    per_group: int | None,
    default_per_group: int = 4,
) -> list[dict]:
    if not records:
        return []
    if per_group is None:
        has_key = any(key_field in r and r.get(key_field) is not None for r in records)
        effective_per_group = default_per_group if has_key else 0
    else:
        effective_per_group = int(per_group)

    if effective_per_group <= 0:
        selected = list(records)
    else:
        groups = defaultdict(list)
        for r in records:
            groups[r.get(key_field, "default")].append(r)
        selected = []
        rng = random.Random(20261110)
        for k in sorted(groups.keys(), key=str):
            items = list(groups[k])
            rng.shuffle(items)
            selected.extend(items[:effective_per_group])

    selected.sort(
        key=lambda r: (
            json.dumps(r.get("tools") or [], sort_keys=True),
            _record_sort_text(r),
        )
    )
    return selected


def _load_jsonl_file(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_split_files(
    dataset_dir: Path | None,
    sft_train_path: Path | None,
    sft_val_path: Path | None,
    dpo_train_path: Path | None,
    dpo_val_path: Path | None,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    explicit_mode = (dataset_dir is None) and (sft_train_path is not None or dpo_train_path is not None)
    base_dir = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR

    sft_train_src = Path(sft_train_path) if sft_train_path is not None else (None if explicit_mode else base_dir / "sft_train.jsonl")
    sft_val_src = Path(sft_val_path) if sft_val_path is not None else (None if explicit_mode else base_dir / "sft_val.jsonl")
    dpo_train_src = Path(dpo_train_path) if dpo_train_path is not None else (None if explicit_mode else base_dir / "dpo_train.jsonl")
    dpo_val_src = Path(dpo_val_path) if dpo_val_path is not None else (None if explicit_mode else base_dir / "dpo_val.jsonl")

    sft_train_rows = _load_jsonl_file(sft_train_src) if (sft_train_src is not None and sft_train_src.is_file()) else []
    if sft_val_src is not None and sft_val_src.is_file():
        sft_val_rows = _load_jsonl_file(sft_val_src)
    elif sft_train_rows:
        sft_train_rows, sft_val_rows = split_train_val(sft_train_rows, val_ratio=val_ratio, seed=seed)
    else:
        sft_val_rows = []

    dpo_train_rows = _load_jsonl_file(dpo_train_src) if (dpo_train_src is not None and dpo_train_src.is_file()) else []
    if dpo_val_src is not None and dpo_val_src.is_file():
        dpo_val_rows = _load_jsonl_file(dpo_val_src)
    elif dpo_train_rows:
        dpo_train_rows, dpo_val_rows = split_train_val(dpo_train_rows, val_ratio=val_ratio, seed=seed)
    else:
        dpo_val_rows = []

    return sft_train_rows, sft_val_rows, dpo_train_rows, dpo_val_rows


def extract_residuals_with_strata(
    sft_train_per_family: int | None = None,
    sft_val_per_family: int | None = None,
    dpo_train_per_pattern: int | None = None,
    dpo_val_per_pattern: int | None = None,
    out_dir: Path = RESIDUAL_DIR,
    residuals_path: Path | None = None,
    strata_exe: Path | None = None,
    strata_root: Path | None = None,
    pack_src: Path = PACK_SRC,
    shard1_gguf: Path | None = None,
    shard2_gguf: Path | None = None,
    expert_profile: Path | None = None,
    mtp_dir: Path | None = None,
    dataset_dir: Path | None = None,
    sft_train_path: Path | None = None,
    sft_val_path: Path | None = None,
    dpo_train_path: Path | None = None,
    dpo_val_path: Path | None = None,
    val_ratio: float = 0.1,
    max_extract_tokens: int = 4000,
    seed: int = 20261110,
) -> Path:
    cache_pt = Path(residuals_path) if residuals_path is not None else (Path(out_dir) / "strata_hc_residuals.pt")
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    sr = Path(strata_root or STRATA_ROOT)
    tok, tmpl = load_strata_tokenizer_and_template(pack_src=pack_src, strata_root=sr)

    sft_tr_all, sft_va_all, dpo_tr_all, dpo_va_all = _resolve_split_files(
        dataset_dir=dataset_dir,
        sft_train_path=sft_train_path,
        sft_val_path=sft_val_path,
        dpo_train_path=dpo_train_path,
        dpo_val_path=dpo_val_path,
        val_ratio=val_ratio,
        seed=seed,
    )

    sft_train_raw = select_stratified_records(sft_tr_all, "task_family", sft_train_per_family, default_per_group=4)
    sft_val_raw = select_stratified_records(sft_va_all, "task_family", sft_val_per_family, default_per_group=1)
    dpo_train_raw = select_stratified_records(dpo_tr_all, "pattern", dpo_train_per_pattern, default_per_group=5)
    dpo_val_raw = select_stratified_records(dpo_va_all, "pattern", dpo_val_per_pattern, default_per_group=1)

    if not (sft_train_raw or sft_val_raw or dpo_train_raw or dpo_val_raw):
        raise ValueError("No SFT or DPO records found for residual extraction.")

    dump_bin = cache_pt.parent / "temp_r_dump.bin"
    serve_log = cache_pt.parent / "strata_extract.log"
    if dump_bin.exists():
        dump_bin.unlink()

    resolved_exe = strata_exe or find_strata_exe(sr)
    resolved_profile = expert_profile or (sr / "data" / "expert-profile.bin")
    resolved_mtp = mtp_dir or MTP_DIR
    max_context = max(8192, max_extract_tokens + 1024)

    env = dict(os.environ, STRATA_PREFILL_DUMP_R_ALL=str(dump_bin))
    cmd = [
        str(resolved_exe),
        "--serve",
        "--short-read", "0",
        "--prompt-cache", "12",
        "--conversation-cache-slots", "12",
        "--pack", str(pack_src),
        "--native", str(shard1_gguf or SHARD1_GGUF),
        "--ple-gguf", str(shard2_gguf or SHARD2_GGUF),
        "--expert-profile", str(resolved_profile),
        "--expert-cache", "2048",
        "--spec", "4",
        "--prefill", "4096",
        "--max-context", str(max_context),
        "--mmap-experts",
    ]
    if resolved_mtp is not None and Path(resolved_mtp).exists():
        cmd.extend(["--mtp", str(resolved_mtp)])

    print(f"[extract] launching {resolved_exe} --serve for residual extraction...", flush=True)
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

            def ensure_pinned(msgs: list[dict], tools: list[dict]) -> None:
                has_sys = bool(msgs and msgs[0].get("role") in ("system", "developer"))
                if not has_sys and not tools:
                    return
                sys_msg = msgs[0] if has_sys else {"role": "system", "content": ""}
                key = (json.dumps(tools, sort_keys=True), str(sys_msg.get("content", "")))
                if key in pinned_headers:
                    return
                sys_only = tmpl.render(messages=[sys_msg], tools=tools or None, add_generation_prompt=False)
                sys_ids = tok.encode(sys_only, parse_special=True)
                warm_msgs = [sys_msg, {"role": "user", "content": "ping"}]
                warm_ids = tok.encode(
                    tmpl.render(messages=warm_msgs, tools=tools or None, add_generation_prompt=False),
                    parse_special=True,
                )
                run_gen(warm_ids, f"pin={len(sys_ids)}")
                pinned_headers.add(key)

            def extract_sft_split(records: list[dict], split_name: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
                out_batches = []
                t0 = time.perf_counter()
                for idx, rec in enumerate(records):
                    msgs, tools = normalize_sft_record(rec)
                    rendered = tmpl.render(messages=msgs, tools=tools or None, add_generation_prompt=False)
                    ids = tok.encode(rendered, parse_special=True)
                    if len(ids) > max_extract_tokens:
                        continue
                    ensure_pinned(msgs, tools)
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
                    ctx_msgs, chosen_msgs, rejected_msgs, tools = normalize_dpo_record(rec)
                    prompt_txt = tmpl.render(messages=ctx_msgs, tools=tools or None, add_generation_prompt=False)
                    prompt_ids = tok.encode(prompt_txt, parse_special=True)
                    c_ids = tok.encode(
                        tmpl.render(messages=ctx_msgs + chosen_msgs, tools=tools or None, add_generation_prompt=False),
                        parse_special=True,
                    )
                    r_ids = tok.encode(
                        tmpl.render(messages=ctx_msgs + rejected_msgs, tools=tools or None, add_generation_prompt=False),
                        parse_special=True,
                    )
                    if max(len(c_ids), len(r_ids)) > max_extract_tokens:
                        continue
                    ensure_pinned(ctx_msgs, tools)
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
                        "id": rec.get("id", f"dpo-{idx}"),
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
    ap.add_argument("--export-pack", action="store_true", help="Export patched dense.bin to a Strata pack")
    ap.add_argument("--patch-active", action="store_true", help="Also patch source pack dense.bin in-place")
    ap.add_argument("--restore-base", action="store_true", help="Restore stock output_hc_base.bin into dense.bin")
    ap.add_argument("--alpha", type=float, default=1.0, help="Interpolation weight between base and trained output_hc")
    ap.add_argument("--strata-config", type=Path, default=None, help="Path to stock Strata install JSON (e.g. strata-iq3_s.json)")
    ap.add_argument("--strata-exe", type=Path, default=None, help="Path to strata.exe")
    ap.add_argument("--strata-root", type=Path, default=None, help="Path to Strata terminal root directory")
    ap.add_argument("--pack-src", type=Path, default=None, help="Source Strata pack directory containing dense.bin")
    ap.add_argument("--pack-dst", type=Path, default=None, help="Destination Strata pack directory")
    ap.add_argument("--shard1-gguf", type=Path, default=None, help="Path to GGUF shard containing output.weight")
    ap.add_argument("--shard2-gguf", type=Path, default=None, help="Path to PLE GGUF shard")
    ap.add_argument("--model-name", type=str, default=None, help="Model name written to exported Strata install JSON")
    ap.add_argument("--install-cfg-out", type=Path, default=None, help="Path to write updated Strata install JSON")
    ap.add_argument("--dataset-dir", type=Path, default=None, help="Directory with {sft,dpo}_{train,val}.jsonl")
    ap.add_argument("--sft-train", type=Path, default=None, help="Path to SFT training JSONL")
    ap.add_argument("--sft-val", type=Path, default=None, help="Path to SFT validation JSONL")
    ap.add_argument("--dpo-train", type=Path, default=None, help="Path to DPO training JSONL")
    ap.add_argument("--dpo-val", type=Path, default=None, help="Path to DPO validation JSONL")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Auto validation split ratio when --*-val is omitted")
    ap.add_argument("--residuals-path", type=Path, default=None, help="Path to cached strata_hc_residuals.pt")
    ap.add_argument("--adapter-out", type=Path, default=MODEL_OUT_DIR / "output_hc_adapter.pt")
    ap.add_argument("--summary-out", type=Path, default=None, help="Path to write JSON training summary")
    ap.add_argument("--history-out", type=Path, default=None, help="Path to write per-step SFT/DPO history JSON")
    ap.add_argument("--max-extract-tokens", type=int, default=4000)
    ap.add_argument("--sft-per-family", type=int, default=None, help="Max records per task_family (0 keeps all)")
    ap.add_argument("--dpo-per-pattern", type=int, default=None, help="Max pairs per pattern (0 keeps all)")
    ap.add_argument("--sft-steps", type=int, default=40)
    ap.add_argument("--dpo-steps", type=int, default=60)
    ap.add_argument("--sft-lr", type=float, default=5e-5)
    ap.add_argument("--dpo-lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = load_strata_install_config(args.strata_config) if args.strata_config is not None else {}
    strata_root = Path(args.strata_root or cfg.get("strata_root") or STRATA_ROOT)
    pack_src = Path(args.pack_src or cfg.get("pack_src") or PACK_SRC)
    shard1_gguf = Path(args.shard1_gguf or cfg.get("shard1_gguf") or SHARD1_GGUF)
    shard2_gguf = Path(args.shard2_gguf or cfg.get("shard2_gguf") or SHARD2_GGUF)
    expert_profile = cfg.get("expert_profile") or (strata_root / "data" / "expert-profile.bin")
    mtp_dir = cfg.get("mtp_dir") or MTP_DIR

    if args.pack_dst is not None:
        pack_dst = Path(args.pack_dst)
    elif pack_src.resolve() == PACK_SRC.resolve():
        pack_dst = PACK_DST
    else:
        pack_dst = pack_src.parent / f"{pack_src.name}_tuned"

    install_cfg_src = args.strata_config if args.strata_config is not None else INSTALL_CFG_SRC
    if args.install_cfg_out is not None:
        install_cfg_dst: Path | None = Path(args.install_cfg_out)
    elif install_cfg_src == INSTALL_CFG_SRC and pack_dst.resolve() == PACK_DST.resolve():
        install_cfg_dst = INSTALL_CFG_DST
    elif install_cfg_src is not None and Path(install_cfg_src).is_file():
        install_cfg_dst = Path(install_cfg_src).parent / f"{Path(install_cfg_src).stem}-tuned.json"
    else:
        install_cfg_dst = None

    if args.model_name is not None:
        model_name = args.model_name
    elif cfg.get("model_name"):
        model_name = f"{cfg['model_name']}-tuned"
    else:
        model_name = "qwen3.8-flash-next-iq3_s-ads"

    if args.restore_base:
        restore_base_pack(pack_src, pack_dst=args.pack_dst)
        print(f"[restore] restored stock output_hc_base.bin into {pack_src / 'dense.bin'}", flush=True)
        return 0

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    torch.set_num_threads(args.threads)
    cache_pt = Path(args.residuals_path) if args.residuals_path is not None else (RESIDUAL_DIR / "strata_hc_residuals.pt")

    if args.extract or (args.train and not cache_pt.exists()):
        sft_val_per = None if args.sft_per_family is None else (0 if args.sft_per_family <= 0 else max(1, args.sft_per_family // 2))
        dpo_val_per = None if args.dpo_per_pattern is None else (0 if args.dpo_per_pattern <= 0 else max(1, args.dpo_per_pattern // 2))
        extract_residuals_with_strata(
            sft_train_per_family=args.sft_per_family,
            sft_val_per_family=sft_val_per,
            dpo_train_per_pattern=args.dpo_per_pattern,
            dpo_val_per_pattern=dpo_val_per,
            out_dir=cache_pt.parent,
            residuals_path=cache_pt,
            strata_exe=args.strata_exe,
            strata_root=strata_root,
            pack_src=pack_src,
            shard1_gguf=shard1_gguf,
            shard2_gguf=shard2_gguf,
            expert_profile=expert_profile,
            mtp_dir=mtp_dir,
            dataset_dir=args.dataset_dir,
            sft_train_path=args.sft_train,
            sft_val_path=args.sft_val,
            dpo_train_path=args.dpo_train,
            dpo_val_path=args.dpo_val,
            val_ratio=args.val_ratio,
            max_extract_tokens=args.max_extract_tokens,
            seed=args.seed if args.seed is not None else 20261110,
        )

    if args.train:
        data = torch.load(cache_pt, weights_only=True)
        head = load_head_from_strata(device="cpu", pack_src=pack_src, shard1_gguf=shard1_gguf, strata_root=strata_root)
        base_head = head.clone_frozen()

        sft_train_all = data.get("sft_train") or []
        sft_val_all = data.get("sft_val") or []
        dpo_train_all = data.get("dpo_train") or []
        dpo_val_all = data.get("dpo_val") or []

        train_tool, train_text = split_tool_and_text_batches(sft_train_all)
        _, val_text = split_tool_and_text_batches(sft_val_all)
        if not train_text:
            train_text = sft_train_all
        if not val_text:
            val_text = sft_val_all

        val_sft_before = eval_sft_batches(head, val_text) if val_text else 0.0
        val_dpo_before = eval_dpo_pairs(head, base_head, dpo_val_all, beta=args.beta) if dpo_val_all else {"loss": 0.0, "margin": 0.0, "accuracy": 0.0}
        print(f"[train] initial val_sft_loss={val_sft_before:.4f}, val_dpo={val_dpo_before}", flush=True)

        if train_text and args.sft_steps > 0:
            sft_res = train_sft_epoch(
                head,
                base_head,
                train_text,
                lr=args.sft_lr,
                anchor_coef=2.0,
                kl_coef=1.0,
                steps=args.sft_steps,
                batch_size=128,
                val_batches=val_text if val_text else None,
                log_every=args.log_every,
                val_every=args.val_every,
            )
            val_sft_after = eval_sft_batches(head, val_text) if val_text else 0.0
            print(f"[train] SFT done: {sft_res.get('initial_loss', 0.0):.4f}->{sft_res.get('final_loss', 0.0):.4f}, val_sft_loss={val_sft_after:.4f}", flush=True)
        else:
            sft_res = {"skipped": True, "steps": 0, "tokens": 0}
            val_sft_after = val_sft_before
            print("[train] skipped SFT (no sft_train batches or sft_steps=0)", flush=True)

        r_tool_train = torch.cat([b[0].float() for b in train_tool], dim=0) if train_tool else None
        if dpo_train_all and args.dpo_steps > 0:
            dpo_res = train_dpo_with_tool_guard(
                head,
                base_head,
                dpo_train_all,
                r_tool_train=r_tool_train,
                lr=args.dpo_lr,
                beta=args.beta,
                anchor_coef=2.0,
                tool_kl_coef=5.0,
                steps=args.dpo_steps,
                val_pairs=dpo_val_all if dpo_val_all else None,
                val_sft_batches=val_text if val_text else None,
                log_every=args.log_every,
                val_every=args.val_every,
            )
            val_dpo_after = eval_dpo_pairs(head, base_head, dpo_val_all, beta=args.beta) if dpo_val_all else {"loss": 0.0, "margin": 0.0, "accuracy": 0.0}
            val_sft_final = eval_sft_batches(head, val_text) if val_text else 0.0
            print(f"[train] DPO done: acc={dpo_res['final_accuracy']:.3f}, val_dpo={val_dpo_after}, val_sft_final={val_sft_final:.4f}", flush=True)
        else:
            dpo_res = {"skipped": True, "steps": 0}
            val_dpo_after = val_dpo_before
            val_sft_final = val_sft_after
            print("[train] skipped DPO (no dpo_train pairs or dpo_steps=0)", flush=True)

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

        sft_history = sft_res.pop("history", [])
        dpo_history = dpo_res.pop("history", [])
        if args.history_out is not None:
            hist_path = Path(args.history_out)
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            hist_path.write_text(json.dumps({"sft": sft_history, "dpo": dpo_history}, indent=2), encoding="utf-8")

        summary = {
            "model": cfg.get("model_name") or "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
            "adapter": str(ckpt_path),
            "sft_train": sft_res,
            "dpo_train": dpo_res,
            "val_sft_before": val_sft_before,
            "val_sft_after_sft": val_sft_after,
            "val_sft_final": val_sft_final,
            "val_dpo_before": val_dpo_before,
            "val_dpo_after": val_dpo_after,
        }
        summary_path = Path(args.summary_out) if args.summary_out is not None else (ckpt_path.parent / "run_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if args.export_pack:
            if args.alpha != 1.0:
                ckpt = torch.load(ckpt_path, weights_only=True)
                w_norm, w_down, w_up = blend_head_weights(base_head, ckpt, alpha=args.alpha)
                head.w_norm.data.copy_(torch.from_numpy(w_norm))
                head.w_down.data.copy_(torch.from_numpy(w_down))
                head.w_up.data.copy_(torch.from_numpy(w_up))
            dst = export_strata_pack(
                head,
                pack_dst,
                also_patch_src=args.patch_active,
                pack_src=pack_src,
                install_cfg_src=install_cfg_src,
                install_cfg_dst=install_cfg_dst,
                model_name=model_name,
            )
            print(f"[export] exported patched Strata pack (alpha={args.alpha}) to {dst}", flush=True)
    elif args.export_pack:
        head = load_head_from_strata(device="cpu", pack_src=pack_src, shard1_gguf=shard1_gguf, strata_root=strata_root)
        ckpt = torch.load(Path(args.adapter_out), weights_only=True)
        w_norm, w_down, w_up = blend_head_weights(head, ckpt, alpha=args.alpha)
        head.w_norm.data.copy_(torch.from_numpy(w_norm))
        head.w_down.data.copy_(torch.from_numpy(w_down))
        head.w_up.data.copy_(torch.from_numpy(w_up))
        dst = export_strata_pack(
            head,
            pack_dst,
            also_patch_src=args.patch_active,
            pack_src=pack_src,
            install_cfg_src=install_cfg_src,
            install_cfg_dst=install_cfg_dst,
            model_name=model_name,
        )
        print(f"[export] exported patched Strata pack (alpha={args.alpha}) to {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
