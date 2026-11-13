"""Launch a frozen benchmark campaign; existing completed trials are never rerun."""
import argparse
import asyncio
import functools
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ads_stack import CAMPAIGN_FATAL, PROFILE, StackController, StackError, StackPaths, bridge_session
from bridge import default_secrets_dir
from campaign_core import Budget, accepted_fingerprints, iter_native_usage, parse_json, seed_recovery_costs
from campaign_worker import run_trial, trial_id
from manifest import load_manifest, load_suite


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    deadline = time.monotonic() + 3
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32) or time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def load_campaign(campaign):
    """Frozen manifest, the raw campaign.json settings it was built from, and the suite tasks."""
    content = (campaign / "campaign.json").read_bytes()
    manifest = load_manifest(campaign)
    if hashlib.sha256(content).hexdigest() != manifest.fingerprint:
        raise ValueError("campaign.json changed while it was being verified")
    settings = parse_json(content.decode("utf-8"))
    return manifest, settings, load_suite(manifest)


def existing_results(campaign, settings, fingerprint, budget):
    results = {}
    accepted = accepted_fingerprints(campaign, settings, fingerprint)
    seed_recovery_costs(campaign, settings.get("setup_cost_records", []), accepted, budget)
    for path in sorted((campaign / "results").glob("*.json")):
        result = parse_json(path.read_text(encoding="utf-8"))
        if result["fingerprint"] not in accepted:
            raise ValueError("Existing trial belongs to a different campaign configuration")
        selector, task_id = result["selector"], result["task"]
        identifier = trial_id({"id": task_id}, {"selector": selector})
        if result["trial_id"] != identifier or path.stem != identifier or task_id not in settings["task_ids"] or selector not in {model["selector"] for model in settings["models"]}:
            raise ValueError("Existing trial identity differs from approved matrix")
        journal = campaign / "results" / (identifier + ".jsonl")
        if hashlib.sha256(journal.read_bytes()).hexdigest() != result["journal_sha256"]:
            raise ValueError("Existing trial evidence was changed")
        records = [parse_json(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]
        if not records or records[0].get("type") != "trial_start" or records[0].get("fingerprint") != result["fingerprint"]:
            raise ValueError("Existing journal does not identify the frozen campaign")
        terminal = records[-1]
        if terminal.get("type") != "result" or {key: value for key, value in result.items() if key != "journal_sha256"} != {key: value for key, value in terminal.items() if key not in ("type", "at")}:
            raise ValueError("Trial sidecar differs from its terminal evidence record")
        recorded_usage = [{"key": entry["key"], "cost": entry["cost"]} for entry in records if entry.get("type") == "usage"]
        if recorded_usage != result["usage_records"]:
            raise ValueError("Trial cost ledger differs from the preserved journal")
        native_costs = dict(iter_native_usage(journal, identifier))
        final_native = None
        for entry in records:
            if entry.get("type") != "message_end" or entry.get("message", {}).get("role") != "assistant":
                continue
            message = entry["message"]
            if message.get("provider") + "/" + message.get("model") != selector:
                raise ValueError("Preserved native response has the wrong model identity")
            final_native = message
        if native_costs != {entry["key"]: entry["cost"] for entry in recorded_usage}:
            raise ValueError("Cost ledger differs from actual native message usage")
        if result["status"] == "completed":
            if not final_native or final_native.get("stopReason") != "stop":
                raise ValueError("Completed trial has no native final turn")
            answer = "\n".join(block["text"] for block in final_native["content"] if block.get("type") == "text" and isinstance(block.get("text"), str))
            if answer != result["answer"] or not any(entry.get("type") == "native_session_verified" and entry.get("isSettled") is True for entry in records):
                raise ValueError("Completed answer or settled state differs from native evidence")
        usage_records = result["usage_records"]
        costs = {}
        for usage in usage_records:
            budget.record(usage["key"], usage["cost"])
            costs[usage["key"]] = usage["cost"]
        if abs(sum(costs.values()) - result["catalog_estimated_cost_usd"]) > 1e-9:
            raise ValueError("Trial cost differs from preserved native usage")
        if result["status"] == "completed" and not result.get("answer", "").strip():
            raise ValueError("Completed trial has no actual final answer")
        results[identifier] = result
        budget.note_failure(result.get("error_category"))
    for journal in (campaign / "results").glob("*.jsonl"):
        if journal.stem not in results:
            raise ValueError("Interrupted trial evidence has no terminal result; preserve and review it before restarting: " + journal.name)
    return results


def campaign_stack(campaign, manifest):
    """The emulator stack for this campaign, gated against its frozen baseline.

    Health probes use the Read key; finish_trial's abort path needs the Recipe key. Both sessions refuse
    a bridge config whose host/port differ from the ones frozen in campaign.json.
    """
    secrets_dir = default_secrets_dir()
    endpoints = manifest.bridge_endpoints
    return StackController.for_campaign(
        campaign, manifest.input_sha256,
        read_session=functools.partial(bridge_session, "read", secrets_dir, endpoints),
        recipe_session=functools.partial(bridge_session, "recipe", secrets_dir, endpoints))


async def launch(campaign, smoke=False, stack=None):
    manifest, settings, tasks = load_campaign(campaign)
    fingerprint = manifest.fingerprint
    if stack is None:
        stack = campaign_stack(campaign, manifest)
    budget = Budget(manifest.cost_threshold_usd)
    planned_trials = len(manifest.task_ids) * len(manifest.models)
    lock_path = campaign / "campaign.lock"
    with lock_path.open("x", encoding="utf-8") as lock:
        json.dump({"pid": os.getpid(), "fingerprint": fingerprint}, lock)
    state = {"campaign_id": settings["campaign_id"], "mode": "smoke" if smoke else "campaign", "status": "running", "pid": os.getpid(), "fingerprint": fingerprint, "started_at": datetime.now(timezone.utc).isoformat(), "planned_trials": planned_trials, "attempted_trials": 0, "completed_trials": 0, "failed_trials": 0, "active_trials": [], "recorded_catalog_cost_usd": 0, "cost_threshold_usd": budget.limit, "stop_reason": None}
    budget_error = None
    closed = False
    try:
        results = existing_results(campaign, settings, fingerprint, budget)
        budget_error = budget.stop_reason
        # Smoke runs every task once for every model in the roster; the smoke campaign is frozen with one cheap model.
        jobs = [(tasks[task_id], model) for task_id in manifest.task_ids for model in manifest.models]
        pending = [(task, model) for task, model in jobs if trial_id(task, model) not in results]
        cursor = 0
        previous = None

        def save_state():
            state["attempted_trials"] = len(results)
            state["completed_trials"] = sum(result["status"] == "completed" for result in results.values())
            state["failed_trials"] = state["attempted_trials"] - state["completed_trials"]
            state["recorded_catalog_cost_usd"] = budget.total
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(campaign / ("smoke-progress.json" if smoke else "progress.json"), state)

        def on_cost(key, cost):
            nonlocal budget_error
            try:
                budget.record(key, cost)
            except ValueError:
                budget_error = "invalid_or_conflicting_native_cost"
                raise
            save_state()

        async def worker():
            nonlocal cursor, budget_error, previous
            while cursor < len(pending):
                if budget_error or not budget.can_schedule or (campaign / "STOP").exists():
                    return
                task, model = pending[cursor]
                cursor += 1
                identifier = trial_id(task, model)
                state["active_trials"].append(identifier)
                save_state()
                try:
                    started_utc = datetime.now(timezone.utc)
                    verified = False

                    async def prepare():
                        nonlocal verified
                        record = await stack.prepare(previous, task)
                        verified = True
                        return record

                    result = await run_trial(campaign=campaign, task=task, model=model, fingerprint=fingerprint,
                                             on_cost=on_cost, stack=prepare)
                    if result["trial_id"] != identifier or result["fingerprint"] != fingerprint:
                        raise ValueError("Worker result identity mismatch")
                    results[identifier] = result
                    budget.note_failure(result.get("error_category"))
                    if budget.stop_reason:
                        budget_error = budget.stop_reason
                    print("TRIAL_SAVED " + json.dumps({"task": result["task"], "selector": result["selector"], "status": result["status"], "duration_s": result["duration_s"], "catalog_estimated_cost_usd": result["catalog_estimated_cost_usd"]}), flush=True)
                    if result.get("error_category") in CAMPAIGN_FATAL:
                        # The trial's own bridge connection was refused (wrong destination or profile)
                        # or preparation found a wrong machine setup; the campaign stops here.
                        previous = None
                        raise StackError(result["error_category"])
                    if not verified:
                        # The bridge is used only on a stack that passed the gate; prepare has
                        # already stopped any stack it owned.
                        previous = None
                        continue
                    # Abort a leftover recipe, keep the run artifacts and restore the recipe library.
                    try:
                        await stack.finish(campaign / "results" / identifier, started_utc)
                        previous = result
                    except StackError as exc:
                        print("STACK_FINISH_FAILED " + json.dumps({"trial_id": identifier, "error_category": exc.code}), flush=True)
                        previous = None
                        if exc.code in CAMPAIGN_FATAL:
                            raise
                finally:
                    state["active_trials"].remove(identifier)
                    save_state()

        save_state()
        print("CAMPAIGN_STARTED " + json.dumps({"pid": os.getpid(), "mode": state["mode"], "planned_trials": planned_trials, "existing_trials": len(results), "parallelism": manifest.parallelism, "cost_threshold_usd": budget.limit}), flush=True)
        await asyncio.gather(*(worker() for _ in range(manifest.parallelism)))
        # Stop the stack this campaign started before declaring an outcome; a stack that does
        # not stop fails the campaign (StackError("stack_stop_failed") through the handler below).
        closed = True
        await stack.close()
        if budget_error:
            state["status"], state["stop_reason"] = "stopped", budget_error
        elif (campaign / "STOP").exists():
            state["status"], state["stop_reason"] = "stopped", "operator_stop_file"
        elif cursor < len(pending):
            state["status"], state["stop_reason"] = "stopped", "estimated_cost_threshold"
        else:
            state["status"] = "smoke_complete" if smoke else "collection_complete"
        save_state()
        atomic_json(campaign / ("smoke-summary.json" if smoke else "collection-summary.json"), {"state": state, "results": list(results.values()), "scoring_status": "not graded"})
        print("CAMPAIGN_SETTLED " + json.dumps(state), flush=True)
        return 1 if smoke and state["failed_trials"] else 0
    except BaseException as exc:
        state["status"] = "interrupted_or_error"
        state["stop_reason"] = exc.code if isinstance(exc, StackError) else "coordinator_exception"
        state["recorded_catalog_cost_usd"] = budget.total
        if not closed:
            closed = True
            try:
                await stack.close()
            except StackError as close_error:
                state["stack_close_error"] = close_error.code
        atomic_json(campaign / ("smoke-progress.json" if smoke else "progress.json"), state)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path, help="Absolute path of the frozen campaign folder (holds campaign.json)")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: run every task once for each model in the manifest, writing smoke-progress.json and smoke-summary.json")
    parser.add_argument("--track", choices=("sputter", "ads", "cluster", "all"), default=None,
                        help="Sub-track to run when --campaign points at a multi-track root")
    parser.add_argument("--verify-only", action="store_true", help="Verify the frozen configuration and existing trial evidence without model calls")
    args = parser.parse_args()
    if not args.campaign.is_absolute():
        parser.error("--campaign must be an absolute path")
    tracks_manifest = args.campaign / "tracks.json"
    if tracks_manifest.is_file():
        selected = ["sputter", "ads", "cluster"] if args.track in (None, "all") else [args.track]
        targets = [args.campaign / name for name in selected]
    elif args.track and args.track != "all" and (args.campaign / args.track / "campaign.json").is_file():
        targets = [args.campaign / args.track]
    else:
        targets = [args.campaign]
    if args.verify_only:
        for target in targets:
            manifest, settings, tasks = load_campaign(target)
            results = existing_results(target, settings, manifest.fingerprint, Budget(manifest.cost_threshold_usd))
            print("CAMPAIGN_VERIFY_PASS " + json.dumps({"campaign": target.name, "models": len(manifest.models), "tasks": len(tasks), "planned_trials": len(manifest.task_ids) * len(manifest.models), "saved_trials": len(results)}))
        return 0
    code = 0
    for target in targets:
        rc_code = asyncio.run(launch(target, args.smoke))
        if rc_code != 0:
            code = rc_code
    return code


if __name__ == "__main__":
    sys.exit(main())
