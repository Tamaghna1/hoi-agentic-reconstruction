#!/usr/bin/env python3
"""Smoke test for Condition E's scheduler change (object_scheduler_type='linear').

Isolated from A/B/C/D and from the other smoke tests: uses its own
results/smoke_test_scheduler/ output dir and the same small window convention
(start_frame=10, end_frame=15) as the other smoke tests.

Reuses the diagnostics-log validation from run_diagnostics_smoke_test.py (JSON
validity, per-key iteration contiguity, termination-record cross-check against
runner.convergence_events) and adds the one new check this experiment actually
needs: that the object optimizer's learning rate genuinely decays as the
'linear' scheduler predicts, not just that a scheduler object was constructed.

Does not touch results/A_released, B_smoothness, C_control_iters800,
D_control_iters1600, E_linear_schedule, or either prior smoke-test output dir.
"""
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/users/anirban/tamaghnam/HOI")
REPO = ROOT / "repo"
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))
import run_experiment as wrapper  # reuse approved config-loading logic, don't duplicate

SMOKE_START = 10
SMOKE_END = 15
SMOKE_OUTDIR = ROOT / "results" / "smoke_test_scheduler"
SMOKE_LOGDIR = ROOT / "logs" / "experiments" / "smoke_test_scheduler"
DIAG_LOG_PATH = SMOKE_LOGDIR / "diagnostics.jsonl"

# Must match CONDITIONS["E"] in run_experiment.py exactly - this smoke test exists
# to validate that exact delta before E is ever submitted as a real job.
SCHEDULER_DELTA = {"object_scheduler_type": "linear"}
EXPECTED_END_FACTOR = 0.1   # LinearLR(start_factor=1.0, end_factor=0.1, ...) per trainer.py's create_scheduler


def build_smoke_config():
    base = wrapper.load_released_config()
    scene_overrides = wrapper.load_scene_overrides()
    effective = dict(base)
    effective.update(scene_overrides)
    assert effective["end_frame"] == wrapper.WINDOW_END, "unexpected: approved window already mutated"
    effective["end_frame"] = SMOKE_END
    effective["result_dir"] = str(SMOKE_OUTDIR)
    effective.update(SCHEDULER_DELTA)
    assert effective["object_scheduler_type"] == "linear"
    return effective


def is_finite_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def validate_record(rec, line_no):
    problems = []
    rtype = rec.get("record_type")
    if rtype not in ("iteration", "termination"):
        return [f"line {line_no}: unknown record_type {rtype!r}"]
    common_required = ["component", "frame_id", "iterations" if rtype == "termination" else "iteration", "budget"]
    for key in common_required:
        if key not in rec:
            problems.append(f"line {line_no}: missing key {key!r}")
    if rec.get("component") not in ("hand", "object"):
        problems.append(f"line {line_no}: unexpected component {rec.get('component')!r}")
    if rtype == "iteration":
        for key in ("total_loss", "gradient_inf", "step_norm", "pose_change"):
            v = rec.get(key)
            if not is_finite_number(v):
                problems.append(f"line {line_no}: {key}={v!r} is not a finite number")
        lrs = rec.get("learning_rates")
        if not isinstance(lrs, list) or not lrs or not all(is_finite_number(v) for v in lrs):
            problems.append(f"line {line_no}: learning_rates invalid: {lrs!r}")
    else:
        for key in ("status", "loss", "gradient_inf", "reason"):
            if key not in rec:
                problems.append(f"line {line_no}: termination record missing key {key!r}")
        if rec.get("status") not in ("converged", "stalled_at_budget", "budget_exhausted"):
            problems.append(f"line {line_no}: unexpected status {rec.get('status')!r}")
    return problems


def main():
    print("=== SCHEDULER SMOKE TEST: building effective config (released baseline + E's scheduler delta + smoke window) ===")
    effective = build_smoke_config()
    print(f"effective window: start_frame={effective['start_frame']}, end_frame={effective['end_frame']}")
    print(f"object_scheduler_type={effective['object_scheduler_type']} (expect 'linear', vs released default 'constant')")

    SMOKE_OUTDIR.mkdir(parents=True, exist_ok=True)
    SMOKE_LOGDIR.mkdir(parents=True, exist_ok=True)
    if DIAG_LOG_PATH.exists():
        DIAG_LOG_PATH.unlink()

    os.environ["AGILE_DIAGNOSTIC_LOG"] = str(DIAG_LOG_PATH)
    print(f"AGILE_DIAGNOSTIC_LOG={DIAG_LOG_PATH}")

    problems_pre = wrapper.check_prerequisites(effective)
    if problems_pre:
        print("PREREQUISITE CHECK FAILED - stopping, not proceeding:")
        for p in problems_pre:
            print(f"  - {p}")
        sys.exit(1)
    print("Prerequisite checks passed.")

    sys.path.insert(0, str(REPO))
    from optimization.config.config import Config
    from optimization.pipelines.trainer import Runner

    print("=== Constructing Config and Runner ===")
    cfg = Config(**effective)
    assert cfg.object_scheduler_type == "linear", f"Config did not retain object_scheduler_type: {cfg.object_scheduler_type!r}"
    runner = Runner(cfg)
    print("=== Runner constructed. Starting train() ===")
    runner.train()
    print("=== train() returned without raising ===")

    print("\n=== VALIDATING DIAGNOSTICS LOG ===")
    if not DIAG_LOG_PATH.is_file():
        print(f"FAIL: diagnostics log not found at {DIAG_LOG_PATH}")
        sys.exit(1)

    all_problems = []
    iter_counts = defaultdict(list)   # (frame_id, component) -> list of (iteration, learning_rates) in file order
    term_counts = defaultdict(int)
    n_lines = 0
    with open(DIAG_LOG_PATH) as f:
        for line_no, line in enumerate(f, start=1):
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                all_problems.append(f"line {line_no}: invalid JSON: {e}")
                continue
            all_problems.extend(validate_record(rec, line_no))
            key = (rec.get("frame_id"), rec.get("component"))
            if rec.get("record_type") == "iteration":
                iter_counts[key].append((rec.get("iteration"), rec.get("learning_rates")))
            elif rec.get("record_type") == "termination":
                term_counts[key] += 1

    print(f"Total diagnostic lines: {n_lines}")
    n_convergence_events = len(getattr(runner, "convergence_events", []))
    n_termination_records = sum(term_counts.values())
    print(f"runner.convergence_events (unmodified pre-existing mechanism): {n_convergence_events}")
    print(f"termination diagnostic records (new mechanism): {n_termination_records}")
    if n_convergence_events != n_termination_records:
        all_problems.append(
            f"MISMATCH: {n_convergence_events} convergence_events vs {n_termination_records} "
            f"termination diagnostic records."
        )

    # ---- The check this smoke test exists for: does the object LR actually decay? ----
    print("\n=== VERIFYING OBJECT LR DECAY (the actual point of Condition E) ===")
    object_keys = [k for k in iter_counts if k[1] == "object"]
    if not object_keys:
        all_problems.append("No object-component iteration records found at all.")
    for key in object_keys:
        rows = iter_counts[key]  # already in file order == iteration order
        first_lrs = rows[0][1]
        last_lrs = rows[-1][1]
        if not (isinstance(first_lrs, list) and isinstance(last_lrs, list) and len(first_lrs) == len(last_lrs)):
            all_problems.append(f"{key}: malformed learning_rates, first={first_lrs!r} last={last_lrs!r}")
            continue
        for i, (lr0, lrN) in enumerate(zip(first_lrs, last_lrs)):
            if lr0 <= 0:
                all_problems.append(f"{key}: optimizer[{i}] initial LR={lr0} is not positive")
                continue
            ratio = lrN / lr0
            # LinearLR(start_factor=1.0, end_factor=0.1, total_iters=train_max_iter) means the
            # factor at the LAST step (index total_iters-1) should be very close to end_factor=0.1.
            # Allow generous tolerance (0.08-0.13) for a short 5-frame, default-iteration-budget run.
            if not (0.08 <= ratio <= 0.13):
                all_problems.append(
                    f"{key}: optimizer[{i}] LR did not decay as expected: first={lr0}, last={lrN}, "
                    f"ratio={ratio:.4f} (expected ~{EXPECTED_END_FACTOR})"
                )
            else:
                print(f"  {key} optimizer[{i}]: LR {lr0} -> {lrN} (ratio {ratio:.4f}, expected ~{EXPECTED_END_FACTOR}) OK")
        # Also confirm it's a genuine monotonic-ish decay, not just endpoints differing (e.g. noise).
        mid_idx = len(rows) // 2
        mid_lrs = rows[mid_idx][1]
        for i, (lr0, lrM, lrN) in enumerate(zip(first_lrs, mid_lrs, last_lrs)):
            if not (lr0 >= lrM >= lrN - 1e-12):
                all_problems.append(
                    f"{key}: optimizer[{i}] LR not monotonically non-increasing: "
                    f"first={lr0}, mid={lrM}, last={lrN}"
                )
            else:
                print(f"  {key} optimizer[{i}]: monotonic decay confirmed (first={lr0} >= mid={lrM} >= last={lrN})")

    if all_problems:
        print(f"\nFAIL: {len(all_problems)} problem(s) found:")
        for p in all_problems[:50]:
            print(f"  - {p}")
        if len(all_problems) > 50:
            print(f"  ... and {len(all_problems) - 50} more")
        sys.exit(1)

    print("\n=== SCHEDULER SMOKE TEST PASSED ===")
    print(f"Object LR confirmed to decay from 1.0x to ~{EXPECTED_END_FACTOR}x monotonically, "
          f"matching the 'linear' scheduler's expected behavior. Logging mechanisms consistent.")


if __name__ == "__main__":
    main()
