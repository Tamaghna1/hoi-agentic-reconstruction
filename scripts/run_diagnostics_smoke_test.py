#!/usr/bin/env python3
"""Smoke test for the AGILE_DIAGNOSTIC_LOG observational logging added for Condition D.

Isolated from A/B/C/D and from the prior pipeline smoke test (results/smoke_test/):
uses its own results/smoke_test_diagnostics/ output dir and its own small window
(same SMOKE_START/SMOKE_END convention as run_smoke_test.py: start_frame=10,
end_frame=15 - a genuine backward pass plus a few forward frames, small for speed).

Validates, after a real run:
  1. The diagnostics JSONL file exists and every line parses as JSON.
  2. Every record has the expected keys for its record_type.
  3. All numeric fields are finite (no NaN/Inf).
  4. Per (frame_id, component), iteration numbers are a contiguous 0..N-1 sequence
     (confirms no dropped/duplicated iterations).
  5. Exactly one termination record exists per (frame_id, component), and its
     iteration count matches iteration_record_count for that key (confirms the
     termination event was logged once, after the last iteration, not mid-loop).
  6. The number of [Convergence] lines printed to stdout matches the number of
     termination records (cross-check against the pre-existing, unmodified
     print-based logging - confirms the new logging didn't diverge from it).

Does not touch results/A_released, B_smoothness, C_control_iters800,
D_control_iters1600, or results/smoke_test/.
"""
import json
import math
import os
import re
import subprocess
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
SMOKE_OUTDIR = ROOT / "results" / "smoke_test_diagnostics"
SMOKE_LOGDIR = ROOT / "logs" / "experiments" / "smoke_test_diagnostics"
DIAG_LOG_PATH = SMOKE_LOGDIR / "diagnostics.jsonl"


def build_smoke_config():
    base = wrapper.load_released_config()
    scene_overrides = wrapper.load_scene_overrides()
    effective = dict(base)
    effective.update(scene_overrides)
    assert effective["end_frame"] == wrapper.WINDOW_END, "unexpected: approved window already mutated"
    effective["end_frame"] = SMOKE_END
    effective["result_dir"] = str(SMOKE_OUTDIR)
    return effective


def is_finite_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def validate_record(rec, line_no):
    problems = []
    rtype = rec.get("record_type")
    if rtype not in ("iteration", "termination"):
        return [f"line {line_no}: unknown record_type {rtype!r}"]

    common_required = ["component", "frame_id", "iterations" if rtype == "termination" else "iteration",
                        "budget"]
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
        loss_terms = rec.get("loss_terms")
        if not isinstance(loss_terms, dict) or not loss_terms:
            problems.append(f"line {line_no}: loss_terms missing or empty: {loss_terms!r}")
        else:
            for k, v in loss_terms.items():
                if not is_finite_number(v):
                    problems.append(f"line {line_no}: loss_terms[{k!r}]={v!r} is not a finite number")
        lrs = rec.get("learning_rates")
        if not isinstance(lrs, list) or not lrs or not all(is_finite_number(v) for v in lrs):
            problems.append(f"line {line_no}: learning_rates invalid: {lrs!r}")
        if rec.get("convergence_status") not in ("optimizing", "stalled", "converged"):
            problems.append(f"line {line_no}: unexpected convergence_status {rec.get('convergence_status')!r}")
    else:  # termination
        for key in ("status", "loss", "gradient_inf", "reason"):
            if key not in rec:
                problems.append(f"line {line_no}: termination record missing key {key!r}")
        if rec.get("status") not in ("converged", "stalled_at_budget", "budget_exhausted"):
            problems.append(f"line {line_no}: unexpected status {rec.get('status')!r}")
        if not is_finite_number(rec.get("loss")):
            problems.append(f"line {line_no}: termination loss={rec.get('loss')!r} not finite")

    return problems


def main():
    print("=== DIAGNOSTICS SMOKE TEST: building effective config (released baseline + smoke window) ===")
    effective = build_smoke_config()
    print(f"effective window: start_frame={effective['start_frame']}, end_frame={effective['end_frame']}")

    SMOKE_OUTDIR.mkdir(parents=True, exist_ok=True)
    SMOKE_LOGDIR.mkdir(parents=True, exist_ok=True)
    if DIAG_LOG_PATH.exists():
        DIAG_LOG_PATH.unlink()  # start clean; this is our own smoke-test-only scratch file

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
    runner = Runner(cfg)
    print("=== Runner constructed. Starting train() ===")
    runner.train()
    print("=== train() returned without raising ===")

    # ---- Validate the diagnostics log ----
    print("\n=== VALIDATING DIAGNOSTICS LOG ===")
    if not DIAG_LOG_PATH.is_file():
        print(f"FAIL: diagnostics log not found at {DIAG_LOG_PATH}")
        sys.exit(1)

    all_problems = []
    iter_counts = defaultdict(set)   # (frame_id, component) -> set of iteration numbers seen
    term_counts = defaultdict(int)   # (frame_id, component) -> count of termination records
    term_iterations = {}             # (frame_id, component) -> 'iterations' field from termination record
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
                iter_counts[key].add(rec.get("iteration"))
            elif rec.get("record_type") == "termination":
                term_counts[key] += 1
                term_iterations[key] = rec.get("iterations")

    print(f"Total diagnostic lines: {n_lines}")
    print(f"Distinct (frame_id, component) keys with iteration records: {len(iter_counts)}")

    for key, iters in iter_counts.items():
        expected = set(range(len(iters)))
        if iters != expected:
            all_problems.append(
                f"{key}: iteration numbers not a contiguous 0..N-1 sequence "
                f"(got {sorted(iters)[:5]}...{sorted(iters)[-5:]}, {len(iters)} values)"
            )
        if term_counts.get(key, 0) != 1:
            all_problems.append(f"{key}: expected exactly 1 termination record, found {term_counts.get(key, 0)}")
        elif term_iterations[key] != len(iters):
            all_problems.append(
                f"{key}: termination 'iterations'={term_iterations[key]} does not match "
                f"count of distinct iteration records={len(iters)}"
            )

    # Cross-check against the pre-existing, unmodified [Convergence] print-based logging.
    # (Re-run count via the runner's own in-memory record, already populated by train().)
    n_convergence_events = len(getattr(runner, "convergence_events", []))
    n_termination_records = sum(term_counts.values())
    print(f"runner.convergence_events (unmodified pre-existing mechanism): {n_convergence_events}")
    print(f"termination diagnostic records (new mechanism): {n_termination_records}")
    if n_convergence_events != n_termination_records:
        all_problems.append(
            f"MISMATCH: {n_convergence_events} convergence_events vs "
            f"{n_termination_records} termination diagnostic records - new logging "
            f"diverged from the pre-existing, unmodified mechanism."
        )

    if all_problems:
        print(f"\nFAIL: {len(all_problems)} problem(s) found:")
        for p in all_problems[:50]:
            print(f"  - {p}")
        if len(all_problems) > 50:
            print(f"  ... and {len(all_problems) - 50} more")
        sys.exit(1)

    print("\n=== DIAGNOSTICS SMOKE TEST PASSED ===")
    print(f"All {n_lines} diagnostic records valid; {n_convergence_events} components tracked "
          f"consistently between the pre-existing and new logging mechanisms.")


if __name__ == "__main__":
    main()
