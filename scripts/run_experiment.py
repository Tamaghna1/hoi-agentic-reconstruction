#!/usr/bin/env python3
"""A/B/C experiment wrapper for AGILE Stage-2 on HO3D MC1.

Loads the RELEASED repo/configs/ho3d.json verbatim, applies MC1's scene overrides
(exactly as run_ho3d.py would), then applies exactly one condition-specific delta.
Never uses bare train.py class defaults as a substitute for the released config.

Usage:
    python3 run_experiment.py --condition {A,B,C} --check-only   # stdlib-only, no AGILE imports
    python3 run_experiment.py --condition {A,B,C}                # full run (requires agile-optim env)
    python3 run_experiment.py --condition {A,B,C} --resume       # skip if already completed

Design requirements this satisfies (see plan.md "FROZEN RESEARCH DECISIONS"):
  1. Loads the exact released configs/ho3d.json.                      -> load_released_config()
  2. Loads the MC1 scene configuration.                                -> load_scene_overrides()
  3. Applies only the approved per-condition override.                 -> CONDITIONS, apply_condition()
  4. Uses the approved frame window.                                   -> WINDOW_START/END in scene overrides
  5. Records the effective configuration for every run.                -> write_effective_config()
  6. Fails if any unexpected configuration difference is detected.     -> assert_single_field_diff()
  7. Uses separate output/log directories for A, B, C.                 -> result_dir()/log_dir()
  8. Supports resumption and clear failure reporting.                  -> --resume, STATUS_* markers
  9. Records commit, env details, dataset identity, seed.              -> write_run_manifest()
 10. Does not download missing models automatically.                   -> check_prerequisites() only checks, never fetches
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/scratch/users/anirban/tamaghnam/HOI")
REPO = ROOT / "repo"
DATA_ROOT = ROOT / "data/HO3D/preprocessed"
MANO_ROOT = ROOT / "assets/mano"
RESULTS_ROOT = ROOT / "results"
LOGS_ROOT = ROOT / "logs"

SCENE = "MC1"
WINDOW_START = 10          # == MC1's contact_frame; fixed by scene design, not overridden here
WINDOW_END = 200            # approved pilot window (frozen decision)
KNOWN_SEED = 42             # Runner.__init__ hard-codes set_random_seed(42) - recorded, not re-chosen

CONDITIONS = {
    "A": {},                                       # released baseline, no delta
    "B": {"object_smoothness_loss": True},          # + vertex smoothness only
    "C": {"object_max_iter_per_frame": 800},        # + compute-matched control only
}

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED"


def load_released_config():
    path = REPO / "configs/ho3d.json"
    if not path.is_file():
        raise FileNotFoundError(f"Released config missing: {path}")
    return json.loads(path.read_text())


def load_scene_overrides():
    path = REPO / "configs/ho3d_scenes.json"
    if not path.is_file():
        raise FileNotFoundError(f"Scene manifest missing: {path}")
    scenes = json.loads(path.read_text())
    if SCENE not in scenes:
        raise ValueError(f"Scene {SCENE} not in released manifest")
    entry = scenes[SCENE]
    if entry.get("contact_frame") != WINDOW_START:
        raise ValueError(
            f"WINDOW_START={WINDOW_START} no longer matches {SCENE}'s released "
            f"contact_frame={entry.get('contact_frame')} - scene manifest may have changed; "
            f"stop and re-verify before proceeding, do not silently adjust."
        )
    return {
        "data_dir": str(DATA_ROOT / "train" / SCENE),
        "mano_model_path": str(MANO_ROOT),
        "start_frame": entry["contact_frame"],
        "end_frame": WINDOW_END,
        # NOTE: result_dir is intentionally NOT set here - it is condition-specific
        # and is added in build_effective_config() below, per condition, so that
        # AGILE's Config.get_result_dir() uses our isolated directory directly
        # instead of falling back to its own auto-generated results/{scene}/{timestamp}_
        # {loss_info} path (a real bug found after A/B/C's first real run - see plan.md).
    }


def assert_single_field_diff(effective, reference, expected_delta_keys, condition):
    diff_keys = sorted(k for k in effective if effective.get(k) != reference.get(k))
    expected = sorted(expected_delta_keys)
    if diff_keys != expected:
        raise RuntimeError(
            f"Condition {condition}: unexpected config diff vs released baseline. "
            f"expected changed keys={expected}, actual changed keys={diff_keys}. "
            f"Refusing to proceed - this would silently change the experiment."
        )


def build_effective_config(condition):
    base = load_released_config()
    scene_overrides = load_scene_overrides()

    reference = dict(base)
    reference.update(scene_overrides)
    # Set explicitly, identically in reference (and therefore effective, via the
    # copy below) so AGILE's Config.get_result_dir() uses our isolated directory
    # directly instead of auto-generating its own results/{scene}/{timestamp}_
    # {loss_info} path. Setting it here (not inside CONDITIONS' per-condition
    # delta) keeps it out of the single-field-diff check - it is infrastructure,
    # not a research setting, and is identical in reference/effective by
    # construction so it can never appear as an unintended diff.
    reference["result_dir"] = str(result_dir(condition))

    delta = CONDITIONS[condition]
    effective = dict(reference)
    effective.update(delta)

    assert_single_field_diff(effective, reference, delta.keys(), condition)
    return effective, reference


def result_dir(condition):
    names = {"A": "A_released", "B": "B_smoothness", "C": "C_control_iters800"}
    return RESULTS_ROOT / names[condition]


def log_dir(condition):
    names = {"A": "A_released", "B": "B_smoothness", "C": "C_control_iters800"}
    d = LOGS_ROOT / "experiments" / names[condition]
    return d


def git_commit(repo_path):
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        commit = out.stdout.strip()
    except Exception as e:
        return {"commit": f"UNAVAILABLE ({e})", "dirty": None, "dirty_files": None}
    try:
        status = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        dirty_files = [l for l in status.stdout.splitlines() if l.strip()]
    except Exception as e:
        return {"commit": commit, "dirty": None, "dirty_files": f"UNAVAILABLE ({e})"}
    return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files}


def env_details():
    details = {"python": sys.version, "conda_prefix": os.environ.get("CONDA_PREFIX", "UNSET")}
    try:
        import torch  # noqa: local import, may not exist yet (env still building)
        details["torch"] = torch.__version__
        details["torch_cuda"] = torch.version.cuda
        details["cuda_available"] = torch.cuda.is_available()
    except Exception as e:
        details["torch"] = f"UNAVAILABLE ({e})"
    # Best-effort versions of other pinned dependencies that matter for reproducibility,
    # beyond torch alone - each independently optional so one missing package doesn't
    # blank out the rest.
    for pkg in ("numpy", "transformers", "trimesh", "scipy", "pytorch3d"):
        try:
            mod = __import__(pkg)
            details[pkg] = getattr(mod, "__version__", "present (no __version__ attr)")
        except Exception as e:
            details[pkg] = f"UNAVAILABLE ({e})"
    return details


def write_run_manifest(condition, effective, outdir):
    manifest = {
        "condition": condition,
        "scene": SCENE,
        "window": {"start_frame": WINDOW_START, "end_frame": WINDOW_END},
        "git_agile_repo": git_commit(REPO),
        "environment": env_details(),
        "known_seed_runner_hardcoded": KNOWN_SEED,
        "dataset_identity": {
            "data_dir": effective["data_dir"],
            "integrity_verification": {
                "method": "scripts/validate_data_and_config.py",
                "slurm_job_id": "73176",
                "result": "897/897 frames verified present across every subfolder, 0 errors, "
                           "2 benign warnings (per-sequence constants, not per-frame data)",
            },
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def write_effective_config(condition, effective, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "effective_config.json").write_text(json.dumps(effective, indent=2, default=str))


def check_prerequisites(effective):
    """Checks only. Never downloads or fetches anything missing."""
    problems = []
    mano_cfg = Path(effective["mano_model_path"]) / "mano_config.yaml"
    if not mano_cfg.is_file():
        problems.append(
            f"MANO not prepared: {mano_cfg} missing. "
            f"Run tools/prepare_mano_assets.py first (not done automatically by this wrapper)."
        )
    data_dir = Path(effective["data_dir"])
    if not data_dir.is_dir():
        problems.append(f"Dataset directory missing: {data_dir}")
    if effective.get("object_feature_loss"):
        dino_path = os.environ.get("DINO_MODEL_PATH")
        if not dino_path or not Path(dino_path).is_dir():
            problems.append(
                "object_feature_loss=True but DINO_MODEL_PATH is unset or missing. "
                "This wrapper will NOT download the gated DINOv3 checkpoint automatically."
            )
    return problems


def status_path(outdir):
    return outdir / "STATUS"


def read_status(outdir):
    p = status_path(outdir)
    return p.read_text().strip() if p.is_file() else None


def write_status(outdir, status):
    outdir.mkdir(parents=True, exist_ok=True)
    status_path(outdir).write_text(status + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=["A", "B", "C"], required=True)
    ap.add_argument("--check-only", action="store_true",
                     help="Validate config/prerequisites and write manifests; do not run AGILE.")
    ap.add_argument("--resume", action="store_true",
                     help="Skip if STATUS already COMPLETE in this condition's result dir.")
    args = ap.parse_args()

    outdir = result_dir(args.condition)
    ldir = log_dir(args.condition)
    ldir.mkdir(parents=True, exist_ok=True)

    if args.resume and read_status(outdir) == STATUS_COMPLETE:
        print(f"[{args.condition}] already COMPLETE at {outdir}, skipping (--resume).")
        return 0

    print(f"[{args.condition}] building effective config from released ho3d.json + {SCENE} scene overrides")
    effective, reference = build_effective_config(args.condition)
    print(f"[{args.condition}] config diff vs released baseline: "
          f"{sorted(k for k in effective if effective.get(k) != reference.get(k))}")

    write_effective_config(args.condition, effective, outdir)
    manifest = write_run_manifest(args.condition, effective, outdir)
    print(f"[{args.condition}] wrote effective_config.json and run_manifest.json to {outdir}")
    print(f"[{args.condition}] environment: {manifest['environment']}")

    problems = check_prerequisites(effective)
    if problems:
        write_status(outdir, STATUS_FAILED)
        print(f"[{args.condition}] FAILED prerequisite checks:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if args.check_only:
        print(f"[{args.condition}] --check-only: all prerequisite checks passed. Not launching AGILE.")
        return 0

    # ---- Real execution path: only reached without --check-only, and only if all
    #      prerequisite checks above passed. Intentionally NOT exercised yet.
    write_status(outdir, STATUS_RUNNING)
    try:
        sys.path.insert(0, str(REPO))
        from optimization.config.config import Config       # noqa: deferred import; verified against run_ho3d.py/train.py
        from optimization.pipelines.trainer import Runner  # noqa: deferred import
        cfg = Config(**effective)
        runner = Runner(cfg)
        runner.train()
        write_status(outdir, STATUS_COMPLETE)
        print(f"[{args.condition}] COMPLETE")
        return 0
    except Exception as e:
        write_status(outdir, STATUS_FAILED)
        print(f"[{args.condition}] FAILED during training: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
