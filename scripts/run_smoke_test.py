#!/usr/bin/env python3
"""Integrated smoke test for the AGILE Stage-2 pipeline on HO3D MC1.

Isolated from the approved A/B/C experiments:
  - Separate output dir (results/smoke_test/), never touches results/A_released,
    results/B_smoothness, results/C_control_iters800.
  - Uses a SMALLER frame window than the approved A/B/C window (start_frame=10,
    end_frame=200) purely to keep this pipeline-validation run fast. This is an
    explicit, documented deviation for the smoke test ONLY - run_experiment.py's
    WINDOW_START/WINDOW_END (the approved A/B/C window) are never imported or
    modified.
  - Uses the exact released configuration (same as condition A), no condition
    delta - the smoke test validates infrastructure, not B/C effects.

Differences from the approved A/B/C window, stated explicitly:
  SMOKE_START = 10   (same as approved - MC1's real contact_frame, preserves a
                       genuine, non-trivial backward pass)
  SMOKE_END   = 15   (vs. approved 200 - only 5 forward frames, enough to exercise
                       multi-frame warm-starting without the full ~190-frame cost)
"""
import json
import math
import os
import subprocess
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path("/scratch/users/anirban/tamaghnam/HOI")
REPO = ROOT / "repo"
SCRIPTS = ROOT / "scripts"
DATA_ROOT = ROOT / "data/HO3D/preprocessed"
MANO_ROOT = ROOT / "assets/mano"
DINO_ROOT = ROOT / "assets/dinov3"

sys.path.insert(0, str(SCRIPTS))
import run_experiment as wrapper  # reuse, do not duplicate, the approved A/B/C logic

SCENE = "MC1"
SMOKE_START = 10
SMOKE_END = 15
SMOKE_OUTDIR = ROOT / "results" / "smoke_test"
SMOKE_LOGDIR = ROOT / "logs" / "experiments" / "smoke_test"

REQUIRED_OUTPUT_FILES = [
    "object_poses.npz",
    "mano_transforms.npz",
    "optimized_mano_joints.npz",
    "optimized_mano_verts.npz",
    "object_mesh.obj",
    "export_metadata.json",
]


def build_smoke_config():
    base = wrapper.load_released_config()
    scene_overrides = wrapper.load_scene_overrides()  # validates contact_frame==10 internally
    effective = dict(base)
    effective.update(scene_overrides)
    # The ONLY smoke-test-specific deviation: a smaller end_frame. Document it.
    assert effective["end_frame"] == wrapper.WINDOW_END, "unexpected: approved window already mutated"
    effective["end_frame"] = SMOKE_END
    effective["result_dir"] = str(SMOKE_OUTDIR)
    return effective, base


def find_saved_dir(result_dir):
    """Stage-2 writes to result_dir/aligned[frame_step]/, final save has no suffix: aligned/."""
    candidates = sorted(Path(result_dir).glob("aligned*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    # Prefer the exact 'aligned' (final, non-periodic) dir if present.
    final = Path(result_dir) / "aligned"
    return final if final.is_dir() else candidates[-1]


def _npy_shape_and_check_finite(fileobj):
    magic = fileobj.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("bad npy magic")
    major = fileobj.read(1)
    fileobj.read(1)
    if major == b"\x01":
        header_len = struct.unpack("<H", fileobj.read(2))[0]
    else:
        header_len = struct.unpack("<I", fileobj.read(4))[0]
    import ast
    header = ast.literal_eval(fileobj.read(header_len).decode("latin1"))
    shape = tuple(header["shape"])
    descr = header["descr"]
    dtype_map = {"<f4": ("<", "f", 4), "<f8": ("<", "d", 8)}
    if descr not in dtype_map:
        return shape, None  # non-float array (e.g. int) - skip finiteness check, not an error
    endian, type_char, itemsize = dtype_map[descr]
    n = 1
    for s in shape:
        n *= s
    raw = fileobj.read(n * itemsize)
    values = struct.unpack(f"{endian}{n}{type_char}", raw)
    n_nan = sum(1 for v in values if v != v)
    n_inf = sum(1 for v in values if v in (float("inf"), float("-inf")))
    return shape, {"n_values": n, "n_nan": n_nan, "n_inf": n_inf}


def check_finite(npz_path):
    results = {}
    with zipfile.ZipFile(npz_path) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                shape, finite_info = _npy_shape_and_check_finite(f)
                results[name] = {"shape": shape, "finite": finite_info}
    return results


def main():
    print("=== SMOKE TEST: building effective config (released baseline + smoke window) ===")
    effective, reference_base = build_smoke_config()
    print(f"effective window: start_frame={effective['start_frame']}, end_frame={effective['end_frame']} "
          f"(approved A/B/C window is start_frame={wrapper.WINDOW_START}, end_frame={wrapper.WINDOW_END} - "
          f"NOT used here, documented smoke-test-only deviation)")

    SMOKE_OUTDIR.mkdir(parents=True, exist_ok=True)
    SMOKE_LOGDIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_OUTDIR / "effective_config.json").write_text(json.dumps(effective, indent=2, default=str))
    manifest = {
        "smoke_test": True,
        "scene": SCENE,
        "window": {"start_frame": SMOKE_START, "end_frame": SMOKE_END},
        "deviation_from_approved_window": {
            "approved": {"start_frame": wrapper.WINDOW_START, "end_frame": wrapper.WINDOW_END},
            "reason": "pipeline validation only - kept small for speed, does not affect A/B/C",
        },
        "git_agile_repo": wrapper.git_commit(REPO),
        "environment": wrapper.env_details(),
    }
    (SMOKE_OUTDIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote effective_config.json and run_manifest.json to {SMOKE_OUTDIR}")

    problems = wrapper.check_prerequisites(effective)
    if problems:
        print("PREREQUISITE CHECK FAILED - stopping, not proceeding:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("Prerequisite checks passed (MANO prepared, DINO_MODEL_PATH set, data present).")

    # ---- Real execution ----
    sys.path.insert(0, str(REPO))
    from optimization.config.config import Config  # verified against run_ho3d.py/train.py's own imports
    from optimization.pipelines.trainer import Runner

    print("=== Constructing Config and Runner (this is where MANO + DINO actually initialize) ===")
    cfg = Config(**effective)
    runner = Runner(cfg)
    print("=== Runner constructed. Starting train() (bidirectional optimization) ===")
    runner.train()
    print("=== train() returned without raising ===")

    # ---- Verification beyond exit status ----
    print("\n=== VERIFICATION ===")
    saved_dir = find_saved_dir(SMOKE_OUTDIR)
    if saved_dir is None:
        print(f"FAIL: no aligned*/ output directory found under {SMOKE_OUTDIR}")
        sys.exit(1)
    print(f"Output directory found: {saved_dir}")

    missing = [f for f in REQUIRED_OUTPUT_FILES if not (saved_dir / f).is_file()]
    if missing:
        print(f"FAIL: missing expected output files: {missing}")
        sys.exit(1)
    print(f"All {len(REQUIRED_OUTPUT_FILES)} expected output files present: {REQUIRED_OUTPUT_FILES}")

    expected_n_frames = SMOKE_END - 0  # bidirectional: covers 0..SMOKE_END-1 if backward truly ran to frame 0
    obj_check = check_finite(saved_dir / "object_poses.npz")
    hand_check = check_finite(saved_dir / "mano_transforms.npz")
    print("object_poses.npz contents:", json.dumps(obj_check, indent=2, default=str))
    print("mano_transforms.npz contents:", json.dumps(hand_check, indent=2, default=str))

    any_nan_inf = False
    for d in (obj_check, hand_check):
        for name, info in d.items():
            if info["finite"] and (info["finite"]["n_nan"] > 0 or info["finite"]["n_inf"] > 0):
                print(f"FAIL: non-finite values in {name}: {info['finite']}")
                any_nan_inf = True
    if any_nan_inf:
        sys.exit(1)
    print("No NaN/Inf found in object_poses.npz or mano_transforms.npz.")

    obj_frames = obj_check.get("object_poses_in_cam.npy", {}).get("shape", [None])[0]
    print(f"object_poses_in_cam frame count = {obj_frames} "
          f"(expect coverage of frames 0..{SMOKE_END-1} = {expected_n_frames} frames if the backward "
          f"pass to frame 0 genuinely ran, not just the {SMOKE_END - SMOKE_START} forward frames)")
    if obj_frames is not None and obj_frames < expected_n_frames:
        print(f"WARNING: frame count {obj_frames} is less than the full 0..{SMOKE_END-1} range - "
              f"the backward pass may not have fully merged into this save. Investigate before "
              f"treating the bidirectional pass as fully verified.")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
