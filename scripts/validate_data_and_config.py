#!/usr/bin/env python3
"""Stdlib-only validation: MC1 data integrity + A/B/C config-diff correctness.

No numpy/torch required - deliberately independent of the agile-optim env build
so this can run in parallel with job 73170 without any dependency race.
"""
import ast
import json
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path("/scratch/users/anirban/tamaghnam/HOI")
MC1 = ROOT / "data/HO3D/preprocessed/train/MC1"
REPO = ROOT / "repo"
NUM_FRAMES = 897
WINDOW_START, WINDOW_END = 10, 200

errors = []
warnings = []


def check(cond, msg):
    if not cond:
        errors.append(msg)


def npy_shape(fileobj):
    magic = fileobj.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("bad npy magic")
    major = fileobj.read(1)
    fileobj.read(1)  # minor
    if major == b"\x01":
        header_len = struct.unpack("<H", fileobj.read(2))[0]
    else:
        header_len = struct.unpack("<I", fileobj.read(4))[0]
    header = fileobj.read(header_len).decode("latin1")
    d = ast.literal_eval(header)
    return tuple(d["shape"])


def npz_first_dims(path):
    dims = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                shape = npy_shape(f)
                dims[name] = shape
    return dims


# ---------------------------------------------------------------------------
print("=== Part 1: MC1 dataset integrity ===")

seq_dirs = {
    "rgb": MC1 / "rgb",
    "mask/hand": MC1 / "mask/hand",
    "mask/object": MC1 / "mask/object",
    "moge/depth": MC1 / "moge/depth",
    "object_pose/ob_in_cam": MC1 / "object_pose/ob_in_cam",
}
for label, d in seq_dirs.items():
    if not d.is_dir():
        errors.append(f"{label}: directory missing: {d}")
        continue
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    check(len(files) == NUM_FRAMES, f"{label}: expected {NUM_FRAMES} files, found {len(files)}")
    expected_names = {f"{i:05d}.png" for i in range(NUM_FRAMES)} if "txt" not in label else None
    if label == "object_pose/ob_in_cam":
        expected_names = {f"{i:05d}.txt" for i in range(NUM_FRAMES)}
    if expected_names is not None:
        missing = sorted(expected_names - set(files))
        extra = sorted(set(files) - expected_names)
        check(not missing, f"{label}: missing frames (first 5): {missing[:5]}")
        check(not extra, f"{label}: unexpected extra files (first 5): {extra[:5]}")
    print(f"  {label}: {len(files)} files" + (" OK" if len(files) == NUM_FRAMES else " MISMATCH"))

static_files = {
    "object_pose/original_mesh.obj": MC1 / "object_pose/original_mesh.obj",
    "object_pose/scaled_mesh.obj": MC1 / "object_pose/scaled_mesh.obj",
    "object_pose/material.mtl": MC1 / "object_pose/material.mtl",
    "object_pose/scales/unified_scale.txt": MC1 / "object_pose/scales/unified_scale.txt",
    "moge/cam_K.txt": MC1 / "moge/cam_K.txt",
}
for label, p in static_files.items():
    ok = p.is_file() and p.stat().st_size > 0
    check(ok, f"{label}: missing or empty: {p}")
    print(f"  {label}: {'OK' if ok else 'MISSING/EMPTY'} ({p.stat().st_size if p.exists() else 0} bytes)")

print("\n  -- NPZ array first-dimension checks (frame axis) --")
npz_files = {
    "wilor/mano_betas.npz": MC1 / "wilor/mano_betas.npz",
    "wilor/mano_global_orient.npz": MC1 / "wilor/mano_global_orient.npz",
    "wilor/mano_hand_pose.npz": MC1 / "wilor/mano_hand_pose.npz",
    "wilor/mano_masks.npz": MC1 / "wilor/mano_masks.npz",
    "wilor/mano_joints_2d.npz": MC1 / "wilor/mano_joints_2d.npz",
    "hand_processed/mano_transforms.npz": MC1 / "hand_processed/mano_transforms.npz",
    "hand_processed/optimized_mano_joints.npz": MC1 / "hand_processed/optimized_mano_joints.npz",
    "hand_processed/optimized_mano_verts.npz": MC1 / "hand_processed/optimized_mano_verts.npz",
}
for label, p in npz_files.items():
    if not p.is_file():
        errors.append(f"{label}: file missing: {p}")
        continue
    try:
        dims = npz_first_dims(p)
    except Exception as e:
        errors.append(f"{label}: failed to parse npy headers: {e}")
        continue
    for arrname, shape in dims.items():
        first_dim_ok = len(shape) > 0 and shape[0] == NUM_FRAMES
        if not first_dim_ok:
            warnings.append(
                f"{label}::{arrname}: shape={shape}, first dim != {NUM_FRAMES} "
                f"(may be a non-per-frame array, e.g. a constant/template - not necessarily an error)"
            )
        print(f"  {label}::{arrname}: shape={shape}")

check(WINDOW_START < WINDOW_END <= NUM_FRAMES, "approved window out of range for this sequence")
check(WINDOW_START >= 0, "approved window start < 0")
print(f"\n  Approved window [{WINDOW_START}, {WINDOW_END}) within [0, {NUM_FRAMES}): "
      f"{'OK' if WINDOW_START < WINDOW_END <= NUM_FRAMES else 'OUT OF RANGE'}")

# ---------------------------------------------------------------------------
print("\n=== Part 2: A/B/C config-diff validation ===")

ho3d_json_path = REPO / "configs/ho3d.json"
scenes_json_path = REPO / "configs/ho3d_scenes.json"
check(ho3d_json_path.is_file(), f"released config missing: {ho3d_json_path}")
check(scenes_json_path.is_file(), f"scene config missing: {scenes_json_path}")

if ho3d_json_path.is_file() and scenes_json_path.is_file():
    base = json.loads(ho3d_json_path.read_text())
    scenes = json.loads(scenes_json_path.read_text())
    check("MC1" in scenes, "MC1 not present in released scene manifest")
    mc1_scene = scenes.get("MC1", {})
    print(f"  MC1 scene entry: {mc1_scene}")
    check(mc1_scene.get("contact_frame") == 10, f"MC1 contact_frame expected 10, found {mc1_scene.get('contact_frame')}")

    # Per-condition scene overrides (same fields run_ho3d.py sets), plus approved window.
    def scene_overrides():
        return {
            "data_dir": str(MC1),  # run_ho3d.py sets data_dir = f"{DATA_ROOT}/train/{scene}" = this exact path
            "mano_model_path": str(ROOT / "assets/mano"),
            "start_frame": mc1_scene.get("contact_frame"),
            "end_frame": WINDOW_END,
        }

    conditions = {
        "A_released": {},
        "B_smoothness": {"object_smoothness_loss": True},
        "C_control_iters": {"object_max_iter_per_frame": 800},
    }

    for name, delta in conditions.items():
        effective = dict(base)
        effective.update(scene_overrides())
        effective.update(delta)

        # Compare against the TRUE released baseline (base + scene overrides only, no delta)
        reference = dict(base)
        reference.update(scene_overrides())

        diff_keys = sorted(k for k in effective if effective.get(k) != reference.get(k))
        expected_diff = sorted(delta.keys())
        ok = diff_keys == expected_diff
        check(ok, f"{name}: unexpected config diff vs released baseline. "
                  f"expected changed keys={expected_diff}, actual changed keys={diff_keys}")
        print(f"  {name}: changed keys vs released baseline = {diff_keys} "
              f"({'OK, matches intended single-field change' if ok else 'MISMATCH'})")

    # Spot-check the released settings the user explicitly asked to be preserved
    must_preserve = {
        "recover_object_pose_from_mask": True,
        "bidirectional_optimization": True,
        "ho_relpose_loss": True,
        "object_feature_loss": True,
        "fix_hand_rotation_to_wilor": True,
        "optimize_hand_pose": False,
        "object_smoothness_type": "vertices",
    }
    print("\n  -- Preserved-setting spot checks (released configs/ho3d.json values) --")
    for k, expected in must_preserve.items():
        actual = base.get(k)
        ok = actual == expected
        check(ok, f"preserved-setting mismatch: {k} expected {expected!r}, found {actual!r}")
        print(f"  {k}: {actual!r} ({'OK' if ok else 'MISMATCH'})")

# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"Errors: {len(errors)}")
for e in errors:
    print(f"  ERROR: {e}")
print(f"Warnings: {len(warnings)}")
for w in warnings:
    print(f"  WARN: {w}")

sys.exit(1 if errors else 0)
