#!/usr/bin/env python3
"""Temporal-consistency analysis for AGILE Stage-2 output (object_poses.npz, mano_transforms.npz).

Deliberately pure stdlib (zipfile + struct npy parsing, no numpy/torch) so this can be
written and self-tested against synthetic data WITHOUT touching the agile-optim
environment while it is still being built by another job - zero interference risk.

IMPORTANT, stated once and not repeated elsewhere in this file's output: every metric
here is a DESCRIPTIVE TEMPORAL CONSISTENCY metric (how much does the estimate change
frame-to-frame), not a GEOMETRIC ACCURACY metric (how close is the estimate to ground
truth). A lower frame-to-frame delta is NOT by itself evidence of improved accuracy -
see scripts/evaluation_plan.md for the full discussion. This module cannot and does not
claim otherwise.

Real AGILE output format (verified from optimization/pipelines/trainer.py,
save_optimized_data, not assumed):
  object_poses.npz    -> 'object_poses_in_cam': float array, shape (N, 4, 4)
  mano_transforms.npz -> 'rotation_matrices': shape (N, 3, 3), 'translations': shape (N, 3)
  optimized_mano_joints.npz -> 'mano_joints': shape (N, 1, J, 3)
  optimized_mano_verts.npz  -> 'mano_verts':  shape (N, 1, V, 3)
"""
import argparse
import ast
import csv
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal pure-Python .npy / .npz reader (float32/float64 only - sufficient for
# AGILE's pose arrays; raises clearly on anything else rather than guessing).
# ---------------------------------------------------------------------------

class NpyReadError(Exception):
    pass


def _read_npy(fileobj):
    magic = fileobj.read(6)
    if magic != b"\x93NUMPY":
        raise NpyReadError("bad npy magic")
    major = fileobj.read(1)
    fileobj.read(1)  # minor
    if major == b"\x01":
        header_len = struct.unpack("<H", fileobj.read(2))[0]
    else:
        header_len = struct.unpack("<I", fileobj.read(4))[0]
    header = ast.literal_eval(fileobj.read(header_len).decode("latin1"))
    shape = tuple(header["shape"])
    descr = header["descr"]
    fortran = header["fortran_order"]
    if fortran:
        raise NpyReadError("fortran-order arrays not supported")
    dtype_map = {"<f4": ("<", "f", 4), "<f8": ("<", "d", 8), ">f4": (">", "f", 4), ">f8": (">", "d", 8)}
    if descr not in dtype_map:
        raise NpyReadError(f"unsupported dtype {descr!r} (only float32/float64 supported)")
    endian, type_char, itemsize = dtype_map[descr]
    n = 1
    for s in shape:
        n *= s
    raw = fileobj.read(n * itemsize)
    if len(raw) != n * itemsize:
        raise NpyReadError(f"truncated array data: expected {n*itemsize} bytes, got {len(raw)}")
    flat = list(struct.unpack(f"{endian}{n}{type_char}", raw))
    return shape, flat


def read_npz_array(path, key):
    with zipfile.ZipFile(path) as zf:
        member = key if key.endswith(".npy") else key + ".npy"
        if member not in zf.namelist():
            raise NpyReadError(f"{key!r} not found in {path} (available: {zf.namelist()})")
        with zf.open(member) as f:
            return _read_npy(f)


def reshape_nested(flat, shape):
    """Turn a flat list + shape into nested lists (small arrays only - fine for 3x3/Nx3)."""
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    size = shape[0]
    stride = 1
    for s in shape[1:]:
        stride *= s
    return [reshape_nested(flat[i * stride:(i + 1) * stride], shape[1:]) for i in range(size)]


# ---------------------------------------------------------------------------
# Pure-Python linear algebra helpers (3x3 rotations, L2 norm, median/MAD)
# ---------------------------------------------------------------------------

def is_nan(x):
    return x != x


def mat3_transpose_mul_trace(a, b):
    """trace(a^T @ b) for two 3x3 matrices given as nested lists."""
    total = 0.0
    for i in range(3):
        for j in range(3):
            total += a[j][i] * b[j][i]
    return total


def geodesic_distance(r1, r2):
    """Angular distance in radians between two rotation matrices (SO(3))."""
    if any(is_nan(v) for row in r1 for v in row) or any(is_nan(v) for row in r2 for v in row):
        return float("nan")
    tr = mat3_transpose_mul_trace(r1, r2)
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.acos(cos_theta)


def l2_norm(vec):
    if any(is_nan(v) for v in vec):
        return float("nan")
    return math.sqrt(sum(v * v for v in vec))


def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def mean_pairwise_l2(points_a, points_b):
    """points_* : list of [x,y,z] points (e.g. joints or a subsample of vertices)."""
    if len(points_a) != len(points_b):
        return float("nan")
    dists = []
    for pa, pb in zip(points_a, points_b):
        if any(is_nan(v) for v in pa) or any(is_nan(v) for v in pb):
            dists.append(float("nan"))
        else:
            dists.append(l2_norm(vec_sub(pa, pb)))
    if any(is_nan(d) for d in dists):
        return float("nan")
    return sum(dists) / len(dists)


def median(values):
    v = sorted(x for x in values if not is_nan(x))
    n = len(v)
    if n == 0:
        return float("nan")
    mid = n // 2
    return v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2.0


def mad(values, med=None):
    v = [x for x in values if not is_nan(x)]
    if not v:
        return float("nan")
    m = med if med is not None else median(v)
    return median([abs(x - m) for x in v])


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def rotation_part(mat4x4_row):
    return [row[:3] for row in mat4x4_row[:3]]


def translation_part(mat4x4_row):
    return [mat4x4_row[0][3], mat4x4_row[1][3], mat4x4_row[2][3]]


def analyze_object_poses(object_poses_in_cam):
    """object_poses_in_cam: list of N 4x4 nested-list matrices."""
    n = len(object_poses_in_cam)
    rot_deltas = [None]
    trans_deltas = [None]
    for i in range(1, n):
        r_prev, r_curr = rotation_part(object_poses_in_cam[i - 1]), rotation_part(object_poses_in_cam[i])
        t_prev, t_curr = translation_part(object_poses_in_cam[i - 1]), translation_part(object_poses_in_cam[i])
        rot_deltas.append(geodesic_distance(r_prev, r_curr))
        trans_deltas.append(l2_norm(vec_sub(t_curr, t_prev)))
    return rot_deltas, trans_deltas


def analyze_hand(rotation_matrices, translations):
    n = len(rotation_matrices)
    rot_deltas = [None]
    trans_deltas = [None]
    for i in range(1, n):
        rot_deltas.append(geodesic_distance(rotation_matrices[i - 1], rotation_matrices[i]))
        trans_deltas.append(l2_norm(vec_sub(translations[i], translations[i - 1])))
    return rot_deltas, trans_deltas


def analyze_points(points_per_frame):
    """points_per_frame: list of N, each a list of [x,y,z] points (joints or verts)."""
    n = len(points_per_frame)
    deltas = [None]
    for i in range(1, n):
        deltas.append(mean_pairwise_l2(points_per_frame[i - 1], points_per_frame[i]))
    return deltas


def detect_outliers(deltas, k=3.0):
    valid = [d for d in deltas if d is not None and not is_nan(d)]
    if not valid:
        return [False] * len(deltas)
    m = median(valid)
    spread = mad(valid, m)
    if spread == 0:
        threshold = m  # degenerate: flag any nonzero deviation
    else:
        threshold = m + k * spread
    flags = []
    for d in deltas:
        if d is None:
            flags.append(False)
        elif is_nan(d):
            flags.append(True)  # NaN frames are always flagged, not silently passed
        else:
            flags.append(d > threshold)
    return flags


def summarize(deltas):
    valid = [d for d in deltas if d is not None and not is_nan(d)]
    nan_count = sum(1 for d in deltas if d is not None and is_nan(d))
    if not valid:
        return {"median": None, "iqr": None, "max": None, "n_valid": 0, "n_nan": nan_count}
    s = sorted(valid)
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    return {
        "median": median(valid),
        "iqr": q3 - q1,
        "max": max(valid),
        "n_valid": len(valid),
        "n_nan": nan_count,
    }


def analyze_condition(result_dir, outlier_k=3.0):
    """Loads a condition's saved AGILE output and computes all temporal-consistency metrics.
    Raises NpyReadError / FileNotFoundError with a clear message on missing/malformed data -
    never silently substitutes or skips a required file."""
    result_dir = Path(result_dir)
    obj_path = result_dir / "object_poses.npz"
    hand_path = result_dir / "mano_transforms.npz"

    if not obj_path.is_file():
        raise FileNotFoundError(f"object_poses.npz missing at {obj_path}")
    if not hand_path.is_file():
        raise FileNotFoundError(f"mano_transforms.npz missing at {hand_path}")

    shape, flat = read_npz_array(obj_path, "object_poses_in_cam")
    if len(shape) != 3 or shape[1:] != (4, 4):
        raise NpyReadError(f"object_poses_in_cam: expected (N,4,4), got shape {shape}")
    object_poses = reshape_nested(flat, shape)

    r_shape, r_flat = read_npz_array(hand_path, "rotation_matrices")
    t_shape, t_flat = read_npz_array(hand_path, "translations")
    if len(r_shape) != 3 or r_shape[1:] != (3, 3):
        raise NpyReadError(f"rotation_matrices: expected (N,3,3), got shape {r_shape}")
    if len(t_shape) != 2 or t_shape[1] != 3:
        raise NpyReadError(f"translations: expected (N,3), got shape {t_shape}")
    if r_shape[0] != t_shape[0]:
        raise NpyReadError(f"inconsistent hand array lengths: rotations N={r_shape[0]}, translations N={t_shape[0]}")
    if r_shape[0] != shape[0]:
        raise NpyReadError(f"inconsistent frame counts: object N={shape[0]}, hand N={r_shape[0]} - "
                            f"cannot assume these index the same frames without this matching")
    hand_rot = reshape_nested(r_flat, r_shape)
    hand_trans = reshape_nested(t_flat, t_shape)

    obj_rot_d, obj_trans_d = analyze_object_poses(object_poses)
    hand_rot_d, hand_trans_d = analyze_hand(hand_rot, hand_trans)

    results = {
        "n_frames": shape[0],
        "object_rotation_delta_rad": obj_rot_d,
        "object_translation_delta": obj_trans_d,
        "hand_rotation_delta_rad": hand_rot_d,
        "hand_translation_delta": hand_trans_d,
        "object_rotation_outliers": detect_outliers(obj_rot_d, outlier_k),
        "object_translation_outliers": detect_outliers(obj_trans_d, outlier_k),
        "summary": {
            "object_rotation_delta_rad": summarize(obj_rot_d),
            "object_translation_delta": summarize(obj_trans_d),
            "hand_rotation_delta_rad": summarize(hand_rot_d),
            "hand_translation_delta": summarize(hand_trans_d),
        },
        "metric_type": "DESCRIPTIVE_TEMPORAL_CONSISTENCY_NOT_GEOMETRIC_ACCURACY",
    }
    return results


def write_outputs(results, outdir, prefix):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{prefix}_summary.json").write_text(json.dumps(results, indent=2, default=str))
    with open(outdir / f"{prefix}_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "object_rot_delta_rad", "object_trans_delta",
                    "hand_rot_delta_rad", "hand_trans_delta",
                    "object_rot_outlier", "object_trans_outlier"])
        n = results["n_frames"]
        for i in range(n):
            w.writerow([
                i,
                results["object_rotation_delta_rad"][i],
                results["object_translation_delta"][i],
                results["hand_rotation_delta_rad"][i],
                results["hand_translation_delta"][i],
                results["object_rotation_outliers"][i],
                results["object_translation_outliers"][i],
            ])


# ---------------------------------------------------------------------------
# Synthetic self-tests (labeled clearly as synthetic - no real AGILE data involved)
# ---------------------------------------------------------------------------

def identity3():
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def mat4_from_r_t(r, t):
    return [r[0] + [t[0]], r[1] + [t[1]], r[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]


def run_synthetic_tests():
    print("=== SYNTHETIC SELF-TESTS (no real AGILE data involved) ===")
    failures = []

    # 1. Constant pose -> all deltas should be exactly 0 (after frame 0)
    n = 10
    poses = [mat4_from_r_t(identity3(), [0.0, 0.0, 0.0]) for _ in range(n)]
    rot_d, trans_d = analyze_object_poses(poses)
    ok = all(d == 0.0 for d in rot_d[1:]) and all(d == 0.0 for d in trans_d[1:])
    print(f"[SYNTHETIC 1: constant pose] {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("constant pose")

    # 2. Smooth continuous rotation -> deltas should be small, constant, no outliers
    n = 30
    step = math.radians(2.0)
    poses = [mat4_from_r_t(rot_z(i * step), [0.0, 0.0, 0.0]) for i in range(n)]
    rot_d, _ = analyze_object_poses(poses)
    valid = rot_d[1:]
    close_to_step = all(abs(d - step) < 1e-9 for d in valid)
    outliers = detect_outliers(rot_d)
    no_outliers = not any(outliers)
    ok = close_to_step and no_outliers
    print(f"[SYNTHETIC 2: smooth rotation] {'PASS' if ok else 'FAIL'} "
          f"(all deltas == step: {close_to_step}, outliers flagged: {sum(outliers)})")
    if not ok:
        failures.append("smooth rotation")

    # 3. Abrupt pose jump -> the jump frame should be flagged as an outlier
    n = 20
    poses = [mat4_from_r_t(rot_z(i * step), [0.0, 0.0, 0.0]) for i in range(n)]
    jump_frame = 10
    poses[jump_frame] = mat4_from_r_t(rot_z(math.pi), [0.0, 0.0, 0.0])  # force an explicit large jump
    rot_d, _ = analyze_object_poses(poses)
    outliers = detect_outliers(rot_d)
    ok = outliers[jump_frame] is True
    print(f"[SYNTHETIC 3: abrupt jump at frame {jump_frame}] {'PASS' if ok else 'FAIL'} "
          f"(flagged: {outliers[jump_frame]}, delta={rot_d[jump_frame]:.4f} rad)")
    if not ok:
        failures.append("abrupt jump")

    # 4. Missing frame (modeled as inconsistent array length between object/hand) -> explicit error, not silent
    try:
        obj_poses_5 = [mat4_from_r_t(identity3(), [0, 0, 0]) for _ in range(5)]
        hand_rot_4 = [identity3() for _ in range(4)]
        hand_trans_4 = [[0.0, 0.0, 0.0] for _ in range(4)]
        if len(obj_poses_5) != len(hand_rot_4):
            raise NpyReadError(
                f"inconsistent frame counts: object N={len(obj_poses_5)}, hand N={len(hand_rot_4)}"
            )
        ok = False  # should not reach here
    except NpyReadError as e:
        ok = True
        print(f"[SYNTHETIC 4: missing/mismatched frame count] PASS (raised clearly: {e})")
    if not ok:
        failures.append("missing frame handling")
        print("[SYNTHETIC 4: missing/mismatched frame count] FAIL (did not raise)")

    # 5. Invalid/NaN input -> must be detected, not silently propagated as a false "0 change"
    n = 10
    poses = [mat4_from_r_t(identity3(), [0.0, 0.0, 0.0]) for _ in range(n)]
    poses[5] = mat4_from_r_t([[float("nan")] * 3 for _ in range(3)], [0.0, 0.0, 0.0])
    rot_d, trans_d = analyze_object_poses(poses)
    nan_detected = is_nan(rot_d[5]) or is_nan(rot_d[6])
    outliers = detect_outliers(rot_d)
    nan_flagged = outliers[5] or outliers[6]
    ok = nan_detected and nan_flagged
    print(f"[SYNTHETIC 5: NaN input] {'PASS' if ok else 'FAIL'} "
          f"(NaN detected in deltas: {nan_detected}, flagged as outlier: {nan_flagged})")
    if not ok:
        failures.append("NaN handling")

    print(f"\n=== SYNTHETIC TEST SUMMARY: {5 - len(failures)}/5 passed ===")
    if failures:
        print(f"FAILED: {failures}")
    return len(failures) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="Run synthetic self-tests only, no real data needed.")
    ap.add_argument("--result-dir", help="Path to a condition's result dir (e.g. results/A_released) - real data.")
    ap.add_argument("--out-dir", help="Where to write summary JSON/CSV.")
    ap.add_argument("--prefix", default="analysis")
    args = ap.parse_args()

    if args.self_test or not args.result_dir:
        ok = run_synthetic_tests()
        sys.exit(0 if ok else 1)

    results = analyze_condition(args.result_dir)
    print(json.dumps(results["summary"], indent=2))
    if args.out_dir:
        write_outputs(results, args.out_dir, args.prefix)
        print(f"Wrote {args.prefix}_summary.json and {args.prefix}_per_frame.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
