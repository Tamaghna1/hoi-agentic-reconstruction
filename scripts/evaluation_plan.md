# MC1 A/B/C Evaluation Plan

Status: prepared, not executed — no experiment output exists yet.

## Metrics available from MC1 alone (no ground truth required)

Computed directly from each condition's saved `object_poses.npz` (`object_poses_in_cam`,
shape (N,4,4)) and `mano_transforms.npz` (`translations` (N,3)) / `optimized_mano_*.npz`
(verts/joints), comparing frame t to frame t-1 **within each condition's own output**:

- Object rotation delta: geodesic distance on SO(3) between consecutive rotation matrices,
  `arccos(clip((trace(R_t^T R_{t-1}) - 1) / 2, -1, 1))`.
- Object translation delta: `||t_t - t_{t-1}||_2`.
- Hand joint/vertex displacement: `||V_t - V_{t-1}||_2` (mean over joints/vertices).
- Second differences of the above (acceleration/jerk proxies) — sharper jitter indicator
  than velocity alone.
- Outlier-frame flags: frames where a delta exceeds `median + k * MAD` of that condition's
  own distribution (k configurable, e.g. 3) — a proxy for optimization failures/divergence.
- Per-condition summary statistics (median, IQR, max) of all the above, for direct A/B/C
  comparison.

This directly answers: "does B (smoothness) or C (more iterations) change the jitter/failure
profile, measurably, relative to A?" — a real, computable, no-GT question.

## Metrics that require official HO3D-v3 ground truth (separate track, not blocking A/B/C)

Using AGILE's own `optimization/evaluation/` code, confirmed from source in an earlier pass:
- Chamfer distance + F-score@5mm/@10mm (`calculate_chamfer_f_scores`).
- MPJPE-style hand joint error (`eval_mpjpe_right`, `eval_mpjpe_right_scaled`).
- `compute_joint3d_error` (`metrics.py`).

These require `HO3D_ROOT` pointing at the official HO3D-v3 release (`github.com/shreyashampali/ho3d`),
architecturally separate from Stage 2's `data_dir` — confirmed not needed to run A/B/C itself,
only to score against ground truth afterward. Not yet acquired; a later, independent step.

## Can temporal consistency be measured without GT? Yes, with an explicit, important limit.

The no-GT metrics above measure **change**, not **correctness**. This is stated plainly because
it is easy to overstate:

**A lower frame-to-frame delta under B or C is NOT by itself evidence of improved accuracy.**
A smoothness prior can produce a trajectory that is simultaneously smoother and *more wrong* —
e.g., lagging behind genuinely fast rotation, or regressing toward an averaged pose that no
longer tracks the true motion. Without GT, there is no way to distinguish "genuinely more
stable because the underlying estimate improved" from "artificially smoothed at the cost of
tracking accuracy," full stop. The MC1-only analysis can establish whether B/C have a
**measurable effect at all** on jitter and failure-frame rate, and nothing stronger than that.
Closing this gap requires the GT track above.

**One partial, weaker no-GT proxy, not currently available**: per-frame reprojection/silhouette
IoU against the *observed* MC1 images would be a 2D consistency check independent of 3D GT —
better than nothing, though still not a full accuracy guarantee. This is **not currently logged**
by AGILE's `save_optimized_data` (confirmed from `trainer.py`: no per-frame loss/IoU history is
saved, only progress-bar text and optional videos). Adding this would mean instrumenting
`run_experiment.py` or the optimizer itself to log it during the run — a real, scoped addition,
not yet implemented, flagged here as a recommendation rather than assumed to already exist.

## Distinguishing improvement from over-smoothing — summary

| Question | Answerable from MC1 alone? |
|---|---|
| Does B/C change the jitter/failure-rate profile at all? | Yes |
| Is a smoother trajectory under B/C more *accurate*? | **No** — requires GT |
| Does C (more iterations) change anything beyond A? | Yes, measurable the same way |
| Full Chamfer/MPJPE comparison to AGILE's own reported numbers | No — requires official HO3D-v3 GT |

## Script (prepared, not run)

`scripts/analyze_temporal_consistency.py` (to be written next) will take `results/<condition>/`
as input, compute the deltas/statistics above per condition, and write a comparison table —
pure numpy/scipy, no torch/CUDA needed, runnable independently once real A/B/C output exists.
Not written yet since there is no real output to validate it against; will be prepared before
the smoke test, not fabricated against synthetic data.
