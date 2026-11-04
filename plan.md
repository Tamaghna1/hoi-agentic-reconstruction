# HOI Project: Training-Free Video-to-3D Hand-Object Interaction

## Current research question
Does AGILE's existing recovery/smoothness mechanisms, enabled vs. not, close the gap
that motivated a proposed backtracking/hypothesis-search extension? Primary experiment:
A (released AGILE config) vs B (+ object_smoothness_loss) vs C (+ doubled
object_max_iter_per_frame), on HO3D scene MC1, preserving bidirectional optimization.

Extensive prior investigation (literature review, AGILE/HOLD source audit, prior-art
search) concluded: backtracking-with-state-reversion is confirmed absent from both
AGILE and HOLD and from external prior art — but AGILE's released config already has
recovery/bidirectional/ho_relpose/feature-loss ON by default, so the real open question
is narrower than first conceived: does smoothness (the one disabled toggle) or extra
compute alone close the gap, before any new backtracking mechanism is justified.

## Experiment design (fixed, do not weaken)
- **A**: exact released `configs/ho3d.json` + MC1 scene config, no changes.
- **B**: A + `object_smoothness_loss=True` (released `object_smoothness_type="vertices"` preserved).
- **C**: A + `object_max_iter_per_frame=800` (compute-matched control).
- All other released settings preserved: `recover_object_pose_from_mask=True`,
  `bidirectional_optimization=True`, `ho_relpose_loss=True` (λ=400),
  `object_feature_loss=True` (λ=10, DINOv3), `fix_hand_rotation_to_wilor=True`,
  `optimize_hand_pose=False`.
- Scene: MC1 (contact_frame=10, 897 frames) — chosen over MC4 (contact_frame=0)
  specifically to exercise a genuine backward pass in bidirectional optimization.
- Proposed window: start_frame=10 (auto, scene default), end_frame≈50 — structurally
  justified (preserves full 10-frame backward pass + short forward segment), NOT yet
  content-verified against actual MC1 frames.
- Backtracking/hypothesis-search mechanism: NOT to be implemented until this baseline
  experiment's results are in.

## Dependency status (as of 2026-11-03)
| Item | Status |
|---|---|
| MANO institutional permission | OBTAINED |
| MANO_RIGHT.pkl file | **DONE** - transferred, SHA-256 verified (45d60aa3...764767) |
| DINOv3 access | **PENDING_APPROVAL** - request submitted, awaiting repo author approval; listing files OK, download not yet possible |
| DINOv3 checkpoint (1.13 GB) | Not yet downloaded |
| MC1 preprocessed data (496.5 MB) | **DONE** - job 73163, 536MB extracted, SHA-256 verified |
| AGILE repo | Cloned, see Experiments log |
| Environment | `envs/latentvans` confirmed UNSUITABLE (torch cu121 vs AGILE's required cu118; no pytorch3d) — fresh `agile-optim` env needed, not yet built |
| CUDA 11.8 toolkit / nvcc on compute nodes | Not yet checked (only confirmed absent on login node, which is expected/irrelevant) |
| Smoke test | Designed (6 steps), not yet run |
| A/B/C experiments | Blocked on all of the above |

## Experiments log

### 2026-11-03 — Project setup
- Created directory structure under `/scratch/users/anirban/tamaghnam/HOI/`:
  `repo/, assets/{mano,dinov3}, data/, results/, logs/, scripts/`.
- Wrote `.gitignore` (excludes assets/, data/, results/, logs/, repo/, wandb/, model/data file extensions).
- Cloned `https://github.com/aim-uofa/AGILE` (depth 1) into `repo/` — 39MB, code only.
- Read `docs/installation.md` in full — exact build recipe confirmed:
  - `agile-download` env (python 3.10, huggingface-hub only) for data download — no GPU needed.
  - `agile-optim` env: python 3.10, torch==2.3.0/torchvision==0.18.0/torchaudio==2.3.0
    (cu118 wheels), pinned pip/setuptools/wheel/ninja/numpy/scipy/six, chumpy 0.70
    (no build isolation), then `requirements.txt`.
  - CUDA extensions built from source at pinned commits: pytorch3d (5043d15...),
    cumesh2sdf (7789ade8...), manopth (4f1dcad1...), via `CUDA_HOME=/usr/local/cuda-11.8`,
    `TORCH_CUDA_ARCH_LIST` per GPU (8.9 RTX4090/Ada6000-equivalent, 8.6 RTX3090/A5000/A6000-equivalent, 8.0 A100).
  - Verification: `python -m unittest discover -s tests -v` (CPU-only capable) + `tools/check_release.py`
    (pure static source-hygiene check, zero GPU/model dependency — useful as a free, immediate sanity check).
- Confirmed `envs/latentvans` unsuitable: Python 3.11.16, torch 2.5.1+cu121 (AGILE needs cu118),
  no pytorch3d. Not modifying this env — building fresh `agile-optim` instead, per user's
  explicit instruction not to touch unrelated/existing environments.
- Submitted MC1 download job:
  - Job 73160 on `long` partition — cancelled before starting (forecast ~24h out, despite
    `sinfo` showing "mixed"/apparently-free capacity — a known KIAC gotcha, confirmed via
    `sbatch --test-only`).
  - Resubmitted as **job 73163 on `short` partition** (forecast same-day) — downloads
    `ho3d/preprocessed/MC1.tar.gz` only (496.5 MB; `generated`/`optimized` archives
    explicitly excluded, not verified as needed) via `tools/download_ho3d.py`, which
    performs its own SHA-256 verification and safe extraction. tmux viewer: `mc1_dl_73163`.
    Status: **pending/running, not yet confirmed complete.**

## Possibilities table
| Idea | Status |
|---|---|
| Backtracking/hypothesis-search with state reversion | Deferred — test A/B/C first |
| Re-entrant grasp-taxonomy phase layer as derived output | Noted as a secondary, modest contribution; not yet started |
| Bimanual/articulated-object extension (AGILE's ARCTIC use is rigid-object-only) | Identified as the strongest remaining novelty angle if A/B/C doesn't resolve the question; not started |

### 2026-11-03 (cont.) — MC1 download complete
- Job 73163 (short partition, cn2): COMPLETED in 2:04.
- `tools/download_ho3d.py` verified SHA-256 and safely extracted
  `ho3d/preprocessed/MC1.tar.gz` -> `data/HO3D/preprocessed/train/MC1/`.
- Extracted structure: hand_processed/, rgb/, mask/, object_pose/, wilor/, moge/ (536 MB total).
- This matches the expected `data_dir=<DATA_ROOT>/train/MC1` layout AGILE's loader expects,
  with DATA_ROOT = /scratch/users/anirban/tamaghnam/HOI/data/HO3D/preprocessed.
- Scratch usage now ~34.6 GB of ~100 GB budget.
- generated/ and optimized/ archives NOT downloaded (not required for Stage 2).

### 2026-11-03 (cont.) — MANO transferred, MC1 visually inspected
- Transferred MANO_RIGHT.pkl from user's Mac (~/Desktop/HOI/mano_v1_2/models/MANO_RIGHT.pkl,
  3,821,356 bytes) to assets/mano/MANO_RIGHT.pkl via scp. SHA-256 verified identical on both ends:
  45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767.
- prepare_mano_assets.py NOT yet run — deferred until agile-optim env build (job 73170) completes,
  since the script needs trimesh/numpy from that environment.
- DINOv3 access request submitted by user, PENDING_APPROVAL (not yet granted). Per user instruction,
  not retrying/polling the gated download; will act once user reports approval.
- Visually inspected 19 MC1 frames spanning the full 897-frame sequence (scp'd locally, viewed
  directly). Finding: continuous single-grasp in-hand rotation of the object throughout — no
  approach, release, regrasp, or contact-topology change anywhere sampled. Rotation/self-occlusion
  increases substantially from ~frame 150 onward vs. the mild rotation in frames 0-50.
  FLAGGED TO USER, awaiting decision: narrow window (~10-50, weak smoothness signal) vs widened
  window (~10-200, captures real rotation/occlusion, ~4x compute) vs different scene. Not yet decided.
- Resolved: hand_processed/ (optimized_mano_joints.npz, optimized_mano_verts.npz) is required
  Stage-1-derived input, NOT ground truth - confirmed via data/loader.py, used alongside wilor/'s
  raw WiLoR output, not a leftover/cache.
- Confirmed via trainer.py: MANO instantiation is unconditional and hardcoded to device="cuda"
  (unlike DINO, gated by cfg.object_feature_loss) - any Runner() construction, including a
  "data-loading-only" smoke test stage, requires both MANO readiness AND a live GPU allocation.

## FROZEN RESEARCH DECISIONS (approved, treat as locked)

- **Dataset**: HO3D MC1 (897 frames, contact_frame=10).
- **Window**: start_frame=10, end_frame=200. Backward pass: full 10-frame span (10->0,
  unaffected by end_frame). Forward pass: 190 frames (10->200).
- **Research scope**: object-pose consistency during continuous in-hand rotation and
  changing self-occlusion. Frames 10-100 are comparatively stable (mild rotation);
  frames 100-200 show substantially more rotation and self-occlusion variation
  (visually confirmed, 32 sampled frames across 0-896).
- **Explicitly NOT claimed**: this experiment does not test regrasping, contact-topology
  transitions, contact ambiguity, or persistent-hypothesis backtracking. The sequence
  contains a single continuous grasp throughout, visually confirmed - no release, no
  regrasp, no second contact event anywhere sampled (0-896).
- **A**: exact released configs/ho3d.json + MC1 scene overrides, no changes.
- **B**: A + object_smoothness_loss=True only (object_smoothness_type="vertices" preserved).
- **C**: A + object_max_iter_per_frame=800 only.
- All other released settings/weights preserved exactly; loaded via wrapper from the
  released JSON, never bare train.py class defaults.

## 2026-11-03 (cont.) — Validation job 73176, wrapper/eval prep

- Submitted job 73176 (validate_mc1_config) on `long`, stdlib-only Python (/usr/bin/python3,
  no conda env - deliberately independent of job 73170's build to avoid any race/conflict).
  COMPLETED in 1s, exit 0. Purpose: MC1 dataset integrity + A/B/C config-diff correctness.
  Result: 0 errors, 2 harmless warnings (hand_processed/mano_transforms.npz's `scales`/
  `rotations` entries are per-sequence constants, not per-frame arrays - by design, not
  a data problem; `translations` entry IS per-frame (897,3), confirmed correct).
  - All 897 frames verified present across rgb/, mask/{hand,object}/, moge/depth/,
    object_pose/ob_in_cam/ - exact match, no missing/extra files.
  - All object_pose/ and moge/ static assets present and non-empty.
  - A/B/C diff logic PROVEN correct: A changes nothing vs released baseline, B changes
    exactly {object_smoothness_loss}, C changes exactly {object_max_iter_per_frame} -
    no accidental combined changes. All "must preserve" settings spot-checked and match
    the released configs/ho3d.json exactly.
  - Script: scripts/validate_data_and_config.py (kept, reusable).
- Resource check: 1 job running (73170) of account `research`'s MaxJobs=2, MaxSubmit=4
  limit before submitting 73176 - within limits throughout (peaked at 2 running).
- Learned from trainer.py (new, targeted fetch): Stage 2's save_optimized_data writes to
  result_dir/aligned[frame_step]/ as .npz (mano_betas, mano_hand_pose, mano_global_orient,
  mano_transforms, mano_faces, optimized_mano_verts, optimized_mano_joints, object_poses,
  camera_data), .obj (object_mesh.obj via trimesh+texture), and export_metadata.json.
  Periodic snapshots every 50 frames in forward mode (previous snapshot deleted, only
  latest kept); final save at sequence end. Object pose saved as 4x4 matrices
  object_poses_in_cam (N,4,4) + object_scale. Hand saved as mano_transforms.npz
  (rotations (N,3) rotvec, rotation_matrices (N,3,3), translations (N,3)).
  **No per-frame loss/convergence history is saved natively** - only progress-bar text
  and optional videos/point-clouds. This directly shapes the evaluation plan below:
  any no-GT temporal-consistency metric must be computed from the saved pose arrays
  themselves, not from a loss log that doesn't exist.

## 2026-11-03 (cont.) - Wrapper and evaluation plan prepared

- scripts/run_experiment.py written: loads released configs/ho3d.json + MC1 scene
  overrides, applies exactly one condition delta, asserts diff-from-baseline matches
  exactly the intended field, writes effective_config.json + run_manifest.json
  (commit hash, env details, known seed=42, dataset identity) per condition, checks
  MANO/DINO/data prerequisites WITHOUT auto-downloading anything, supports --resume
  via STATUS markers, lazy-imports AGILE only past all checks.
- Validated via job 73178 (wrapper_check_only, long, stdlib-only python3, COMPLETED
  exit 0:0): ran --check-only for A/B/C. All three built the exact correct config
  diff ([], [object_smoothness_loss], [object_max_iter_per_frame]). All three
  CORRECTLY FAILED prerequisite checks (MANO not prepared, DINO_MODEL_PATH unset) -
  this is the intended fail-closed behavior, proving the wrapper refuses to proceed
  without real prerequisites and never auto-downloads anything missing.
- scripts/evaluation_plan.md written: no-GT metrics (frame-to-frame object
  rotation/translation deltas via SO(3) geodesic distance, hand joint/vertex
  displacement, outlier-frame detection via median+k*MAD) computable from the
  released repo own saved object_poses.npz/mano_transforms.npz; GT-requiring metrics
  identified from optimization/evaluation/ (Chamfer, F-score, MPJPE) needing separate
  HO3D_ROOT. Explicit limitation documented: no-GT deltas measure CHANGE not
  CORRECTNESS - a smoother trajectory under B/C is not by itself evidence of improved
  accuracy; closing that gap requires the GT track.
  scripts/analyze_temporal_consistency.py not yet written (no real experiment output
  exists yet to validate it against).
- Resource usage stayed within account research limits (MaxJobs=2, MaxSubmit=4)
  throughout: peaked at 2 concurrent (73170 plus one validation job), never exceeded.

## 2026-11-03 (cont.) - Analysis script, reproducibility audit, job 73170 investigation

- scripts/analyze_temporal_consistency.py written: pure stdlib (zipfile+struct npy
  parsing, no numpy/torch), deliberately independent of the agile-optim env to avoid
  any interference with job 73170's build. Computes SO(3) geodesic rotation deltas,
  translation deltas, hand joint/vertex displacement, outlier detection
  (median + k*MAD), explicit handling of mismatched frame counts and NaN input
  (raises/flags clearly, never silently propagates). Loader verified against the
  REAL AGILE save format from a targeted trainer.py fetch (object_poses.npz:
  object_poses_in_cam (N,4,4); mano_transforms.npz: rotation_matrices (N,3,3),
  translations (N,3)).
- Validated via job 73180 (analysis_selftest, long, stdlib-only, COMPLETED exit 0:0):
  5/5 synthetic tests passed - constant pose (zero deltas), smooth rotation (deltas
  match step size, no false outliers), abrupt jump (correctly flagged, delta=2.83 rad
  matches the forced pi rotation), mismatched frame count (raised clear error, did not
  proceed), NaN input (detected and flagged as outlier, not silently treated as zero
  change). All labeled synthetic, no real AGILE data involved.
- Reproducibility audit of run_experiment.py, findings and fixes applied:
  - A/B/C output isolation: confirmed safe by construction (result_dir()/log_dir()
    map 1:1 to condition name, no shared paths).
  - Resumption isolation: confirmed safe (STATUS file lives inside each condition's
    own result dir only).
  - Config diff guarantee: PROVEN via job 73178/73182 (hard assertion + actual test
    run), not just asserted in code.
  - WEAKNESS FOUND AND FIXED: git commit was recorded but repo dirty-state was not -
    added git status --porcelain check, now recorded as dirty: true/false +
    dirty_files list in run_manifest.json.
  - WEAKNESS FOUND AND FIXED: dataset identity only referenced a path, not the
    independent integrity verification - now references job 73176's ID and result
    directly in the manifest.
  - WEAKNESS FOUND AND FIXED: environment capture only recorded torch; now also
    attempts numpy/transformers/trimesh/scipy/pytorch3d versions (each independently
    optional, graceful UNAVAILABLE per-package rather than failing the whole capture).
  - Re-validated via job 73182 (wrapper_check_only, COMPLETED exit 0:0): confirmed
    git_agile_repo (commit 4b0d0cdc583f8eebed84cc74da362f49565d8105, dirty: false)
    and dataset_identity.integrity_verification both correctly populated in the
    written run_manifest.json.
  - NOTED, not changed: results/ is fully gitignored per the user's own explicit
    instruction to exclude "generated results" - effective_config.json/run_manifest.json
    therefore will not be committed to GitHub even though they are small metadata, not
    data/checkpoints. Flagging as a deliberate consequence of existing policy, not a bug.
- Job 73170 investigation (no disruptive action taken):
  - Still on the same "Downloading open3d-0.19.0 (447.7 MB)" log line across multiple
    checks spanning from elapsed 25:30 to 32:33 (~7 min with zero new output).
  - sstat -j 73170 returned empty (no live resource-usage plugin data available on
    this cluster for this check).
  - Direct SSH to the allocated compute node (cn1) was attempted and denied
    (publickey/password refused) - no way to inspect the process directly.
  - Checked /home/anirban/tamaghnam/.cache/pip size as an indirect progress proxy:
    unchanged at exactly 3.0G across the check interval - a point of concern, though
    not conclusive (this may not be where pip buffers an in-progress download).
  - No error, retry, or timeout message anywhere in stdout or stderr - a genuine
    network stall usually does eventually surface something, so complete silence is
    a point in favor of "still working," weighed against the unchanged cache size.
  - CONCLUSION: evidence is mixed, not conclusive either way. No action taken.
    Decision threshold set: if elapsed reaches ~45-50 min total with still no stage
    progression, treat as likely-stalled and report a concrete recovery recommendation
    (safe recovery = cancel + resubmit with pip's progress forced on and a resumable
    download flag, NOT a blind resubmission of the same command) rather than guessing
    now. TimeLimit=3:00:00, elapsed 32:33 - ample remaining wall time either way.

## 2026-11-03 (cont.) - DINOv3 access confirmed, job 73170 threshold check

- User confirmed DINOv3 access is now GRANTED on the HF model page for
  facebook/dinov3-vitl16-pretrain-lvd1689m. Status updated:
  PENDING_APPROVAL -> ACCESS_GRANTED_NOT_YET_DOWNLOADED.
  Per explicit staging instruction, download/verification deferred until job 73170
  completes AND the environment is independently verified - not started yet.
- Job 73170 at elapsed 36:14: still on the same open3d download line, pip cache
  unchanged at 3.0G across three consecutive readings (~11+ min span). This has NOT
  yet crossed the stated 45-50 min total-elapsed threshold for escalation. No action
  taken. A bounded watcher (10 min timeout) has been set to either catch completion
  or formally cross the threshold, landing at ~46 min elapsed.

## 2026-11-03 (cont.) - Job 73170 threshold resolved: slow, not stalled

- Bounded watcher confirmed at elapsed 46:47: open3d (447.7 MB) DID complete -
  "447.7/447.7 MB 194.8 kB/s 0:31:23" - just very slow (~195 KB/s average), not
  stalled. pip cache grew 3.0G -> 3.5G confirming real progress occurred throughout
  the silent period. Decision to wait rather than escalate/cancel was correct.
  Threshold concern resolved; reverting to normal-progress monitoring. Recent
  package download speeds trending slower (182-195 KB/s for the last two vs
  0.66-1.9 MB/s for earlier ones) - noted as a general connection-speed risk for
  remaining stages (CUDA toolkit + 3 source builds), not an emergency.

## 2026-11-04 - Job 73170 COMPLETED successfully, full real verification done

- Job 73170: COMPLETED, elapsed 01:12:38, exit 0:0. All 8 stages finished clean.
  Stage 5 (CUDA 11.8 toolkit via conda nvidia/label/cuda-11.8.0): confirmed installed,
  nvcc at envs/agile-optim/bin/nvcc, "release 11.8, V11.8.89" - exact target version.
  Stage 6: pytorch3d 0.7.8, torchcumesh2sdf 0.0.0, manopth 0.0.1 all built successfully.
  Stage 7 (pip check): "No broken requirements found."
  Stage 8 (check_release.py): {"python_files": 165, "ho3d_scenes": 18, "problems": []}.
- REAL verification (not trusting exit code alone, per explicit instruction) via job
  73190 with a live GPU (short/cn2, RTX A5000):
  - torch 2.3.0+cu118, torch.version.cuda=11.8, cuda_available=True, GPU detected.
    Real CUDA matmul executed successfully on-GPU.
  - pytorch3d 0.7.8: real CUDA op (knn_points) executed successfully on-GPU.
  - manopth: constructed a ManoLayer from the ACTUAL transferred MANO_RIGHT.pkl and
    ran a real forward pass - verts shape (1,778,3), joints shape (1,21,3), correct
    MANO dimensions. Genuine evidence the transferred MANO file is valid.
  - pip check: clean (re-confirmed with everything installed).
- Found and resolved one issue: torchcumesh2sdf raised ImportError (libc10.so not
  found) when imported standalone without torch imported first. Follow-up job 73192
  confirmed root cause via ldd: the extension's .so does not bake libtorch's
  libraries into its own RPATH and relies on torch being loaded in-process first -
  the standard, expected pattern for PyTorch C++/CUDA extensions. With torch imported
  first, the extension imports cleanly and exposes get_sdf/get_udf/get_collide/
  free_cached_memory. CONCLUSION: this was an artifact of the standalone test script,
  not a real build/packaging failure - AGILE's own code always imports torch first,
  so this would never occur in a real run. No fix needed to the build itself.
  - Minor, non-blocking note: envs/agile-optim/lib64/libcudart.so* was not found at
    that path during verification - torch uses its own bundled CUDA runtime
    (nvidia-cuda-runtime-cu11 pip package) independent of the conda toolkit's lib64,
    which is why this did not affect torch/pytorch3d functioning. Not investigated
    further since it did not block anything that was actually tested.
- STATUS: environment build fully verified, not just exit-code-trusted. Proceeding
  immediately to MANO asset preparation (already authorized).

## 2026-11-04 (cont.) - MANO preparation complete and verified

- Job 73197 (short/cn2, after moving off long which was genuinely stuck with reason
  Priority): COMPLETED, 22s, exit 0:0.
- tools/prepare_mano_assets.py ran cleanly against the transferred MANO_RIGHT.pkl:
  generated mano_config.yaml (127B), finger_vertices_mask.npz (1072B), mano_hand.obj
  (49559B, 1552 faces including the documented 14-triangle wrist cap, 778 vertices -
  matches docs/installation.md's stated spec exactly). finger_mask_imported: true.
- AGILE's own optimization.models.mano_wrapper.MANO class (the exact class
  Runner.__init__ constructs, not manopth directly) initialized successfully from
  the generated mano_config.yaml. This is the authoritative verification.
- MANO dependency: FULLY COMPLETE AND VERIFIED (file transferred + checksummed,
  assets prepared, AGILE's own class loads it correctly).
- Proceeding to DINOv3 setup now that access has been confirmed granted by the user
  and both the environment and MANO are ready.

## 2026-11-04 (cont.) - DINOv3: access granted but NOT authenticated on cluster

- Scratch space checked before attempting DINOv3 download: 696M used in HOI/, 79T free
  filesystem-wide - ample room for the 1.13GB checkpoint.
- Job 73198/73199 (check_hf_auth, short/cn2, COMPLETED exit 0:0): checked HF
  authentication via huggingface_hub.whoami() - NEVER printed/logged the token itself,
  only the library's success/failure result.
- RESULT: NOT AUTHENTICATED. LocalTokenNotFoundError - no HF token is configured on
  the KIAC cluster yet, despite the user's access APPROVAL being granted on the HF
  model page. These are two separate things: approval (done) vs. a credential actually
  present on this machine (not done).
- BLOCKED on user action: user must personally run 'hf auth login' interactively on
  KIAC (after activating envs/agile-optim) to store a token in their own
  ~/.cache/huggingface/token. Not something this assistant can or will do on their
  behalf, consistent with "never put my HF token in scripts, logs, GitHub, or plan.md".
- Once done, re-run the same whoami()-based check (reusable: scripts/check_hf_auth.sh)
  to confirm, then proceed to the resumable download.

## 2026-11-04 (cont.) - DINOv3 downloaded and fully verified: ALL PREREQUISITES COMPLETE

- HF authentication re-checked after user ran hf auth login themselves: job 73201,
  AUTHENTICATED as Tamaghna1, GATED REPO ACCESS CONFIRMED - model_info retrieved,
  6 files listed matching expectation exactly. Token itself never read/printed/logged
  by this assistant at any point.
- Job 73204 (download_dinov3, short/cn2): COMPLETED in 31s (25s actual transfer -
  far faster than the env-build downloads, different CDN). hf download --local-dir
  used (AGILE's own documented, inherently-resumable approach). All 6 expected files
  present: config.json (745B), .gitattributes (1519B), LICENSE.md (7503B),
  model.safetensors (1,212,559,808B), preprocessor_config.json (585B), README.md
  (14520B). Total 1.2G at assets/dinov3/.
- Job 73206 (verify_dinov3, short/cn2, GPU): COMPLETED, 4:46 elapsed, exit 0:0.
  - File size cross-check: model.safetensors actual=1212559808, exactly matches the
    HF metadata read earlier. MATCH.
  - AGILE's own optimization.utils.feature.DINOWrapper class (the exact class
    Runner.__init__ constructs when object_feature_loss=True) initialized
    successfully and ran a forward pass with no error/traceback anywhere in the logs.
  - Fallback raw transformers.AutoModel/AutoImageProcessor test: loaded successfully,
    ran a real forward pass. last_hidden_state shape = (1, 201, 1024). Verified by
    hand: 224x224 image / patch16 = 14x14 = 196 patches + 1 CLS + 4 register tokens
    = 201 - EXACT match to AGILE's documented "ViT-L/16 with four register tokens"
    spec. Hidden dim 1024 matches ViT-L. Strong, numerically-verified confirmation
    the checkpoint is genuinely correct and functional, not just importable.
- DINOv3 dependency: FULLY COMPLETE AND VERIFIED.

## STATUS: ALL FOUR PREREQUISITES NOW COMPLETE AND INDEPENDENTLY VERIFIED
  1. Environment (agile-optim): built, CUDA 11.8 confirmed, all 3 extensions verified
     with real GPU execution.
  2. MANO: transferred, checksummed, prepared, AGILE's own MANO class verified.
  3. DINOv3: access granted, authenticated, downloaded, AGILE's own DINOWrapper
     class verified with exact numeric confirmation.
  4. MC1 dataset: downloaded, 897/897 frames verified present and well-formed.
  Per the execution gates: integrated smoke test is now unblocked. Not yet started -
  awaiting user confirmation to proceed, since this moves into new territory
  (real bidirectional optimization on actual data, more substantial GPU time).

## 2026-11-04 (cont.) - INTEGRATED SMOKE TEST: PASSED

### Attempt 1 (job 73209): FAILED, root cause identified and fixed
- Failed in 13s: ImportError, 'from optimization.config import Config' is wrong -
  optimization/config/__init__.py is empty, Config lives in the config.py submodule.
- Root cause verified against AGILE's own run_ho3d.py and train.py, both of which use
  'from optimization.config.config import Config'.
- IMPORTANT: the identical bug existed in the APPROVED A/B/C wrapper
  (scripts/run_experiment.py), hidden behind --check-only's early exit - this would
  have made all three A/B/C conditions fail identically at launch. Fixed in BOTH
  scripts/run_smoke_test.py and scripts/run_experiment.py. This is an infrastructure
  bug fix (a Python import path), not a change to any research setting, dependency,
  loss, or preserved configuration field - the released config values themselves were
  never touched.

### Attempt 2 (job 73211): PASSED
- COMPLETED, elapsed 00:09:17, exit 0:0. Resource usage: MaxRSS/MaxVMSize were blank
  in sacct (this cluster's accounting plugin does not report detailed memory stats,
  consistent with sstat returning empty earlier in this project) - elapsed time is
  the only hard resource metric available.
- Window used: start_frame=10, end_frame=15 (SMOKE-TEST-ONLY deviation from the
  approved A/B/C window of start_frame=10, end_frame=200, documented in
  results/smoke_test/run_manifest.json; the approved window in
  scripts/run_experiment.py's WINDOW_START/WINDOW_END was never touched).
- Config/git/environment captured: results/smoke_test/{effective_config.json,
  run_manifest.json,config.json} - git commit recorded (dirty-state tracked),
  environment package versions captured, dataset identity referenced job 73176's
  verification.
- W&B: attempted, failed non-fatally exactly as designed ('No module named wandb' -
  wandb package is not installed in agile-optim; this means the nonfatal-failure
  behavior works correctly but there is currently no real W&B tracking value. Not
  blocking; install wandb in agile-optim later if real tracking is wanted).

### Verification evidence (beyond exit status, per explicit requirement)
- MANO: "Initialized MANO model from .../assets/mano" - confirmed in logs.
- DINO: "Loading DINO model from .../assets/dinov3... DINO model loaded." - confirmed.
  Further confirmed ACTIVE (not just loaded): per-iteration optimization logs show
  feature_loss values (e.g. feature_loss=5.9358) changing every iteration, direct
  evidence object_feature_loss (DINO) is genuinely contributing to the real
  optimization, not merely present in config. ho_loss (ho_relpose_loss) similarly
  shown active in the same per-iteration logs.
- CUDA extensions: mesh downsampling (30158->5002 vertices, "Mesh watertight: True"),
  rendering/optimization loop ran at several it/s on GPU - consistent with pytorch3d/
  cumesh2sdf genuinely executing, not silently falling back to CPU or a no-op path.
- Bidirectional pass: BOTH directions genuinely ran and were explicitly logged:
  "[Bidirectional] FORWARD pass: frames 10 -> end" (processed frame 10) then
  "[Bidirectional] BACKWARD pass: frames 0 -> 9 (seeded from contact frame)"
  (processed frames 5 and 0), then explicitly merged:
  "[Bidirectional] Saving combined result: 2 backward + 1 forward = 3 frames".
- Output files: all 6 expected files present in results/smoke_test/aligned/, PLUS
  additional genuine outputs not in the original minimum checklist: textured
  object_mesh.obj + material_0.png + Bake_Mat.png, object_assets/{display,
  optimization_proxy}.glb, camera_data.npz, mano_betas/mano_faces/
  mano_global_orient/mano_hand_pose.npz, hand+object optimization progress videos
  (.mp4) per processed frame, HO contact point clouds (.ply), export_metadata.json.
  This is a rich, complete real-run output, not a stub.
- Finite values: object_poses.npz and mano_transforms.npz checked value-by-value
  (pure-stdlib npy parsing) - zero NaN, zero Inf across all arrays.
- No silent fallback: no "skipping"/"falling back" messages anywhere in the logs;
  the conditional recovery mechanism (recover_object_pose_from_mask) was not
  triggered for these frames (IoU threshold not breached) - a legitimate, expected
  non-firing of a conditional safety net, not a sign of a masked problem.

### Finding requiring a documented correction to MY OWN verification script
- My run_smoke_test.py's verification logic WARNED that the saved frame count (3)
  was less than the full 0..14 range (15), based on an assumption that every frame
  in the processed window gets saved. This assumption was WRONG for this released
  config: AGILE's own logs show "Using frames from 10 to 15 with stride 5" and
  "Using frames from 0 to 10 with stride 5" - the released config's hold_mode
  (hold_mode=True, hold_mode_interval=5) intentionally processes/saves only every
  5th frame as a keyframe, not every frame in the window. With a 5-frame-wide smoke
  window this leaves only 1 forward keyframe (frame 10) + 2 backward keyframes
  (frames 5, 0) = 3 total - EXACTLY what was saved. This is correct, expected
  behavior of the exact released configuration, not a pipeline defect. The real
  approved A/B/C window (10-200) will yield far more keyframes under the same
  stride-5 sampling (~38 forward + 2 backward), so this stride-related sparsity is
  mainly an artifact of choosing a deliberately tiny smoke-test window, not
  something that limits interpretation of the real A/B/C runs.

### Observation flagged, not resolved (not clearly a problem, not clearly fine)
- Every processed frame's convergence status was "stalled_at_budget" (hand) or
  "budget_exhausted" (object) - meaning each frame used its full iteration budget
  (200 for hand, 400-800 for object) without hitting an early-stop convergence
  criterion. Loss values show plausible optimization dynamics (no explosion, no
  NaN), and this is plausibly expected behavior for cold-start frames processed
  with no prior-frame warm start (true of frame 10 as the sequence anchor in both
  the smoke test AND the real A/B/C runs, and of frame 0 as the far end of the
  backward sweep). Frame 0's hand gradient_inf (132.3) was notably larger than
  frame 10's (0.017) or frame 5's (0.046) - flagged as a specific numeric
  observation worth watching in the real A/B/C runs, not a confirmed problem, since
  there is no established baseline for what gradient_inf range is normal in this
  codebase. NOT something this assistant changed settings to work around.

### Outstanding, separate from the smoke test itself
- No git repository exists yet for this project (confirmed: 'fatal: not a git
  repository' in HOI/), and 'gh' is not installed on this cluster. Per the user's
  instruction to commit and push code/docs, a repo name/remote decision is needed
  before this can be done - flagged as a question for the user, not resolved
  unilaterally.

## CONCLUSION: smoke test PASSED on the second attempt (first attempt's failure was
a script bug, fixed and verified, not a pipeline/dependency/environment problem).
Every stage (MANO, DINO, CUDA extensions, data loading, bidirectional optimization,
output writing) is now genuinely verified working on real data with real compute.
Pipeline is ready for the approved A/B/C runs, pending user authorization to launch
them and pending resolution of the git repo/remote question for the commit+push step.

## 2026-11-04 (cont.) - GitHub repository setup

- Repo created by user: https://github.com/Tamaghna1/hoi-agentic-reconstruction
- Verified remote state before touching anything: 'git ls-remote' returned exit 0,
  empty output - confirmed genuinely empty repo (not an auth failure), safe to push
  without any overwrite risk.
- Found an untracked results/-adjacent directory, debug/ (AGILE's own generated
  mesh/texture debug output from the smoke test), added to .gitignore alongside
  *.obj/*.mtl/*.glb/*.ply/*.mp4 extensions for defense in depth.
- 'git init' in HOI/ (was NOT already a repo, confirmed first). Remote 'origin' set
  to the exact URL above. Branch renamed to 'main'.
- Staged explicitly (.gitignore, plan.md, scripts/ only - never git add -A).
  git status before commit showed exactly these three items as untracked; nothing
  from assets/, data/, logs/, repo/, results/, debug/ appeared at all.
- Scanned the staged diff for credential patterns before committing - one harmless
  false positive (plain-English text describing an SSH permission error in plan.md).
- COMMITTED locally: 493cb07 "Initial commit: HOI/AGILE project infrastructure,
  scripts, and plan.md" (21 files, 2255 insertions).
- PUSH FAILED: "Invalid username or token. Password authentication is not supported
  for Git operations." The cluster's stored credential (credential.helper=store) is
  stale/invalid. Checked Mac-side fallback: gh is not authenticated there either, no
  credential helper configured. BLOCKED on user action - did not attempt any token-based
  workaround. User must either (a) run a manual git push on the cluster once,
  entering username + a fresh PAT when prompted, letting credential.helper cache it,
  or (b) run 'gh auth login' on the Mac so pushes can happen from there instead.
  Commit is safe, unpushed, locally on the cluster - nothing lost.
- Re-ran scripts/run_wrapper_check.sh (long -> short partition, also found and fixed:
  it never exported DINO_MODEL_PATH, a gap in the check SCRIPT's own job environment,
  not evidence DINO itself was missing - DINO was already independently verified
  working earlier). Job 73225, COMPLETED exit 0:0: all three conditions (A/B/C) now
  pass EVERY prerequisite check cleanly, config diffs still exactly correct
  (A: none, B: [object_smoothness_loss], C: [object_max_iter_per_frame]).

## STATUS: pre-experiment audit PASSED. Repository setup PARTIAL - committed locally,
push blocked on user's GitHub authentication action (cluster or Mac side). Per the
user's own stated two-part gate ("if repository setup and the pre-experiment audit
both pass, proceed with A/B/C"), NOT launching the real A/B/C experiments yet -
waiting for either the push to succeed or explicit instruction to proceed anyway.

## 2026-11-04 (cont.) - GitHub push resolved via dedicated SSH deploy key

- Inspected ~/.ssh first: no existing keys, no config - nothing at risk of being
  overwritten.
- Generated a dedicated Ed25519 keypair (~/.ssh/hoi_agentic_reconstruction_deploy_ed25519),
  permissions 600 (private)/644 (public). Private key never printed/transmitted.
  Wrote ~/.ssh/config scoping it to Host github.com (IdentitiesOnly yes).
- Gave the user the public key only, with exact GitHub deploy-key UI instructions.
  Did not attempt any push until the user confirmed the key was added with write
  access.
- Tested SSH auth: "Hi Tamaghna1/hoi-agentic-reconstruction! You've successfully
  authenticated" - confirms the key is correctly scoped to exactly this repo.
- Re-verified before pushing: remote still genuinely empty (git ls-remote exit 0,
  no output), branch main at exactly 493cb07, no unexpected remote content.
- Switched origin to git@github.com:Tamaghna1/hoi-agentic-reconstruction.git,
  pushed with 'git push -u origin main' - no force flag used, no history rewrite.
- VERIFIED: local HEAD and origin/main both equal
  493cb073bbfe162c87d0734a948e3b630d78d432 exactly; confirmed visible via
  'git ls-remote origin'.

## STATUS: GitHub push RESOLVED AND VERIFIED. Repository: 
https://github.com/Tamaghna1/hoi-agentic-reconstruction, commit 493cb073bbfe162c87d0734a948e3b630d78d432 on branch main.
Both gates from the prior instruction (repository setup + pre-experiment audit) are
now satisfied. Proceeding to re-verify prerequisites still pass, then launch the
approved A/B/C experiments.

## 2026-11-04 (cont.) - A/B/C EXPERIMENTS LAUNCHED

Both gates satisfied (repo setup verified pushed, pre-experiment audit passed clean).
Final prerequisite spot-check before launch: MANO/DINO/MC1 all present, 0 jobs
running, confirmed clean slate.

- scripts/run_condition_{A,B,C}.sh written: each activates agile-optim, sets
  CUDA_HOME/TORCH_CUDA_ARCH_LIST=8.6/DINO_MODEL_PATH, attempts W&B logging
  (nonfatal), then runs 'python scripts/run_experiment.py --condition {A,B,C}'
  (the real, non-check-only path) - released config + approved window
  (start_frame=10, end_frame=200) + exactly one condition-specific delta each,
  proven correct by the final wrapper_check_only validation (job 73225).
- Submitted all three via tmux_job.sh (each with its own 3-window log/err/status
  viewer): 
  - Job 73230 (exp_A, released baseline) - RUNNING on cn2.
  - Job 73231 (exp_B, +object_smoothness_loss) - RUNNING on cn3.
  - Job 73232 (exp_C, +object_max_iter_per_frame=800) - PENDING, reason
    AssocMaxJobsLimit - correctly queued, respecting the account's MaxJobs=2
    running-job cap (MaxSubmit=4 not exceeded with all 3 submitted). Will start
    automatically once A or B completes.
  - Partition list for all three: short,long,a100,medium,ada (broadest available,
    scheduler picks whichever frees first) - gres=gpu:1, mem=24G, cpus=4,
    time=6:00:00 (generous estimate based on the smoke test's 3-keyframe/9min
    pace scaled to the ~40-keyframe real window, with margin).
  - Output dirs: results/{A_released,B_smoothness,C_control_iters800}/ - isolated,
    verified non-overlapping by construction (result_dir()/log_dir() map 1:1 to
    condition name).

## 2026-11-04 (cont.) - A and B COMPLETED; output-isolation bug found and fixed

### Condition A (job 73230): COMPLETED, exit 0:0, elapsed 01:12:33 (4346s)
### Condition B (job 73231): COMPLETED, exit 0:0, elapsed 00:57:10 (3405s)
Both: "[Bidirectional] Saving combined result: 2 backward + 38 forward = 40 frames",
matching expectation (stride-5 hold_mode over the 10-200 window). Final object
scales: A=0.9299, B=0.9219 (plausibly different given B's extra smoothness term;
not yet rigorously compared - see next step).

### FINDING: A/B/C output isolation was NOT actually achieved as claimed, root cause fixed
- results/A_released/ and results/B_smoothness/ contain ONLY my wrapper's own
  effective_config.json/run_manifest.json - the REAL AGILE output (aligned/,
  object_poses.npz, videos, etc.) landed instead under
  results/MC1/<timestamp>_<loss_info>/ - a path AGILE auto-generated itself.
- ROOT CAUSE (verified against AGILE's actual source, not assumed): AGILE's
  Config.result_dir field was never set by run_experiment.py's
  load_scene_overrides() - only my OWN result_dir(condition) helper function
  existed, used solely for where to write MY wrapper's manifest files, never
  inserted into the effective config dict passed to Config(**effective). With
  result_dir=None, Config.get_result_dir() (confirmed via source fetch of
  optimization/config/config.py) auto-generates results/{scene}/{timestamp}_
  {loss_info}[_{mode_info}], where loss_info/mode_info are built from which loss
  flags are enabled (mask, depth, normal, contact, smoothness, feature/type,
  lastframe, handscale) - NOT from object_max_iter_per_frame, meaning condition C's
  auto-generated name is distinguishable from A's ONLY by timestamp.
- CONSEQUENCE, verified not assumed: no actual data collision occurred between A
  and B (different timestamps, and B's name additionally includes
  "smoothness0.0002"). C (still running under the OLD buggy Config object
  constructed before this fix) will also land in an auto-generated path, distinct
  from A's only by timestamp - confirmed distinct by directory listing
  (results/MC1/1104_021134_mask5.0_feature10.0_typelocal_lastframe/ exists
  separately from A's 1104_011226_... directory).
- ACTUAL CURRENT DATA LOCATIONS (verified via filesystem, not assumed):
  - Condition A: results/MC1/1104_011226_mask5.0_feature10.0_typelocal_lastframe/
  - Condition B: results/MC1/1104_011508_mask5.0_smoothness0.0002_feature10.0_typelocal_lastframe/
  - Condition C (in progress): results/MC1/1104_021134_mask5.0_feature10.0_typelocal_lastframe/
- FIX APPLIED (verified via --check-only re-run, not consuming a GPU job slot):
  load_scene_overrides() no longer sets result_dir; build_effective_config() now
  sets reference["result_dir"] = str(result_dir(condition)) BEFORE effective is
  copied from reference, so it is identical in both and does not appear in the
  single-field-diff check. Re-ran --check-only for A and B: diffs still exactly
  correct ([], [object_smoothness_loss]); confirmed effective_config.json's
  result_dir field now correctly resolves to e.g.
  /scratch/users/anirban/tamaghnam/HOI/results/A_released. This fix is NOT yet
  applied to the already-running condition C (its Config object was constructed
  before the fix existed - not interfering with a running job).
- NOT YET DECIDED: whether to move the already-completed A/B (and eventually C)
  output from their auto-generated locations into the originally-intended
  results/{A_released,B_smoothness,C_control_iters800}/ directories, or to
  document and use the actual auto-generated paths going forward. Flagged to user,
  not decided unilaterally - results/ is gitignored/not version-controlled, so any
  move would need explicit care.

## 2026-11-04 (cont.) - User decision on output locations: leave in place (option 2)

User explicitly chose NOT to move/rename/copy/reorganize any existing A/B/C output.
All data stays exactly where AGILE actually wrote it. No files moved, renamed, or
deleted at any point in this project.

### Consolidated record: wrapper manifest dirs vs AGILE's actual output dirs
Two genuinely distinct things exist per condition, serving different purposes:

1. **Wrapper manifest directories** (results/{A_released,B_smoothness,C_control_iters800}/)
   - Created and written by scripts/run_experiment.py itself.
   - Contain ONLY: effective_config.json, run_manifest.json (git commit/dirty state,
     environment package versions, dataset identity + integrity-verification
     reference). These are small, our own bookkeeping files - NOT AGILE's
     optimization output.

2. **AGILE's actual output directories** (results/MC1/<timestamp>_<loss_info>[_<mode_info>]/)
   - Created and written by AGILE's own Config.get_result_dir() auto-generation
     (root cause below), containing the REAL optimization results: aligned/
     (object_poses.npz, mano_transforms.npz, optimized_mano_{joints,verts}.npz,
     object_mesh.obj, export_metadata.json, textures), plus
     {hand,object}_optimization_videos/, pcd_vis/, videos/, contact_pcd.ply.
   - Exact paths, verified on disk:
     - A: results/MC1/1104_011226_mask5.0_feature10.0_typelocal_lastframe/
     - B: results/MC1/1104_011508_mask5.0_smoothness0.0002_feature10.0_typelocal_lastframe/
     - C: results/MC1/1104_021134_mask5.0_feature10.0_typelocal_lastframe/ (pending
       final confirmation once job 73232 completes)

### Root cause (verified against AGILE source, repeated here for a single authoritative record)
scripts/run_experiment.py's load_scene_overrides() never set Config.result_dir in
the dict passed to Config(**effective) - only an internal result_dir(condition)
helper existed, used solely for where OUR wrapper's own manifest files were
written. With Config.result_dir=None, AGILE's Config.get_result_dir()
(optimization/config/config.py) auto-generates results/{scene}/{timestamp}_
{loss_info}[_{mode_info}], where loss_info/mode_info are built from which loss
flags are enabled (mask, depth, normal, contact, smoothness, feature/type,
lastframe, handscale) - object_max_iter_per_frame (condition C's delta) is NOT
one of the incorporated fields, so C's directory is distinguished from A's by
timestamp alone.

### No output collision occurred - verified on disk, not assumed
Each condition's auto-generated directory name is distinct (different timestamps;
B additionally differs via its smoothness-weight component). Confirmed via direct
filesystem listing (find ... -type d) before any analysis was attempted - A's and
B's directories are separate, non-overlapping, each containing a complete,
independent set of output files.

### Fix commit
3f8720e5a096ba3c0300d0f0fd4a3206198a926c (pushed to origin/main) - 
load_scene_overrides() no longer sets result_dir; build_effective_config() now
sets reference["result_dir"] = str(result_dir(condition)) before effective is
copied from reference (so it is identical in both, never appears in the
single-field-diff check). 

### Re-verification that the fix is real (item 4 of the user's latest instructions)
Re-ran --check-only for conditions A and B after the fix (no GPU job consumed):
diffs still exactly correct ([], [object_smoothness_loss]); directly inspected the
written effective_config.json and confirmed result_dir now resolves to
/scratch/users/anirban/tamaghnam/HOI/results/A_released (and equivalently for
other conditions) - CONFIRMED the corrected wrapper explicitly sets
Config.result_dir for any future experiment launched through it. This fix was
NOT applied retroactively to the already-running condition C (its Config object
was constructed before the fix existed; a running job was not touched).

## 2026-11-04 (cont.) - Condition C COMPLETED; full A/B/C temporal-consistency analysis

### Condition C (job 73232): COMPLETED, exit 0:0, elapsed 01:59:23 (7138s) - notably
longer than A/B, consistent with double the object iteration budget.
"[Bidirectional] Saving combined result: 2 backward + 38 forward = 40 frames",
"Optimized frame range: 0 to 39" - identical frame structure to A and B. Final
object scale 0.9288 (reported as diagnostic only, see comparison below).

### Output completeness verified (item 2) - all three conditions
All three aligned/ directories (actual AGILE output paths, see prior entries)
contain the identical 16-item file set: Bake_Mat.png, camera_data.npz,
export_metadata.json, mano_betas.npz, mano_faces.npz, mano_global_orient.npz,
mano_hand_pose.npz, mano_transforms.npz, material_0.png, material.mtl,
object_assets/, object_assets.json, object_mesh.obj, object_poses.npz,
optimized_mano_joints.npz, optimized_mano_verts.npz. Frame correspondence
confirmed identical across A/B/C: all three report "frame range 0 to 39 (total:
40 frames)" with the same 2-backward+38-forward composition.

### Temporal-consistency analysis (scripts/analyze_temporal_consistency.py, run
against the REAL aligned/ directories, not the wrapper manifest dirs)
Object rotation/translation deltas (median, IQR, max over 39 frame-to-frame
transitions per condition), zero NaN in all cases:
| Condition | obj rot median (rad) | obj rot IQR | obj trans median | obj trans IQR |
|---|---|---|---|---|
| A | 0.17868 | 0.11517 | 0.013007 | 0.013062 |
| B | 0.18042 | 0.11196 | 0.013080 | 0.013167 |
| C | 0.17998 | 0.11080 | 0.013039 | 0.012930 |

OBSERVATION: object rotation/translation deltas are nearly identical across all
three conditions - B's added smoothness loss did NOT produce a measurably lower
frame-to-frame delta relative to A at this level of precision. This is a real,
reportable finding, not a null result being hidden.

Hand rotation delta: EXACTLY 0.0 (median, IQR, max all 0.0, 39/39 valid, 0 NaN) in
ALL THREE conditions. Verified numerically, not inferred - every per-frame
rotation_matrices entry is identical across the full 40-frame sequence in every
condition. Consistent with fix_hand_rotation_to_wilor=True, though the exact
mechanism (whether this field represents absolute orientation vs. a reference-
relative transform) was not further verified against source - flagged as an open
question, not asserted as fully explained.

### Iteration-budget convergence (item 9) - comprehensive, not just the smoke-test sample
Extracted ALL [Convergence] log records (80 per condition = 40 frames x 2
components; 240 total across A/B/C). Result: 100% of all 240 component-level
optimizations report either "stalled_at_budget" or "budget_exhausted" - ZERO
early-stopping convergence anywhere, in any condition, for either hand or object.
This is the same pattern observed in the smoke test, now confirmed universal
across the full approved window, not an artifact of the smoke test's tiny window.

Status tally by condition/component:
- A: hand 40/40 stalled_at_budget; object 27 budget_exhausted@400 + 12
  stalled_at_budget@400 + 1 budget_exhausted@800 (first frame, 
  object_max_iter_first_frame=800 per released config).
- B: hand 40/40 stalled_at_budget; object 22 budget_exhausted@400 + 17
  stalled_at_budget@400 + 1 budget_exhausted@800 (first frame).
- C: hand 40/40 stalled_at_budget; object 32 budget_exhausted@800 + 8
  stalled_at_budget@800 (all 40 at the condition's 800-iteration delta).

Notable, flagged without over-interpretation: C's object optimizations show a
HIGHER proportion reported as "budget_exhausted" (32/40=80%) than A's (28/40=70%)
despite double the iteration budget. The precise semantic distinction between
"budget_exhausted" and "stalled_at_budget" was not verified against AGILE's
convergence-checking source code in this pass - this comparison is reported as an
observed pattern, not a conclusion about whether C's extra iterations helped or
hurt convergence, since the status labels' exact criteria are not fully
understood yet.

## SOURCE-LEVEL DIAGNOSTIC INVESTIGATION (2026-11-04)

Investigation requested before any Condition D authorization. All A/B/C outputs
left untouched (read-only inspection + log/config analysis only). No saved data
was modified. No new experiments launched.

### 1. Convergence semantics — traced to source

**Files**: `optimization/optim/convergence.py` (`ConvergenceMonitor.observe`, lines 17-31),
`optimization/optim/object_optimizer.py` (lines 312, 543-549), `optimization/optim/hand_optimizer.py`
(lines 293, 440-446).

`ConvergenceMonitor.observe(loss, step, gradient, phase)` keeps a 5-sample sliding window
(`deque(maxlen=5)`) of `(loss, step_norm, max_grad)` per iteration. On each call, once the
window is full, it computes:
- `small_step = max(steps) < step_tol`   (`step_tol` = `cfg.online_converge_thresh` = `1e-4`, explicitly set in the released `ho3d.json`, confirmed identical in A and C's `effective_config.json`)
- `stable_loss = max(losses)-min(losses) <= loss_rtol * max(1, abs(losses[0]))`  (`loss_rtol` = `cfg.converge_loss_rtol`, **dataclass default `1e-5`** — not present as an explicit key in the released `ho3d.json` or our `effective_config.json` dumps; governs via `Config`'s dataclass default, confirmed by reading `config.py` line 322, not by our own manifest, which only records explicitly-set keys — **flagged below as a manifest completeness gap**)
- gradient check: `max(grads) < grad_tol`  (`grad_tol` = `cfg.converge_grad_tol`, dataclass default `1e-3`, config.py line 323, same manifest caveat)

Returns, in this exact priority:
- `'converged'` only if **all three** (small step AND stable loss AND small gradient) hold.
- `'stalled'` if `small_step` holds but at least one of the other two does not.
- `'optimizing'` otherwise (step itself not yet small).

The per-frame optimization loop (`object_optimizer.py` train_one_frame, `hand_optimizer.py`
train_one_frame) calls `convergence.observe(...)` every iteration and breaks early only when
`convergence_status == 'converged'`. After the loop exits (by `break` or by exhausting
`train_max_iter`), the final status recorded to `[Convergence]` is:
```
status = 'converged' if converged else ('stalled_at_budget' if convergence_status == 'stalled' else 'budget_exhausted')
```
(`object_optimizer.py` line 544; `hand_optimizer.py` line 441 — identical logic, independently
implemented per component.)

**Answers, direct from source, not inferred from naming:**
- **Are the three statuses mutually exclusive?** Yes, by construction — a single `if/elif/else`
  on the *last* observed `convergence_status`; no code path can produce two simultaneously.
- **Does `stalled_at_budget` mean progress stopped, or just that the budget was reached?**
  Neither, precisely — it means the **parameter step size** was small (<1e-4) on the final
  5-iteration window, but loss and/or gradient did **not** clear their thresholds. The code's
  own comment (`object_optimizer.py` line ~536, duplicated at `hand_optimizer.py` line ~390):
  *"LR decay can make updates tiny while the objective gradient is large. Report that as
  stalled and retain the configured iteration budget."* This is an explicit acknowledgment
  that `stalled_at_budget` can be an **LR-schedule artifact** (the Adam scheduler decays step
  size toward the end of the fixed `train_max_iter` budget regardless of whether the true
  gradient has gone to zero) rather than genuine convergence or a genuine plateau. It is
  **not** a certificate that optimization "stopped making progress" in the loss/gradient
  sense — only that *parameter updates* became small.
- **`budget_exhausted`** means the step size itself was *still above* `step_tol` when the
  iteration budget ran out — i.e., the optimizer was still taking non-negligible steps when
  the loop ended. This is the stronger signal of a genuinely insufficient iteration budget.
- **Are gradients, losses, and parameter changes actually checked?** Yes, all three, but only
  `'converged'` requires all three simultaneously; `'stalled'` requires only the step-size
  criterion, which is why it can fire even with a large gradient still present.
- **Does the optimizer retain useful per-iteration information not saved in our outputs?**
  Yes. Per-iteration loss/step/gradient values exist transiently inside `ConvergenceMonitor`'s
  5-sample window and are discarded after each frame. There IS a TensorBoard path
  (`runner.tb_writer`, `trainer.py` line 88, logging per-iteration loss/pose_change per frame
  in both optimizers) — but **we confirmed no `tfevents` files exist anywhere under
  `results/` or `logs/` for A/B/C** (checked via `find -iname "*tfevents*"`, empty), meaning
  `cfg.tb_dir`/logging was never configured for these runs. The only persisted per-iteration
  signal is the rounded-to-4-decimals `desc` string in stdout/stderr (e.g.
  `alpha_loss=0.4283 feature_loss=5.8663 smoothness_loss=0.0000 ho_loss=0.1192`) and the single
  final `[Convergence]` summary event per frame/component. Full per-iteration trajectories are
  **not recoverable** from existing outputs.

**Manifest completeness gap (new finding, not a correctness bug):** our own
`run_experiment.py`'s `effective_config.json` only records keys explicitly present in the
released `ho3d.json` + scene overrides + condition delta. `converge_loss_rtol` and
`converge_grad_tol` are never explicitly set by the released config, so they are **absent**
from our manifest (not `null`-valued — simply missing keys), and the actual governing values
come from `Config`'s Python dataclass defaults. This means our manifest alone cannot fully
reconstruct the effective runtime configuration for these two fields — verified instead by
reading `config.py` directly. Noted as a reproducibility-documentation gap; does not affect
correctness of the completed A/B/C runs.

### 2. Invariant hand rotation — traced to source, finding CORRECTED from "unexplained"

**Files**: `optimization/optim/hand_optimizer.py` (`_initialize_hand_pose`, lines 41-56; main
loop lines 238, 290, 298, 451), `optimization/pipelines/trainer.py`
(`save_optimized_data`, lines ~853-868).

- `hand_optimizer.py` line 238: `rvec = torch.nn.Parameter(torch.zeros(3, device=device),
  requires_grad=not cfg.fix_hand_rotation_to_wilor)` — under `fix_hand_rotation_to_wilor=True`
  (the released setting used by all of A/B/C), `rvec.requires_grad=False`: **rotation is never
  optimized**, only translation (`tvec`) is. This is deliberate, documented in `config.py`'s
  own comment on the field (lines ~232-236): *"Keep each frame's original WiLoR global
  rotation exactly: skip hand PnP initialization and rotation updates... Translation remains
  optimizable."*
- `_initialize_hand_pose` (line ~46-49): `original_R = runner.trainset[frame_id]['mano_global_orient']`
  — this **is** read fresh per frame from the dataset (the per-frame WiLoR estimate genuinely
  varies frame to frame; rotation is not globally constant in the underlying data).
- In the per-iteration loop (line 298, repeated at line 451): `new_R = base_R if
  cfg.fix_hand_rotation_to_wilor else rotation_vector_to_matrix_safe(...)` — `new_R` is
  literally `base_R` (the per-frame WiLoR rotation), held fixed for all iterations of that
  frame's hand optimization.
- **The decisive step is at export**, `trainer.py` `save_optimized_data`:
  ```python
  if self.cfg.fix_hand_rotation_to_wilor:
      if not torch.equal(hand_R, mano_global_orient):
          raise ValueError('Fixed WiLoR hand rotation changed before export')
      # The saved transform is relative to the original WiLoR
      # orientation. Identity preserves it without conversion.
      hand_rvec = torch.zeros(3, dtype=hand_R.dtype)
      hand_R = torch.eye(3, dtype=hand_R.dtype)
  ```
  The code **explicitly overwrites** the saved rotation with `torch.eye(3)` / `zeros(3)`
  whenever `fix_hand_rotation_to_wilor=True`, by design: the saved field is defined as "rotation
  relative to this frame's own original WiLoR orientation," which is trivially identity when
  rotation was never changed from that same per-frame WiLoR value. This identity matrix is
  then appended to `final_hand_rotations` and written to `mano_transforms.npz`'s
  `rotation_matrices` field (`trainer.py` line 903) — **the exact field our
  `analyze_temporal_consistency.py` reads** (confirmed in that script's own header comment
  and `analyze_hand()` function, which consumes `rotation_matrices`/`translations` from
  `mano_transforms.npz`).

**Direct answers:**
- **Why exactly invariant?** Because the export code hard-codes `hand_R = eye(3)` for every
  frame under this config branch — not because the underlying hand orientation is actually
  constant, and not because of any bug in our analysis.
- **Are hand translation and articulation optimized independently of rotation?** Yes —
  translation (`tvec`) is always optimized via its own Adam optimizer regardless of
  `fix_hand_rotation_to_wilor`; MANO pose articulation is optimized only if `optimize_hand_pose`
  is separately enabled (`False` in the released config — not optimized here at all).
- **Is the saved rotation copied from WiLoR, initialized once, or recomputed per frame?**
  Recomputed (re-read) per frame from `trainset[frame_id]['mano_global_orient']`, then
  immediately redefined as identity *relative to itself* at save time.
- **Is our analysis script reading the correct field/convention?** Yes — verified field name
  (`rotation_matrices` in `mano_transforms.npz`) and verified, via source, that this field's
  value is guaranteed to be `eye(3)` under the released config, not an analysis bug.
- **Is a zero rotation delta an expected consequence of the released config, or a possible
  serialization/analysis issue?** **Expected consequence of the released config**, confirmed
  by source, not an analysis issue. The true per-frame absolute WiLoR rotation (which does
  vary) is separately preserved verbatim in `mano_global_orient.npz` (`trainer.py` line
  858-859, saved straight from `self.parser.mano_global_orient`, unmodified, identical input
  data across A/B/C since it's not optimized) — but that field was not what our
  temporal-consistency script analyzed, and since it is identical pre-optimization input data
  across all three conditions, it carries no information relevant to comparing A/B/C anyway.

**Correction to the previous report**: the earlier framing ("hand rotation delta exactly 0.0,
flagged as a real finding not fully explained mechanistically") is now **fully explained** by
source and should no longer be treated as an open question. It is a by-design consequence of
`fix_hand_rotation_to_wilor=True`'s export convention, carries zero information about actual
hand-rotation smoothness, and should not be used as evidence of anything about the optimizer's
behavior in any condition.

### 3. A/B/C comparison audit

- **Was B's smoothness loss actually active and contributing?** Confirmed empirically, not
  just structurally. Searched A/B/C's full stderr logs (which contain the real per-iteration
  `desc` strings) for `"[Object].*smoothness_loss="`:
  - A: **0** occurrences. C: **0** occurrences. B: **31,200** occurrences.
  (A/C's `object_smoothness_loss=False`, so the term is never added to the object loss dict at
  all — confirmed absent, not just zero-valued.) This cleanly confirms the manipulation was
  isolated to B only, as intended — no leakage into A or C.
  (Separately, `"[Hand].*smoothness_loss="` appears in all three — ~15,600/14,003/15,600 times
  — but this is `hand_smoothness_loss`, an always-on, non-condition-varied default loss
  (`config.py` line 151, `True` by default) that the hand optimizer's own
  `_to_legacy_loss_dict` happens to display under the same generic key name `smoothness_loss`.
  Confirmed this is a **display-key naming collision only**, unrelated to the object
  smoothness manipulation under test — cross-checked `object_optimizer.py`'s `_to_legacy_loss_dict`
  key map, lines 226-238: `'object_smoothness': 'smoothness_loss'`.)
  - **Precision limitation, newly discovered**: every one of the 31,200 `[Object]` smoothness
    print lines in B reads exactly `smoothness_loss=0.0000` (4-decimal rounding in the log
    format string). `object_smoothness_lambda=0.0002` (confirmed in B's `effective_config.json`,
    matches the released default, `config.py` line 146). A representative full line:
    `Frame 15 [Object] Iter 249: alpha_loss=0.4283 feature_loss=5.8663 smoothness_loss=0.0000 ho_loss=0.1192`
    — i.e. the *weighted* smoothness term is at least an order of magnitude smaller than the
    other three simultaneously-active losses at essentially every iteration sampled, and its
    true (unrounded) magnitude cannot be recovered from the existing logs (no TensorBoard
    writer was active — see Section 1). **We can confirm the loss term was structurally wired
    in, correctly weighted per the released config, and fired on effectively every eligible
    frame — but we cannot confirm from existing logs whether its gradient contribution was
    ever large enough, relative to the other active losses, to meaningfully influence the
    optimization trajectory.** This is a plausible, source-grounded explanation for the
    near-identical A vs. B temporal-consistency metrics reported previously: a released
    smoothness weight (2e-4) small enough, relative to mask (5.0)/feature (10.0)/ho_relpose
    (400.0), that it may simply be too weak to produce a measurable effect — not necessarily
    evidence that smoothness regularization is ineffective in general.
- **Did `recover_object_pose_from_mask` (True in the released config) ever disable B's
  smoothness loss for any frame?** (`object_optimizer.py` line ~375: `if recovered:
  loss_cfg.object_smoothness_loss = False`.) Checked `[Object recovery]` print events
  (`recovery.py` line 97) in all three logs: **0 occurrences in A, B, and C.** Recovery never
  triggered in this scene/window; this confound does not apply here.
- **Do the reported temporal metrics use the same frames/coordinate convention across A/B/C?**
  Yes — all three conditions' logs report identical `"Optimized frame range: 0 to 39 (total: 40
  frames)"` and identical `2 backward + 38 forward` composition (already verified in the prior
  report); `analyze_temporal_consistency.py` reads the same `object_poses.npz`/
  `mano_transforms.npz` field names and the same camera-frame convention (`object_poses_in_cam`,
  a 4x4 matrix in camera coordinates per `trainer.py`'s save code) for every condition — no
  per-condition divergence in what's being read.
- **Do the 40 saved frames represent the expected keyframes from the approved MC1 window and
  stride?** Traced the hold-mode index mapping: `megasam.py` line 67,
  `hold_indices = np.arange(0, num_frames_before_hold, hold_mode_interval)` subsamples the
  **entire** sequence by stride 5 *before* windowing, producing a reduced index space where
  subsampled index `i` corresponds to actual frame `5*i`. `trainer.py` line 1307 enforces
  `contact_frame % hold_mode_interval == 0`; our approved `start_frame=10` → subsampled index 2.
  Approved `end_frame=200` → subsampled index 40. Forward pass from index 2 to 40 → **38
  frames**; backward pass from index 2 down to index 0 → **2 frames**. This exactly matches the
  observed `2 backward + 38 forward = 40 total` in all three conditions' logs — confirmed
  correct, not coincidental.
- **Limitations in interpreting B's null result (consolidated):** (1) the smoothness weight's
  true unrounded magnitude is unverifiable from existing logs (see above) — the null result may
  reflect an under-powered released weight rather than a genuine absence of effect; (2) all 240
  component-optimizations across A/B/C exhausted their iteration budget without satisfying the
  strict convergence criteria (Section 1), so none of A/B/C represents a settled optimum —
  differences (or their absence) between conditions are confounded with incomplete optimization
  under the current fixed-budget/threshold combination; (3) the specific metric compared
  (object pose geodesic/L2 deltas) is sensitive mainly to the rotation/translation smoothness
  loss's influence on *pose*, while the implemented `vertices`-type smoothness loss
  (`optimization/losses/smoothness.py` lines 112-119, `temporal_smoothness_loss_vertices`)
  penalizes vertex displacement under a *rotation-difference* operator (`dR = prev_R - curr_R`,
  not a geodesic distance) — a different, non-equivalent mathematical quantity from the
  geodesic metric we used to compare conditions, so even a real smoothness effect might not be
  fully captured by our chosen comparison metric.

### 4. Condition D — evaluated, NOT launched, presented for authorization

Given the above, a higher-iteration-budget condition is scientifically justified **to answer a
narrower question than originally framed**: not "does more compute make AGILE more accurate,"
but **"does the universal `budget_exhausted`/`stalled_at_budget` non-convergence resolve with a
substantially larger budget, or is it structural (e.g., threshold values too strict for this
loss combination, or a genuine non-zero-gradient plateau)?"** This is answerable without new
code, using only the existing `[Convergence]` status field and loss/gradient values already
logged.

**Proposed Condition D (recommendation only — awaiting authorization):**
- Same MC1 sequence, same approved frame window (`start_frame=10, end_frame=200`).
- Same released configuration as A (baseline, no smoothness) — isolates the iteration-budget
  question from the smoothness question.
- Only `object_max_iter_per_frame` changes, from A's `400` to **`1600`** (4x A, 2x C).
  `hand_max_iter_per_frame` left unchanged at `200` (hand already hits `stalled_at_budget`
  40/40 across all conditions at this budget; scaling hand iterations is a separate,
  independently justified follow-up question, not bundled into D).
- **Budget multiplier justification**: from A (400 iters, 1:12:33) vs. C (800 iters, 1:59:23),
  the marginal per-iteration object-optimization cost is ≈ (1:59:23 − 1:12:33) / 400 iters /
  40 frames ≈ 0.176 sec/iteration. A further doubling to 1600 (adding 800 iters/frame over C)
  is estimated to add ≈ 800 × 0.176 × 40 ≈ 94 minutes over C's runtime, for an estimated total
  **wall-clock ≈ 3h 30min** (bounded estimate; actual cluster contention/GPU variance not
  included). This is a single, modest, interpretable step — not an open-ended search — chosen
  to be large enough to meaningfully test the "needs more compute" hypothesis without
  committing to a much longer, harder-to-schedule run before we know whether the hypothesis
  even holds.
- **Interpretation criteria, fixed in advance:**
  - If the proportion of `budget_exhausted` (object) drops substantially relative to C's 80%
    (32/40) — e.g., most frames now report `'converged'` or at least `'stalled_at_budget'`
    rather than `'budget_exhausted'` — this supports "needs more compute," and a convergence
    analysis under relaxed/longer budget becomes meaningful.
  - If the `budget_exhausted`/`stalled_at_budget` proportions are essentially unchanged from C
    despite 2x more iterations, this supports "structural" (threshold/landscape issue, not
    simply insufficient time), and further increasing the budget alone would not be a
    productive next step — the convergence thresholds themselves, or the loss landscape, would
    need separate investigation instead.
  - Temporal-consistency metrics (geodesic/L2 deltas) will be recomputed under the same
    protocol as A/B/C for completeness, but per Section 3's caveats, will be interpreted only
    as descriptive, not as an accuracy claim, consistent with the standing evaluation-plan
    limitation.

**This recommendation is not launched. Awaiting explicit authorization before submitting any
Condition D job.**

## CONDITION D — AUTHORIZED, IN PROGRESS (2026-11-04)

User reviewed the source-level investigation and authorized Condition D: released A
configuration with exactly `object_max_iter_per_frame: 400 -> 1600` changed, plus
observational-only per-iteration diagnostic logging added beforehand, tested in isolation
first. B and C were not touched. A/B/C outputs were not modified.

### Diagnostic logging implementation

Added `repo/optimization/optim/diagnostics.py` (new file): a `log_record(dict)` function that
appends one JSON line to the path named by the `AGILE_DIAGNOSTIC_LOG` env var, or is a complete
no-op if that var is unset (so A/B/C's exact prior behavior is reproduced with zero code-path
difference when it's unset). Patched `object_optimizer.py` and `hand_optimizer.py` to call
`log_record` once per iteration (loss terms, gradient_inf, step_norm, pose_change, learning
rates, convergence_status) and once per frame/component at termination (status + a static,
human-readable `reason` string per status, plus the existing `event` dict fields). The only
change to pre-existing lines was hoisting one already-made call,
`optimizer_step_norm(optimizers, step_before)`, into a local variable (`_diag_step_norm`) to
avoid computing it twice — the value, call site, and inputs are unchanged, and it still feeds
`convergence.observe()` identically. No gradient, optimizer, scheduler, loss, or convergence
computation was altered. Full diff reviewed before committing — 46 insertions across both
files, zero deletions of existing logic lines. Committed **locally inside `repo/`'s own git**
(commit `dd135b4`, on top of upstream `4b0d0cd`) for traceability via the existing
`git_commit(REPO)` manifest mechanism — **not pushed** to `aim-uofa/AGILE` upstream (we don't
have, and shouldn't use, write access there). `repo/` remains gitignored in our own tracked
repo by design (vendored external checkout); this local commit is the only persistence
mechanism for the patch, recorded here for recoverability if `repo/` is ever reset.

Extended our own `scripts/run_experiment.py` (tracked in our git) with
`CONDITIONS["D"] = {"object_max_iter_per_frame": 1600}` and the matching `result_dir`/`log_dir`
name (`D_control_iters1600`) and `--condition` choice. This is the only change needed to make D
a first-class condition under the exact same single-field-diff enforcement
(`assert_single_field_diff`) already governing A/B/C.

### Diagnostic logging smoke test

New isolated script `scripts/run_diagnostics_smoke_test.py` + `scripts/run_diagnostics_smoke_test.sh`,
output to `results/smoke_test_diagnostics/` (separate from `results/smoke_test/` and from
A/B/C/D) — released config, smoke window `start_frame=10, end_frame=15` (same convention as the
original pipeline smoke test). Validates, after a real GPU run: every JSONL line parses;
required keys present per record type; all numeric fields finite; per-(frame_id, component)
iteration numbers form a contiguous `0..N-1` sequence; exactly one termination record per key
with a matching `iterations` count; and a cross-check that the count of termination records
from the **new** mechanism exactly matches `len(runner.convergence_events)` from the
**pre-existing, unmodified** mechanism.

**Job 73232... correction, job 73366**, `short` partition, **COMPLETED, 00:12:35, exit 0:0**.
Result: **PASSED** — 2206 total diagnostic lines, all valid; 6 distinct (frame_id, component)
keys, each with correct contiguous iteration numbering and exactly one matching termination
record; **6 convergence_events (old mechanism) == 6 termination records (new mechanism)**,
confirming the new logging introduced zero divergence from the pre-existing behavior.

### Condition D pre-flight validation (before submission)

- `python3 scripts/run_experiment.py --condition D --check-only` (run on the login node —
  inspection only, no GPU, no AGILE/torch import on this path): config diff vs released
  baseline reported as exactly `['object_max_iter_per_frame']`; prerequisite checks (MANO,
  DINO, data) passed.
- Compared `results/D_control_iters1600/effective_config.json` against `results/A_released/`'s:
  differs in exactly `object_max_iter_per_frame` (400->1600) and `result_dir` (the
  intentionally-isolated infra field, identical mechanism as A/B/C's fix). `D_control_iters1600/`
  result dir confirmed freshly created, empty of any prior training output, no collision risk
  with A/B/C.
- Scratch quota: 79T free / 85T total on `/scratch` - no concern for D's output (~same order of
  magnitude as A/B/C, each a few hundred MB) or diagnostics JSONL (estimated tens of MB given
  roughly 1600*40 + 200*40 ~ 72,000 anticipated iteration records for D, each a few hundred
  bytes).
- Partition load at submission time: cluster fairly busy on `a100`/`ada` from other users; job
  submitted with the same `short,long,a100,medium,ada` multi-partition fallback used by A/B/C.
- Runtime estimate (recorded before submission): from A (400 obj-iters, 1:12:33) vs. C (800
  obj-iters, 1:59:23), marginal per-iteration object cost ~= (1:59:23-1:12:33)/400/40 frames ~=
  0.176 s/iter. D adds 800 more iters/frame over C: ~800*0.176*40 ~= 94 min over C's time ->
  estimated total **~3h30m**, comfortably inside the 8-hour walltime requested.

## CONDITION D — RESULTS AND B-SMOOTHNESS REVISIT (2026-11-04, continued)

### Job outcome
Job 73371 (`exp_D`), partition `short`, completed successfully: exit 0:0, elapsed 03:29:30
(estimate was ~3h30m — matched closely). STATUS=COMPLETE. Output verified complete at
`results/D_control_iters1600/aligned/` (same 16-file set as A/B/C, 40 frames: 2 backward + 38
forward, matching the approved window's expected keyframe count exactly as in A/B/C).

### Convergence status distribution — object optimizer (40 frames each)
| Condition | budget (iters) | budget_exhausted | stalled_at_budget |
|---|---|---|---|
| A | 400  | 28/40 (70%) | 12/40 (30%) |
| C | 800  | 32/40 (80%) |  8/40 (20%) |
| D | 1600 | 38/40 (95%) |  2/40 (5%)  |

This is the OPPOSITE of the "more iterations resolves budget exhaustion" hypothesis motivating
D. Root cause, established from D's new per-iteration diagnostic log
(`logs/experiments/D_control_iters1600/diagnostics.jsonl`, 71,280 records) plus source:

1. **The object optimizer's learning rate never decays**, in A/B/C/D alike. `object_scheduler_type`
   (a field distinct from the hand optimizer's `online_scheduler_type`) defaults to `"constant"`
   in `optimization/config/config.py:329`, and the released `configs/ho3d.json` never overrides
   it. `create_scheduler(..., scheduler_type='constant')` returns
   `ConstantLR(optimizer, factor=1.0, ...)` — confirmed empirically: D's diagnostics show
   `learning_rates == [0.002, 0.001]` at every logged checkpoint from iteration 0 through the
   final iteration 1599, for every frame checked.
2. **Object loss plateaus within the first ~100–300 iterations** regardless of total budget:
   across all 39 complete object frames in D, 36/39 show <2% loss change between iteration 300
   and the final iteration 1599 (checked explicitly, not inferred).
3. **Step size oscillates noisily around the convergence threshold** (`online_converge_thresh =
   1e-4`) for the remainder of the budget, without a decaying trend — e.g. frame 50's step_norm
   cycles between ~3e-5 and ~4e-4 for 1300+ iterations; aggregated over the last 200 iterations
   alone, per-frame threshold-crossing counts range from 0 to 45.
4. **Conclusion**: `budget_exhausted` vs. `stalled_at_budget` is substantially determined by
   which side of this flat, noisy oscillation the terminal iteration happens to land on — not by
   whether the optimizer has genuine remaining progress to make. A longer budget simply samples
   more iterations from this noisy process, so it has more chances to land above threshold at
   termination purely by chance. This quantitatively explains the monotonic 70%→80%→95% trend.

### Temporal-consistency metrics (final saved outputs, actual `analyze_temporal_consistency.py`
run on A/C/D's real data — see "A/B/C path correction" note below)
| Condition | object_rotation_delta_rad (median) | object_translation_delta (median) |
|---|---|---|
| A | 0.17868 | 0.013007 |
| C | 0.17998 | 0.013039 |
| D | 0.17978 | 0.013039 |

Differences across A/C/D are ~0.1–0.7% of the median value — noise-level, not material. Hand
translation/rotation deltas are identical across A/C/D to displayed precision (expected: object
iteration budget does not affect the hand optimizer). **This directly confirms, from the actual
saved outputs (not just per-iteration diagnostics), that the 4x increase in object iteration
budget (400→1600) produced no detectable change in the final reconstructed object or hand
trajectories.**

### Explicit non-claims (per authorization)
This is a finding about convergence-status mechanics and iteration-budget sufficiency only. It
does **not** make, and should not be read as making, any claim about geometric accuracy against
ground truth. The *increase* in budget_exhausted counts with more iterations is not evidence of
worse reconstruction, any more than a decrease would have been evidence of better reconstruction
— the temporal-consistency numbers above are what settle the "did anything actually change"
question, and they say no.

### A/B/C path correction (process note, not a data-integrity issue)
While preparing this comparison, `results/A_released/`, `results/B_smoothness/`, and
`results/C_control_iters800/` were found to contain only `effective_config.json`,
`run_manifest.json`, `STATUS` — no `aligned/` output. This is NOT data loss: it is the
previously-documented `Config.get_result_dir()` auto-path bug (see the "A/B/C output isolation"
findings earlier in this file) — A/B/C's real outputs were saved, before the fix, to
AGILE's own auto-generated paths:
  - A → `results/MC1/1104_011226_mask5.0_feature10.0_typelocal_lastframe/aligned`
  - B → `results/MC1/1104_011508_mask5.0_smoothness0.0002_feature10.0_typelocal_lastframe/aligned`
  - C → `results/MC1/1104_021134_mask5.0_feature10.0_typelocal_lastframe/aligned`
D (run after the `build_effective_config()` fix) saved correctly to its intended
`results/D_control_iters1600/aligned`. All data for all four conditions is present and intact;
this note exists so a future reader does not mistake the isolated `results/{A,B,C}_*/` directories
for the complete outputs.

### B smoothness-loss magnitude — final answer (authorization item 5)
Re-verified via disambiguated log parsing (`[Object]`-prefixed lines only, vs. the earlier
non-disambiguated grep that conflated them with the unrelated, always-on `[Hand]`
`hand_smoothness_loss` term): **all 31,200** `[Object]` smoothness_loss print occurrences in B's
log read exactly `0.0000` at 4-decimal precision, with zero exceptions. No higher-precision
record exists anywhere in B's artifacts: no alternate print statement in
`object_optimizer.py`/`losses/__init__.py` emits the raw/unweighted value, no TensorBoard writer
was active (no `tfevents` files), and `results/B_smoothness/` holds no per-iteration history
artifact. What the evidence *does* support, as a bound rather than a point estimate: a value that
rounds to 0.0000 at every one of 31,200 samples implies the weighted contribution stayed below
~5e-5 throughout the entire run — two to three orders of magnitude below the co-active
`feature_loss` (~5.8) and `alpha_loss` (~0.4) at the same iterations. This is consistent with
(but does not rigorously prove) negligible practical influence; Adam's per-parameter adaptive
step-size normalization means a persistently-directional but tiny gradient could in principle
still exert some influence that a raw-magnitude argument would not capture. No new experiment was
run to resolve this further, per instruction.

### Next scientifically justified direction (not run, not authorized — offered per item 6)
Given the object optimizer's LR is flat by the released config's own `object_scheduler_type` and
loss plateaus within ~300 iterations regardless of total budget, further increases in
`object_max_iter_per_frame` alone are not a promising direction — this experiment (D) has now
demonstrated that. The data suggest the more productive lever, if iteration-budget-related
experiments are to continue, would be testing an actual *decaying* `object_scheduler_type` (e.g.
`"sequential"`, already implemented and already used by the hand optimizer, but never applied to
the object optimizer in the released config) — not raw iteration count — since a flat LR acting
on a noisy, plateaued loss is the direct, source-grounded explanation for the oscillating
step-size behavior observed here. This is a new, falsifiable hypothesis, offered as a candidate
next direction only; no experiment has been run or authorized beyond A/B/C/D.

## CONDITION E — DECAYING OBJECT-OPTIMIZER LR SCHEDULE (2026-11-04/05)

### Scheduler implementation (inspected fresh from source before any config change, per authorization item 1)
Confirmed three **distinct** scheduler-type config fields exist in `optimization/config/config.py`,
not one: `object_scheduler_type` (read explicitly by `object_optimizer.py` for all its
optimizers — rot/trans/scale), `hand_scheduler_type` (read by `hand_optimizer.py`'s main
trans/rot optimizers), and `online_scheduler_type` (fallback inside `Runner.create_scheduler()`,
used only for the hand's *optional* scale/intrinsics/pose schedulers). All three default
differently (`"constant"`, `"sequential"`, `"sequential"` respectively); the released
`ho3d.json` never overrides the first two, i.e. both sit at their dataclass defaults.
`create_scheduler()` dispatches on the type string to a real torch scheduler
(`ConstantLR`/`SequentialLR`/`LinearLR`/`ExponentialLR`, plus an unsupported `'sequential2'`
`LambdaLR` variant not in either field's `Literal` type) and raises `NotImplementedError` for
anything else — so only genuinely implemented options were considered. Confirmed
`scheduler.step()` fires unconditionally once per iteration, in lockstep with `optimizer.step()`,
identically in `object_optimizer.py` and `hand_optimizer.py` — there is no secondary interval or
condition gating it.

**Selected: `object_scheduler_type = "linear"`** — `LinearLR(start_factor=1.0, end_factor=0.1,
total_iters=train_max_iter)`. Chosen over `"sequential"` (multi-phase, already used by hand) and
`"exponential"` (requires deriving a gamma) as the simplest option that still produces gradual,
monotonic decay to a materially smaller terminal LR, with no additional config fields required.

### Condition E definition
Single-field delta from A: `{"object_scheduler_type": "linear"}`. Everything else (MC1 window
10–200, `object_max_iter_per_frame=400`, all loss weights, hand optimizer config, MANO/DINOv3
assets, seed) identical to A. Verified via `assert_single_field_diff()` and a manual
`effective_config.json` diff against A's: the only changed key is `object_scheduler_type`
(`'constant'` → `'linear'`).

### Scheduler smoke test (job 73447, isolated at `results/smoke_test_scheduler/`)
COMPLETED, exit 0:0, elapsed 00:09:50. Explicitly checked LR *decay*, not just scheduler
construction: for every `(frame_id, 'object')` optimizer in the 3-frame smoke run, LR ratio
(last iteration / first iteration) was 0.1001–0.1002 (expected ~0.1), and decay was confirmed
monotonic (first ≥ mid ≥ last) for every optimizer. Diagnostics mechanism cross-checked:
`runner.convergence_events` (6, pre-existing mechanism) exactly matched new termination
diagnostic records (6). PASSED before the real job was submitted.

### Pre-flight checks (all passed, authorization item 4)
Single-field diff confirmed against A; `results/E_linear_schedule/` confirmed isolated/empty of
real output before submission; scratch usage trivial (2.6G total project); `sbatch --test-only`
queue forecast (~12h wait) was a scheduling matter only, not a resource problem.

### Job outcome
Job 73449 (`exp_E`), completed successfully: exit 0:0, elapsed 01:11:20 (started
2026-11-04T21:24:10, ended 2026-11-04T22:35:30). STATUS=COMPLETE. Output verified complete at
`results/E_linear_schedule/aligned/` — same 16-file set as A/B/C/D, 40 frames (2 backward + 38
forward), matching the approved window's expected keyframe count exactly as in A/B/C/D.

### LR trajectory — genuinely decays (authorization item 4/5, question 1)
Pulled the **real** run's diagnostics log (`logs/experiments/E_linear_schedule/diagnostics.jsonl`,
24,480 records), not just the smoke test. All 40 distinct object frame-runs show monotonic LR
decay from the first to the last logged iteration, with ratio 0.1001–0.1003 (median 0.1002),
matching the `linear` scheduler's expected `end_factor=0.1` exactly. (Frame 10, the
backward/forward seed frame, is optimized twice — 800 total records — and each of its two
400-iteration passes independently shows the same ~0.1x decay, confirming the scheduler
re-initializes correctly on each call rather than carrying state across passes.) **Answer to
question 1: yes, the LR decays exactly as the selected scheduler specifies.**

### Loss trajectory — still plateaus early (question 2)
Computed percent change in total object loss from iteration 300 to the final iteration, across
all 40 frame-runs: median 0.48%, mean 0.74%, max 3.97%; 37/40 frame-runs show <2% change. This
is essentially the same plateau timing documented for D under the flat (constant) LR. **Answer
to question 2: no — the decaying schedule does not cause loss to continue improving past the
~100–300 iteration mark; the plateau is a property of the loss landscape/budget, not the LR
schedule.**

### Gradient/step-size behavior (question 3) and convergence-status proportions (question 4)
| Condition | object_scheduler_type | budget (iters) | budget_exhausted | stalled_at_budget |
|---|---|---|---|---|
| A | constant | 400  | 28/40 (70%) | 12/40 (30%) |
| C | constant | 800  | 32/40 (80%) |  8/40 (20%) |
| D | constant | 1600 | 38/40 (95%) |  2/40 (5%)  |
| E | **linear** | 400 | **1/40 (2.5%)** | **39/40 (97.5%)** |

Step magnitude settles far more smoothly under the decaying schedule: mean `step_norm` across
frames falls from 0.00222 (first 100 iterations) to 9.14e-5 (last 100 iterations) in E — a ~24x
reduction — versus D's constant-LR 0.0024 → 1.56e-4 over the equivalent first-400-iteration
window (~15x, and per D's own diagnostics, this decrease does not continue smoothly but
oscillates for the remaining 1200 iterations). At the exact terminal iteration, E's `step_norm`
median across all 40 frame-runs is 2.57e-5 (only 1/40 above the `online_converge_thresh=1e-4`
threshold), versus D's terminal-iteration median of 1.84e-4 (straddling the threshold, which is
exactly why D's 95% `budget_exhausted` rate is so high). **Answer to question 3: yes, gradients
and step magnitudes settle far more consistently (monotonically, not just on average) under the
decaying schedule than under the flat one.**

**Important caveat on question 4's result, required by the authorization's "do not assume a
scheduler is effective merely because it decays" and "distinguish convergence behavior from
reconstruction accuracy" instructions:** the near-total flip from `budget_exhausted` to
`stalled_at_budget` is a **mechanical, expected consequence of imposing a 10x LR decay on the
step-size threshold test**, not evidence that the optimizer found a better optimum or that more
genuine progress occurred. Because `step_norm` scales with `learning_rate × gradient`, forcing
the LR down to 0.1x by the final iteration will push `step_norm` under almost any fixed threshold
near the end of the run **regardless of whether the underlying gradient has actually stopped
changing** — this is precisely the caveat already present in the diagnostic code's own
`stalled_at_budget` reason string ("may reflect LR-schedule decay rather than a true plateau").
The loss-plateau result above (question 2) is the more reliable signal here, and it shows no
material change from the flat-LR conditions.

### Final trajectory and temporal consistency (questions 5 and 6, real saved outputs — `analyze_temporal_consistency.py`)
| Condition | object_rotation_delta_rad (median) | object_translation_delta (median) | hand deltas |
|---|---|---|---|
| A | 0.17868 | 0.013007 | baseline |
| E | 0.18025 | 0.013105 | identical to A |

Differences: object rotation +0.88%, object translation +0.75% — noise-level, consistent with
the ~0.1–0.9% spread already seen across A/C/D. Hand rotation/translation deltas are identical
to displayed precision (expected: this condition only changes the object optimizer's scheduler).
**Answer to question 5: no material change in the final object trajectory. Answer to question 6:
temporal consistency is effectively unchanged, not improved or worsened.**

### Does the schedule produce meaningful benefit within the same budget? (question 7)
No. The decaying schedule is correctly implemented, genuinely decays (question 1), and produces
much smoother, more monotonic step-size behavior (question 3) — but it does not change when the
loss plateaus (question 2) or what the final reconstructed trajectory looks like (questions 5–6).
The large shift in `budget_exhausted`/`stalled_at_budget` proportions (question 4) is better
understood as the convergence-status *heuristic* becoming a direct readout of the imposed LR
curve, rather than as a signal of improved optimization. Within a 400-iteration budget on this
scene/window, switching the object optimizer from a constant to a linearly-decaying LR schedule
has no detected effect on the reconstruction, only on the internal diagnostics' interpretation of
convergence.

### Explicit non-claims (per authorization)
This is a finding about optimizer/scheduler convergence dynamics and the convergence-status
heuristic's sensitivity to LR shape — not a claim about geometric accuracy against ground truth.
No ground-truth evaluation was performed for E, consistent with none having been performed for
A/B/C/D.

### Limitations
Single scene/window (MC1, frames 10–200); one decaying schedule (`linear`) out of three valid
options (`sequential`, `exponential` untested); one iteration budget (400, matching A) — a
decaying schedule combined with a *different* budget was not tested, since the authorization
scoped E to A's exact 400-iteration budget. Diagnostics give per-iteration optimizer internals
only, not pixel/mesh-level accuracy.

### Recommended next direction (not run, not authorized — offered per item 6)
Two independent levers have now been tested against the same null result: raising the object
iteration budget 4x (C/D) and imposing a 10x-decaying LR schedule on the existing budget (E).
Neither changed the final reconstructed trajectory beyond noise level, and the loss plateau at
~100–300 iterations is invariant to both. This converges on a different conclusion than either
experiment alone suggested: the bottleneck is unlikely to be the object optimizer's convergence
procedure at all. A more promising next step would be examining what the loss is actually
optimizing — e.g. whether `feature_loss`/`alpha_loss`'s local optimum at ~iteration 300 reflects
a genuine geometric optimum or a limitation of the loss formulation/feature representation
itself — or evaluating reconstruction quality against ground-truth object poses directly, which
no experiment in this series (A–E) has yet done. This is a new candidate direction only; no
experiment has been run or authorized beyond A/B/C/D/E.

## GROUND-TRUTH EVALUATION AUDIT — BLOCKED (2026-11-05)

Per authorization, this was a READ-ONLY audit of GT availability, metric validity, and frame
correspondence for A–E, run before any evaluation. No Slurm job was submitted, no files were
modified, nothing was downloaded. **Conclusion: the gate fails — no genuine official ground
truth exists on this cluster for MC1. Evaluation was NOT run, per the authorization's explicit
"stop before running the evaluation" instruction.**

### 1. GT availability — none found; everything present is a model prediction/initialization, not GT
| Asset | Found locally? | What it actually is | Path |
|---|---|---|---|
| Object 6D pose | **No** | `object_pose/ob_in_cam/*.txt` — **FoundationPose's own predicted pose** (its own naming convention, `ob_in_cam`), used only as optimizer initialization per `optimization/README.md` ("WiLoR, MoGe, FoundationPose ... object pose initialization") | `data/HO3D/preprocessed/train/MC1/object_pose/` |
| Object mesh | **No** | `object_pose/{original_mesh.obj,scaled_mesh.obj}` — a reconstructed/scanned mesh used for FoundationPose tracking, not an official YCB CAD model in GT-aligned coordinates | same dir |
| Hand pose/shape | **No** | `wilor/*.npz` — **WiLoR's predicted** MANO params, used only as a rotation/translation initialization (`fix_hand_rotation_to_wilor`, `config.py:241`) | `data/HO3D/preprocessed/train/MC1/wilor/` |
| Camera intrinsics | Present, but only the camera actually used for reconstruction; not independently cross-checked against an official calibration file (none exists locally to check against) | `moge/cam_K.txt`, also duplicated into every `camera_data.npz["intrinsic"]` | both present |
| Frame IDs | **Yes** — see item 3 below | `camera_data.npz["prediction_frame_ids"]` | every `aligned/` dir |

No `meta/*.pkl` and no `models/<obj>/textured_simple.obj` exist anywhere on this filesystem
(`find data/ -iname "*.pkl"` returns nothing under `HO3D/`; the only `.pkl` anywhere is the
unrelated `assets/mano/MANO_RIGHT.pkl` model file). This was verified by searching, not assumed
absent.

**Decisive confirmation from AGILE's own source and docs — the repo itself tells us GT is
intentionally excluded from what we have:**
- `optimization/utils/dataset_utils.py:27`: `HO3D_ROOT = os.environ.get('HO3D_ROOT',
  'datasets/HO3D/raw')` — a dedicated GT loader class (`Ho3dReader.get_gt_pose`/`get_mano_data`/
  `get_gt_mesh`) exists in the codebase and expects `{HO3D_ROOT}/<seq>/meta/*.pkl` and
  `{HO3D_ROOT}/models/<obj>/textured_simple.obj` — a **separate dataset directory that was never
  downloaded** on this cluster. `HO3D_ROOT` is currently unset; no such directory exists under
  this user's scratch or home.
- `docs/installation.md:98`: *"For evaluation, obtain the original HO3D annotations and object
  models ... Ground truth is not needed for optimization or visualization."*
- `docs/ho3d-results.md`: *"MANO model files, external network weights and evaluation ground
  truth are not included."*

### 2. Metric validity — formulas exist in-repo, but literally nothing is computable without GT
AGILE ships a real, already-implemented evaluation protocol at `optimization/evaluation/protocols.py`
(`METRIC_PROTOCOLS`, versioned `evaluation-v2-gt-cv-robust-icp`):
- `cd_icp` (cm²) — object Sim(3) alignment via PCA+FPFH+ICP, bidirectional nearest-neighbor
  Chamfer distance.
- `cd_right` (cm²) — HOLD-style: object pose minus hand-wrist pose, **no fitted rotation/scale**
  — the one candidate metric that would not risk trivially hiding translation error via
  over-permissive alignment.
- `cd_interaction_hand_sim3_per_frame` (cm²) — per-frame Sim(3) fit to 21 GT hand joints, then
  applied to the object.
- `mpjpe_ra_r` (mm) — wrist-translation-only alignment, canonical 21 joints.
- `mpjpe_right_scaled` (mm) — wrist-translation + one global least-squares scale, explicitly
  marked "auxiliary" in the code (i.e. the authors themselves flag it as the least trustworthy
  of the set).

`protocols.validate_inputs`/`valid_mask`/`check_frame_ids` **hard-require** a genuine binary
`is_valid` GT mask and matching `frame_ids` between prediction and GT arrays — there is no
fallback or degraded mode. This is not a judgment call: the code itself will refuse to execute
without real GT frame-validity data, which we do not have. **Zero metrics are currently
computable**, not because of ambiguity in method but because the required GT input tensors do
not exist.

### 3. Frame correspondence — solved, high confidence, independently cross-checked
Every `aligned/camera_data.npz` contains an authoritative `prediction_frame_ids` int64 array,
shape `(40,)`, documented in that condition's `export_metadata.json` as
`"frame_ids_source": "saved-prediction-frame-ids"` — not inferred from loader logs. Pulled and
compared for two independently-run conditions (A and E):
```
prediction_frame_ids == [0, 5, 10, 15, 20, ..., 190, 195]   (identical for A and E)
```
Mapping: **output index `i` → original video frame `5*i`**. Output index 2 → original frame 10,
which equals `WINDOW_START`/the scene's `contact_frame` — consistent with the known approved
window. Because this is identical across two separately-run jobs, it is a deterministic property
of the sampling code, not something that could have drifted per-run. Camera coordinate
convention (OpenCV, metres) is also consistent across all saved `camera_data.npz` files.

### Gate decision: BLOCKED — evaluation not run
Frame correspondence and coordinate conventions are fully verified and evaluation-ready. The
sole blocker is the GT asset itself: official HO3D annotations (`meta/*.pkl` with
`objTrans`/`objRot`/`objCorners3D`/`handPose`/`handTrans`/`handBeta`/`handJoints3D`) and object
CAD models were never downloaded to this cluster — AGILE's own installation docs confirm this
was an intentional omission for the optimization-only workflow we've been running. What exists
locally under `object_pose/` and `wilor/` are FoundationPose's and WiLoR's own *predictions*,
used only as optimizer initialization; treating either as GT would be circular (comparing AGILE's
optimized output against the same model's own unoptimized guess) and was explicitly ruled out by
the authorization's "never treat predictions as ground truth" instruction.

**What acquiring GT would require**: the official HO-3D_v3 release. Public mirrors list the full
archive at roughly 32–67 GB (varies by mirror/version), academic-research-only license, hosted at
the TU Graz project page (per the GitHub repo `shreyashampali/ho3d`) — this is far over the
~1GB threshold the user set for unprompted downloads, and whether a smaller annotations-only
subset (just `meta/*.pkl` for the MC1 sequence, no RGB/depth/already-have-those) is separately
obtainable has not been verified. **No download has been attempted or will be attempted without
explicit user confirmation of source and size.**

### Recommended next step (not run, not authorized — offered per the authorization's final item)
Three options, none executed:
1. Obtain official HO3D GT (likely the only way to get a real geometric-accuracy number) — needs
   the user to confirm an acceptable size/subset and source before any download is attempted.
2. Proceed with the loss-landscape investigation (examining why `feature_loss`/`alpha_loss`
   plateau at ~iteration 300, per Condition E's recommended direction) — this does not require
   GT and can proceed on existing data.
3. Identify a different dataset/benchmark already available locally (if any) with usable GT,
   as a substitute for MC1-specific evaluation.
No experiment or evaluation has been run beyond what was already authorized (A–E, this audit).

## LOSS-LANDSCAPE AND OPTIMIZATION-DYNAMICS INVESTIGATION (2026-11-05)

Read-only investigation into why the object optimizer's loss plateaus at ~iteration 100–300 and
why Conditions D (constant LR) and E (linearly-decaying LR) converge to essentially the same
reconstruction. No reconstruction job was run; no files were modified beyond throwaway scratch
scripts off-repo. Built on D's and E's existing diagnostics.jsonl (71,280 and 24,480 records
respectively) plus fresh source inspection.

### A. The `ho_loss` discrepancy — resolved, not a logging bug
Condition E's raw job stderr shows a third printed term, `ho_loss`, alongside `alpha_loss` and
`feature_loss`, that earlier spot-checks of `diagnostics.jsonl` had not surfaced. Traced to
source: `ho_relpose_loss=True`, `ho_relpose_loss_lambda=400.0` (`config.py:267,269`;
`configs/ho3d.json:117-118`) is a real, heavily-weighted, actively-backpropagated loss term.
`object_optimizer.py:392-399` sums `loss = Σ loss_dict.values()` and backpropagates it;
`log_record` at `object_optimizer.py:492-497` captures `loss_terms={k: float(v) for k,v in
loss_dict.items() if v is not None}` — the exact same dict, so `ho_loss` IS present in
diagnostics wherever it is active. It is only absent for `frame_id=10` (the bidirectional
seed/contact frame), because `ho_relpose_loss` requires a previous frame to compare against
(`object_optimizer.py:318-321`: `if cfg.ho_relpose_loss and not is_first_frame and ...`).
Verified empirically: all 39/40 non-seed frames carry `ho_loss` in both D's and E's logs; frame
10 alone lacks it. **No logging gap — prior plateau analyses used all 40 frames in aggregate and
are unaffected; this only explains why a single-frame spot-check could miss the term.**

### B. Loss decomposition — `feature_loss` dominates and is the term that plateaus
Weights: `object_mask_lambda=5.0` → `alpha_loss`; `object_feature_loss_lambda=10.0` →
`feature_loss`; `ho_relpose_loss_lambda=400.0` → `ho_loss`. Median weighted values across all 40
frames (D's iterations 0–399, matching E's full budget, for a fair comparison — D's and E's
numbers agree to within ~1% at every checkpoint):

| iter | alpha_loss | feature_loss | ho_loss | total | feature_loss share of total |
|---|---|---|---|---|---|
| 0   | 1.01 | 6.01 | 2.41 | 9.43 | 64% |
| 50  | 0.59 | 5.76 | 0.65 | 7.01 | 82% |
| 100 | 0.48 | 5.75 | 0.29 | 6.53 | 88% |
| 200 | 0.40 | 5.66 | 0.21 | 6.27 | 90% |
| 300 | 0.40 | 5.62 | 0.21 | 6.22 | 90% |
| 399 | 0.40 | 5.55 | 0.21 | 6.15 | 90% |

- **`ho_loss` resolves fast**: 2.41 → 0.21 by iteration ~150–200 (a 91% drop). Despite its large
  nominal weight (400.0), its raw unweighted magnitude is tiny (~0.015→0.0005 m mean weighted
  hand-vertex distance) — it is a quickly-satisfied anchor term, not the cause of the late
  plateau.
- **`feature_loss` dominates (82–90% of the weighted total) and is nearly flat after iteration
  ~100** (5.75 → 5.55, a ~3.5% drop over the remaining 300 iterations). **This is the term
  responsible for the overall plateau** — both the largest contributor and the one that stops
  moving.
- `alpha_loss` keeps improving slowly throughout (1.01 → 0.40, a ~60% reduction) but is a minor
  contributor (6–10% of the total), so continued improvement there cannot move the total loss
  much on its own.
- Per-component gradient conflict is **not measurable** with the current diagnostics (only the
  summed `gradient_inf` is logged, not per-term gradients) — this is an explicit logging gap,
  not evidence either way about term conflict.
- `feature_loss` is `mean(|1 − cosine_similarity|)` over DINOv3-feature correspondence points
  (`utils/feature.py:125-186`), masked to non-occluded, in-bounds points; the reference
  similarity map is refreshed only every `compute_feature_every_iter=20` iterations
  (`config.py:218`), though the loss is computed/backpropagated against the cached map every
  single iteration in between refreshes.

### C. Optimization dynamics — genuine but incomplete flattening; convergence thresholds are miscalibrated for the object optimizer
Median `gradient_inf` (unscaled by LR — a direct read on the loss landscape, independent of
schedule) across the same 40 frames, D vs E:

| iter | 0 | 50 | 100 | 150 | 200 | 300 | 399 |
|---|---|---|---|---|---|---|---|
| D | 149.5 | 18.7 | 7.0 | 2.1 | 4.0 | 3.2 | 1.23 |
| E | 151.3 | 19.5 | 5.9 | 2.8 | 3.1 | 3.0 | 0.82 |

Both show a genuine ~100–150x drop in raw gradient magnitude by iteration 300–400 — real
flattening of the landscape, not purely an artifact of E's LR schedule (`gradient_inf` is not
LR-scaled). **But the gradient does not vanish**: median sits at 0.8–1.2 at the final iteration,
with individual frames spiking much higher (e.g. D's frame 20 at iter399: `gradient_inf=18.7`).
This is a **shallow, slowly-decaying plateau region, not a hard stationary point**.

**Threshold miscalibration — directly demonstrated, not inferred**: `converge_grad_tol=1e-3`
(`config.py:323`) vs. observed object `gradient_inf` of roughly 0.8–150 throughout — the
threshold is ~1000x smaller than the actual gradient scale, so the `'converged'` branch
(`convergence.py:29`: requires `max(grads) < grad_tol`) is essentially unreachable for the
object optimizer as currently configured. `'stalled_at_budget'` only checks `step_norm <
step_tol` (`convergence.py:27-31`) — never gradient magnitude or loss stability — which is why
E's imposed LR decay alone can flip the stalled/exhausted label without the landscape itself
changing (consistent with the Condition E report's caveat). Confirmed empirically: **zero
`'converged'` terminations occurred for the object optimizer across either D or E, for any
frame.**

**Contrast with the hand optimizer** (sampled frame 25, Condition E): `joints_2d_loss` is flat
from iteration 0 (4.1290 → 4.1294 → 4.1290 — essentially no real movement even at the start of
the run), yet `gradient_inf` fully decays toward zero (0.09 → 70 → 50 → 4.36 → 0.52 → 0.0126 by
iteration 199) — a clean true convergence under the *same* fixed `1e-3` threshold. This shows a
concrete scale-mismatch between the two optimizers' convergence criteria: the fixed threshold
is reasonably matched to the hand optimizer's gradient scale but not the object optimizer's —
this is a calibration issue specific to the object optimizer's loss/gradient magnitude, not
evidence about hand-pose freedom or quality.

### D. D vs E matched-frame comparison
At the same iteration count (399), across all 40 common frame_ids: median `total_loss`
percent-difference between D and E is **0.755%** (mean 1.18%, max 4.47% at frame 85). This
closely matches the final-trajectory temporal-consistency difference already measured for
Condition E (object rotation +0.88%, translation +0.75%) — strong internal cross-validation that
**D and E reach essentially the same solution when compared at matched iteration count**,
independent of which LR schedule produced it. Per-frame `gradient_inf` at iteration 399 varies
more between D and E (e.g. frame 20: D=18.7 vs E=2.1) — consistent with D's flat-LR regime
producing noisier late-stage gradient/step behavior that E's imposed decay smooths out, without
that noise translating into a materially different final loss. No individual frame's difference
(max 4.47%) is large enough to warrant a dedicated follow-up beyond flagging frame 85 as the
largest outlier observed.

### E. Bottleneck classification — demonstrated vs. plausible-but-untested, explicitly separated
| Hypothesis | Status | Evidence |
|---|---|---|
| Loss saturation / insensitive dominant component | **DEMONSTRATED** | `feature_loss` is 82–90% of the weighted total and drops only ~3.5% from iteration 50→399 — simultaneously the largest contributor and the one that stops moving, directly explaining the plateau's timing. |
| Poor conditioning / flat directions | **DEMONSTRATED** | `gradient_inf` drops ~100–150x by iteration 300 but plateaus at a nonzero median (~1–3, not →0) in both D and E, independent of schedule — a shallow-basin signature, not a scheduling artifact. |
| Conflicting objectives | **UNTESTABLE with current diagnostics** | Only the summed `gradient_inf` is logged, not per-term gradients. An explicit logging gap, not evidence either way. |
| Fixed convergence thresholds mis-scaled for the object optimizer | **DEMONSTRATED** | `converge_grad_tol=1e-3` is ~1000x below the observed object `gradient_inf` scale; the hand optimizer, whose gradient scale is much smaller, reaches near-zero gradient at its own plateau under the identical fixed threshold — a direct scale-calibration mismatch specific to the object optimizer. |
| Initialization limitations (WiLoR/FoundationPose) | **PLAUSIBLE, UNTESTABLE** | No GT exists locally (per the completed GT audit) to check absolute initialization accuracy; `ho_loss`'s fast resolution is consistent with, but does not prove, a reasonable relative initialization. |
| Insufficient object parameter freedom | **NOT SUPPORTED** | `optimize_object_scale=True` (isotropic); rotation, translation, and scale are all free parameters — this is not the limiting factor. |
| Rendering/correspondence/visibility limitations | **PLAUSIBLE, source-supported, not isolated** | `feature_loss` masks occluded/out-of-bounds points and only refreshes its reference similarity map every 20 iterations (`compute_feature_every_iter=20`) while backpropagating every iteration against the cached map — a plausible but currently unmeasured contributor to the loss's insensitivity. |

### F. Two proposed next experiments (NOT run — require explicit authorization before any job is submitted)

**1. Per-component gradient instrumentation + short diagnostic run (recommended, highest
priority).** Hypothesis: the plateau reflects a genuinely flat/insensitive `feature_loss`
landscape, not cancellation between conflicting terms. Variable changed: add diagnostic logging
of each loss term's own gradient norm w.r.t. object pose parameters (via a separate
`torch.autograd.grad` call per term before the combined backward pass), in addition to the
existing summed `gradient_inf`. Expected observations: if `feature_loss`'s own per-term gradient
also plateaus small while its loss value stays elevated (~5.5), that confirms genuine flatness.
If instead some per-term gradient remains large while the summed `gradient_inf` stays small,
that would reveal term cancellation — a different, currently invisible mechanism. Compute/data:
reuse the existing smoke-test scale (3–5 frames, same MC1 window) — purely additive
instrumentation, isolated output directory, no new lettered reconstruction condition required.
Directly closes the "conflicting objectives" gap identified in Parts B/E. Risk/limitation: adds
one extra backward pass per loss term per iteration at smoke-test scale only — not something to
run at full A–E scale without separately re-checking the compute budget.

**2. Vary `compute_feature_every_iter` (20 → 1) on a short diagnostic run.** Hypothesis: part of
the plateau is staleness in the cached feature-similarity reference map, not pure landscape
flatness. Variable changed: that one config field only. Expected observations: if loss keeps
improving past iteration ~300 under constant (every-iteration) refresh, staleness is a real
contributor to the plateau; if the plateau timing is unchanged, this rules staleness out and
reinforces the flat-landscape explanation as sufficient on its own. Compute/data: same MC1
window, short diagnostic run only. Risk/limitation: substantially higher per-iteration compute
(DINOv3 feature resampling becomes the per-iteration bottleneck instead of an amortized cost) —
should be run on very few frames/iterations only, and is itself a config change requiring
explicit sign-off before any job runs, consistent with "do not launch a new reconstruction
experiment" scoping.

### Explicit non-claims
No reconstruction experiment was run in this investigation. No claim is made that either
proposed experiment, if run, would necessarily change the final reconstruction — both are
diagnostic/explanatory, aimed at distinguishing which of the DEMONSTRATED/PLAUSIBLE hypotheses
above is the dominant mechanism, not at improving reconstruction quality directly. No ground
truth was used or is available (per the prior GT audit) for any claim in this section.
