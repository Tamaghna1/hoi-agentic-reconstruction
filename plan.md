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
