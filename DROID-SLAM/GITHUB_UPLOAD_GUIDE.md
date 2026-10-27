# GitHub Upload Guide — Final Direct R/V/P Release

## What belongs in the repository

Commit the source required to build and run the final model:

```text
README.md
.gitignore
environment*.yaml, requirements*.txt, setup.py
demo.py, train.py
src/
droid_slam/depth_video.py
droid_slam/droid.py
droid_slam/droid_backend.py
droid_slam/droid_frontend.py
droid_slam/droid_net.py
droid_slam/factor_graph.py
droid_slam/logger.py
droid_slam/motion_filter.py
droid_slam/modules/extractor.py
droid_slam/geom/ba.py
droid_slam/geom/losses.py
droid_slam/geom/imu_factor.py
droid_slam/data_readers/base.py
droid_slam/data_readers/factory.py
droid_slam/data_readers/euroc.py
droid_slam/data_readers/euroc_protocol.py
droid_slam/data_readers/imu_preintegration.py
droid_slam/data_readers/sampling.py
droid_slam/data_readers/tartan_v2.py
droid_slam/imu_contract.py
droid_slam/imu_prior.py
droid_slam/imu_residual.py
droid_slam/imu_runtime.py
droid_slam/vi_initialization.py
tools/evaluate_euroc_recon.py
tools/prepare_euroc_original_protocol_assets.py
tools/export_euroc_original_protocol_motion_filter_manifests.sh
tools/imu_preintegrate.py
tools/run_final_direct_rvp_train.sh
tools/run_final_v1_03_evaluation.sh
configs/final_direct_rvp.env.example
models/FINAL_CHECKPOINT.md
results/V1_03_FINAL_RESULT.md
FINAL_PPT_TECHNICAL_BRIEF.md
presentation_assets/final_direct_rvp_architecture.svg
presentation_assets/final_direct_rvp_training.svg
LICENSE
```

Keep the already tracked upstream DROID files and third-party submodules. Do
not remove their licenses or attribution.

## What must not be committed

```text
logs/
outputs/
datasets/
checkpoints/
*.pth
*.ply
large reconstruction .pth files
intermediate experiment scripts, queues, ablation reports, and old diagrams
machine-specific .env files
```

The revised `.gitignore` enforces this for all untracked research artifacts.
It is safe to retain these files locally; they are merely excluded from the
public release.

## Model weights

The selected learned checkpoint is only 16.7 MB, but a GitHub Release asset is
still preferable to putting a binary model into ordinary source history. Its
filename and SHA-256 are recorded in `models/FINAL_CHECKPOINT.md`.

The original `droid.pth` is an upstream dependency and is not redistributed.

## Suggested staging command

Review first, then stage only the curated release paths:

```bash
git add README.md .gitignore demo.py train.py setup.py environment.yaml environment_novis.yaml \
  requirements.txt requirements_frozen.txt LICENSE \
  src/droid.cpp src/droid_kernels.cu \
  droid_slam/depth_video.py droid_slam/droid.py droid_slam/droid_backend.py \
  droid_slam/droid_frontend.py droid_slam/droid_net.py droid_slam/factor_graph.py \
  droid_slam/logger.py droid_slam/motion_filter.py droid_slam/modules/extractor.py \
  droid_slam/geom/ba.py droid_slam/geom/losses.py droid_slam/geom/imu_factor.py \
  droid_slam/data_readers/base.py droid_slam/data_readers/factory.py \
  droid_slam/data_readers/euroc.py droid_slam/data_readers/euroc_protocol.py \
  droid_slam/data_readers/imu_preintegration.py droid_slam/data_readers/sampling.py \
  droid_slam/data_readers/tartan_v2.py droid_slam/imu_contract.py \
  droid_slam/imu_prior.py droid_slam/imu_residual.py droid_slam/imu_runtime.py \
  droid_slam/vi_initialization.py \
  tools/evaluate_euroc_recon.py \
  tools/prepare_euroc_original_protocol_assets.py \
  tools/export_euroc_original_protocol_motion_filter_manifests.sh \
  tools/imu_preintegrate.py \
  tools/run_final_direct_rvp_train.sh \
  tools/run_final_v1_03_evaluation.sh \
  configs models results presentation_assets/final_direct_rvp_architecture.svg \
  presentation_assets/final_direct_rvp_training.svg \
  FINAL_PPT_TECHNICAL_BRIEF.md GITHUB_UPLOAD_GUIDE.md

git status --short
git diff --cached --check
```

Before pushing, attach the learned checkpoint to a GitHub Release and add its
download URL to `models/FINAL_CHECKPOINT.md` if one is available.
