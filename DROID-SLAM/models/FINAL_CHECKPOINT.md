# Final Checkpoint Distribution

The final learned checkpoint is intentionally not committed as a normal Git
object. Upload it as a **GitHub Release asset** (or another versioned model
hosting location), then place the downloaded file anywhere locally and set
`LEARNED_WEIGHTS` before running `tools/run_final_v1_03_evaluation.sh`.

| field | value |
|---|---|
| filename | `learned_attenuation_rvp_quality_pretrain_joint2000_001000.pth` |
| size | 16,702,138 bytes |
| SHA-256 | `0b01773923e4585c05299a347567edf2c9e6347175360f65bf03de3947682574` |
| download | [GitHub Release asset](https://github.com/huikyoung03/SLAM/releases/download/final-direct-rvp-2026-10-27/learned_attenuation_rvp_quality_pretrain_joint2000_001000.pth) |
| model | frozen visual DROID + direct learned R/V/P IMU attenuation |

The upstream visual `droid.pth` is also required and should be obtained from
the original DROID-SLAM release rather than duplicated in this repository.
