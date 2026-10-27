# Final V1_03 Result

## Protocol

- Sequence: EuRoC `V1_03_difficult` (not used as a training sequence)
- Training sequences: `MH_03_medium`, `MH_04_difficult`
- Input/tracking: strict original-DROID EuRoC protocol, rectified camera, tracking stride 2, dense pose filling stride 1
- Runtime schedule: local visual DBA 3 → local full 15DoF DBA 2 → global visual DBA 7 → global full 15DoF BA 12
- Evaluation: whole-trajectory dense Sim(3)-aligned ATE RMSE on 2,149 timestamps

## Result

| Model | Sim(3) ATE RMSE ↓ | Dense poses |
|---|---:|---:|
| Original visual DROID (`droid.pth`) | 0.018560 m | 2,149 |
| Final direct R/V/P learned IMU attenuation | **0.018467 m** | 2,149 |

Absolute reduction: `0.000093 m` (0.093 mm). Relative reduction: `0.501%`.

The final learned checkpoint is
`learned_attenuation_rvp_quality_pretrain_joint2000_001000.pth`.
The observed difference is small and comes from one held-out sequence / one run, so it is reported as a numerical observation rather than a general statistical claim.
