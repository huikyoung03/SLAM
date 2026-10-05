set -e

SESSION=/home/ubuntu/SLAM/uploads/session_1781694669396
IMGDIR=$SESSION/images

cd /home/ubuntu/DROID-SLAM

echo "SESSION=$SESSION"
echo "IMGDIR=$IMGDIR"

echo "=== Baseline ==="
python demo.py \
  --imagedir=$IMGDIR \
  --calib=$SESSION/calib.txt \
  --stride=3 \
  --imu_mode=off \
  --disable_vis \
  --reconstruction_path=$SESSION/reconstruction_baseline_s3.pth

echo "=== Make IMU-selected turn dense WebP ==="
python tools/make_imu_selected_session_webp.py \
  --session $SESSION \
  --image_dir $IMGDIR \
  --out_dir $SESSION/imu_selected_turn_dense_webp \
  --base_stride 3 \
  --turn_percentile 70 \
  --extreme_percentile 98

SEL=$SESSION/imu_selected_turn_dense_webp

echo "=== IMU-selected visual only ==="
python demo.py \
  --imagedir=$SEL/images \
  --calib=$SEL/calib.txt \
  --stride=1 \
  --imu_mode=off \
  --disable_vis \
  --reconstruction_path=$SEL/reconstruction_imu_selected_visualonly.pth

echo "=== IMU-selected residual w002 ==="
python demo.py \
  --imagedir=$SEL/images \
  --calib=$SEL/calib.txt \
  --stride=1 \
  --imu_prior=$SEL/imu_prior.csv \
  --imu_mode=off \
  --imu_residual \
  --imu_residual_weight=0.02 \
  --imu_residual_window=3 \
  --imu_residual_compose_order=prev_dq \
  --imu_residual_debug \
  --disable_vis \
  --reconstruction_path=$SEL/reconstruction_imu_selected_residual_w002_win3_prevdq.pth

cp $SEL/imu_residual_debug.csv \
   $SEL/imu_residual_debug_selected_residual_w002_win3_prevdq.csv

echo "=== IMU-selected residual w001 ==="
python demo.py \
  --imagedir=$SEL/images \
  --calib=$SEL/calib.txt \
  --stride=1 \
  --imu_prior=$SEL/imu_prior.csv \
  --imu_mode=off \
  --imu_residual \
  --imu_residual_weight=0.01 \
  --imu_residual_window=3 \
  --imu_residual_compose_order=prev_dq \
  --imu_residual_debug \
  --disable_vis \
  --reconstruction_path=$SEL/reconstruction_imu_selected_residual_w001_win3_prevdq.pth

cp $SEL/imu_residual_debug.csv \
   $SEL/imu_residual_debug_selected_residual_w001_win3_prevdq.csv

echo "=== Make drop extreme WebP ==="
python tools/make_imu_selected_session_webp.py \
  --session $SESSION \
  --image_dir $IMGDIR \
  --out_dir $SESSION/imu_selected_drop_extreme_webp \
  --base_stride 3 \
  --turn_percentile 70 \
  --extreme_percentile 98 \
  --drop_extreme

SEL2=$SESSION/imu_selected_drop_extreme_webp

echo "=== Drop extreme visual only ==="
python demo.py \
  --imagedir=$SEL2/images \
  --calib=$SEL2/calib.txt \
  --stride=1 \
  --imu_mode=off \
  --disable_vis \
  --reconstruction_path=$SEL2/reconstruction_imu_selected_drop_visualonly.pth

echo "=== Drop extreme residual w001 ==="
python demo.py \
  --imagedir=$SEL2/images \
  --calib=$SEL2/calib.txt \
  --stride=1 \
  --imu_prior=$SEL2/imu_prior.csv \
  --imu_mode=off \
  --imu_residual \
  --imu_residual_weight=0.01 \
  --imu_residual_window=3 \
  --imu_residual_compose_order=prev_dq \
  --imu_residual_debug \
  --disable_vis \
  --reconstruction_path=$SEL2/reconstruction_imu_selected_drop_residual_w001_win3_prevdq.pth

cp $SEL2/imu_residual_debug.csv \
   $SEL2/imu_residual_debug_selected_drop_residual_w001_win3_prevdq.csv

echo "=== Metrics: original baseline ==="
TURN_ORIG=$(SESSION=$SESSION python - <<'PY'
import os
import pandas as pd
from pathlib import Path
session = Path(os.environ["SESSION"])
df = pd.read_csv(session / "imu_prior.csv")
print(len(df)//2)
PY
)

python tools/compare_recon_metrics.py \
  --turn_frame $TURN_ORIG \
  $SESSION/reconstruction_baseline_s3.pth

echo "=== Metrics: selected turn dense ==="
TURN_SEL=$(SEL=$SEL python - <<'PY'
import os
import pandas as pd
from pathlib import Path
sel = Path(os.environ["SEL"])
df = pd.read_csv(sel / "imu_prior.csv")
print(len(df)//2)
PY
)

python tools/compare_recon_metrics.py \
  --turn_frame $TURN_SEL \
  $SEL/reconstruction_imu_selected_visualonly.pth \
  $SEL/reconstruction_imu_selected_residual_w002_win3_prevdq.pth \
  $SEL/reconstruction_imu_selected_residual_w001_win3_prevdq.pth

echo "=== Metrics: drop extreme ==="
TURN_SEL2=$(SEL2=$SEL2 python - <<'PY'
import os
import pandas as pd
from pathlib import Path
sel = Path(os.environ["SEL2"])
df = pd.read_csv(sel / "imu_prior.csv")
print(len(df)//2)
PY
)

python tools/compare_recon_metrics.py \
  --turn_frame $TURN_SEL2 \
  $SEL2/reconstruction_imu_selected_drop_visualonly.pth \
  $SEL2/reconstruction_imu_selected_drop_residual_w001_win3_prevdq.pth

echo "DONE"
