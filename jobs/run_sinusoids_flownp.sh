# Train FlowNP baselines on sinusoids and sinusoids-dist-shift
# =============================================================
# This script does NOT touch the original INP/NP training code (models/train.py).
# It uses baselines/train_flownp_trending_sinusoids_dist_shift.py.

# (Optional) activate your env first:
#   conda env create -f environment.yaml
#   conda activate inps

# ---------------------------
# Distribution shift dataset
# ---------------------------
python baselines/train_flownp_trending_sinusoids_dist_shift.py \
  --project-name INPs_sinusoids \
  --dataset set-trending-sinusoids-dist-shift \
  --knowledge-type b \
  --use-knowledge True \
  --batch-size 64 \
  --num-epochs 200 \
  --lr 1e-4 \
  --noise 0.2 \
  --min-num-context 0 \
  --max-num-context 10 \
  --num-targets 100 \
  --test-num-z-samples 32 \
  --run-name-prefix flownp_dist_shift \
  --seed 0

# ---------------------------
# In-distribution dataset
# ---------------------------
# If you also want FlowNP on the in-distribution sinusoids dataset, you can run:
# python baselines/train_flownp_trending_sinusoids_dist_shift.py \
#   --project-name INPs_sinusoids \
#   --dataset set-trending-sinusoids \
#   --knowledge-type full \
#   --use-knowledge True \
#   --batch-size 64 \
#   --num-epochs 200 \
#   --lr 1e-4 \
#   --noise 0.2 \
#   --min-num-context 0 \
#   --max-num-context 10 \
#   --num-targets 100 \
#   --test-num-z-samples 32 \
#   --run-name-prefix flownp \
#   --seed 0
