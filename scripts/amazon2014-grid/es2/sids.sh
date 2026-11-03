#!/usr/bin/env bash
set -euo pipefail

representations=${1:?Usage: scripts/amazon2014-grid/es2/sids.sh REPRESENTATIONS NEW_SID_DIR CATEGORY METHOD SID_SEED [cpu|cuda]}
output=${2:?NEW_SID_DIR is required}
category=${3:?CATEGORY is required}
method=${4:?METHOD is required}
seed=${5:?SID_SEED is required}
device=${6:-cuda}
export PYTHONPATH=src OMP_NUM_THREADS=1 POLARS_MAX_THREADS=4 NVIDIA_TF32_OVERRIDE=0

# Final ES2 recipes: full catalog, three semantic levels plus collision suffix.
case "$category" in
  beauty) batch=16384; epochs=20310; layer_epochs=1354 ;;
  toys) batch=16384; epochs=20611; layer_epochs=1375 ;;
  sports) batch=18357; epochs=13388; layer_epochs=893 ;;
  *) echo 'CATEGORY must be beauty, toys or sports' >&2; exit 2 ;;
esac
set -- --codebook-size=256 --codebook-size=256 --codebook-size=256 \
  --fit-backend=torch --fit-batch-size=65536
case "$method" in
  rq_vae)
    eval_batch=8192
    set -- "$@" --hidden-dim-factors=2.6666666666666665 --hidden-dim-factors=8 \
      --hidden-dim-factors=16 --hidden-dim-factors=32 \
      --no-codebook-normalize --codebook-kmeans-init --codebook-distance-mode=l2 \
      --beta=0.25 --learning-rate=0.001 --no-latent-l2 --mlp-bias \
      --joint-normalization --optimizer-name=adagrad --gradient-clip=0 \
      --epochs="$epochs" --training-batch-size="$batch"
    ;;
  rvq)
    eval_batch=1024
    set -- "$@" --learning-rate=0.001 --normalize-residuals --init-buffer-size=3072 \
      --loss-reduction=sum --loss-scale=0.125 --epochs-per-layer="$layer_epochs" \
      --training-batch-size="$batch"
    ;;
  rq_kmeans_full)
    eval_batch=8192
    set -- "$@" --balanced=0 --max-iter=100 --n-init=3 --tolerance=0.0001 \
      --max-gpu-memory-fraction=0.8
    ;;
  *) echo 'METHOD must be rq_vae, rvq or rq_kmeans_full' >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$output")"
mkdir "$output"
.venv/bin/python -m item_representations.semantic_ids fit-sid-mapper \
  --items="$representations/train/items.pq" --embeddings="$representations/train/embeddings.npy" \
  --output-dir="$output/train" --method="$method" --random-state="$seed" \
  --device="$device" --eval-batch-size="$eval_batch" "$@"
.venv/bin/python -m item_representations.semantic_ids apply-sid-mapper \
  --checkpoint="$output/train/checkpoint.pkl" --items="$representations/inference/items.pq" \
  --embeddings="$representations/inference/embeddings.npy" --output-dir="$output/inference" \
  --device="$device" --eval-batch-size="$eval_batch"
