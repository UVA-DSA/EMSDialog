#!/usr/bin/env bash
set -euo pipefail

# ======= CONFIG =======
PYTHON=${PYTHON:-python}
TRAIN_SCRIPT="../static_train.py"

BASE="/scratch/zar8jw/Conversation_Generation"
DATA_BASE="$BASE/data"
OUT_BASE="$BASE/train_output/static_train/Qwen3-4B"

MODEL_NAME="Qwen/Qwen3-4B"
DATASETS=("ours")
LRS=("1e-4" "2e-4" "3e-4" "4e-4" "5e-4" "6e-4" "7e-4" "8e-4" "9e-4" "1e-3")
SEEDS=("0" "1" "42")

# Set to 1 to disable Weights & Biases across all runs.
WANDB_DISABLE=${WANDB_DISABLE:-0}

# Optional GPU pin
# export CUDA_VISIBLE_DEVICES=0

# ======= LOGGING =======
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUT_BASE/logs/ours_grid_seeds_$STAMP"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
run_cmd() {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  log "RUN [$name] -> $TRAIN_SCRIPT $*"
  # shellcheck disable=SC2068
  $PYTHON "$TRAIN_SCRIPT" $@ 2>&1 | tee -a "$logfile"
}

maybe_wandb() {
  if [[ "$WANDB_DISABLE" == "1" ]]; then
    echo "--nowandb"
  else
    echo "--use_wandb"
  fi
}

# ======= SWEEP =======
for ds in "${DATASETS[@]}"; do
  DATA_GLOB="$DATA_BASE/$ds"
  if [[ ! -d "$DATA_GLOB" ]]; then
    log "WARN: dataset not found: $DATA_GLOB (skipping)"
    continue
  fi

  for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      LR_TAG="${lr}"
      RUN_TAG="${ds}-seed${seed}-${LR_TAG}"
      OUT_DIR="$OUT_BASE/${ds}_seed${seed}-${LR_TAG}"
      mkdir -p "$OUT_DIR"

      # ---- Sentinel: skip if results already exist ----
      METRICS_FILE="$OUT_DIR/_turnwise_metrics.json"
      if [[ -f "$METRICS_FILE" && "${FORCE_REDO:-0}" != "1" ]]; then
        log "SKIP [$RUN_TAG] found results: $METRICS_FILE"
        continue
      fi

      # ---- Train ----
      run_cmd "train_${RUN_TAG}" \
        --data_glob="$DATA_GLOB" \
        --model_name="$MODEL_NAME" \
        --lr "$lr" \
        --seed "$seed" \
        --output_dir="$OUT_DIR" \
        $(maybe_wandb) \
        --wandb_run="$RUN_TAG"

      # ---- Test ----
      run_cmd "test_${RUN_TAG}" \
        --test_only \
        --model_name="$MODEL_NAME" \
        --test_checkpoint="$OUT_DIR" \
        --save_preds="$OUT_DIR"

      # ---- Cleanup: delete checkpoints to save disk ----
      log "CLEANUP [$RUN_TAG] deleting checkpoints under $OUT_DIR"
      find "$OUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" -print -exec rm -rf {} \; || true
      # If your script dumps big model files directly in OUT_DIR, you can also uncomment:
      # rm -f "$OUT_DIR"/pytorch_model*.bin "$OUT_DIR"/adapter_model* 2>/dev/null || true
    done
  done
done

log "All runs complete. Logs at: $LOG_DIR"
