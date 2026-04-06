#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
SCRIPT="../dynamic_train.py"   # change if your script name differs

# --- config ---
ROOT_DATA="../../data"                         # expects $ROOT_DATA/{ours,ours-gemini,0-shot,cot,notechat,...}
OUT_ROOT="../../train_output/dynamic_train/Qwen3-4B"
MODEL_NAME="Qwen/Qwen3-4B"

DATASETS=("ours")
LRS=("1e-4" "2e-4" "3e-4" "4e-4" "5e-4" "6e-4" "7e-4" "8e-4" "9e-4" "1e-3")
SEEDS=("0" "1" "42")

USE_WANDB=1   # set 0 to disable

# --- logging ---
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUT_ROOT}/logs/ours_grid_seeds_${STAMP}"
mkdir -p "$LOG_DIR"

log () { echo "[$(date '+%F %T')] $*"; }

run_cmd () {
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}.log"
  log "RUN [$name] -> $PYTHON $SCRIPT $*"
  # shellcheck disable=SC2068
  $PYTHON "$SCRIPT" $@ 2>&1 | tee -a "$logfile"
}

maybe_wandb () {
  if [[ "${USE_WANDB}" == "1" ]]; then
    echo "--use_wandb"
  fi
}

# For your setup (custom nn.Module wrapper), success marker is pytorch_model.bin.
train_done () {
  local out_dir="$1"
  [[ -f "${out_dir}/pytorch_model.bin" ]]
}

# Your test() writes _turnwise_metrics.json
test_done () {
  local out_dir="$1"
  [[ -f "${out_dir}/_turnwise_metrics.json" ]]
}

cleanup_ckpts () {
  local out_dir="$1"
  log "CLEANUP: removing checkpoint-* under $out_dir"
  find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print -exec rm -rf {} + || true
}

for ds in "${DATASETS[@]}"; do
  DATA_DIR="${ROOT_DATA}/${ds}"
  [[ -d "$DATA_DIR" ]] || { log "SKIP: dataset not found: $DATA_DIR"; continue; }

  for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      TAG="${ds}_seed${seed}-${lr}"
      OUT_DIR="${OUT_ROOT}/${TAG}"
      mkdir -p "$OUT_DIR"

      # ---- train (keep checkpoints during training so you can resume) ----
      if train_done "$OUT_DIR"; then
        log "SKIP train_${TAG}: found pytorch_model.bin"
      else
        run_cmd "train_${TAG}" \
          --data_glob="$DATA_DIR" \
          --model_name="$MODEL_NAME" \
          --epochs=10 \
          --dynamic \
          --output_dir="$OUT_DIR" \
          --seed="$seed" \
          --lr "$lr" \
          $(maybe_wandb) \
          --wandb_run="${TAG}"
      fi

      # ---- test (IMPORTANT: pass --model_name to avoid default 0.6B mismatch) ----
      if test_done "$OUT_DIR"; then
        log "SKIP test_${TAG}: found _turnwise_metrics.json"
      else
        run_cmd "test_${TAG}" \
          --test_only \
          --model_name="$MODEL_NAME" \
          --test_checkpoint="$OUT_DIR" \
          --save_preds="$OUT_DIR" \
          --seed="$seed"
      fi

      # ---- delete checkpoints AFTER testing succeeded ----
      if test_done "$OUT_DIR"; then
        cleanup_ckpts "$OUT_DIR"
      else
        log "NO CLEANUP for ${TAG}: test output missing (${OUT_DIR}/_turnwise_metrics.json)"
      fi
    done
  done
done

log "All grid runs completed. Logs: $LOG_DIR"
