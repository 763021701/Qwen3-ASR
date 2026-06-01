#!/usr/bin/env bash
# Start balanced_none SFT in background (survives SSH disconnect).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/balanced_none_sft.yaml}"
LOG_DIR="${LOG_DIR:-outputs/balanced_none_speed_sft/logs}"
mkdir -p "$LOG_DIR"
NOHUP_LOG="${LOG_DIR}/nohup_train.out"
PID_FILE="${LOG_DIR}/train.pid"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Training already running (PID $old_pid). Log: $NOHUP_LOG"
    exit 0
  fi
fi

nohup conda run -n qwen3_asr_ft --no-capture-output \
  python tools/qwen3_asr_pipeline.py \
  --config "$CONFIG" \
  --stage train \
  >>"$NOHUP_LOG" 2>&1 &

echo $! >"$PID_FILE"
echo "Started background training PID=$(cat "$PID_FILE")"
echo "  nohup log: $NOHUP_LOG"
echo "  train log:  $LOG_DIR/train.log"
echo "Monitor: tail -f $LOG_DIR/train.log"
