#!/usr/bin/env bash
# Run the remaining unified-7 experiments after scheme 1 (which is complete):
# stage 1 synth-dominant -> eval -> stage 2 real-dominant -> eval -> summary.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python}"
LOG="$REPO/outputs/poc_unified7_stages_chain.log"
mkdir -p outputs/poc_unified7_stage1_synth80_1ep/logs \
         outputs/poc_unified7_stage2_real75_1ep/logs
exec >>"$LOG" 2>&1

ts() { date -u +%FT%TZ; }
best_ckpt() {
    "$PY" - "$1" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["checkpoints"][0]["path"])
EOF
}

echo "[stages] start $(ts)"

# Stage 1: synthetic-dominant, 1 epoch, from base.
echo "[stages] stage1 start $(ts)"
"$PY" tools/qwen3_asr_pipeline.py --config configs/poc_unified7_stage1_synth80_1ep.yaml --stage train \
    > outputs/poc_unified7_stage1_synth80_1ep/logs/driver.log 2>&1 &
PID=$!
wait "$PID" || { echo "[stages] ERROR: stage1 failed"; exit 1; }
grep -q "train_runtime" outputs/poc_unified7_stage1_synth80_1ep/logs/driver.log || {
    echo "[stages] ERROR: stage1 ended without train_runtime"; exit 1; }
BEST_S1="$(best_ckpt outputs/poc_unified7_stage1_synth80_1ep/best_cer_checkpoints.json)"
echo "[stages] stage1 best=$BEST_S1 $(ts)"
sleep 15
"$PY" tools/eval_unified7_scheme.py --model "$BEST_S1" \
    --out-dir outputs/poc_unified7_stage1_synth80_1ep/eval_best

# Stage 2: real-dominant, 1 epoch, initialised from stage-1 best.
"$PY" - "$BEST_S1" <<'EOF'
import pathlib, re, sys
path = pathlib.Path("configs/poc_unified7_stage2_real75_1ep.yaml")
text = path.read_text(encoding="utf-8")
text, n = re.subn(r"(?m)^  model_path: .*$", f"  model_path: {sys.argv[1]}", text, count=1)
if n != 1:
    raise SystemExit("stage2 model_path line not found")
path.write_text(text, encoding="utf-8")
print(f"[stages] stage2 init -> {sys.argv[1]}")
EOF
echo "[stages] stage2 start $(ts)"
"$PY" tools/qwen3_asr_pipeline.py --config configs/poc_unified7_stage2_real75_1ep.yaml --stage train \
    > outputs/poc_unified7_stage2_real75_1ep/logs/driver.log 2>&1 &
PID=$!
wait "$PID" || { echo "[stages] ERROR: stage2 failed"; exit 1; }
grep -q "train_runtime" outputs/poc_unified7_stage2_real75_1ep/logs/driver.log || {
    echo "[stages] ERROR: stage2 ended without train_runtime"; exit 1; }
BEST_S2="$(best_ckpt outputs/poc_unified7_stage2_real75_1ep/best_cer_checkpoints.json)"
echo "[stages] stage2 best=$BEST_S2 $(ts)"
sleep 15
"$PY" tools/eval_unified7_scheme.py --model "$BEST_S2" \
    --out-dir outputs/poc_unified7_stage2_real75_1ep/eval_best

# Collate scheme 1 + both stages.
"$PY" - <<'EOF'
import json
from pathlib import Path
runs = {
    "direct_mix_2ep": "outputs/poc_unified7_directmix_2ep/eval_best/summary.json",
    "stage1_synth80_1ep": "outputs/poc_unified7_stage1_synth80_1ep/eval_best/summary.json",
    "stage2_real75_1ep": "outputs/poc_unified7_stage2_real75_1ep/eval_best/summary.json",
}
out = {}
for name, path in runs.items():
    p = Path(path)
    out[name] = json.loads(p.read_text()) if p.is_file() else {"missing": path}
Path("outputs/poc_unified7_comparison.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
EOF
echo "[stages] done $(ts)"
