#!/usr/bin/env bash
# AuT-only stage 1, then full-parameter stage 2 on all train data.
# Launch detached with setsid. Failures stop the chain.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python}"
LOG="$REPO/outputs/poc_autonly_noleak_chain.log"
S1_DIR="$REPO/outputs/poc_autonly_noleak_stage1_realvc_1ep"
S2_DIR="$REPO/outputs/poc_autonly_noleak_stage2_all_1ep"
mkdir -p "$S1_DIR/logs" "$S2_DIR/logs" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

ts() { date -u +%FT%TZ; }
best_ckpt() {
    "$PY" - "$1" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["checkpoints"][0]["path"])
EOF
}

echo "[chain] start $(ts)"
"$PY" tools/build_autonly_manifests.py --out-root data/poc_autonly_noleak || { echo "[chain] ERROR: manifest build failed"; exit 1; }

echo "[chain] stage1 train $(ts)"
"$PY" tools/qwen3_asr_pipeline.py --config configs/poc_autonly_noleak_stage1_realvc_1ep.yaml --stage train \
    > "$S1_DIR/logs/driver.log" 2>&1 || { echo "[chain] ERROR: stage1 failed"; exit 1; }
grep -q "train_runtime" "$S1_DIR/logs/driver.log" || {
    echo "[chain] ERROR: stage1 ended without train_runtime"; exit 1; }

BEST_S1="$(best_ckpt "$S1_DIR/best_cer_checkpoints.json")"
echo "[chain] stage1 best=$BEST_S1 $(ts)"
"$PY" tools/eval_unified7_scheme.py --model "$BEST_S1" --out-dir "$S1_DIR/eval_best" \
    > "$S1_DIR/logs/eval_best.log" 2>&1 || { echo "[chain] ERROR: stage1 eval failed"; exit 1; }

find "$S1_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' ! -path "$BEST_S1" -print -exec rm -rf {} +
echo "[chain] stage1 pruned to $BEST_S1"

"$PY" - "$BEST_S1" <<'EOF'
import pathlib, re, sys
path = pathlib.Path("configs/poc_autonly_noleak_stage2_all_1ep.yaml")
text = path.read_text(encoding="utf-8")
text, n = re.subn(r"(?m)^  model_path: .*$", f"  model_path: {sys.argv[1]}", text, count=1)
if n != 1:
    raise SystemExit("stage2 model_path line not found")
path.write_text(text, encoding="utf-8")
print(f"[chain] stage2 init -> {sys.argv[1]}")
EOF

echo "[chain] stage2 train $(ts)"
"$PY" tools/qwen3_asr_pipeline.py --config configs/poc_autonly_noleak_stage2_all_1ep.yaml --stage train \
    > "$S2_DIR/logs/driver.log" 2>&1 || { echo "[chain] ERROR: stage2 failed"; exit 1; }
grep -q "train_runtime" "$S2_DIR/logs/driver.log" || {
    echo "[chain] ERROR: stage2 ended without train_runtime"; exit 1; }

BEST_S2="$(best_ckpt "$S2_DIR/best_cer_checkpoints.json")"
echo "[chain] stage2 best=$BEST_S2 $(ts)"
"$PY" tools/eval_unified7_scheme.py --model "$BEST_S2" --out-dir "$S2_DIR/eval_best" \
    > "$S2_DIR/logs/eval_best.log" 2>&1 || { echo "[chain] ERROR: stage2 eval failed"; exit 1; }

"$PY" - <<'EOF'
import json
from pathlib import Path
runs = {
    "noleak_stage1": "outputs/poc_autonly_noleak_stage1_realvc_1ep/eval_best/summary.json",
    "noleak_stage2": "outputs/poc_autonly_noleak_stage2_all_1ep/eval_best/summary.json",
    "leaked_stage2": "outputs/poc_autonly_stage2_all_1ep/eval_best/summary.json",
    "unified7_stage2": "outputs/poc_unified7_stage2_real75_1ep/eval_best/summary.json",
}
out = {}
for name, path in runs.items():
    p = Path(path)
    out[name] = json.loads(p.read_text()) if p.is_file() else {"missing": path}
Path("outputs/poc_autonly_noleak_comparison.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
EOF
echo "[chain] done $(ts)"
