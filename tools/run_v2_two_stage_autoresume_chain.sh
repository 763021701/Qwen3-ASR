#!/usr/bin/env bash
# Auto-resuming balanced-data two-stage chain.
# Each stage retries up to 8 times; a crash resumes from checkpoint-latest
# (hardlinked every save by KeepBestCheckpointsCallback) or the newest numeric
# checkpoint. This survives the environment-suspend DataLoader failures.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python}"
LOG="$REPO/outputs/poc_v2_two_stage_autoresume_chain.log"
CFG1="$REPO/configs/poc_v2_stage1_synth80_1ep.yaml"
CFG1_AUTO="$REPO/configs/poc_v2_stage1_synth80_1ep.autoresume.yaml"
CFG2="$REPO/configs/poc_v2_stage2_real75_1ep.yaml"
CFG2_AUTO="$REPO/configs/poc_v2_stage2_real75_1ep.autoresume.yaml"
S1_DIR="$REPO/outputs/poc_v2_stage1_synth80_1ep"
S2_DIR="$REPO/outputs/poc_v2_stage2_real75_1ep"
mkdir -p "$S1_DIR/logs" "$S2_DIR/logs"
exec >>"$LOG" 2>&1

ts() { date -u +%FT%TZ; }
best_ckpt() {
    "$PY" - "$1" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["checkpoints"][0]["path"])
EOF
}
latest_ckpt() {
    "$PY" - "$1" <<'EOF'
import json, os, sys
directory = sys.argv[1]
best = (-1, "")
latest = os.path.join(directory, "checkpoint-latest")
if os.path.isdir(latest):
    try:
        step = int(json.load(open(os.path.join(latest, "trainer_state.json")))["global_step"])
        best = (step, latest)
    except (OSError, ValueError, KeyError, TypeError):
        pass
if os.path.isdir(directory):
    for name in os.listdir(directory):
        suffix = name[len("checkpoint-"):] if name.startswith("checkpoint-") else ""
        if suffix.isdigit() and int(suffix) > best[0]:
            best = (int(suffix), os.path.join(directory, name))
print(best[1] if best[1] else "")
EOF
}

echo "[auto] start $(ts)"

# Stage 1: synth-dominant.
stage1_ok=0
for attempt in $(seq 1 8); do
    CKPT="$(latest_ckpt "$S1_DIR")"
    if [ -n "$CKPT" ]; then
        "$PY" tools/make_resume_config.py --base "$CFG1" --out "$CFG1_AUTO" --checkpoint "$CKPT"
    else
        cp "$CFG1" "$CFG1_AUTO"
    fi
    echo "[auto] stage1 attempt=$attempt ckpt=${CKPT:-fresh} $(ts)"
    "$PY" tools/qwen3_asr_pipeline.py --config "$CFG1_AUTO" --stage train \
        > "$S1_DIR/logs/driver.log" 2>&1 || true
    if grep -q "train_runtime" "$S1_DIR/logs/driver.log"; then
        stage1_ok=1
        break
    fi
    cp "$S1_DIR/logs/driver.log" "$S1_DIR/logs/driver.attempt${attempt}.log"
    echo "[auto] stage1 attempt=$attempt failed; retrying $(ts)"
done
[ "$stage1_ok" = "1" ] || { echo "[auto] ERROR: stage1 exhausted retries"; exit 1; }

BEST_S1="$(best_ckpt "$S1_DIR/best_cer_checkpoints.json")"
echo "[auto] stage1 best=$BEST_S1 $(ts)"
sleep 10
"$PY" tools/eval_unified7_scheme.py --model "$BEST_S1" --out-dir "$S1_DIR/eval_best" \
    > "$S1_DIR/logs/eval_best.log" 2>&1 || { echo "[auto] ERROR: stage1 eval failed"; exit 1; }
rm -rf "$S1_DIR/checkpoint-latest"

# Stage 2: real-dominant, initialised from stage-1 best.
stage2_ok=0
for attempt in $(seq 1 8); do
    CKPT="$(latest_ckpt "$S2_DIR")"
    if [ -n "$CKPT" ]; then
        "$PY" tools/make_resume_config.py --base "$CFG2" --out "$CFG2_AUTO" --checkpoint "$CKPT"
    else
        "$PY" tools/make_resume_config.py --base "$CFG2" --out "$CFG2_AUTO" --init-from "$BEST_S1"
    fi
    echo "[auto] stage2 attempt=$attempt ckpt=${CKPT:-init:$BEST_S1} $(ts)"
    "$PY" tools/qwen3_asr_pipeline.py --config "$CFG2_AUTO" --stage train \
        > "$S2_DIR/logs/driver.log" 2>&1 || true
    if grep -q "train_runtime" "$S2_DIR/logs/driver.log"; then
        stage2_ok=1
        break
    fi
    cp "$S2_DIR/logs/driver.log" "$S2_DIR/logs/driver.attempt${attempt}.log"
    echo "[auto] stage2 attempt=$attempt failed; retrying $(ts)"
done
[ "$stage2_ok" = "1" ] || { echo "[auto] ERROR: stage2 exhausted retries"; exit 1; }

BEST_S2="$(best_ckpt "$S2_DIR/best_cer_checkpoints.json")"
echo "[auto] stage2 best=$BEST_S2 $(ts)"
sleep 10
"$PY" tools/eval_unified7_scheme.py --model "$BEST_S2" --out-dir "$S2_DIR/eval_best" \
    > "$S2_DIR/logs/eval_best.log" 2>&1 || { echo "[auto] ERROR: stage2 eval failed"; exit 1; }
rm -rf "$S2_DIR/checkpoint-latest"

"$PY" - <<'EOF'
import json
from pathlib import Path
runs = {
    "old_stage2_real75_1ep": "outputs/poc_unified7_stage2_real75_1ep/eval_best/summary.json",
    "v2_stage1_synth80_1ep": "outputs/poc_v2_stage1_synth80_1ep/eval_best/summary.json",
    "v2_stage2_real75_1ep": "outputs/poc_v2_stage2_real75_1ep/eval_best/summary.json",
}
out = {}
for name, path in runs.items():
    p = Path(path)
    out[name] = json.loads(p.read_text()) if p.is_file() else {"missing": path}
Path("outputs/poc_v2_two_stage_comparison.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
EOF
echo "[auto] done $(ts)"
