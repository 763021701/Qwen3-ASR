#!/usr/bin/env bash
# Three small-data probes, sequential, with auto-resume on environment-suspend
# DataLoader failures. Each run: train 1 epoch -> best CER checkpoint -> eval on
# POC53/radiology95/holdout19.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/root/autodl-tmp/miniconda3/envs/qwen3_asr_ft/bin/python}"
LOG="$REPO/outputs/poc_small_experiments_chain.log"
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

run_experiment() {
    local name="$1" cfg="$2" out="$3"
    local auto="$cfg.autoresume.yaml"
    mkdir -p "$out/logs"
    local ok=0
    for attempt in 1 2 3 4; do
        local ckpt
        ckpt="$(latest_ckpt "$out")"
        if [ -n "$ckpt" ]; then
            "$PY" tools/make_resume_config.py --base "$cfg" --out "$auto" --checkpoint "$ckpt"
        else
            cp "$cfg" "$auto"
        fi
        echo "[small] $name attempt=$attempt ckpt=${ckpt:-fresh} $(ts)"
        "$PY" tools/qwen3_asr_pipeline.py --config "$auto" --stage train \
            > "$out/logs/driver.log" 2>&1 || true
        if grep -q "train_runtime" "$out/logs/driver.log"; then
            ok=1
            break
        fi
        cp "$out/logs/driver.log" "$out/logs/driver.attempt${attempt}.log"
        echo "[small] $name attempt=$attempt failed; retrying $(ts)"
    done
    [ "$ok" = "1" ] || { echo "[small] ERROR: $name exhausted retries"; return 1; }

    local best
    best="$(best_ckpt "$out/best_cer_checkpoints.json")"
    echo "[small] $name best=$best $(ts)"
    sleep 5
    "$PY" tools/eval_unified7_scheme.py --model "$best" --out-dir "$out/eval_best" \
        > "$out/logs/eval_best.log" 2>&1 || { echo "[small] ERROR: $name eval failed"; return 1; }
    rm -rf "$out/checkpoint-latest"
}

echo "[small] start $(ts)"
"$PY" tools/build_small_experiment_manifests.py || { echo "[small] ERROR: manifest build failed"; exit 1; }

run_experiment realraw_noaug \
    "$REPO/configs/poc_small_realraw_noaug_1ep.yaml" \
    "$REPO/outputs/poc_small_realraw_noaug_1ep" || exit 1
run_experiment realdenoise_aug \
    "$REPO/configs/poc_small_realdenoise_aug_1ep.yaml" \
    "$REPO/outputs/poc_small_realdenoise_aug_1ep" || exit 1
run_experiment synth_ab_aug \
    "$REPO/configs/poc_small_synth_ab_aug_1ep.yaml" \
    "$REPO/outputs/poc_small_synth_ab_aug_1ep" || exit 1

"$PY" - <<'EOF'
import json
from pathlib import Path
runs = {
    "realraw_noaug": "outputs/poc_small_realraw_noaug_1ep/eval_best/summary.json",
    "realdenoise_aug": "outputs/poc_small_realdenoise_aug_1ep/eval_best/summary.json",
    "synth_ab_aug": "outputs/poc_small_synth_ab_aug_1ep/eval_best/summary.json",
}
out = {}
for name, path in runs.items():
    p = Path(path)
    out[name] = json.loads(p.read_text()) if p.is_file() else {"missing": path}
Path("outputs/poc_small_experiments_eval.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
EOF
echo "[small] done $(ts)"
