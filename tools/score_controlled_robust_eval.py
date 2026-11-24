#!/usr/bin/env python3
"""Report absolute error, clean-to-corrupt degradation and paired model gains."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys

_SNAPSHOT_BENCH = Path(__file__).resolve().parents[2]/"hmasr_benchmark"
_DEFAULT_BENCH = (_SNAPSHOT_BENCH if (_SNAPSHOT_BENCH/"src/asr_benchmark").exists()
                  else Path("/root/autodl-tmp/workspace/project/hmasr_benchmark"))
BENCH = Path(os.environ.get("HMASR_BENCHMARK_ROOT", str(_DEFAULT_BENCH)))
sys.path.insert(0, str(BENCH/"src"))
sys.path.insert(0, str(BENCH/"third_party"))


def derived_metrics(base_clean, lora_clean, base_error, lora_error):
    """Inputs are fractions; differences are percentage points, RER is percent."""
    return dict(
        baseline_clean_percent=100*base_clean, lora_clean_percent=100*lora_clean,
        baseline_error_percent=100*base_error, lora_error_percent=100*lora_error,
        baseline_degradation_pp=100*(base_error-base_clean),
        lora_degradation_pp=100*(lora_error-lora_clean),
        degradation_advantage_pp=100*((base_error-base_clean)-(lora_error-lora_clean)),
        absolute_error_delta_pp=100*(lora_error-base_error),
        relative_error_reduction_percent=100*(base_error-lora_error)/base_error if base_error else None,
    )


def normalized_noise_auc(e20, e10, e0):
    """Trapezoid area on 0..20 dB divided by the 20 dB span."""
    return (e20 + 2*e10 + e0)/4


def load_predictions(path):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            pair = row["source_id"]
            if pair in result:
                raise ValueError(f"Duplicate speech pair: {path}: {pair}")
            result[pair] = row
    return result


def write_csv(path, rows):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(root):
    from asr_benchmark.compare import check_compatible
    from asr_benchmark.metrics import compute_metrics
    root = Path(root).resolve()
    base_run = json.loads((root/"baseline/run.json").read_text())
    lora_run = json.loads((root/"final/run.json").read_text())
    if base_run["status"] != "completed" or lora_run["status"] != "completed":
        raise ValueError("Both inference runs must finish before robustness scoring")
    check_compatible(base_run, lora_run)
    specs = json.loads((root/"dataset_map.json").read_text())
    bm = json.loads((root/"baseline/metrics.json").read_text())["datasets"]
    lm = json.loads((root/"final/metrics.json").read_text())["datasets"]
    if set(specs) != set(bm) or set(specs) != set(lm):
        raise ValueError("Missing dataset metrics")
    predictions = {}
    result, family_rows, auc_rows = [], [], []
    by_source = defaultdict(dict)
    for name, spec in specs.items():
        by_source[(spec["split"], spec["corpus"])][spec["condition"]] = name
        predictions[name] = (
            load_predictions(root/"baseline/predictions"/(name+".jsonl")),
            load_predictions(root/"final/predictions"/(name+".jsonl")),
        )
        a,b = predictions[name]
        if set(a) != set(b) or len(a) != spec["n_utts"]:
            raise ValueError("Missing or duplicate paired predictions")
        for pair in a:
            if a[pair]["reference_normalized"] != b[pair]["reference_normalized"]:
                raise ValueError("Model references differ")
        for rows, stored in [(a, bm[name]), (b, lm[name])]:
            observed = compute_metrics(spec["metric"],
                                       [r["reference_normalized"] for r in rows.values()],
                                       [r["hypothesis_normalized"] for r in rows.values()])
            for field in ["value", "substitutions", "deletions", "insertions",
                          "n_ref_units", "num_pairs", "num_skipped_empty_ref"]:
                if abs(observed[field] - stored[field]) > 1e-12:
                    raise ValueError(f"Cached metrics differ from predictions: {name}: {field}")
        if bm[name]["n_ref_units"] != lm[name]["n_ref_units"]:
            raise ValueError("Model reference-unit counts differ")
    for (split, corpus), conditions in by_source.items():
        clean_name = conditions["clean"]
        cb,cl = predictions[clean_name]
        for condition, name in conditions.items():
            spec = specs[name]
            pb,pl = predictions[name]
            if set(pb) != set(cb):
                raise ValueError("Corrupted/clean samples differ")
            for pair in pb:
                if pb[pair]["reference_normalized"] != cb[pair]["reference_normalized"]:
                    raise ValueError("Augmentation changed a reference")
            row = dict(split=split, corpus=corpus, condition=condition, metric=spec["metric"],
                       n_utts=spec["n_utts"], n_ref_units=bm[name]["n_ref_units"],
                       n_scored_utts=spec["n_utts"]-bm[name]["num_skipped_empty_ref"],
                       n_skipped_empty_ref=bm[name]["num_skipped_empty_ref"])
            row.update(derived_metrics(bm[clean_name]["value"], lm[clean_name]["value"],
                                       bm[name]["value"], lm[name]["value"]))
            result.append(row)
            family_ids = defaultdict(list)
            for line in Path(spec["manifest"]).open():
                sample = json.loads(line)
                noise = sample.get("trace",{}).get("noise")
                if noise:
                    family_ids[noise["source_family"]].append(sample["source_id"])
            for family, pairs in sorted(family_ids.items()):
                def error(pred):
                    return compute_metrics(spec["metric"],
                                           [pred[p]["reference_normalized"] for p in pairs],
                                           [pred[p]["hypothesis_normalized"] for p in pairs])["value"]
                group = dict(split=split, corpus=corpus, condition=condition, metric=spec["metric"],
                             noise_family=family, n_utts=len(pairs),
                             n_scored_utts=sum(bool(cb[p]["reference_normalized"].strip()) for p in pairs),
                             n_skipped_empty_ref=sum(not cb[p]["reference_normalized"].strip() for p in pairs))
                group.update(derived_metrics(error(cb),error(cl),error(pb),error(pl)))
                family_rows.append(group)
        b_auc = normalized_noise_auc(*(bm[conditions[f"noise_snr{s}"]]["value"] for s in [20,10,0]))
        l_auc = normalized_noise_auc(*(lm[conditions[f"noise_snr{s}"]]["value"] for s in [20,10,0]))
        auc_rows.append(dict(split=split,corpus=corpus,metric=spec["metric"],
                             baseline_normalized_noise_curve_auc_percent=100*b_auc,
                             lora_normalized_noise_curve_auc_percent=100*l_auc,
                             absolute_delta_pp=100*(l_auc-b_auc)))
    result.sort(key=lambda r:(r["split"],r["corpus"],r["condition"]))
    for name, rows in [("robustness_metrics",result),("noise_family_metrics",family_rows),
                       ("noise_curve_auc",auc_rows)]:
        (root/(name+".json")).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
        write_csv(root/(name+".csv"),rows)
    summary = dict(
        complete=True, paired_datasets=len(result),
        canonical_metrics_recomputed_from_all_paired_predictions=True,
        paired_references_verified=True,
        noise_family_groups=len(family_rows), noise_curve_auc_groups=len(auc_rows),
        metric_definitions={
            "absolute_error":"WER/CER on the exact same rendered audio; lower is better.",
            "degradation":"Corrupted WER/CER minus the same model's clean WER/CER, percentage points.",
            "degradation_advantage":"Base degradation minus LoRA degradation; positive means smaller extra degradation for LoRA.",
            "relative_error_reduction":"(base_corrupted_error - lora_corrupted_error) / base_corrupted_error * 100; positive favors LoRA.",
            "noise_curve_auc":"Trapezoid integral of error on SNR 0/10/20 dB, divided by 20 dB; lower is better.",
        },
        interpretation="Absolute error and degradation must be read together. WER and CER are not pooled into a single score. Dev and test are reported separately.",
        empty_reference_policy="The canonical scorer excludes references that normalize to empty. The same speech IDs are excluded for both models and every matched condition; selected and scored counts are reported separately.",
        scored_utterances_by_split_corpus=[
            dict(split=split, corpus=corpus, n_selected=specs[c["clean"]]["n_utts"],
                 n_scored=specs[c["clean"]]["n_utts"]-bm[c["clean"]]["num_skipped_empty_ref"],
                 n_skipped_empty_ref=bm[c["clean"]]["num_skipped_empty_ref"])
            for (split,corpus),c in by_source.items()
        ],
        generation=json.loads((root/"generation_summary.json").read_text()),
    )
    (root/"robustness_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    report = ["# 受控鲁棒性评测", "",
              "基座与 1000h LoRA 使用同一批固定音频；dev/test 分开报告。",
              "错误率为 WER/CER 百分比，越低越好；额外退化为相对各自 clean 的百分点增幅。",
              "RER 为相对基座的错误率降低比例，正值表示 LoRA 更好。",
              "clean 为原始录音并采用统一电平控制，可能已有背景噪声。每组抽取数与有效计分数见各表。", ""]
    order = ["clean", "noise_snr20", "noise_snr10", "noise_snr0", "rir_real", "rir_simulated",
             "rir_real_noise_snr10", "rir_real_noise_snr0", "rir_simulated_noise_snr10", "rir_simulated_noise_snr0"]
    for split in ["test", "dev"]:
        for corpus in sorted({r["corpus"] for r in result}):
            rows = {r["condition"]:r for r in result if r["split"]==split and r["corpus"]==corpus}
            if not rows:
                continue
            first = next(iter(rows.values()))
            report += [f"## {split} / {corpus} ({first['metric'].upper()})", "",
                       f"抽取 {first['n_utts']} 句；实际计分 {first['n_scored_utts']} 句，标准化后为空 {first['n_skipped_empty_ref']} 句。", "",
                       "| 条件 | 基座错误率 % | LoRA 错误率 % | 基座额外退化 pp | LoRA 额外退化 pp | RER % |",
                       "|---|---:|---:|---:|---:|---:|"]
            for condition in order:
                if condition not in rows:
                    continue
                r = rows[condition]
                rer = r["relative_error_reduction_percent"]
                rer_text = f"{rer:.2f}" if rer is not None else "—"
                report.append(f"| {condition} | {r['baseline_error_percent']:.2f} | {r['lora_error_percent']:.2f} | "
                              f"{r['baseline_degradation_pp']:.2f} | {r['lora_degradation_pp']:.2f} | {rer_text} |")
            report.append("")
    report += ["## 解释范围", "",
               "绝对错误率和额外退化需一起阅读；clean 变差也可能让额外退化看起来变小。",
               "WER 与 CER 不混成一个总分。噪声来源子集和 SNR 曲线面积见同目录 CSV/JSON。",
               ("混响使用预处理 RIR 的固定 dry/wet 分量比例 6 dB；实测 dev/test 各只有两个保守来源组。"
                if any(r["condition"].startswith("rir_") for r in result)
                else "本轮仅叠加固定 test 噪声，不加入 RIR、变速或 SpecAugment。"),
               "所有成对预测、参考文本及缓存指标均已复核；音频和素材隔离检查见 suite_verification.json。", ""]
    (root/"robustness_report.md").write_text("\n".join(report))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir",required=True)
    args=parser.parse_args()
    score(args.output_dir)


if __name__=="__main__":
    main()
