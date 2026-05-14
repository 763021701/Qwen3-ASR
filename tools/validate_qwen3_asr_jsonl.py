#!/usr/bin/env python3
# coding=utf-8

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from qwen_asr.inference.utils import normalize_language_spec, split_language_spec, validate_language_spec


_LABEL_RE = re.compile(r"^language\s+([^<]+)<asr_text>(.*)$", re.DOTALL)


@dataclass
class ValidationIssue:
    line: int
    code: str
    message: str


@dataclass
class ValidationReport:
    path: str
    valid: bool
    total_lines: int
    valid_records: int
    issues: List[ValidationIssue]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_supported_languages(repo_root: Optional[str] = None) -> List[str]:
    root = repo_root or _repo_root()
    utils_path = os.path.join(root, "qwen_asr", "inference", "utils.py")
    with open(utils_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=utils_path)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "SUPPORTED_LANGUAGES":
                value = ast.literal_eval(node.value)
                return list(value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SUPPORTED_LANGUAGES":
                    value = ast.literal_eval(node.value)
                    return list(value)
    raise RuntimeError(f"SUPPORTED_LANGUAGES not found in {utils_path}")


def _validate_language_spec_against_allowed(lang_norm: str, allowed: Optional[Set[str]]) -> None:
    """Validate normalized spec; if allowed is set, each atom must be in allowed (except sole None)."""
    if allowed is None:
        validate_language_spec(lang_norm)
        return
    atoms = split_language_spec(lang_norm)
    if len(atoms) == 1 and atoms[0] == "None":
        return
    for a in atoms:
        if a == "None":
            raise ValueError(
                "Invalid language spec: 'None' may only appear as the sole language spec, "
                f"got {lang_norm!r}."
            )
        if a not in allowed:
            raise ValueError(f"Unsupported language {a!r}.")


def parse_label_language(text: str) -> Optional[str]:
    match = _LABEL_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip()


def iter_jsonl(path: str) -> Iterable[tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                yield line_no, None, "empty_line"
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, None, f"invalid_json: {exc}"
                continue
            if not isinstance(obj, dict):
                yield line_no, None, "not_object"
                continue
            yield line_no, obj, None


def validate_jsonl(
    path: str,
    expected_language: str = "",
    check_audio: bool = True,
    supported_languages: Optional[List[str]] = None,
) -> ValidationReport:
    issues: List[ValidationIssue] = []
    total = 0
    valid_records = 0
    allowed: Optional[Set[str]] = None
    if supported_languages is not None:
        allowed = set(supported_languages)

    exp_norm = ""
    exp_raw = (expected_language or "").strip()
    if exp_raw:
        try:
            exp_norm = normalize_language_spec(exp_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid expected_language / --language: {exc}") from exc
        _validate_language_spec_against_allowed(exp_norm, allowed)

    if not os.path.isfile(path):
        return ValidationReport(
            path=path,
            valid=False,
            total_lines=0,
            valid_records=0,
            issues=[ValidationIssue(0, "missing_jsonl", f"JSONL file not found: {path}")],
        )

    for line_no, obj, parse_error in iter_jsonl(path):
        total += 1
        if parse_error:
            code = parse_error.split(":", 1)[0]
            issues.append(ValidationIssue(line_no, code, parse_error))
            continue

        assert obj is not None
        record_ok = True
        audio = obj.get("audio")
        text = obj.get("text")

        if not isinstance(audio, str) or not audio.strip():
            issues.append(ValidationIssue(line_no, "missing_audio", "Field 'audio' must be a non-empty string."))
            record_ok = False
        elif check_audio and not os.path.isfile(audio):
            issues.append(ValidationIssue(line_no, "audio_not_found", f"Audio file not found: {audio}"))
            record_ok = False

        if not isinstance(text, str) or not text.strip():
            issues.append(ValidationIssue(line_no, "missing_text", "Field 'text' must be a non-empty string."))
            record_ok = False
        else:
            lang = parse_label_language(text)
            if lang is None:
                issues.append(
                    ValidationIssue(
                        line_no,
                        "invalid_text_format",
                        "Field 'text' must match 'language {Name[,Name...]}<asr_text>{transcript}'.",
                    )
                )
                record_ok = False
            else:
                try:
                    lang_norm = normalize_language_spec(lang)
                except ValueError as exc:
                    issues.append(
                        ValidationIssue(
                            line_no,
                            "invalid_language_spec",
                            f"Invalid language spec in label: {exc}",
                        )
                    )
                    record_ok = False
                else:
                    if exp_norm and lang_norm != exp_norm:
                        issues.append(
                            ValidationIssue(
                                line_no,
                                "language_mismatch",
                                f"Expected language {exp_norm!r}, got {lang_norm!r}.",
                            )
                        )
                        record_ok = False
                    try:
                        _validate_language_spec_against_allowed(lang_norm, allowed)
                    except ValueError as exc:
                        issues.append(
                            ValidationIssue(
                                line_no,
                                "unsupported_language",
                                str(exc),
                            )
                        )
                        record_ok = False

        if record_ok:
            valid_records += 1

    return ValidationReport(
        path=path,
        valid=not issues and total > 0,
        total_lines=total,
        valid_records=valid_records,
        issues=issues,
    )


def report_to_dict(report: ValidationReport) -> Dict[str, Any]:
    out = asdict(report)
    out["issues"] = [asdict(issue) for issue in report.issues]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Qwen3-ASR finetuning JSONL manifests.")
    p.add_argument("--jsonl", required=True, help="Manifest path to validate.")
    p.add_argument(
        "--language",
        default="",
        help="Expected Qwen3-ASR language label (single or comma-separated), e.g. Uyghur or Chinese,English.",
    )
    p.add_argument("--check_audio", type=int, default=1, choices=(0, 1), help="Check audio files exist.")
    p.add_argument("--output_report", default="", help="Optional JSON report output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_jsonl(
        path=args.jsonl,
        expected_language=args.language.strip(),
        check_audio=bool(args.check_audio),
    )
    payload = report_to_dict(report)

    if args.output_report:
        parent = os.path.dirname(os.path.abspath(args.output_report))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not report.valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
