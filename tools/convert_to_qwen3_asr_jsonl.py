#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
import os
import random
import re
from typing import Dict, Iterable, Iterator, List, Optional


from qwen_asr.inference.utils import normalize_language_spec, validate_language_spec


SPEECH_PATH_RE = re.compile(r"<\|startofspeech\|>!(.*?)<\|endofspeech\|>")

# Qwen3-ASR training text prefix: "language {LABEL}<asr_text>..." (see finetuning/README.md).
DEFAULT_CV_LOCALE_TO_LANGUAGE: Dict[str, str] = {
    "ug": "Uyghur",
    "tr": "Turkish",
    "uz": "Uzbek",
    "zh-CN": "Chinese",
}
GENERIC_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Fun-ASR-Nano style data into Qwen3-ASR finetuning jsonl."
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output Qwen3-ASR jsonl path.")

    parser.add_argument("--wav_scp", type=str, default="", help="Kaldi-style wav.scp file.")
    parser.add_argument("--text_file", type=str, default="", help="Kaldi-style text file.")
    parser.add_argument("--funasr_jsonl", type=str, default="", help="Fun-ASR-Nano messages jsonl file.")
    parser.add_argument(
        "--switchlingua_csv",
        type=str,
        default="",
        help="SwitchLingua-style metadata CSV (columns include file_name, text). Mutually exclusive with other sources.",
    )
    parser.add_argument(
        "--switchlingua_audio_dir",
        type=str,
        default="",
        help="Directory containing audio files named like file_name in the SwitchLingua CSV.",
    )
    parser.add_argument("--cv_tsv", type=str, default="", help="Common Voice train/dev/test tsv file.")
    parser.add_argument(
        "--cv_clips_dir",
        type=str,
        default="",
        help="Common Voice clips directory. If empty, use dirname(cv_tsv)/clips.",
    )
    parser.add_argument(
        "--cv_multilingual_root",
        type=str,
        default="",
        help="Common Voice corpus root with per-locale subfolders (e.g. .../cv-corpus-25.0-2026-03-09). "
        "Reads {root}/{locale}/{split}.tsv for each locale in --cv_locales. Mutually exclusive with other sources.",
    )
    parser.add_argument(
        "--cv_locales",
        type=str,
        default="",
        help="Comma-separated locale directory names under --cv_multilingual_root, e.g. ug,tr,uz,zh-CN.",
    )
    parser.add_argument(
        "--cv_split",
        type=str,
        default="train",
        help="TSV split name under each locale (used with --cv_multilingual_root), e.g. train, dev, test.",
    )
    parser.add_argument(
        "--cv_languages",
        type=str,
        default="",
        help="Optional comma-separated language labels parallel to --cv_locales (same length). "
        "If empty, built-in defaults are used for known locales (ug, tr, uz, zh-CN).",
    )
    parser.add_argument(
        "--cv_balance_cap_locale",
        type=str,
        default="",
        help="If set (e.g. ug), use ALL samples from this anchor locale; each other locale is uniformly "
        "subsampled to at most the anchor's valid row count (multilingual / cv_multilingual_root only).",
    )
    parser.add_argument(
        "--cv_balance_seed",
        type=int,
        default=42,
        help="RNG seed for subsampling when --cv_balance_cap_locale is set.",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="None",
        help="Language prefix to inject into text, e.g. English, Chinese, None, or comma-separated code-switching (Chinese,English).",
    )
    parser.add_argument(
        "--keep_prompt",
        type=int,
        default=0,
        help="Keep non-empty system prompt as Qwen3-ASR prompt field. 0 or 1.",
    )
    parser.add_argument(
        "--drop_generic_system_prompt",
        type=int,
        default=1,
        help='Drop the generic prompt "You are a helpful assistant.". 0 or 1.',
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, only convert the first N samples (all modes).",
    )
    return parser.parse_args()


def read_kaldi_mapping(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid line in {path}:{line_no}: {raw_line.rstrip()}")
            utt_id, value = parts
            mapping[utt_id] = value
    return mapping


def normalize_target_text(text: str, language: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty transcript is not allowed.")
    if text.startswith("language ") and "<asr_text>" in text:
        return text
    return f"language {language}<asr_text>{text}"


def iter_from_wav_and_text(wav_scp: str, text_file: str, language: str) -> Iterable[dict]:
    wav_map = read_kaldi_mapping(wav_scp)
    text_map = read_kaldi_mapping(text_file)

    wav_keys = set(wav_map)
    text_keys = set(text_map)
    if wav_keys != text_keys:
        missing_in_wav = sorted(text_keys - wav_keys)
        missing_in_text = sorted(wav_keys - text_keys)
        msg = []
        if missing_in_wav:
            msg.append(f"missing in wav.scp: {missing_in_wav[:5]}")
        if missing_in_text:
            msg.append(f"missing in text: {missing_in_text[:5]}")
        raise ValueError("Utterance ids do not match between wav.scp and text file. " + "; ".join(msg))

    for utt_id in sorted(wav_map):
        yield {
            "audio": wav_map[utt_id],
            "text": normalize_target_text(text_map[utt_id], language),
        }


def extract_message_content(messages, role: str) -> str:
    for item in messages:
        if item.get("role") == role:
            return str(item.get("content", ""))
    return ""


def extract_audio_path_from_user_content(user_content: str) -> str:
    match = SPEECH_PATH_RE.search(user_content)
    if not match:
        raise ValueError(f"Cannot find speech path in user content: {user_content}")
    return match.group(1).strip()


def _clean_switchlingua_transcript(text: str) -> str:
    """Collapse internal newlines / runs of whitespace for a single-line jsonl transcript."""
    t = (text or "").strip()
    if not t:
        return ""
    return " ".join(t.split())


def iter_from_switchlingua_csv(
    csv_path: str,
    audio_dir: str,
    language: str,
) -> Iterator[dict]:
    """
    Read SwitchLingua-style CSV: ``file_name`` (audio basename), ``text`` (reference transcript).

    Audio paths are ``os.path.join(audio_dir, file_name)``.
    """
    if not os.path.isfile(csv_path):
        raise ValueError(f"SwitchLingua CSV not found: {csv_path}")
    if not os.path.isdir(audio_dir):
        raise ValueError(f"SwitchLingua audio dir not found: {audio_dir}")
    audio_root = os.path.abspath(audio_dir)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "file_name" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(
                f"SwitchLingua CSV must have columns 'file_name' and 'text'; got {reader.fieldnames!r}"
            )
        for line_no, row in enumerate(reader, start=2):
            fn = (row.get("file_name") or "").strip()
            transcript = _clean_switchlingua_transcript(row.get("text") or "")
            if not fn or not transcript:
                continue
            audio_abs = os.path.abspath(os.path.join(audio_root, fn))
            yield {
                "audio": audio_abs,
                "text": normalize_target_text(transcript, language),
            }


def iter_from_funasr_jsonl(
    jsonl_path: str,
    language: str,
    keep_prompt: bool,
    drop_generic_system_prompt: bool,
) -> Iterable[dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError(f"Invalid messages field in {jsonl_path}:{line_no}")

            system_prompt = extract_message_content(messages, "system").strip()
            user_content = extract_message_content(messages, "user").strip()
            assistant_content = extract_message_content(messages, "assistant").strip()

            audio_path = extract_audio_path_from_user_content(user_content)
            output = {
                "audio": audio_path,
                "text": normalize_target_text(assistant_content, language),
            }

            if keep_prompt and system_prompt:
                if not (drop_generic_system_prompt and system_prompt == GENERIC_SYSTEM_PROMPT):
                    output["prompt"] = system_prompt

            yield output


def _iter_common_voice_records(tsv_path: str, clips_dir: str, language: str) -> Iterator[dict]:
    if not os.path.isfile(tsv_path):
        raise ValueError(f"Common Voice tsv not found: {tsv_path}")
    if not os.path.isdir(clips_dir):
        raise ValueError(f"Common Voice clips dir not found: {clips_dir}")

    with open(tsv_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Empty tsv file: {tsv_path}")
        headers = header_line.rstrip("\n").split("\t")
        col_idx = {name: i for i, name in enumerate(headers)}
        if "path" not in col_idx or "sentence" not in col_idx:
            raise ValueError(f"Expected columns 'path' and 'sentence' in {tsv_path}, got: {headers}")

        for line_no, raw_line in enumerate(f, start=2):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            path_i = col_idx["path"]
            sent_i = col_idx["sentence"]
            rel_audio = parts[path_i].strip() if path_i < len(parts) else ""
            sentence = parts[sent_i].strip() if sent_i < len(parts) else ""

            if not rel_audio:
                raise ValueError(f"Missing 'path' at {tsv_path}:{line_no}")
            if not sentence:
                continue

            audio_abs = os.path.abspath(os.path.join(clips_dir, rel_audio))
            yield {
                "audio": audio_abs,
                "text": normalize_target_text(sentence, language),
            }


def _iter_reservoir_capped(source: Iterable[dict], cap: int, rng: random.Random) -> Iterator[dict]:
    """Uniform subsample of size min(population, cap) in one pass (Vitter-style reservoir)."""
    if cap < 0:
        raise ValueError("cap must be non-negative.")
    reservoir: List[dict] = []
    for i, item in enumerate(source):
        if i < cap:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < cap:
                reservoir[j] = item
    rng.shuffle(reservoir)
    yield from reservoir


def iter_from_common_voice_tsv(
    tsv_path: str,
    clips_dir: str,
    language: str,
    max_samples: int = 0,
) -> Iterable[dict]:
    count = 0
    for rec in _iter_common_voice_records(tsv_path, clips_dir, language):
        yield rec
        count += 1
        if max_samples > 0 and count >= max_samples:
            return


def _parse_csv_fields(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_cv_locale_language(loc: str, language_labels: Optional[List[str]], index: int) -> str:
    if language_labels is not None:
        return language_labels[index]
    lang = DEFAULT_CV_LOCALE_TO_LANGUAGE.get(loc)
    if not lang:
        known = ", ".join(sorted(DEFAULT_CV_LOCALE_TO_LANGUAGE))
        raise ValueError(
            f"No built-in language label for locale {loc!r}. "
            f"Pass --cv_languages with one label per locale, or extend DEFAULT_CV_LOCALE_TO_LANGUAGE. "
            f"Built-in locales: {known}"
        )
    return lang


def iter_from_cv_multilingual_corpus(
    corpus_root: str,
    locales: List[str],
    split: str,
    language_labels: Optional[List[str]],
    balance_cap_locale: str = "",
    balance_seed: int = 42,
) -> Iterable[dict]:
    if not os.path.isdir(corpus_root):
        raise ValueError(f"Common Voice corpus root not found: {corpus_root}")
    if not locales:
        raise ValueError("No locales provided for multilingual Common Voice conversion.")
    if language_labels is not None and len(language_labels) != len(locales):
        raise ValueError(
            f"--cv_languages length ({len(language_labels)}) must match --cv_locales ({locales})."
        )

    anchor_key = balance_cap_locale.strip()
    anchor_records: Optional[List[dict]] = None
    cap = 0
    rng: Optional[random.Random] = None

    if anchor_key:
        if anchor_key not in locales:
            raise ValueError(
                f"--cv_balance_cap_locale {anchor_key!r} must appear in --cv_locales {locales!r}."
            )
        anchor_idx = locales.index(anchor_key)
        anchor_lang = _resolve_cv_locale_language(anchor_key, language_labels, anchor_idx)
        anchor_tsv = os.path.join(corpus_root, anchor_key, f"{split}.tsv")
        anchor_clips = os.path.join(corpus_root, anchor_key, "clips")
        anchor_records = list(_iter_common_voice_records(anchor_tsv, anchor_clips, anchor_lang))
        cap = len(anchor_records)
        if cap == 0:
            raise ValueError(f"Anchor locale {anchor_key!r} produced 0 valid training rows.")
        rng = random.Random(balance_seed)
        print(
            f"Balanced multilingual: anchor locale={anchor_key!r}, cap={cap}, subsample_seed={balance_seed}."
        )

    for i, loc in enumerate(locales):
        lang = _resolve_cv_locale_language(loc, language_labels, i)
        tsv_path = os.path.join(corpus_root, loc, f"{split}.tsv")
        clips_dir = os.path.join(corpus_root, loc, "clips")

        if anchor_key and loc == anchor_key:
            assert anchor_records is not None
            print(f"Locale {loc}: using all anchor samples ({len(anchor_records)}).")
            yield from anchor_records
        elif anchor_key:
            assert rng is not None
            before_stream = _iter_common_voice_records(tsv_path, clips_dir, lang)
            out = list(_iter_reservoir_capped(before_stream, cap, rng))
            print(f"Locale {loc}: subsampled to {len(out)} rows (cap={cap}).")
            yield from out
        else:
            yield from iter_from_common_voice_tsv(tsv_path, clips_dir, lang, max_samples=0)


def main():
    args = parse_args()
    try:
        language = normalize_language_spec(args.language.strip())
        validate_language_spec(language)
    except ValueError as exc:
        raise SystemExit(f"Invalid --language: {exc}") from exc

    use_kaldi = bool(args.wav_scp or args.text_file)
    use_funasr_jsonl = bool(args.funasr_jsonl)
    use_cv_tsv = bool(args.cv_tsv)
    use_cv_multilingual = bool(args.cv_multilingual_root.strip())
    use_switchlingua = bool(args.switchlingua_csv.strip())

    source_modes = (
        int(use_kaldi)
        + int(use_funasr_jsonl)
        + int(use_cv_tsv)
        + int(use_cv_multilingual)
        + int(use_switchlingua)
    )
    if source_modes != 1:
        raise ValueError(
            "Provide exactly one source mode: Kaldi, FunASR jsonl, Common Voice tsv, "
            "Common Voice multilingual root, or SwitchLingua CSV."
        )

    if use_kaldi:
        if not args.wav_scp or not args.text_file:
            raise ValueError("Both --wav_scp and --text_file are required for Kaldi-style conversion.")
        records = iter_from_wav_and_text(args.wav_scp, args.text_file, language)
    elif use_funasr_jsonl:
        records = iter_from_funasr_jsonl(
            args.funasr_jsonl,
            language=language,
            keep_prompt=bool(args.keep_prompt),
            drop_generic_system_prompt=bool(args.drop_generic_system_prompt),
        )
    elif use_cv_tsv:
        clips_dir = args.cv_clips_dir.strip() or os.path.join(os.path.dirname(os.path.abspath(args.cv_tsv)), "clips")
        records = iter_from_common_voice_tsv(
            args.cv_tsv,
            clips_dir=clips_dir,
            language=language,
            max_samples=args.max_samples,
        )
    elif use_cv_multilingual:
        if not args.cv_locales.strip():
            raise ValueError("--cv_locales is required when using --cv_multilingual_root.")
        locales = list(dict.fromkeys(_parse_csv_fields(args.cv_locales)))  # deduplicate, preserve order
        langs_csv = args.cv_languages.strip()
        language_labels: Optional[List[str]] = _parse_csv_fields(langs_csv) if langs_csv else None
        split = args.cv_split.strip() or "train"
        records = iter_from_cv_multilingual_corpus(
            os.path.abspath(args.cv_multilingual_root.strip()),
            locales,
            split,
            language_labels,
            balance_cap_locale=args.cv_balance_cap_locale,
            balance_seed=args.cv_balance_seed,
        )
    elif use_switchlingua:
        if not args.switchlingua_audio_dir.strip():
            raise ValueError("--switchlingua_audio_dir is required when using --switchlingua_csv.")
        records = iter_from_switchlingua_csv(
            os.path.abspath(args.switchlingua_csv.strip()),
            os.path.abspath(args.switchlingua_audio_dir.strip()),
            language,
        )
    else:
        raise ValueError("Invalid source mode.")

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for item in records:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if args.max_samples > 0 and count >= args.max_samples:
                break

    print(f"Converted {count} samples to {args.output_file}")


if __name__ == "__main__":
    main()
