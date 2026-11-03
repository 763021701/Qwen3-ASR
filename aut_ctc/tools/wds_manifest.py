"""Build a JSONL manifest from local WebDataset shards (<key>.flac + <key>.json / <key>.txt) without extracting them.

Every entry points at its audio member: {"id", "audio": <shard path>, "tar_offset", "tar_size", "duration", "text",
"lang", "source", ...}; ManifestDataset reads that byte range directly.

  python -m aut_ctc.tools.wds_manifest --shards 'shards/WenetSpeech/TRAIN/*.tar' --lang zh --source wenetspeech \
      --out data/train_200h/wenetspeech.jsonl --hours 47 --seed 0
Options: --hours samples that many hours at random (seeded) from all clips; --min_confidence drops clips whose json
"confidence" is lower (pseudo-labelled sources), taking it from --confidence_tsv ("<key>\t<confidence>" lines) when the
shards do not carry it; --where key=v1|v2 keeps clips whose json field is one of the values
(repeatable, e.g. KeSpeech style=Mandarin|Accent); --keep_fields copies json fields into the entries.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import tarfile

import soundfile as sf

AUDIO_EXTS = ("flac", "wav", "mp3", "opus", "ogg")


def audio_duration(tf, member) -> float:
    """FLAC: from the 42-byte STREAMINFO header (no need to read the audio); otherwise decode the header via soundfile."""
    f = tf.extractfile(member)
    head = f.read(42)
    if head[:4] == b"fLaC":
        bits = int.from_bytes(head[18:26], "big")  # sample rate (20 bits) | channels, bps (8 bits) | total samples (36)
        rate, total = bits >> 44, bits & ((1 << 36) - 1)
        if rate and total:
            return total / rate
    return sf.info(io.BytesIO(head + f.read())).duration


def shard_entries(path: str, lang: str, source: str, keep_fields=()):
    """Yield (entry, json metadata) for every audio member of one shard."""
    with tarfile.open(path) as tf:
        groups = {}
        for m in tf:
            if m.isfile():
                key, _, ext = m.name.rpartition(".")
                groups.setdefault(key, {})[ext] = m
        for key, members in groups.items():
            audio = next((members[e] for e in AUDIO_EXTS if e in members), None)
            if audio is None:
                continue
            meta = json.loads(tf.extractfile(members["json"]).read()) if "json" in members else {}
            text = (tf.extractfile(members["txt"]).read().decode("utf-8") if "txt" in members
                    else meta.get("text") or "").strip()
            duration = audio_duration(tf, audio)
            entry = {"id": f"{source}/{key.rsplit('/', 1)[-1]}", "audio": os.path.abspath(path),
                     "tar_offset": audio.offset_data, "tar_size": audio.size, "duration": round(duration, 3),
                     "text": text, "lang": lang, "source": source}
            entry.update({k: meta[k] for k in keep_fields if k in meta})
            yield entry, meta


def main():
    p = argparse.ArgumentParser("WebDataset shards -> manifest")
    p.add_argument("--shards", nargs="+", required=True, help="shard paths or glob patterns")
    p.add_argument("--lang", required=True, help="zh / yue / en / wuyu / ... (decides Chinese vs English TN)")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=0, help="random subset of this many hours (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_confidence", type=float, default=None)
    p.add_argument("--confidence_tsv", default=None, help="key<TAB>confidence for shards without it")
    p.add_argument("--where", nargs="*", default=[], help="json field filters, e.g. style=Mandarin|Accent")
    p.add_argument("--keep_fields", nargs="*", default=[])
    args = p.parse_args()

    shards = sorted({s for pattern in args.shards for s in glob.glob(pattern)})
    where = [w.split("=", 1) for w in args.where]
    confidence = {}
    if args.confidence_tsv:
        with open(args.confidence_tsv, encoding="utf-8") as f:
            confidence = {k: float(v) for k, v in (line.rstrip("\n").split("\t") for line in f)}
    entries, dropped = [], 0
    for shard in shards:
        for entry, meta in shard_entries(shard, args.lang, args.source, args.keep_fields):
            conf = meta.get("confidence", confidence.get(entry["id"].split("/", 1)[1], 0.0 if confidence else 1.0))
            low_conf = args.min_confidence is not None and float(conf) < args.min_confidence
            if low_conf or any(str(meta.get(k)) not in v.split("|") for k, v in where):
                dropped += 1
                continue
            entries.append(entry)
    total = sum(e["duration"] for e in entries) / 3600
    if args.hours and args.hours < total:
        random.Random(args.seed).shuffle(entries)
        kept, acc = [], 0.0
        for e in entries:
            if acc >= args.hours * 3600:
                break
            kept.append(e)
            acc += e["duration"]
        entries = kept
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{args.source}: {len(shards)} shards, {total:.1f} h available ({dropped} clips filtered), wrote "
          f"{len(entries)} clips / {sum(e['duration'] for e in entries) / 3600:.1f} h -> {args.out}")


if __name__ == "__main__":
    main()
