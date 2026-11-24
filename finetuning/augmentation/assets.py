"""Strict readers for frozen augmentation asset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SPLITS = ("train", "dev", "test")


def read_asset_manifest(
    manifest_path: str, kind: str, split: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Read one split only; an explicit manifest never falls back to a directory."""
    manifest = Path(manifest_path).absolute()
    if split is not None and split not in SPLITS:
        raise ValueError(f"Invalid augmentation split: {split!r}")
    rows = []
    seen = set()
    observed = set()
    with manifest.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{manifest}:{number}"
            if row.get("schema_version") != 1 or row.get("kind") != kind:
                raise ValueError(f"{where}: expected schema_version=1, kind={kind}")
            actual_split = row.get("split")
            if actual_split not in SPLITS or (split is not None and actual_split != split):
                raise ValueError(f"{where}: split {actual_split!r} does not match {split!r}")
            observed.add(actual_split)
            if len(observed) != 1:
                raise ValueError(f"{where}: mixed augmentation splits are forbidden")
            if not all(row.get(key) for key in ("asset_id", "origin_ids", "group_id", "path")):
                raise ValueError(f"{where}: missing asset identity or path")
            path = Path(row["path"])
            if not path.is_absolute():
                path = manifest.parent / path
            path = path.resolve(strict=True)
            if not path.is_file() or path.suffix.lower() not in {".wav", ".flac"}:
                raise ValueError(f"{where}: not an audio file: {path}")
            if path in seen:
                raise ValueError(f"{where}: duplicate canonical path: {path}")
            if kind == "rir" and row.get("source") not in {"real", "simulated"}:
                raise ValueError(f"{where}: invalid RIR source")
            seen.add(path)
            # Fingerprints and alias inventories belong to the audit. Retaining
            # them in every simulator would multiply hundreds of MB by workers.
            runtime_fields = ("asset_id", "origin_ids", "group_id", "kind", "split",
                              "source", "category", "room_id")
            rows.append({**{key: row[key] for key in runtime_fields if key in row},
                         "path": path})
    if not rows:
        raise ValueError(f"Empty augmentation manifest: {manifest}")
    return rows
