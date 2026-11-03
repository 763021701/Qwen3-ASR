"""List / download ModelScope dataset files in parallel, logging throughput (uses the stored ModelScope login).

  python -m aut_ctc.tools.ms_download list huangpppu/WenetSpeech --out meta/WenetSpeech.files.json
  python -m aut_ctc.tools.ms_download get jobs.txt --root /root/autodl-tmp/huangpu/aut_ctc_data/shards --workers 6
jobs.txt: one "<namespace/dataset> <path in repo> [size [sha256]]" per line. Files land in <root>/<dataset>/<path>;
complete files are skipped, and every new file is checked against the sha256 when one is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def list_files(repo: str) -> list[dict]:
    from modelscope.hub.api import HubApi

    api, files, page = HubApi(), [], 1
    while True:
        batch = api.get_dataset_files(repo_id=repo, revision="master", root_path="/", recursive=True,
                                      page_number=page, page_size=1000)
        files += [{"path": f["Path"], "size": int(f.get("Size") or 0), "sha256": f.get("Sha256") or ""}
                  for f in batch if f.get("Type") != "tree"]
        if len(batch) < 1000:
            return files
        page += 1


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _on_disk(path: Path) -> int:
    for p in (path, path.with_name(path.name + ".incomplete")):
        if p.exists():
            return p.stat().st_size
    return 0


def download(jobs, root: Path, workers: int, retries: int = 10) -> None:
    # The SDK gives each file a 1 h budget by default; a 4 GB shard at ~1 MiB/s per connection needs longer.
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_TOTAL_TIMEOUT", str(12 * 3600))
    from modelscope.hub.file_download import dataset_file_download

    targets = [(repo, path, size, root / repo.split("/")[-1] / path, sha) for repo, path, size, sha in jobs]
    todo = [t for t in targets if not (t[3].exists() and (not t[2] or t[3].stat().st_size == t[2]))]
    print(f"{len(targets)} files, {len(targets) - len(todo)} already complete, {len(todo)} to download "
          f"({sum(t[2] for t in todo) / 2**30:.1f} GiB)", flush=True)
    done = threading.Event()

    def monitor():
        last, t_last = sum(_on_disk(t[3]) for t in todo), time.time()
        while not done.wait(30):
            now, t_now = sum(_on_disk(t[3]) for t in todo), time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {now / 2**30:.2f} GiB on disk, {(now - last) / 2**20 / (t_now - t_last):.1f} MiB/s",
                  flush=True)
            last, t_last = now, t_now

    def fetch(job):
        repo, path, size, dst, sha = job
        t0 = time.time()
        for attempt in range(1, retries + 1):
            try:
                dataset_file_download(repo, path, local_dir=str(root / repo.split("/")[-1]))
                break
            except Exception as e:  # deadline / network errors: the SDK keeps a resumable .incomplete file
                print(f"retry {attempt}/{retries} {repo}/{path}: {type(e).__name__}: {str(e)[:150]}", flush=True)
                if attempt == retries:
                    print(f"FAILED {repo}/{path} after {retries} attempts", flush=True)
                    return
                time.sleep(min(60, 10 * attempt))
        got, dt = dst.stat().st_size, time.time() - t0
        bad = (size and got != size) or (sha and sha256_of(dst) != sha)
        if bad:
            dst.unlink()
        print(f"{'FAILED (size/sha256 mismatch, removed)' if bad else 'done'} {repo}/{path} {got / 2**20:.0f} MiB "
              f"in {dt:.0f} s ({got / 2**20 / dt:.1f} MiB/s)", flush=True)

    threading.Thread(target=monitor, daemon=True).start()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for fut in [pool.submit(fetch, j) for j in todo]:
            fut.result()
    done.set()
    total = sum(t[2] for t in todo)
    print(f"all done: {total / 2**30:.1f} GiB in {time.time() - t0:.0f} s ({total / 2**20 / (time.time() - t0):.1f} MiB/s)",
          flush=True)


def main():
    p = argparse.ArgumentParser("ModelScope dataset files")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list")
    pl.add_argument("repo")
    pl.add_argument("--out", required=True)
    pg = sub.add_parser("get")
    pg.add_argument("jobs")
    pg.add_argument("--root", required=True)
    pg.add_argument("--workers", type=int, default=6)
    pg.add_argument("--retries", type=int, default=10)
    args = p.parse_args()
    if args.cmd == "list":
        files = list_files(args.repo)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(files, indent=0), encoding="utf-8")
        print(f"{args.repo}: {len(files)} files, {sum(f['size'] for f in files) / 2**30:.1f} GiB -> {args.out}")
    else:
        jobs = []
        for line in open(args.jobs, encoding="utf-8"):
            parts = line.split()
            if parts and not parts[0].startswith("#"):
                jobs.append((parts[0], parts[1], int(parts[2]) if len(parts) > 2 else 0, parts[3] if len(parts) > 3 else ""))
        os.makedirs(args.root, exist_ok=True)
        download(jobs, Path(args.root), args.workers, args.retries)


if __name__ == "__main__":
    main()
