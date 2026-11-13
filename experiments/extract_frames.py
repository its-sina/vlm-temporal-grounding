"""Decode, sample and resize the frames of every video once, writing lossless frame caches.

Writes <frame-cache-dir>/<vid>.npz (uint8 frames [N, H, W, 3] and the duration). Existing files are
skipped, so the script can be re-run after an interruption.

  python experiments/extract_frames.py --dataset charades --split test --fps 1.0 --thumb 336 --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import find_video, load_samples  # noqa: E402
from src.data.frames import load_frames, save_frame_cache  # noqa: E402
from src.utils import paths  # noqa: E402


def extract_one(vid: str, video_dir: Path, out_dir: Path, fps: float, thumb: int) -> tuple[str, str]:
    dst = out_dir / f"{vid}.npz"
    if dst.exists():
        return vid, "skip"
    video = find_video(video_dir, vid)
    if video is None:
        return vid, "missing"
    frames, duration = load_frames(video, fps, (thumb, thumb))
    tmp = out_dir / f"{vid}.tmp.npz"
    save_frame_cache(tmp, frames, duration)
    os.replace(tmp, dst)
    return vid, "ok"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="charades")
    ap.add_argument("--split", default="test")
    ap.add_argument("--annotations", help="annotation JSON (default: <data-dir>/<dataset>/annotations/<split>.json)")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--frame-cache-dir", type=Path, help="output directory (default: under $VTG_CACHE_DIR)")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--thumb", type=int, default=336, help="frame size in px (320 for Qwen3-VL, 448 for InternVL)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--vids", help="optional file with one video id per line")
    a = ap.parse_args(argv)

    data_root = a.data_dir or paths.data_dir()
    annotations = Path(a.annotations) if a.annotations else paths.default_annotation_file(a.dataset, a.split, data_root)
    video_dir = a.video_dir or paths.default_video_dir(a.dataset, data_root)
    out_dir = a.frame_cache_dir or paths.frame_cache_dir(a.dataset, a.fps, a.thumb)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted({s.vid for s in load_samples(annotations)})
    if a.vids:
        keep = set(Path(a.vids).read_text().split())
        vids = [v for v in vids if v in keep]
    t0, stats = time.time(), {}
    with ProcessPoolExecutor(a.workers) as ex:
        futures = [ex.submit(extract_one, v, video_dir, out_dir, a.fps, a.thumb) for v in vids]
        for i, fut in enumerate(as_completed(futures)):
            vid, status = fut.result()
            stats[status] = stats.get(status, 0) + 1
            if status == "missing":
                print(f"missing video: {vid}", flush=True)
            if i % 100 == 0:
                print(f"{i}/{len(vids)} {stats} {time.time() - t0:.0f}s", flush=True)
    print(f"done {stats} -> {out_dir}")


if __name__ == "__main__":
    main()
