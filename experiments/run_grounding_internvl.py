"""Temporal grounding evaluation on InternVL3.5 (delimiter study on a model with 1-D RoPE).

  --method full      grid images (448-px cells; InternVL's own tiling gives one tile per frame plus a
                     thumbnail of the whole grid), variant native
  --method frameseq  per-frame sequence (each frame its own 448-px image, shown once, last-occurrence
                     window labels), --variant sep|neutral|gap|block|raster (see src/models/internvl.py)
  --method video     InternVL's own video template ("Frame{i}: <image>" lines, 1-based) on the same frames

Output format as experiments/run_grounding.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import load_samples, select_samples  # noqa: E402
from src.data.frames import FrameSource  # noqa: E402
from src.evaluation.metrics import compute_iou  # noqa: E402
from src.grounding.parsing import frames_to_seconds, parse_timestamps  # noqa: E402
from src.grounding.prompts import NUMPRO_PROMPT, build_frameseq_messages, build_grid_messages  # noqa: E402
from src.models.internvl import VARIANTS, InternVLRunner, raster_permutation  # noqa: E402
from src.representations.grid import GridConfig, build_grids, make_layout  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.io import parse_shard, read_jsonl_tolerant, write_run_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML file with default argument values")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-8B-HF", help="local directory or Hub id")
    ap.add_argument("--method", default="frameseq", choices=["full", "frameseq", "video"])
    ap.add_argument("--variant", default="sep", choices=list(VARIANTS))
    ap.add_argument("--dataset", default="charades")
    ap.add_argument("--split", default="test")
    ap.add_argument("--annotations")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--frame-cache-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--thumb", type=int, default=448)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--exclude-vids")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    return ap


FPS = 1.0


def main(argv=None) -> None:
    a = parse_args_with_config(build_parser(), argv)
    if (a.method in ("full", "video")) != (a.variant == "native"):
        raise SystemExit("--variant native goes with --method full/video, the other variants with frameseq")
    data_root = a.data_dir or paths.data_dir()
    a.annotations = (Path(a.annotations) if a.annotations
                     else paths.default_annotation_file(a.dataset, a.split, data_root))
    a.video_dir = a.video_dir or paths.default_video_dir(a.dataset, data_root)
    a.frame_cache_dir = a.frame_cache_dir or paths.frame_cache_dir(a.dataset, FPS, a.thumb)
    out_dir = Path(a.output_dir or paths.results_dir()) / a.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    si, sn = parse_shard(a.shard)
    cfg = GridConfig(4, 3, FPS, a.stride, (a.thumb, a.thumb))
    if si == 0:
        write_run_config(out_dir, vars(a), grid=cfg.name)
    samples = select_samples(load_samples(a.annotations), a.subset, a.exclude_vids, (si, sn))
    pred_path = out_dir / f"pred_shard{si}.jsonl"
    done = {rec["uid"] for rec in read_jsonl_tolerant(pred_path)}
    todo = [s for s in samples if s.uid not in done]
    print(f"[{a.exp}] shard {si}/{sn}: {len(samples)} samples, {len(todo)} to do", flush=True)
    if not todo:
        return
    r = InternVLRunner(a.model)
    get_frames = FrameSource(a.frame_cache_dir, a.video_dir, FPS, a.thumb)

    def one(s):
        t0 = time.perf_counter()
        frames, dur = get_frames(s.vid)
        if a.method == "video":
            # InternVL3.5's video template, built through the image path so that frames keep the
            # 448-px input size; fed with the same 1-fps frames as all other methods.
            body = "\n".join(f"Frame{i + 1}: {r.processor.image_token}" for i in range(len(frames)))
            text = f"<|im_start|>user\n{body}\n{NUMPRO_PROMPT.format(s.query)}<|im_end|>\n<|im_start|>assistant\n"
            inp = r.processor(text=[text], images=list(frames), crop_to_patches=False, return_tensors="pt").to(r.device)
            tp = time.perf_counter() - t0
            out = r.run(inp, variant="native", max_new_tokens=a.max_new_tokens, t_preprocess=tp)
            return out, {"n_frames": len(frames), **out.extra}
        if a.method == "full":
            layout, grids = build_grids(frames, dur, cfg)
            inp = r.prepare(build_grid_messages(layout, grids, s.query), crop_to_patches=True)
            perm = None
        else:
            layout = make_layout(len(frames), dur, cfg)
            inp = r.prepare(build_frameseq_messages(layout, frames, s.query, "last", "window"), crop_to_patches=False)
            perm = raster_permutation(layout, layout.frame_owner("last")) if a.variant == "raster" else None
        tp = time.perf_counter() - t0
        out = r.run(inp, variant=a.variant, raster=perm, max_new_tokens=a.max_new_tokens, t_preprocess=tp)
        return out, {"n_windows": len(layout.windows), "n_frames": layout.n_frames, **out.extra}

    for s in todo[:a.warmup]:
        one(s)
    with open(pred_path, "a") as f:
        for i, s in enumerate(todo):
            out, info = one(s)
            pred = parse_timestamps(out.text)
            pred_s = frames_to_seconds(pred, FPS)
            extra = {}
            if a.method == "video":
                # 1-based "Frame{i}" numbers: frame i is second i-1 (same index-to-second rule as grids)
                pred_s = [max(0, p - 1) for p in pred]
                alt = [max(0, pred[0] - 1), pred[1]]          # sensitivity: inclusive end
                extra = {"pred_alt_incl_end": alt, "iou_alt_incl_end": compute_iou(alt, s.gt)}
            iou = compute_iou(pred_s, s.gt)
            rec = {"uid": s.uid, "vid": s.vid, "query": s.query, "gt": s.gt, "duration": s.duration,
                   "ans_txt": out.text, "pred": pred_s, "iou": iou, "pred_raw": pred,
                   "iou_upstream": compute_iou(pred, s.gt),
                   "n_input_tokens": out.n_input_tokens, "n_visual_tokens": out.n_visual_tokens,
                   "n_visual_tokens_encoded": out.n_visual_tokens_encoded, "n_generated": out.n_generated,
                   "t_preprocess": out.t_preprocess, "t_vision": out.t_vision, "t_prefill": out.t_prefill,
                   "t_decode": out.t_decode, "t_model": out.t_model, "t_total": out.t_total,
                   "peak_mem_gb": out.peak_mem_gb, **info, **extra}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[{a.exp}] {i}/{len(todo)} iou={iou:.2f} t={rec['t_model']:.2f}s {out.text!r}", flush=True)
    print(f"[{a.exp}] shard {si} done", flush=True)


if __name__ == "__main__":
    main()
