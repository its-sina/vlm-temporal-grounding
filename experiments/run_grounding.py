"""Temporal grounding evaluation on Qwen-VL models (resumable and shardable).

Methods (``--method BASE[+dedup]``):

  seq        native video input
  full       grid images, standard joint encoding (T2SGrid)
  iso        per-frame encoding: every unique frame encoded alone, tokens placed into the grid cells
  hybrid     first ``--depth`` ViT blocks per frame, remaining blocks per grid (Qwen2-VL only;
             depth 0 == full, depth 32 == iso)
  frameseq   no grid: each frame its own image, shown once, with delimiter tokens between frames
  bbridge    non-overlapping grids + bridging grids at window boundaries (requires stride == cols*rows)
  +dedup     drop repeated frame copies (``--dedup first|last|center``, ``+pad`` drops padding cells)

ContiGrid is ``--method iso+dedup --dedup last+pad`` with overlapping windows (see configs/).

Examples:
  python experiments/run_grounding.py --config configs/contigrid_qwen2vl_charades.yaml --exp contigrid
  python experiments/run_grounding.py --config configs/baselines/t2sgrid_qwen2vl_charades.yaml --exp t2sgrid --shard 0/4

Frames come from the cache written by experiments/extract_frames.py (or are decoded on the fly when
``--video-dir`` is given). Output: <output-dir>/<exp>/pred_shard{i}.jsonl and config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.annotations import find_video, load_samples, select_samples  # noqa: E402
from src.data.frames import FrameSource  # noqa: E402
from src.grounding.parsing import frames_to_seconds, parse_timestamps, parse_timestamps_float  # noqa: E402
from src.grounding.prompts import build_frameseq_messages, build_grid_messages, build_seq_messages  # noqa: E402
from src.evaluation.metrics import compute_iou  # noqa: E402
from src.models.qwen_vl import QwenVLRunner, RunOutput  # noqa: E402
from src.representations import token_layout, vision_reuse  # noqa: E402
from src.representations.grid import GridConfig, build_grids, make_layout, use_frame_index_labels  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.config import parse_args_with_config  # noqa: E402
from src.utils.io import parse_shard, read_jsonl_tolerant, write_run_config  # noqa: E402

BASE_METHODS = ("seq", "full", "iso", "hybrid", "frameseq", "bbridge")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML file with default argument values")
    ap.add_argument("--exp", required=True, help="run name (output subdirectory)")
    ap.add_argument("--model", default=os.environ.get("VTG_MODEL", "Qwen/Qwen2-VL-7B-Instruct"),
                    help="local checkpoint directory or Hugging Face Hub id")
    # data
    ap.add_argument("--dataset", default="charades", help="dataset name (used for default paths)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--annotations", help="annotation JSON (default: <data-dir>/<dataset>/annotations/<split>.json)")
    ap.add_argument("--data-dir", type=Path, help="dataset root (default: $VTG_DATA_DIR or ./data)")
    ap.add_argument("--video-dir", type=Path, help="video directory (default: <data-dir>/<dataset>/videos)")
    ap.add_argument("--frame-cache-dir", type=Path, help="decoded-frame cache (default: under $VTG_CACHE_DIR)")
    ap.add_argument("--output-dir", type=Path, help="results root (default: $VTG_RESULTS_DIR or ./results)")
    ap.add_argument("--subset", type=int, default=0, help="deterministic subset size (0 = full split)")
    ap.add_argument("--exclude-vids", help="file of video ids to drop after subset selection")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th sample starting at i")
    # representation
    ap.add_argument("--method", default="full", help=f"one of {BASE_METHODS}, optionally with +dedup")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--stride", type=int, default=7, help="window stride in frames (< cols*rows: overlap)")
    ap.add_argument("--fps", type=float, default=1.0, help="frame sampling rate")
    ap.add_argument("--thumb", type=int, default=336,
                    help="cell size in px (Qwen3-VL: a multiple of 32, e.g. 320, for cell-aligned methods)")
    ap.add_argument("--depth", type=int, default=0, help="hybrid: number of ViT blocks run per frame")
    ap.add_argument("--dedup", default="first", help="first|last|center[+pad] (also the frameseq policy)")
    ap.add_argument("--bridge-grid", default="3x2", help="bbridge: bridging grid COLSxROWS")
    ap.add_argument("--labels", default="upstream", choices=["upstream", "frames"],
                    help="upstream: partial-window label in seconds (T2SGrid); frames: every label a frame index")
    ap.add_argument("--pos", default="grid", choices=["grid", "flatcells", "cellmajor", "cellmajor_gap"],
                    help="M-RoPE layout: grid | flatcells (each kept cell its own 2D block) | cellmajor "
                         "(flatcells + tokens reordered cell by cell) | cellmajor_gap (cellmajor + 2 unused "
                         "positions between frames, i.e. per-frame-sequence positions without delimiter tokens)")
    ap.add_argument("--label", default="window", choices=["window", "frame"], help="frameseq: label style")
    ap.add_argument("--sep", default="keep", choices=["keep", "neutral"],
                    help="frameseq: keep the inner delimiter pairs | neutral: replace their ids by newline "
                         "tokens for the embedding lookup only (same positions and token count)")
    ap.add_argument("--vfps", type=float, default=1.0, help="seq: video sampling fps passed to qwen-vl-utils")
    ap.add_argument("--prompt", default="numpro", choices=["numpro", "mo"],
                    help="numpro: frame numbers (seq: fps 1 only) | mo: times in seconds")
    # generation
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2, help="untimed warm-up samples (not recorded)")
    ap.add_argument("--hf-generate", action="store_true",
                    help="diagnostic: decode with model.generate and the checkpoint generation config "
                         "(method full only, seeded per sample; timings not comparable)")
    return ap


def resolve_paths(a) -> None:
    data_root = a.data_dir or paths.data_dir()
    a.annotations = (Path(a.annotations) if a.annotations
                     else paths.default_annotation_file(a.dataset, a.split, data_root))
    a.video_dir = a.video_dir or paths.default_video_dir(a.dataset, data_root)
    a.frame_cache_dir = a.frame_cache_dir or paths.frame_cache_dir(a.dataset, a.fps, a.thumb)
    a.output_dir = a.output_dir or paths.results_dir()


class Evaluator:
    def __init__(self, a, runner: QwenVLRunner):
        self.a = a
        self.r = runner
        self.cfg = GridConfig(a.cols, a.rows, a.fps, a.stride, (a.thumb, a.thumb))
        self.base, _, extra = a.method.partition("+")
        if self.base not in BASE_METHODS or extra not in ("", "dedup"):
            raise ValueError(f"unknown method {a.method!r}")
        self.dedup = extra == "dedup"
        if a.prompt == "mo" and self.base != "seq" and a.labels != "frames":
            raise ValueError("time prompts on grids need unit-consistent labels (--labels frames)")
        if a.sep == "neutral" and self.base != "frameseq":
            raise ValueError("--sep neutral applies to --method frameseq only")
        self.frames = FrameSource(a.frame_cache_dir, a.video_dir, a.fps, a.thumb)

    def layout_for(self, frames, duration):
        layout = make_layout(len(frames), duration, self.cfg)
        return use_frame_index_labels(layout) if self.a.labels == "frames" else layout

    def __call__(self, s) -> tuple[RunOutput, dict]:
        a, r, base = self.a, self.r, self.base
        t0 = time.perf_counter()
        if base == "seq":
            video = find_video(a.video_dir, s.vid)
            if video is None:
                raise FileNotFoundError(f"video {s.vid!r} not found in {a.video_dir}")
            inp = r.prepare(build_seq_messages(str(video), s.query, a.vfps, a.prompt))
            tp = time.perf_counter() - t0
            info = {"video_grid_thw": inp["video_grid_thw"][0].tolist()}
            return r.run(inp, max_new_tokens=a.max_new_tokens, t_preprocess=tp), info

        frames, dur = self.frames(s.vid)
        t_io = time.perf_counter() - t0
        t0 = time.perf_counter()
        if base == "frameseq":
            layout = self.layout_for(frames, dur)
            policy = a.dedup.replace("+pad", "")
            inp = r.prepare(build_frameseq_messages(layout, frames, s.query, policy, a.label))
        else:
            layout, grids = build_grids(frames, dur, self.cfg)
            if a.labels == "frames":
                use_frame_index_labels(layout)
            inp = r.prepare(build_grid_messages(layout, grids, s.query, time_prompt=(a.prompt == "mo")))
            if a.hf_generate:
                if base != "full" or self.dedup:
                    raise ValueError("--hf-generate supports --method full only")
                torch.manual_seed(zlib.crc32(s.uid.encode()))
                nv = int((inp["input_ids"] == r.image_token_id).sum())
                out = RunOutput(text=r.generate_reference(inp, a.max_new_tokens),
                                n_input_tokens=int(inp["input_ids"].shape[1]),
                                n_visual_tokens=nv, n_visual_tokens_encoded=nv)
                return out, {"t_io": t_io, "n_windows": len(layout.windows), "n_frames": layout.n_frames}

        prepared = None
        if base in ("hybrid", "iso"):
            prepared = vision_reuse.prepare_unique_frames(r, frames, layout)
        elif base == "bbridge":
            bcols, brows = (int(x) for x in a.bridge_grid.lower().split("x"))
            prepared = vision_reuse.prepare_boundary_bridge(r, frames, layout, bcols, brows)
        tp = time.perf_counter() - t0
        info = {"t_io": t_io, "n_windows": len(layout.windows), "n_frames": layout.n_frames}

        vision_fn = None
        if base == "hybrid":
            def vision_fn(rr, ii):
                e, st = vision_reuse.hybrid_encode(rr, frames, layout, a.depth, prepared)
                info.update(st)
                return e
        elif base == "iso":
            def vision_fn(rr, ii):
                e, st = vision_reuse.iso_encode(rr, frames, layout, prepared)
                info.update(st)
                return e
        elif base == "bbridge":
            def vision_fn(rr, ii):
                e, st = vision_reuse.boundary_bridge_encode(rr, ii, layout, prepared)
                info.update(st)
                return e

        policy = a.dedup if self.dedup else None
        keep_fn = token_layout.dedup_keep_fn(layout, r.image_token_id, a.dedup) if self.dedup else None
        pos_fn = None
        if a.pos == "flatcells":
            pos_fn = token_layout.flatcell_pos_fn(layout, r.image_token_id, policy)
        elif a.pos in ("cellmajor", "cellmajor_gap"):
            wrap, keep_fn, pos_fn = token_layout.cellmajor_fns(layout, r.image_token_id, policy,
                                                               gap=2 if a.pos == "cellmajor_gap" else 0)
            inner_fn = vision_fn or (lambda rr, ii: rr.encode_images(ii["pixel_values"], ii["image_grid_thw"]))

            def vision_fn(rr, ii):
                return wrap(inner_fn(rr, ii))

        ids_fn = None
        if a.sep == "neutral":
            nl = r.processor.tokenizer.encode("\n", add_special_tokens=False)
            assert len(nl) == 1, nl
            inner = token_layout.inner_separator_mask(inp["input_ids"][0], r.image_token_id,
                                                      r.model.config.vision_start_token_id,
                                                      r.model.config.vision_end_token_id)
            info["n_sep_replaced"] = int(inner.sum())

            def ids_fn(ids, inner=inner, nl=nl[0]):
                ids = ids.clone()
                ids[0, inner] = nl
                return ids
        out = r.run(inp, vision_fn=vision_fn, keep_fn=keep_fn, pos_fn=pos_fn, ids_fn=ids_fn,
                    max_new_tokens=a.max_new_tokens, t_preprocess=tp)
        return out, info

    def record(self, s, out: RunOutput, info: dict) -> dict:
        """Score one answer. Grid answers are frame indices (converted to seconds); native-video and
        time-prompt answers are seconds (fractional values kept)."""
        a = self.a
        seconds_answer = self.base == "seq" or a.prompt == "mo"
        pred = parse_timestamps(out.text)
        if self.base == "seq":
            pred_s = pred
        elif a.prompt == "mo":
            pred_s = parse_timestamps_float(out.text)
        else:
            pred_s = frames_to_seconds(pred, a.fps)
        iou = compute_iou(pred_s, s.gt)
        rec = {"uid": s.uid, "vid": s.vid, "query": s.query, "gt": s.gt, "duration": s.duration,
               "ans_txt": out.text, "pred": pred_s, "iou": iou, "pred_raw": pred,
               "iou_upstream": compute_iou(pred, s.gt)}   # raw answer vs seconds, as in the reference code
        if seconds_answer:
            pred_f = parse_timestamps_float(out.text)
            rec.update(pred_float=pred_f, iou_float=compute_iou(pred_f, s.gt))
        rec.update(n_input_tokens=out.n_input_tokens, n_visual_tokens=out.n_visual_tokens,
                   n_visual_tokens_encoded=out.n_visual_tokens_encoded, n_generated=out.n_generated,
                   t_preprocess=out.t_preprocess, t_vision=out.t_vision, t_prefill=out.t_prefill,
                   t_decode=out.t_decode, t_model=out.t_model, t_total=out.t_total,
                   peak_mem_gb=out.peak_mem_gb, **info)
        return rec


def main(argv=None) -> None:
    a = parse_args_with_config(build_parser(), argv)
    resolve_paths(a)
    out_dir = Path(a.output_dir) / a.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    si, sn = parse_shard(a.shard)
    cfg = GridConfig(a.cols, a.rows, a.fps, a.stride, (a.thumb, a.thumb))
    if si == 0:
        write_run_config(out_dir, vars(a), grid=cfg.name)

    samples = select_samples(load_samples(a.annotations), a.subset, a.exclude_vids, (si, sn))
    pred_path = out_dir / f"pred_shard{si}.jsonl"
    done = {rec["uid"] for rec in read_jsonl_tolerant(pred_path)}
    todo = [s for s in samples if s.uid not in done]
    print(f"[{a.exp}] shard {si}/{sn}: {len(samples)} samples, {len(todo)} to do", flush=True)
    if not todo:
        return

    evaluator = Evaluator(a, QwenVLRunner(a.model))
    for s in todo[:a.warmup]:   # CUDA / kernel initialisation must not pollute latency
        evaluator(s)
    with open(pred_path, "a") as f:
        for i, s in enumerate(todo):
            out, info = evaluator(s)
            rec = evaluator.record(s, out, info)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[{a.exp}] {i}/{len(todo)} iou={rec['iou']:.2f} t={rec['t_model']:.2f}s {out.text!r}",
                      flush=True)
    print(f"[{a.exp}] shard {si} done", flush=True)


if __name__ == "__main__":
    main()
