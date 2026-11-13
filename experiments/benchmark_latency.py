"""Isolated latency and memory benchmark (one GPU, batch size 1, the same samples for every method).

Per sample and method (CUDA-synchronised):
  t_video   video decode + sampling + resize (moviepy); shared by all methods
  t_prep    CPU preprocessing. Per-frame methods process only the unique frames and build the token
            sequence directly (no composed grid pixels).
  t_vision, t_prefill, t_decode   GPU stages
  e2e       t_video + t_prep + t_vision + t_prefill + t_decode
plus peak GPU memory, LLM visual tokens and the answer IoU (sanity check).

  python experiments/benchmark_latency.py --annotations data/charades/annotations/test.json \
      --video-dir data/charades/videos --n 300 --out results/latency_qwen2vl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.annotations import dev_subset, find_video, load_samples  # noqa: E402
from src.data.frames import load_frames  # noqa: E402
from src.evaluation.metrics import compute_iou  # noqa: E402
from src.grounding.parsing import frames_to_seconds, parse_timestamps  # noqa: E402
from src.grounding.prompts import build_frameseq_messages, build_grid_messages  # noqa: E402
from src.models.qwen_vl import QwenVLRunner  # noqa: E402
from src.representations.grid import GridConfig, compose, make_layout  # noqa: E402
from src.representations.token_layout import dedup_keep_fn  # noqa: E402
from src.representations.vision_reuse import iso_encode, prepare_unique_frames  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.io import git_revision, write_jsonl  # noqa: E402

THUMB = 336
METHODS = {  # name: (base, stride, dedup policy or None)
    "t2sgrid_s12": ("full", 12, None),
    "t2sgrid_s7": ("full", 7, None),
    "t2sgrid_s6": ("full", 6, None),
    "dedup_last_pad_s7": ("full", 7, "last+pad"),
    "per_frame_s12": ("iso", 12, None),
    "dedup_grid_s7": ("iso", 7, "last+pad"),
    "dedup_grid_s6": ("iso", 6, "last+pad"),
    "frameseq_s7": ("frameseq", 7, None),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("VTG_MODEL", "Qwen/Qwen2-VL-7B-Instruct"))
    ap.add_argument("--dataset", default="charades")
    ap.add_argument("--annotations", help="default: <data-dir>/<dataset>/annotations/test.json")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", type=Path, help="default: <results>/latency")
    a = ap.parse_args(argv)
    data_root = a.data_dir or paths.data_dir()
    annotations = Path(a.annotations) if a.annotations else paths.default_annotation_file(a.dataset, "test", data_root)
    video_dir = a.video_dir or paths.default_video_dir(a.dataset, data_root)
    out_dir = a.out or paths.results_dir() / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = QwenVLRunner(a.model)
    tok = r.processor.tokenizer
    samples = dev_subset(load_samples(annotations), a.n)

    # shared video decoding cost, measured once per video
    vcache = {}
    for s in samples:
        if s.vid not in vcache:
            t0 = time.perf_counter()
            fr, dur = load_frames(find_video(video_dir, s.vid), 1.0, (THUMB, THUMB))
            vcache[s.vid] = (fr, dur, time.perf_counter() - t0)

    def build_inputs_direct(messages, grid_thws, pv):
        """Token sequence without running the image processor on composed grids (same text as the processor)."""
        text = r.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pad = "<|image_pad|>"
        parts = text.split(pad)
        assert len(parts) == len(grid_thws) + 1
        m2 = r.merge ** 2
        text = "".join(p + pad * (int(t.prod()) // m2) for p, t in zip(parts[:-1], grid_thws)) + parts[-1]
        enc = tok([text], return_tensors="pt")
        return {"input_ids": enc["input_ids"].to(r.device), "attention_mask": enc["attention_mask"].to(r.device),
                "image_grid_thw": torch.stack(list(grid_thws)).to(r.device), "pixel_values": pv}

    def one(s, name, check=False):
        base, stride, dedup = METHODS[name]
        cfg = GridConfig(4, 3, 1.0, stride, (THUMB, THUMB))
        frames, dur, _ = vcache[s.vid]
        t0 = time.perf_counter()
        layout = make_layout(len(frames), dur, cfg)
        vision_fn, keep_fn = None, None
        if base == "full":
            grids = [compose(w, frames, cfg) for w in layout.windows]
            inp = r.prepare(build_grid_messages(layout, grids, s.query))
        elif base == "frameseq":
            inp = r.prepare(build_frameseq_messages(layout, frames, s.query, "last", "window"))
        else:  # per-frame: process unique frames only, token ids built directly for the grid layout
            uniq, (pv, thw) = prepare_unique_frames(r, frames, layout)
            gthw = torch.tensor([1, cfg.rows * THUMB // 14, cfg.cols * THUMB // 14])
            msgs = build_grid_messages(layout, [None] * len(layout.windows), s.query)
            inp = build_inputs_direct(msgs, [gthw] * len(layout.windows), pv)
            if check:  # token ids must equal the processor's (one-time check)
                ref = r.prepare(build_grid_messages(layout, [compose(w, frames, cfg) for w in layout.windows], s.query))
                assert torch.equal(ref["input_ids"], inp["input_ids"]), "direct tokenisation mismatch"
            prepared = (uniq, (pv, thw))

            def vision_fn(rr, ii):
                return iso_encode(rr, frames, layout, prepared)[0]
        if dedup:
            keep_fn = dedup_keep_fn(layout, r.image_token_id, dedup)
        t_prep = time.perf_counter() - t0
        out = r.run(inp, vision_fn=vision_fn, keep_fn=keep_fn, t_preprocess=t_prep)
        pred = frames_to_seconds(parse_timestamps(out.text), 1.0)
        return {"uid": s.uid, "method": name, "iou": compute_iou(pred, s.gt), "ans_txt": out.text,
                "t_video": vcache[s.vid][2], "t_prep": t_prep, "t_vision": out.t_vision, "t_prefill": out.t_prefill,
                "t_decode": out.t_decode, "n_generated": out.n_generated, "n_visual_tokens": out.n_visual_tokens,
                "peak_mem_gb": out.peak_mem_gb}

    names = a.methods.split(",")
    meta = {"model": a.model, "n": a.n, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "git": git_revision(), "time": time.strftime("%F %T"), "python": platform.python_version()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    recs = []
    for name in names:
        for i, s in enumerate(samples[:a.warmup]):
            one(s, name, check=(i == 0))
        for s in samples:
            recs.append(one(s, name))
        print(f"done {name}", flush=True)
    write_jsonl(out_dir / "per_sample.jsonl", recs)

    def aggregate(name):
        rows = [x for x in recs if x["method"] == name]
        g = lambda k: np.array([x[k] for x in rows])  # noqa: E731
        model = g("t_vision") + g("t_prefill") + g("t_decode")
        e2e = model + g("t_prep") + g("t_video")
        return {"method": name, "n": len(rows), "mIoU": 100 * g("iou").mean(),
                "vis_tokens": g("n_visual_tokens").mean(),
                "t_video": g("t_video").mean(), "t_prep": g("t_prep").mean(), "t_vision": g("t_vision").mean(),
                "t_prefill": g("t_prefill").mean(), "t_decode": g("t_decode").mean(),
                "t_model": model.mean(), "t_model_std": model.std(), "t_model_p50": float(np.median(model)),
                "e2e": e2e.mean(), "e2e_std": e2e.std(), "peak_mem_gb": g("peak_mem_gb").max(),
                "n_generated": g("n_generated").mean()}

    rows = [aggregate(n) for n in names]
    ref_name = "t2sgrid_s7" if "t2sgrid_s7" in names else names[0]
    ref = next(x for x in rows if x["method"] == ref_name)
    for x in rows:
        x["speedup_model"] = ref["t_model"] / x["t_model"]
        x["speedup_e2e"] = ref["e2e"] / x["e2e"]
        print(f"{x['method']:>20}  mIoU {x['mIoU']:6.2f}  tokens {x['vis_tokens']:7.0f}  "
              f"model {x['t_model']:.3f}s ({x['speedup_model']:.2f}x)  e2e {x['e2e']:.2f}s ({x['speedup_e2e']:.2f}x)  "
              f"mem {x['peak_mem_gb']:.1f} GB")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
