"""Descriptive analysis of where LLM attention goes with and without inter-frame delimiters.

For each query, two matched inputs are built:
  (a) per-frame sequence with delimiter tokens between consecutive frames of a window
  (c) the same sequence with the inner delimiters deleted, all other tokens keeping their positions
and one prefill pass with eager attention is run. A forward hook on every self-attention module reduces
the head-averaged attention weights to

  vis_self     mass a visual token puts on visual tokens of its own frame
  vis_cross    ... on visual tokens of other frames of the same window
  vis_prevwin  ... on visual tokens of frames of earlier windows
  vis_delim    ... on inner delimiter tokens ((a) only)
  vis_text     ... on all remaining tokens (text, special tokens, attention sink)

for visual queries of every frame except the first of each window, and for the last prompt token:

  ans_vis      total mass on visual tokens
  ans_gt       share of that mass on frames inside the ground-truth moment
  ans_gt_norm  ans_gt divided by the share of visual tokens inside the moment (1 = uniform)
  ans_delim    mass on inner delimiters.

Reported per layer third and over all layers, with paired bootstrap CIs for (c) - (a). Descriptive
only: it shows where attention goes, not that this causes accuracy differences.

  python experiments/attention/attention_analysis.py --model-kind qwen2 --n 150 --out results/attention_qwen2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.attention.sequence import MODEL_PRESETS, build_frame_sequence, load_runner  # noqa: E402
from src.data.annotations import dev_subset, load_samples  # noqa: E402
from src.data.frames import FrameSource  # noqa: E402
from src.representations.grid import GridConfig  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.io import write_jsonl  # noqa: E402

KEYS = ["vis_self", "vis_cross", "vis_prevwin", "vis_delim", "vis_text",
        "ans_vis", "ans_gt", "ans_gt_norm", "ans_delim"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-kind", choices=list(MODEL_PRESETS), required=True)
    ap.add_argument("--model", help="checkpoint directory or Hub id (default: preset for --model-kind)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="skip queries with longer inputs (eager attention memory); skipped queries are reported")
    ap.add_argument("--dataset", default="charades")
    ap.add_argument("--annotations")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--frame-cache-dir", type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    default_model, thumb = MODEL_PRESETS[a.model_kind]
    model_name = a.model or default_model
    cfg = GridConfig(4, 3, 1.0, 7, (thumb, thumb))
    data_root = a.data_dir or paths.data_dir()
    annotations = Path(a.annotations) if a.annotations else paths.default_annotation_file(a.dataset, "test", data_root)
    get_frames = FrameSource(a.frame_cache_dir or paths.frame_cache_dir(a.dataset, 1.0, thumb),
                             a.video_dir or paths.default_video_dir(a.dataset, data_root), 1.0, thumb)

    r = load_runner(a.model_kind, model_name)
    lm = r.model.model.language_model
    lm.config._attn_implementation = "eager"          # attention weights needed; the ViT keeps FlashAttention
    attn_mods = [mod for name, mod in lm.named_modules() if name.endswith("self_attn")]
    n_layers = len(attn_mods)
    state: dict = {}

    def make_hook(li):
        def hook(mod, args, out):
            w = out[1]                                # [1, H, T, T]
            assert w is not None, "eager attention weights missing"
            w = w[0].float().mean(0)                  # head-averaged [T, T]
            wq = w[state["vis_q"]]
            rec = {"vis_self": (wq * state["same_frame"]).sum(1), "vis_cross": (wq * state["same_win_other"]).sum(1),
                   "vis_prevwin": (wq * state["prev_win"]).sum(1), "vis_delim": wq[:, state["delim"]].sum(1)}
            rec["vis_text"] = 1 - rec["vis_self"] - rec["vis_cross"] - rec["vis_prevwin"] - rec["vis_delim"]
            state["layers"][li] = {k: float(v.mean()) for k, v in rec.items()}
            last = w[-1]
            vis_mass = float(last[state["vis"]].sum())
            gt_mass = float(last[state["vis_gt"]].sum())
            state["layers"][li].update({
                "ans_vis": vis_mass, "ans_gt": gt_mass / max(vis_mass, 1e-9),
                "ans_gt_norm": (gt_mass / max(vis_mass, 1e-9)) / max(state["gt_share"], 1e-9),
                "ans_delim": float(last[state["delim"]].sum())})
            return (out[0], None)
        return hook

    for i, mod in enumerate(attn_mods):
        mod.register_forward_hook(make_hook(i))

    @torch.no_grad()
    def analyse(s, seq, drop_inner):
        keep = ~seq.inner if drop_inner else torch.ones_like(seq.inner)
        ids, fid, win, vis, inner_k = seq.ids[keep], seq.fid[keep], seq.win[keep], seq.vis[keep], seq.inner[keep]
        e = seq.emb[:, keep]
        p = seq.pos[..., keep]
        first_of_win = torch.zeros_like(vis)
        for w_ in win[vis].unique():
            fr = fid[(win == w_) & vis].min()
            first_of_win |= (fid == fr) & vis
        q = torch.nonzero(vis & ~first_of_win).flatten()
        fq, wq = fid[q][:, None], win[q][:, None]
        keys_vis = vis[None, :]
        g0, g1 = s.gt
        gt_frames = [f for f in range(int(fid.max()) + 1) if g0 - 1 < f < g1]   # frame f = second f at 1 fps
        vis_gt = vis & torch.isin(fid, torch.tensor(gt_frames or [-5], device=ids.device))
        state.update(vis_q=q, same_frame=((fid[None, :] == fq) & keys_vis).float(),
                     same_win_other=((win[None, :] == wq) & (fid[None, :] != fq) & keys_vis).float(),
                     prev_win=((win[None, :] < wq) & keys_vis).float(), delim=inner_k, vis=vis, vis_gt=vis_gt,
                     gt_share=float(vis_gt.sum()) / max(int(vis.sum()), 1), layers=[None] * n_layers)
        kw = ({"visual_pos_masks": vis[None], "deepstack_visual_embeds": seq.extra["ds"]}
              if "ds" in seq.extra else {})
        # explicit all-ones mask: without it, transformers may infer packed sequences from non-monotonic
        # M-RoPE positions (Qwen3-VL) and build a block-diagonal mask
        lm(inputs_embeds=e, position_ids=p, attention_mask=torch.ones(1, e.shape[1], dtype=torch.long, device=e.device),
           use_cache=False, **kw)
        return state["layers"], len(ids)

    samples = dev_subset(load_samples(annotations), a.n)
    recs, skipped = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        frames, dur = get_frames(s.vid)
        seq = build_frame_sequence(r, frames, dur, s.query, cfg)
        if int(seq.inner.sum()) == 0:
            continue
        if a.max_tokens and len(seq.ids) > a.max_tokens:
            skipped.append((s.uid, int(len(seq.ids))))
            continue
        for variant, drop in (("a_sep", False), ("c_gap", True)):
            layers, n_tokens = analyse(s, seq, drop)
            recs.append({"uid": s.uid, "variant": variant, "T": n_tokens, "layers": layers})
        if i % 25 == 0:
            print(f"{i}/{len(samples)} {time.time() - t0:.0f}s", flush=True)
    write_jsonl(a.out / "per_query.jsonl", recs)

    groups = {"early": range(0, n_layers // 3), "mid": range(n_layers // 3, 2 * n_layers // 3),
              "late": range(2 * n_layers // 3, n_layers), "all": range(n_layers)}
    by: dict = {}
    for x in recs:
        for k in KEYS:
            for g, rg in groups.items():
                by.setdefault((k, g, x["variant"]), {})[x["uid"]] = float(np.mean([x["layers"][li][k] for li in rg]))
    rng = np.random.default_rng(0)
    summary = {"model": model_name, "n_queries": len({x["uid"] for x in recs}), "n_layers": n_layers,
               "metrics": {}, "skipped_long": skipped, "max_tokens": a.max_tokens}
    for k in KEYS:
        for g in groups:
            sep, gap = by[(k, g, "a_sep")], by[(k, g, "c_gap")]
            u = sorted(set(sep) & set(gap))
            av, cv = np.array([sep[x] for x in u]), np.array([gap[x] for x in u])
            d = cv - av
            bs = d[rng.integers(0, len(u), (10000, len(u)))].mean(1)
            res = {"a": float(av.mean()), "c": float(cv.mean()), "diff": float(d.mean()),
                   "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            summary["metrics"][f"{k}/{g}"] = res
            print(f"{k:>12} {g:>5}  (a) {res['a']:.4f}  (c) {res['c']:.4f}  "
                  f"(c)-(a) {res['diff']:+.4f} [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
