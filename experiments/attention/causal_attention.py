"""Causal interventions on inter-frame delimiter tokens (temporal grounding accuracy).

Every condition feeds the LLM the same per-frame sequence; conditions differ only in the attention
mask (prefill and decoding), the delimiter embeddings (R*) or the delimiters' value vectors (V*).
See src/attention/interventions.py for the condition definitions. Answers are greedy, parsed as frame
indices at 1 fps. Output: one directory per condition, <output-dir>/<prefix>_<cond>/pred_shard{i}.jsonl.

  python experiments/attention/causal_attention.py --model-kind qwen2 --prefix causal_qwen2 --n 500 \
      --conds A,A_nodelim,Rsp,V0.5 --shard 0/2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.attention.interventions import (CONDS, REPLACEMENT_TEXT, allowed, install_value_scaling,  # noqa: E402
                                         text_blocks_delimiters, value_scale_of)
from src.attention.sequence import MODEL_PRESETS, build_frame_sequence, load_runner  # noqa: E402
from src.data.annotations import dev_subset, load_samples  # noqa: E402
from src.data.frames import FrameSource  # noqa: E402
from src.evaluation.metrics import compute_iou  # noqa: E402
from src.grounding.parsing import parse_timestamps  # noqa: E402
from src.representations.grid import GridConfig  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.io import parse_shard, write_run_config  # noqa: E402

DEFAULT_CONDS = CONDS + list(REPLACEMENT_TEXT) + ["V0", "V0.5", "V2"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-kind", choices=list(MODEL_PRESETS), required=True)
    ap.add_argument("--model", help="checkpoint directory or Hub id (default: preset for --model-kind)")
    ap.add_argument("--prefix", required=True, help="output directory prefix")
    ap.add_argument("--n", type=int, default=500, help="deterministic subset size")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--conds", default=",".join(CONDS), help=f"comma-separated, e.g. {','.join(DEFAULT_CONDS)}")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dataset", default="charades")
    ap.add_argument("--annotations")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--frame-cache-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    conds = a.conds.split(",")
    for c in conds:
        if c not in CONDS and c not in REPLACEMENT_TEXT and value_scale_of(c) is None:
            raise SystemExit(f"unknown condition {c!r}")
    si, sn = parse_shard(a.shard)
    default_model, thumb = MODEL_PRESETS[a.model_kind]
    cfg = GridConfig(4, 3, 1.0, 7, (thumb, thumb))
    data_root = a.data_dir or paths.data_dir()
    annotations = Path(a.annotations) if a.annotations else paths.default_annotation_file(a.dataset, "test", data_root)
    get_frames = FrameSource(a.frame_cache_dir or paths.frame_cache_dir(a.dataset, 1.0, thumb),
                             a.video_dir or paths.default_video_dir(a.dataset, data_root), 1.0, thumb)
    out_root = a.output_dir or paths.results_dir()

    r = load_runner(a.model_kind, a.model or default_model)
    lm = r.model.model.language_model
    lm.config._attn_implementation = "sdpa"     # custom 4-D masks (FlashAttention cannot take them); ViT unchanged
    emb_layer = lm.get_input_embeddings()
    neg = torch.finfo(torch.bfloat16).min
    value_state = install_value_scaling(lm)     # inactive unless a V<alpha> condition sets it (prefill only)
    tokenizer = r.processor.tokenizer
    replacement_ids = {}
    for name, text in REPLACEMENT_TEXT.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError((text, ids))
        replacement_ids[name] = ids[0]

    samples = dev_subset(load_samples(annotations), a.n)[si::sn]
    outs, done = {}, None
    for c in conds:
        d = out_root / f"{a.prefix}_{c}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"pred_shard{si}.jsonl"
        ids = {json.loads(line)["uid"] for line in open(p)} if p.exists() else set()
        done = ids if done is None else done & ids
        if si == 0:
            write_run_config(d, vars(a), cond=c)
        outs[c] = open(p, "a")
    todo = [s for s in samples if s.uid not in done]
    print(f"[{a.prefix}] shard {si}/{sn}: {len(samples)} samples, {len(todo)} to do", flush=True)

    @torch.no_grad()
    def generate(cond, seq):
        emb, pos, inner = seq.emb, seq.pos, seq.inner
        ok = allowed(cond, inner, seq.vis, seq.fid, seq.win)
        mask = torch.zeros(ok.shape, dtype=emb.dtype, device=emb.device).masked_fill_(~ok, neg)[None, None]
        kw = ({"visual_pos_masks": seq.vis[None], "deepstack_visual_embeds": seq.extra["ds"]}
              if "ds" in seq.extra else {})
        from transformers import DynamicCache

        cache = DynamicCache()
        if cond in replacement_ids:      # embedding lookup only; same positions, causal mask
            emb = emb.clone()
            emb[0, inner] = emb_layer(torch.tensor([replacement_ids[cond]], device=emb.device)).to(emb.dtype)
        alpha = value_scale_of(cond)
        if alpha is not None:
            value_state["pos"], value_state["alpha"] = torch.nonzero(inner).flatten(), alpha
        try:
            out = lm(inputs_embeds=emb, position_ids=pos, attention_mask=mask, past_key_values=cache,
                     use_cache=True, **kw)
        finally:
            value_state["pos"] = None
        tok = int(r.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
        gen, length = [tok], emb.shape[1]
        nxt = int(pos.max()) + 1
        key_ok = torch.ones(length, dtype=torch.bool, device=emb.device)
        if text_blocks_delimiters(cond):     # generated tokens are text queries
            key_ok[inner] = False
        while len(gen) < a.max_new_tokens and tok not in r.eos:
            e = emb_layer(torch.tensor([[tok]], device=emb.device))
            p = (torch.full((3, 1, 1), nxt, device=emb.device, dtype=pos.dtype) if pos.dim() == 3
                 else torch.tensor([[nxt]], device=emb.device))
            key_ok = torch.cat([key_ok, torch.ones(1, dtype=torch.bool, device=emb.device)])
            m = torch.zeros(1, 1, 1, length + 1, dtype=emb.dtype, device=emb.device).masked_fill_(
                ~key_ok[None, None, None], neg)
            o = lm(inputs_embeds=e, position_ids=p, attention_mask=m, past_key_values=cache, use_cache=True,
                   cache_position=torch.tensor([length], device=emb.device))
            length += 1
            nxt += 1
            tok = int(r.model.lm_head(o.last_hidden_state[:, -1]).argmax(-1))
            gen.append(tok)
        return r.processor.batch_decode([gen], skip_special_tokens=True)[0], len(gen)

    t0 = time.time()
    for i, s in enumerate(todo):
        frames, dur = get_frames(s.vid)
        seq = build_frame_sequence(r, frames, dur, s.query, cfg)
        for c in conds:
            t1 = time.perf_counter()
            text, n_gen = generate(c, seq)
            pred = parse_timestamps(text)
            rec = {"uid": s.uid, "vid": s.vid, "query": s.query, "gt": s.gt, "duration": s.duration, "ans_txt": text,
                   "pred": [float(x) for x in pred], "iou": compute_iou(pred, s.gt), "cond": c,
                   "n_input_tokens": int(len(seq.ids)), "n_visual_tokens": int(seq.vis.sum()),
                   "n_inner_delim": int(seq.inner.sum()), "n_generated": n_gen, "t_model": time.perf_counter() - t1}
            outs[c].write(json.dumps(rec) + "\n")
            outs[c].flush()
        if i % 25 == 0:
            print(f"[{a.prefix}] {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    for f in outs.values():
        f.close()
    print(f"[{a.prefix}] shard {si} done", flush=True)


if __name__ == "__main__":
    main()
