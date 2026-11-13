"""Visual-token organization for grid inputs of Qwen-VL models.

With cells whose side is a multiple of ``2 * patch_size`` (e.g. 336 px for the 14-px patches of
Qwen2/2.5-VL, 320 px for the 16-px patches of Qwen3-VL), every cell maps to a fixed block of merged
visual tokens (12 x 12 = 144 tokens for 336 px on Qwen2-VL), so each frame owns a well-defined
token block inside its grid image. This module provides:

* index maps from cells to merged tokens and to ViT patches;
* token keep-masks: frame deduplication across overlapping windows (keep the first, last or most
  central copy of every frame) and removal of black padding cells;
* M-RoPE position rewrites and token reorderings used in layout ablations;
* detection of the inter-frame delimiter tokens of per-frame sequences.

All maps are in the processor's patch order: (h_block, w_block, mh, mw) with spatial merge 2.
"""

from __future__ import annotations

import torch

from src.representations.grid import GridConfig, GridLayout

_PATCH_SIZE = 14


def set_patch_size(patch: int) -> None:
    """Set the ViT patch size of the loaded model (one model per process; done by the runner)."""
    global _PATCH_SIZE
    _PATCH_SIZE = int(patch)


def patch_size() -> int:
    return _PATCH_SIZE


def merged_token_px() -> int:
    """Side length in pixels of one merged visual token (2 x 2 patches)."""
    return 2 * _PATCH_SIZE


def cell_token_index(cfg: GridConfig, merge: int = 2, patch: int | None = None) -> torch.Tensor:
    """[k, t] merged-token indices (raster order of the grid image) of each cell."""
    unit = (patch or _PATCH_SIZE) * merge
    cw, ch = cfg.thumb[0] // unit, cfg.thumb[1] // unit
    assert cfg.thumb[0] % unit == 0 and cfg.thumb[1] % unit == 0, "cells must align with merge blocks"
    width = cfg.cols * cw
    idx = []
    for ci in range(cfg.k):
        r, c = divmod(ci, cfg.cols)
        rows = torch.arange(r * ch, (r + 1) * ch)
        cols = torch.arange(c * cw, (c + 1) * cw)
        idx.append((rows[:, None] * width + cols[None, :]).reshape(-1))
    return torch.stack(idx)


def cell_patch_index(cfg: GridConfig, merge: int = 2, patch: int | None = None) -> torch.Tensor:
    """[k, t * merge^2] patch indices of each cell inside the grid's patch sequence, ordered exactly
    like the patch sequence of the same frame encoded alone."""
    tok = cell_token_index(cfg, merge, patch)
    m2 = merge * merge
    return (tok[:, :, None] * m2 + torch.arange(m2)[None, None, :]).reshape(cfg.k, -1)


# --------------------------------------------------------------------------- deduplication
DEDUP_POLICIES = ("first", "last", "center")


def visual_keep_mask(layout: GridLayout, policy: str, allow_all: bool = True) -> torch.Tensor:
    """bool [n_windows * tokens_per_window]: visual tokens kept under a deduplication policy.

    policy: ``first`` | ``last`` | ``center`` (copy whose cell is closest to the grid centre) |
    ``all`` (keep every copy; only if ``allow_all``). A ``+pad`` suffix also drops the tokens of
    black padding cells; otherwise padding is kept.
    """
    cfg = layout.cfg
    drop_pad = policy.endswith("+pad")
    policy = policy.replace("+pad", "")
    tok = cell_token_index(cfg)
    tpw = cfg.k * tok.shape[1]
    centre_r, centre_c = (cfg.rows - 1) / 2, (cfg.cols - 1) / 2

    def dist_to_centre(wc):
        r, c = divmod(wc[1], cfg.cols)
        return (r - centre_r) ** 2 + (c - centre_c) ** 2

    keep_cell: set[tuple[int, int]] = set()
    for _, lst in layout.frame_occurrences().items():
        if policy == "first":
            keep_cell.add(lst[0])
        elif policy == "last":
            keep_cell.add(lst[-1])
        elif policy == "center":
            keep_cell.add(min(lst, key=dist_to_centre))
        elif policy == "all" and allow_all:
            keep_cell.update(lst)
        else:
            raise ValueError(policy)
    keep = torch.zeros(len(layout.windows), tpw, dtype=torch.bool)
    for w in layout.windows:
        for ci, f in enumerate(w.cells):
            if (f is None and not drop_pad) or (w.index, ci) in keep_cell:
                keep[w.index, tok[ci]] = True
    return keep.reshape(-1)


def dedup_keep_fn(layout: GridLayout, image_token_id: int, policy: str = "first"):
    """Keep-function that drops the visual tokens of repeated frame copies before the LLM.

    Every frame is kept once (policy ``first``/``last``/``center``); ``+pad`` also drops padding
    cells. M-RoPE positions of the kept tokens are left unchanged, so dropped cells leave their
    positions unused. Returns ``keep_fn(input_ids, positions, embeddings) -> bool mask``.
    """
    vis_keep = visual_keep_mask(layout, policy, allow_all=False)

    def keep_fn(input_ids, pos, emb):
        keep = torch.ones_like(input_ids, dtype=torch.bool)
        keep[input_ids == image_token_id] = vis_keep.to(keep.device)
        return keep
    return keep_fn


# --------------------------------------------------------------------------- layout ablations
def flatcell_pos_fn(layout: GridLayout, image_token_id: int, policy: str | None = None):
    """Position-only ablation: keep the grid token sequence but rewrite M-RoPE positions as if every
    kept cell were a separate image placed one after another (t = base, h = base + row,
    w = base + col; next base = base + cell side), text positions continuing sequentially.

    policy: deduplication policy whose dropped cells receive no positions (None = keep all cells).
    """
    cfg = layout.cfg
    tok = cell_token_index(cfg)
    ch, cw = cfg.thumb[1] // merged_token_px(), cfg.thumb[0] // merged_token_px()
    tpw = cfg.k * tok.shape[1]
    keep = visual_keep_mask(layout, policy) if policy else torch.ones(len(layout.windows) * tpw, dtype=torch.bool)
    keep = keep.view(len(layout.windows), tpw)
    yy = torch.arange(ch).repeat_interleave(cw)
    xx = torch.arange(cw).repeat(ch)

    def pos_fn(input_ids, pos):
        ids = input_ids.cpu()
        new = torch.empty_like(pos.cpu())              # [3, L]
        nxt, i, g, length = 0, 0, 0, ids.shape[0]
        while i < length:
            if ids[i] != image_token_id:
                new[:, i] = nxt
                nxt += 1
                i += 1
                continue
            blk = torch.empty(3, tpw, dtype=new.dtype)
            for ci in range(cfg.k):
                if not keep[g, tok[ci][0]]:
                    blk[:, tok[ci]] = nxt              # token is dropped later; value irrelevant
                    continue
                blk[0, tok[ci]] = nxt
                blk[1, tok[ci]] = nxt + yy
                blk[2, tok[ci]] = nxt + xx
                nxt += max(ch, cw)
            new[:, i:i + tpw] = blk
            i += tpw
            g += 1
        return new.to(pos.device)
    return pos_fn


def cellmajor_fns(layout: GridLayout, image_token_id: int, policy: str | None = None, gap: int = 0):
    """Sequence-order ablation: reorder every grid's visual tokens from raster order (rows of
    neighbouring cells interleaved) to cell-major order (each cell's tokens contiguous), with flat
    per-cell M-RoPE positions. This is a per-frame sequence without per-frame delimiter tokens.

    gap > 0 leaves ``gap`` unused positions between consecutive kept cells of a window. With
    ``gap=2`` the positions equal those of the per-frame sequence, whose delimiter pair between two
    frames occupies two positions, so the delimiter tokens are removed while positions are held fixed.

    Returns ``(wrap_vision, keep_fn, pos_fn)``; ``wrap_vision`` reorders per-image embeddings.
    """
    cfg = layout.cfg
    tok = cell_token_index(cfg)
    perm = tok.reshape(-1)                            # cell-major order -> raster index
    tpw = perm.numel()
    nw = len(layout.windows)
    keep_r = (visual_keep_mask(layout, policy) if policy else torch.ones(nw * tpw, dtype=torch.bool)).view(nw, tpw)
    keep_c = keep_r[:, perm].reshape(-1)              # keep mask in cell-major order
    ch, cw = cfg.thumb[1] // merged_token_px(), cfg.thumb[0] // merged_token_px()
    per = ch * cw
    yy = torch.arange(ch).repeat_interleave(cw)
    xx = torch.arange(cw).repeat(ch)

    def wrap_vision(embs):
        return [e[perm.to(e.device)] for e in embs]

    def keep_fn(input_ids, pos, emb):
        keep = torch.ones_like(input_ids, dtype=torch.bool)
        keep[input_ids == image_token_id] = keep_c.to(keep.device)
        return keep

    def pos_fn(input_ids, pos):
        ids = input_ids.cpu()
        new = torch.empty_like(pos.cpu())
        nxt, i, g, length = 0, 0, 0, ids.shape[0]
        kc = keep_c.view(nw, tpw)
        while i < length:
            if ids[i] != image_token_id:
                new[:, i] = nxt
                nxt += 1
                i += 1
                continue
            first = True
            for ci in range(cfg.k):
                sl = slice(i + ci * per, i + (ci + 1) * per)
                if kc[g, ci * per] and not first:
                    nxt += gap
                new[0, sl] = nxt
                new[1, sl] = nxt + yy
                new[2, sl] = nxt + xx
                if kc[g, ci * per]:
                    nxt += max(ch, cw)
                    first = False
            i += tpw
            g += 1
        return new.to(pos.device)
    return wrap_vision, keep_fn, pos_fn


# --------------------------------------------------------------------------- delimiters
def inner_separator_mask(ids: torch.Tensor, img: int, vs: int, ve: int) -> torch.Tensor:
    """Bool mask of the ``<vision_end><vision_start>`` pairs directly between two images of the same
    window (image token, ve, vs, image token: no label text in between)."""
    m = torch.zeros_like(ids, dtype=torch.bool)
    i = torch.nonzero((ids[1:-2] == ve) & (ids[2:-1] == vs) & (ids[:-3] == img) & (ids[3:] == img)).flatten() + 1
    m[i] = True
    m[i + 1] = True
    return m
