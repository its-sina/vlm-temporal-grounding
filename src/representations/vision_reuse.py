"""Per-frame visual encoding for overlapping temporal windows.

Overlapping windows contain the same frame several times. Instead of encoding every grid image,
the frames can be encoded once and their merged tokens placed into the cell positions of each
window (``iso_encode``). This works with any Qwen-VL vision tower and is the encoding used by
ContiGrid.

Structural facts of the Qwen2-VL vision tower that make this exact:

1. With cells aligned to the 2 x 2 merge blocks, each frame owns a fixed token block (see
   ``token_layout.cell_token_index``).
2. ``patch_embed`` (Conv3d with kernel == stride) and the patch-merger MLP act per patch / per
   2 x 2 block, independent of position.
3. Vision RoPE is relative, so attention restricted to one frame gives the same output wherever
   the frame sits in the grid.

``hybrid_encode`` interpolates between joint grid encoding and per-frame encoding on Qwen2-VL:
patch embedding and the first ``l`` ViT blocks run once per unique frame with frame-local
attention, the remaining blocks run per window with full grid attention.
``l = 0`` reproduces the standard grid encoding and ``l = depth`` per-frame encoding.

``boundary_bridge_encode`` is an ablation for non-overlapping windows: boundary frames take their
features from small jointly encoded grids that straddle each window boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from src.representations.grid import GridConfig, GridLayout, Window, compose
from src.representations.token_layout import (cell_patch_index, cell_token_index, merged_token_px,
                                              patch_size)

# The image processor rejects aspect ratios above 200: stack at most this many frames per call.
_MAX_STACKED_FRAMES = 128


def _cu_seqlens(lengths: list[int], device) -> torch.Tensor:
    return F.pad(torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))


def _vision_rope(visual, thw: torch.Tensor):
    rope = visual.rot_pos_emb(thw)
    emb = torch.cat((rope, rope), dim=-1)
    return emb.cos(), emb.sin()


def _run_blocks(visual, h, cu_seqlens, pos_emb, lo, hi):
    for blk in visual.blocks[lo:hi]:
        h = blk(h, cu_seqlens=cu_seqlens, position_embeddings=pos_emb)
    return h


class FrameEncoder:
    """Encodes unique frames (plus one black padding frame) through patch_embed + blocks[:l]."""

    def __init__(self, runner):
        self.runner = runner
        self.visual = runner.model.model.visual
        self.image_processor = runner.processor.image_processor

    def pixel_values(self, imgs: list[Image.Image]):
        """Pixel values of equally sized frames in one processor call.

        Frames stacked vertically (width = one frame) yield each frame's patches contiguously and in
        its own patch order, so the result is identical to processing every frame separately while
        avoiding the per-image processor overhead. Falls back to per-image processing otherwise.
        """
        w, h = imgs[0].size
        if len(imgs) > _MAX_STACKED_FRAMES:
            parts = [self.pixel_values(imgs[i:i + _MAX_STACKED_FRAMES])
                     for i in range(0, len(imgs), _MAX_STACKED_FRAMES)]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        unit, patch = merged_token_px(), patch_size()
        if all(im.size == (w, h) for im in imgs) and w % unit == 0 and h % unit == 0:
            stack = Image.new("RGB", (w, h * len(imgs)))
            for i, im in enumerate(imgs):
                stack.paste(im, (0, h * i))
            out = self.image_processor(images=[stack], return_tensors="pt")
            pv = out["pixel_values"]
            if pv.shape[0] == len(imgs) * (w // patch) * (h // patch):   # not resized by max_pixels
                thw = torch.tensor([[1, h // patch, w // patch]] * len(imgs))
                return pv.to(self.runner.device), thw.to(self.runner.device)
        out = self.image_processor(images=imgs, return_tensors="pt")
        return out["pixel_values"].to(self.runner.device), out["image_grid_thw"].to(self.runner.device)

    @torch.no_grad()
    def frames_to_depth(self, imgs, depth: int):
        """Run patch_embed and the first ``depth`` blocks with frame-local attention.

        imgs: list of PIL frames, or a precomputed (pixel_values, grid_thw) tuple.
        Returns [n_frames, patches, D].
        """
        pv, thw = imgs if isinstance(imgs, tuple) else self.pixel_values(imgs)
        n_img = int(thw.shape[0])
        v = self.visual
        h = v.patch_embed(pv.type(v.dtype))
        n_per = int(thw[0].prod())
        if depth > 0:
            h = _run_blocks(v, h, _cu_seqlens([n_per] * n_img, h.device), _vision_rope(v, thw), 0, depth)
        return h.view(n_img, n_per, -1)


def prepare_unique_frames(runner, frames: list[Image.Image], layout: GridLayout):
    """CPU preprocessing for per-frame encoding: pixel values of each unique frame (+ a black cell
    if the layout has padding). Kept outside the timed vision stage, like grid preprocessing."""
    cfg = layout.cfg
    needs_black = any(c is None for w in layout.windows for c in w.cells)
    uniq = sorted({f for w in layout.windows for f in w.frames})
    imgs = [frames[f] for f in uniq] + ([Image.new("RGB", cfg.thumb)] if needs_black else [])
    return uniq, FrameEncoder(runner).pixel_values(imgs)


@torch.no_grad()
def iso_encode(runner, frames: list[Image.Image], layout: GridLayout,
               prepared=None) -> tuple[list[torch.Tensor], dict]:
    """Per-frame encoding: every unique frame (and the black cell) is encoded as its own image by the
    model's standard vision path; its merged tokens are scattered into the cell positions of each
    overlapping window. On Qwen2-VL this equals ``hybrid_encode`` with ``l = depth``."""
    uniq, (pv, thw) = prepared if prepared is not None else prepare_unique_frames(runner, frames, layout)
    per_frame = runner.encode_images(pv, thw)               # list of [t, D]
    slot = {f: i for i, f in enumerate(uniq)}
    black = len(uniq)
    tok = cell_token_index(layout.cfg).to(per_frame[0].device)   # [k, t]
    tpw = tok.numel()
    out = []
    for w in layout.windows:
        g = torch.empty(tpw, per_frame[0].shape[-1], device=per_frame[0].device, dtype=per_frame[0].dtype)
        for ci, f in enumerate(w.cells):
            g[tok[ci]] = per_frame[slot[f] if f is not None else black]
        out.append(g)
    return out, {"n_unique_frames": len(uniq)}


@torch.no_grad()
def hybrid_encode(runner, frames: list[Image.Image], layout: GridLayout, depth: int,
                  prepared=None) -> tuple[list[torch.Tensor], dict]:
    """Per-window embeddings with the first ``depth`` ViT blocks shared per frame (Qwen2-VL only)."""
    assert patch_size() == 14, "hybrid_encode uses Qwen2-VL ViT internals; use iso_encode on other models"
    cfg = layout.cfg
    v = runner.model.model.visual
    enc = FrameEncoder(runner)
    uniq, pv = prepared if prepared is not None else prepare_unique_frames(runner, frames, layout)
    cached = enc.frames_to_depth(pv, depth)  # [U(+1), P, D]
    slot = {f: i for i, f in enumerate(uniq)}
    black_slot = len(uniq)
    pidx = cell_patch_index(cfg).to(cached.device)  # [k, P]
    n_patches = pidx.shape[1]
    nwin = len(layout.windows)
    per_window = cfg.k * n_patches
    gh, gw = cfg.rows * cfg.thumb[1] // 14, cfg.cols * cfg.thumb[0] // 14
    grid = torch.empty(nwin, per_window, cached.shape[-1], device=cached.device, dtype=cached.dtype)
    for wi, w in enumerate(layout.windows):
        for ci, f in enumerate(w.cells):
            grid[wi, pidx[ci]] = cached[slot[f] if f is not None else black_slot]
    h = grid.view(nwin * per_window, -1)
    if depth < len(v.blocks):
        thw = torch.tensor([[1, gh, gw]] * nwin, device=h.device)
        h = _run_blocks(v, h, _cu_seqlens([per_window] * nwin, h.device), _vision_rope(v, thw), depth, len(v.blocks))
    out = v.merger(h)
    stats = {"vit_tokens_shared": int(cached.shape[0] * cached.shape[1]) if depth > 0 else 0,
             "vit_tokens_window": nwin * per_window if depth < len(v.blocks) else 0,
             "n_unique_frames": len(uniq)}
    return list(out.split(per_window // 4)), stats


# --------------------------------------------------------------------------- boundary bridging
def bridge_windows(n_frames: int, k: int, half: int) -> list[list[int | None]]:
    """For every boundary b = k, 2k, ... < n_frames of the non-overlapping tiling, the frames
    [b - half, b + half) (clipped; missing cells are None = black)."""
    out = []
    b = k
    while b < n_frames:
        out.append([f if 0 <= f < n_frames else None for f in range(b - half, b + half)])
        b += k
    return out


def prepare_boundary_bridge(runner, frames, layout: GridLayout, bcols: int = 3, brows: int = 2):
    """CPU preprocessing: pixel values of the bridging grids (bcols x brows cells each)."""
    cfg = layout.cfg
    half = bcols * brows // 2
    bcfg = GridConfig(bcols, brows, cfg.fps, bcols * brows, cfg.thumb)
    wins = bridge_windows(layout.n_frames, cfg.k, half)
    if not wins:
        return bcfg, wins, None
    imgs = [compose(Window(i, cells, 0, 0), frames, bcfg) for i, cells in enumerate(wins)]
    ip = runner.processor.image_processor(images=imgs, return_tensors="pt")
    return bcfg, wins, (ip["pixel_values"].to(runner.device), ip["image_grid_thw"].to(runner.device))


@torch.no_grad()
def boundary_bridge_encode(runner, inputs, layout: GridLayout, prepared) -> tuple[list[torch.Tensor], dict]:
    """Encode the non-overlapping grids as usual plus one small bridging grid per window boundary;
    boundary-adjacent frames take their features from the bridging grid."""
    cfg = layout.cfg
    assert cfg.stride == cfg.k, "boundary bridging expects the non-overlapping layout (stride == k)"
    bcfg, wins, bpv = prepared
    base = runner.encode_images(inputs["pixel_values"], inputs["image_grid_thw"])
    if not wins:
        return base, {"n_bridges": 0}
    btok = cell_token_index(bcfg).to(base[0].device)
    tok = cell_token_index(cfg).to(base[0].device)
    bemb = runner.encode_images(*bpv)
    replacement = {}
    for cells, e in zip(wins, bemb):
        for ci, f in enumerate(cells):
            if f is not None:
                replacement[f] = e[btok[ci]]
    out = []
    for w, e in zip(layout.windows, base):
        e = e.clone()
        for ci, f in enumerate(w.cells):
            if f is not None and f in replacement:
                e[tok[ci]] = replacement[f]
        out.append(e)
    return out, {"n_bridges": len(wins), "bridge_tokens": int(sum(x.shape[0] for x in bemb))}
