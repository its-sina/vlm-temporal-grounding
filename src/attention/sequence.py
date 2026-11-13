"""LLM inputs of the per-frame sequence for attention experiments (Qwen-VL and InternVL)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.attention.interventions import frame_and_window_ids
from src.grounding.prompts import build_frameseq_messages
from src.representations.grid import GridConfig, make_layout
from src.representations.token_layout import inner_separator_mask

# Model presets: default checkpoint and the cell size that aligns with its visual tokens.
MODEL_PRESETS = {
    "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", 336),
    "qwen3": ("Qwen/Qwen3-VL-8B-Instruct", 320),
    "internvl": ("OpenGVLab/InternVL3_5-8B-HF", 448),
}


def load_runner(kind: str, model_path: str):
    if kind == "internvl":
        from src.models.internvl import InternVLRunner
        return InternVLRunner(model_path)
    from src.models.qwen_vl import QwenVLRunner
    return QwenVLRunner(model_path)


@dataclass
class FrameSequence:
    ids: torch.Tensor          # [T] token ids
    emb: torch.Tensor          # [1, T, D] input embeddings with visual features inserted
    pos: torch.Tensor          # position ids ([3, 1, T] M-RoPE or [1, T])
    inner: torch.Tensor        # [T] inner delimiter mask
    vis: torch.Tensor          # [T] visual-token mask
    fid: torch.Tensor          # [T] frame index of visual tokens
    win: torch.Tensor          # [T] window index of visual tokens
    extra: dict = field(default_factory=dict)   # Qwen3-VL: {"ds": deepstack embeddings}


@torch.no_grad()
def build_frame_sequence(runner, frames, duration: float, query: str, cfg: GridConfig) -> FrameSequence:
    """Per-frame sequence (each frame once, last-occurrence window labels) with delimiters present."""
    layout = make_layout(len(frames), duration, cfg)
    owner = layout.frame_owner("last")
    msgs = build_frameseq_messages(layout, frames, query, "last", "window")
    lm = runner.model.model.language_model
    extra = {}
    if runner.model_type == "internvl":
        inp = runner.prepare(msgs, crop_to_patches=False)
        ids = inp["input_ids"][0]
        feats = runner.encode(inp["pixel_values"])
        emb = lm.get_input_embeddings()(ids[None])
        vis = ids == runner.image_token_id
        emb[0, vis] = feats.reshape(-1, feats.shape[-1]).to(emb.dtype)
        pos = torch.arange(len(ids), device=ids.device)[None]
        inner = runner.inner_separators(ids)
    else:
        inp = runner.prepare(msgs)
        ids = inp["input_ids"][0]
        ie = torch.cat(runner.encode_images(inp["pixel_values"], inp["image_grid_thw"]))
        emb = lm.get_input_embeddings()(ids[None])
        vis = ids == runner.image_token_id
        ie, deepstack = runner.split_deepstack(ie, emb.dtype)
        if deepstack is not None:
            extra["ds"] = deepstack
        emb[0, vis] = ie.to(emb.dtype)
        pos, _ = runner.model.model.get_rope_index(inp["input_ids"], image_grid_thw=inp["image_grid_thw"],
                                                   attention_mask=inp["attention_mask"])
        inner = inner_separator_mask(ids, runner.image_token_id, runner.model.config.vision_start_token_id,
                                     runner.model.config.vision_end_token_id)
    fid, win = frame_and_window_ids(vis, owner)
    n_frames_seen = int(fid.max()) + 1
    assert n_frames_seen == len(frames), (n_frames_seen, len(frames))
    return FrameSequence(ids, emb, pos, inner, vis, fid, win, extra)
