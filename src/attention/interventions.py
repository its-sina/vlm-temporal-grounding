"""Causal interventions on inter-frame delimiter tokens.

All conditions feed the LLM the same per-frame sequence (each frame once, last-occurrence window
labels, delimiter tokens between consecutive frames of a window, original position ids). They differ
only in the attention mask, the delimiter embeddings or the delimiters' value vectors, so token count,
visual features and positions are identical across conditions.

Attention-mask conditions (``CONDS``):

  A               causal mask (no intervention)
  A_nodelim       no token may attend to inner delimiters; in a causal LM this equals deleting the
                  delimiters with the remaining positions kept
  A_nodelim_vis   only visual queries are blocked from the delimiters
  A_nodelim_text  only non-visual queries (labels, question, answer) are blocked from the delimiters
  C_nocross       A_nodelim + visual queries may not attend to other frames of the same window
  C_noprevwin     A_nodelim + visual queries may not attend to frames of earlier windows
  A_nocross       A + the same within-window cross-frame block (delimiters present)

Other conditions (causal mask of A):

  Rnl / Rsp / Rbar  every inner delimiter's input embedding replaced by that of a newline, a space
                    or a '|' token (``REPLACEMENT_TEXT``)
  V<alpha>          the delimiters' value vectors are multiplied by alpha in every layer during
                    prefill; attention weights are unchanged (``install_value_scaling``)

Self-attention (the diagonal) is never blocked.

Tensors are 1-D over the token sequence:
  inner : bool, inner delimiter tokens (between two frames of the same window)
  vis   : bool, visual tokens
  fid   : int, frame index of each visual token (-1 otherwise)
  win   : int, window index of each visual token (-1 otherwise)
"""

from __future__ import annotations

import torch

CONDS = ["A", "A_nodelim", "A_nodelim_vis", "A_nodelim_text", "C_nocross", "C_noprevwin", "A_nocross"]
REPLACEMENT_TEXT = {"Rnl": "\n", "Rsp": " ", "Rbar": "|"}


def allowed(cond: str, inner, vis, fid, win) -> torch.Tensor:
    """bool [T, T] allowed-attention matrix (row = query, column = key) for the prefill."""
    n = len(vis)
    ok = torch.ones(n, n, dtype=torch.bool, device=vis.device).tril()
    if cond in ("A_nodelim", "C_nocross", "C_noprevwin"):
        ok[:, inner] = False
    elif cond == "A_nodelim_vis":
        ok[vis[:, None] & inner[None, :]] = False
    elif cond == "A_nodelim_text":
        ok[(~vis)[:, None] & inner[None, :]] = False
    vv = vis[:, None] & vis[None, :]
    if cond in ("C_nocross", "A_nocross"):
        ok[vv & (win[:, None] == win[None, :]) & (fid[:, None] != fid[None, :])] = False
    if cond == "C_noprevwin":
        ok[vv & (win[None, :] < win[:, None])] = False
    ok.fill_diagonal_(True)
    return ok


def text_blocks_delimiters(cond: str) -> bool:
    """Whether generated (text) tokens are blocked from the inner delimiters during decoding."""
    return cond in ("A_nodelim", "A_nodelim_text", "C_nocross", "C_noprevwin")


def value_scale_of(cond: str) -> float | None:
    """'V0.5' -> 0.5; None for the other conditions."""
    return float(cond[1:]) if cond.startswith("V") else None


def install_value_scaling(lm) -> dict:
    """Register forward hooks on every decoder layer's value projection.

    While ``state["pos"]`` is set (prefill only), the value vectors at those token positions are
    multiplied by ``state["alpha"]`` in every layer. Queries, keys and the softmax are untouched, so the
    delimiters absorb exactly the same attention mass; only what they pass on is scaled.
    Returns the mutable state dict.
    """
    state = {"pos": None, "alpha": 1.0}

    def hook(mod, inp, out):
        if state["pos"] is None or out.shape[1] == 1:
            return None
        out = out.clone()
        out[:, state["pos"]] = out[:, state["pos"]] * state["alpha"]
        return out
    for layer in lm.layers:
        layer.self_attn.v_proj.register_forward_hook(hook)
    return state


def frame_and_window_ids(vis: torch.Tensor, owner: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame index of every visual token (runs of visual tokens, chronological) and its window.

    owner: frame index -> window index that shows it (each frame appears once in the sequence).
    """
    fid = torch.full(vis.shape, -1, dtype=torch.long, device=vis.device)
    run, prev = -1, False
    for t in range(len(vis)):
        v = bool(vis[t])
        if v and not prev:
            run += 1
        if v:
            fid[t] = run
        prev = v
    win = torch.full_like(fid, -1)
    win[vis] = torch.tensor([owner[int(f)] for f in fid[vis]], device=vis.device, dtype=torch.long)
    return fid, win
