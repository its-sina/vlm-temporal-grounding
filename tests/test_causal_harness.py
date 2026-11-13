"""CPU validation of the causal attention-intervention harness (experiments/attention/causal_attention.py).

Covers: (1) detection of inner delimiters, (2) the mask rules of every condition, (3) the claim that blocking all
attention to a set of tokens is computationally the deletion of those tokens with the remaining position ids kept,
checked on a tiny randomly initialised Qwen2-VL language model with 3-D M-RoPE positions, and (4) value scaling.
Nothing here loads real weights or uses a GPU.
"""
import pytest
import torch

from src.attention.interventions import CONDS, allowed, frame_and_window_ids, text_blocks_delimiters
from src.representations.token_layout import inner_separator_mask

IMG, VS, VE, TXT, LAB = 9, 7, 8, 1, 2


def toy_sequence():
    """text | label0 | Ds f0 De Ds f1 De Ds f2 De | label1 | Ds f3 De Ds f4 De | question.  Frames have 3 tokens.
    Windows: frames 0-2 -> window 0, frames 3-4 -> window 1."""
    seq, fid, win = [], [], []

    def add(tok, f=-1, w=-1):
        seq.append(tok)
        fid.append(f)
        win.append(w)
    for tok in (TXT, TXT, LAB):
        add(tok)
    for f in range(3):
        add(VS)
        for _ in range(3):
            add(IMG, f, 0)
        add(VE)
    add(LAB)
    for f in (3, 4):
        add(VS)
        for _ in range(3):
            add(IMG, f, 1)
        add(VE)
    for _ in range(3):
        add(TXT)
    ids = torch.tensor(seq)
    return ids, ids == IMG, torch.tensor(fid), torch.tensor(win)


def test_inner_delimiters_only_between_frames_of_a_window():
    ids, vis, fid, win = toy_sequence()
    inner = inner_separator_mask(ids, IMG, VS, VE)
    pos = torch.nonzero(inner).flatten().tolist()
    # window 0: two (De, Ds) pairs between f0|f1 and f1|f2; window 1: one pair between f3|f4; none across the label
    assert len(pos) == 6
    for p in pos:
        assert int(ids[p]) in (VS, VE)
    assert all(int(ids[p]) == VE for p in pos[0::2]) and all(int(ids[p]) == VS for p in pos[1::2])
    # the outer delimiters of each window (first Ds, last De) are not inner delimiters
    first_vs = int(torch.nonzero(ids == VS)[0])
    last_ve = int(torch.nonzero(ids == VE)[-1])
    assert not inner[first_vs] and not inner[last_ve]


def test_frame_and_window_ids_match_toy_sequence():
    ids, vis, fid, win = toy_sequence()
    f2, w2 = frame_and_window_ids(vis, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
    assert torch.equal(f2, fid) and torch.equal(w2, win)


@pytest.mark.parametrize("cond", CONDS)
def test_masks_are_causal_and_keep_self_attention(cond):
    ids, vis, fid, win = toy_sequence()
    inner = inner_separator_mask(ids, IMG, VS, VE)
    ok = allowed(cond, inner, vis, fid, win)
    assert not torch.triu(ok, 1).any()                 # no attention to the future
    assert ok.diagonal().all()                          # self-attention never blocked
    if cond == "A":
        assert torch.equal(ok, torch.ones_like(ok).tril())


def test_condition_rules():
    ids, vis, fid, win = toy_sequence()
    inner = inner_separator_mask(ids, IMG, VS, VE)
    causal = torch.ones(len(ids), len(ids), dtype=torch.bool).tril()
    offdiag = ~torch.eye(len(ids), dtype=torch.bool)
    A = allowed("A", inner, vis, fid, win)
    assert torch.equal(A, causal)
    nd = allowed("A_nodelim", inner, vis, fid, win)
    assert not (nd[:, inner] & offdiag[:, inner]).any()                   # nobody else reads a delimiter
    assert torch.equal(nd[:, ~inner], causal[:, ~inner])                  # everything else untouched
    nv = allowed("A_nodelim_vis", inner, vis, fid, win)
    assert not (nv[vis][:, inner]).any()
    assert torch.equal(nv[~vis], causal[~vis])                            # text queries still read delimiters
    nt = allowed("A_nodelim_text", inner, vis, fid, win)
    assert torch.equal(nt[vis], causal[vis])
    assert not (nt[~vis][:, inner] & offdiag[~vis][:, inner]).any()
    nc = allowed("C_nocross", inner, vis, fid, win)
    same_win_other_frame = vis[:, None] & vis[None, :] & (win[:, None] == win[None, :]) & (fid[:, None] != fid[None, :])
    assert not (nc & same_win_other_frame).any()
    assert torch.equal(nc & ~same_win_other_frame, nd & ~same_win_other_frame)  # otherwise identical to A_nodelim
    npw = allowed("C_noprevwin", inner, vis, fid, win)
    earlier = vis[:, None] & vis[None, :] & (win[None, :] < win[:, None])
    assert not (npw & earlier).any() and torch.equal(npw & ~earlier, nd & ~earlier)
    an = allowed("A_nocross", inner, vis, fid, win)
    assert torch.equal(an, A & ~same_win_other_frame)
    # own-frame attention survives every condition
    own = vis[:, None] & vis[None, :] & (fid[:, None] == fid[None, :]) & causal
    for c in CONDS:
        assert (allowed(c, inner, vis, fid, win) | ~own).all()
    blocked = [c for c in CONDS if text_blocks_delimiters(c)]
    assert blocked == ["A_nodelim", "A_nodelim_text", "C_nocross", "C_noprevwin"]


def _tiny_qwen2vl_lm(attn):
    pytest.importorskip("transformers")
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLTextConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel
    cfg = Qwen2VLTextConfig(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]})
    cfg._attn_implementation = attn
    torch.manual_seed(0)
    return Qwen2VLTextModel(cfg).eval()


def _mrope_positions(ids):
    """3-D positions in the style of Qwen2-VL: text advances all three axes; image tokens get (t, h, w)."""
    pos = torch.zeros(3, len(ids), dtype=torch.long)
    nxt = k = 0
    for i, t in enumerate(ids.tolist()):
        if t == IMG:
            pos[:, i] = torch.tensor([nxt, nxt + k // 2, nxt + k % 2])
            k += 1
            if k == 3:
                nxt += 2
                k = 0
        else:
            pos[:, i] = nxt
            nxt += 1
    return pos[:, None, :]


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_blocking_attention_equals_deletion_with_positions_kept(attn):
    """fp32: masking every attention edge INTO the delimiters gives the same hidden states at every other position as
    deleting the delimiters and keeping the remaining tokens' position ids."""
    lm = _tiny_qwen2vl_lm(attn)
    ids, vis, fid, win = toy_sequence()
    inner = inner_separator_mask(ids, IMG, VS, VE)
    pos = _mrope_positions(ids)
    emb = lm.embed_tokens(ids[None]) + 0.1 * torch.randn(1, len(ids), 64, generator=torch.Generator().manual_seed(1))
    ok = allowed("A_nodelim", inner, vis, fid, win)
    mask = torch.zeros(ok.shape).masked_fill_(~ok, torch.finfo(torch.float32).min)[None, None]
    with torch.no_grad():
        h_mask = lm(inputs_embeds=emb, position_ids=pos, attention_mask=mask).last_hidden_state[0]
        keep = ~inner
        n_kept = int(keep.sum())
        causal = torch.zeros(n_kept, n_kept).masked_fill_(
            ~torch.ones(n_kept, n_kept, dtype=torch.bool).tril(), torch.finfo(torch.float32).min)[None, None]
        h_del = lm(inputs_embeds=emb[:, keep], position_ids=pos[:, :, keep], attention_mask=causal).last_hidden_state[0]
        h_full = lm(inputs_embeds=emb, position_ids=pos).last_hidden_state[0]
    assert torch.allclose(h_mask[keep], h_del, atol=1e-5, rtol=1e-5)
    # sanity: the delimiters do matter when attended (otherwise the test would be vacuous)
    assert (h_full[keep] - h_del).abs().max() > 1e-3


def test_renumbering_changes_the_computation():
    """Deleting tokens with renumbered positions is not the same computation as deletion with kept positions."""
    lm = _tiny_qwen2vl_lm("eager")
    ids, vis, fid, win = toy_sequence()
    keep = ~inner_separator_mask(ids, IMG, VS, VE)
    pos = _mrope_positions(ids)[:, :, keep]
    emb = lm.embed_tokens(ids[None])[:, keep]
    with torch.no_grad():
        h_kept = lm(inputs_embeds=emb, position_ids=pos).last_hidden_state
        h_ren = lm(inputs_embeds=emb, position_ids=_mrope_positions(ids[keep])).last_hidden_state
    assert (h_kept - h_ren).abs().max() > 1e-4


def test_value_scaling_keeps_attention_and_scales_contribution():
    """V<alpha>: attention probabilities are identical to the unmodified run; alpha=1 is a no-op; alpha=0 removes the
    delimiters' contribution (differs from both the unmodified run and from masking, which also frees their mass)."""
    from src.attention.interventions import install_value_scaling, value_scale_of
    lm = _tiny_qwen2vl_lm("eager")
    st = install_value_scaling(lm)
    ids, vis, fid, win = toy_sequence()
    inner = inner_separator_mask(ids, IMG, VS, VE)
    pos = _mrope_positions(ids)
    emb = lm.embed_tokens(ids[None]) + 0.1 * torch.randn(1, len(ids), 64, generator=torch.Generator().manual_seed(1))
    keep = ~inner
    ok = allowed("A_nodelim", inner, vis, fid, win)
    mask = torch.zeros(ok.shape).masked_fill_(~ok, torch.finfo(torch.float32).min)[None, None]

    def run(alpha=None, m=None):
        st["pos"] = None if alpha is None else torch.nonzero(inner).flatten()
        st["alpha"] = alpha
        try:
            o = lm(inputs_embeds=emb, position_ids=pos, attention_mask=m, output_attentions=True)
        finally:
            st["pos"] = None
        return o.last_hidden_state[0], o.attentions
    with torch.no_grad():
        h0, a0 = run()
        h1, _ = run(1.0)
        hz, az = run(0.0)
        hm, _ = run(None, mask)
    assert value_scale_of("V0.5") == 0.5 and value_scale_of("A") is None
    assert torch.allclose(h0, h1, atol=1e-6)
    assert torch.allclose(a0[0], az[0], atol=1e-6)            # first layer: identical attention probabilities
    assert (hz[keep] - h0[keep]).abs().max() > 1e-4
    assert (hz[keep] - hm[keep]).abs().max() > 1e-4
