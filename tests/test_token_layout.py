import os

import pytest
import torch
from PIL import Image

from src.representations.grid import GridConfig, make_layout
from src.representations.token_layout import (cellmajor_fns, cell_patch_index, cell_token_index, dedup_keep_fn,
                                              flatcell_pos_fn, visual_keep_mask)

IMG = 9


def test_cell_token_index_partition():
    tok = cell_token_index(GridConfig())
    assert tok.shape == (12, 144)
    assert sorted(tok.reshape(-1).tolist()) == list(range(1728))
    assert tok[1, 0].item() == 12 and tok[4, 0].item() == 12 * 48   # cells (0, 1) and (1, 0)


def test_cell_patch_index_partition():
    p = cell_patch_index(GridConfig())
    assert p.shape == (12, 576)
    assert sorted(p.reshape(-1).tolist()) == list(range(6912))


def test_cells_must_align_with_merge_blocks():
    with pytest.raises(AssertionError):
        cell_token_index(GridConfig(thumb=(320, 320)))   # 320 is not a multiple of 28


@pytest.mark.parametrize("policy", ["first+pad", "last+pad", "center+pad"])
def test_dedup_keeps_each_frame_once(policy):
    lay = make_layout(31, 30.96, GridConfig(stride=7))
    keep = dedup_keep_fn(lay, IMG, policy)(torch.full((len(lay.windows) * 1728,), IMG), None, None)
    assert int(keep.sum()) == 31 * 144


def test_dedup_without_pad_suffix_keeps_padding():
    lay = make_layout(31, 30.96, GridConfig(stride=7))
    keep = dedup_keep_fn(lay, IMG, "last")(torch.full((len(lay.windows) * 1728,), IMG), None, None)
    assert int(keep.sum()) == (31 + 2) * 144     # two black cells in the partial window


def test_dedup_last_keeps_latest_copy_and_text():
    lay = make_layout(31, 30.96, GridConfig(stride=7))
    mask = visual_keep_mask(lay, "last+pad").view(len(lay.windows), 1728)
    tok = cell_token_index(lay.cfg)
    assert not mask[0, tok[7]].any() and mask[1, tok[0]].all()     # frame 7: window 0 cell 7 -> window 1 cell 0
    ids = torch.tensor([1, 2] + [IMG] * (len(lay.windows) * 1728) + [3])
    keep = dedup_keep_fn(lay, IMG, "last+pad")(ids, None, None)
    assert keep[0] and keep[1] and keep[-1]


def test_dedup_rejects_unknown_policy():
    lay = make_layout(31, 30.96, GridConfig(stride=7))
    with pytest.raises(ValueError):
        dedup_keep_fn(lay, IMG, "all")
    assert int(visual_keep_mask(lay, "all").sum()) == len(lay.windows) * 1728


def _grid_ids(lay):
    return torch.tensor([1] + sum(([5] + [IMG] * 1728 + [6] for _ in lay.windows), []) + [2])


def test_flatcell_positions_one_block_per_kept_cell():
    lay = make_layout(24, 24.0, GridConfig(stride=12))
    ids = _grid_ids(lay)
    pos = flatcell_pos_fn(lay, IMG)(ids, torch.zeros(3, len(ids), dtype=torch.long))
    first = pos[:, 2:2 + 1728]
    tok = cell_token_index(lay.cfg)
    assert first[0, tok[0]].unique().numel() == 1 and first[0, tok[1]].min() == first[0, tok[0]].max() + 12
    assert int(pos[0, -1]) == int(pos[0].max())


def test_cellmajor_gap_spacing():
    lay = make_layout(24, 24.0, GridConfig(stride=12))
    ids = _grid_ids(lay)
    wrap, keep_fn, pos_fn = cellmajor_fns(lay, IMG, None, gap=2)
    pos = pos_fn(ids, torch.zeros(3, len(ids), dtype=torch.long))
    block = pos[0, 2:2 + 1728].view(12, 144)
    assert torch.equal(block[1, 0] - block[0, 0], torch.tensor(12 + 2))
    emb = [torch.arange(1728)[:, None].float()]
    assert torch.equal(wrap(emb)[0][:144, 0].long(), cell_token_index(lay.cfg)[0])
    assert keep_fn(ids, None, None).all()


def test_stacked_frame_preprocessing_is_identical():
    """Stacked per-frame preprocessing equals per-image processing (needs a local Qwen2-VL processor)."""
    path = os.environ.get("VTG_QWEN2VL_PROCESSOR")
    if not path or not os.path.isdir(path):
        pytest.skip("set VTG_QWEN2VL_PROCESSOR to a local Qwen2-VL checkpoint directory")
    from transformers import AutoProcessor

    from src.representations.vision_reuse import FrameEncoder

    class Runner:
        processor = AutoProcessor.from_pretrained(path, use_fast=True)
        device = "cpu"

        class model:  # noqa: N801
            class model:  # noqa: N801
                visual = None

    for n in (9, 260):   # > 200 frames exercises chunked stacking (processor aspect-ratio limit)
        imgs = [Image.new("RGB", (336, 336), (i * 7 % 255, i * 13 % 255, 40)) for i in range(n)]
        pv, thw = FrameEncoder(Runner).pixel_values(imgs)
        ref = Runner.processor.image_processor(images=imgs, return_tensors="pt")
        assert torch.equal(pv, ref["pixel_values"]) and torch.equal(thw, ref["image_grid_thw"])
