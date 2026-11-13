import numpy as np
from PIL import Image

from src.data.frames import load_frame_cache, sample_times, save_frame_cache
from src.representations.grid import GridConfig, build_grids, make_layout, use_frame_index_labels


def test_layout_overlapping_windows_and_partial_tail():
    lay = make_layout(31, 30.96, GridConfig(stride=7))
    assert [w.cells[0] for w in lay.windows] == [0, 7, 14, 21]
    assert lay.windows[-1].partial and lay.windows[-1].label_end == 30
    assert lay.windows[-1].cells[-2:] == [None, None]   # frames 21..30, then 2 black cells
    assert lay.total_frames_text == 30


def test_layout_no_overlap_exact_multiple():
    lay = make_layout(24, 24.0, GridConfig(stride=12))
    assert len(lay.windows) == 2 and not lay.windows[-1].partial


def test_sample_times_float_accumulation():
    assert len(sample_times(30.96, 1.0)) == 31
    assert len(sample_times(3.0, 1.0)) == 3
    assert sample_times(2.0, 2.0) == [0.0, 0.5, 1.0, 1.5]


def test_grid_name():
    assert GridConfig().name == "g43_f1.0_s7_r1_l0_ts336x336"


def test_compose_places_frames_row_major():
    cfg = GridConfig(thumb=(28, 28))
    frames = [Image.new("RGB", (28, 28), (i, 0, 0)) for i in range(13)]
    _, grids = build_grids(frames, 13.0, cfg)
    a = np.asarray(grids[0])
    assert a[0, 0, 0] == 0 and a[0, 28, 0] == 1 and a[28, 0, 0] == 4
    assert np.asarray(grids[-1])[-1, -1].sum() == 0  # black padding


def test_partial_window_label_units():
    lay = make_layout(61, 121.4, GridConfig(4, 3, 0.5, 12))   # 0.5 fps
    assert lay.windows[-1].partial and lay.windows[-1].label_end == 121   # reference: seconds
    use_frame_index_labels(lay)
    assert lay.windows[-1].label_end == 60                             # unit-consistent: frame index


def test_frame_owner_policies():
    lay = make_layout(31, 30.96, GridConfig(stride=7))
    last, first = lay.frame_owner("last"), lay.frame_owner("first")
    assert first[8] == 0 and last[8] == 1
    assert sorted(last) == list(range(31))


def test_frame_cache_roundtrip_is_lossless(tmp_path):
    frames = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (8, 8, 3), dtype=np.uint8)) for i in range(3)]
    save_frame_cache(tmp_path / "v.npz", frames, 2.5)
    back, dur = load_frame_cache(tmp_path / "v.npz")
    assert dur == 2.5
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(frames, back))
