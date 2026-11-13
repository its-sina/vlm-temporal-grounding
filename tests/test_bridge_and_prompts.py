from PIL import Image

from src.grounding.prompts import build_frameseq_messages, build_grid_messages, build_seq_messages
from src.representations.grid import GridConfig, build_grids, make_layout, use_frame_index_labels
from src.representations.vision_reuse import bridge_windows


def test_bridge_windows():
    assert bridge_windows(31, 12, 3) == [[9, 10, 11, 12, 13, 14], [21, 22, 23, 24, 25, 26]]
    assert bridge_windows(12, 12, 3) == []
    assert bridge_windows(14, 12, 3) == [[9, 10, 11, 12, 13, None]]


def test_frameseq_messages_each_frame_once_chronological():
    frames = [Image.new("RGB", (336, 336), (i, 0, 0)) for i in range(31)]
    lay = make_layout(31, 30.96, GridConfig(4, 3, 1.0, 7))
    for policy in ("last", "first"):
        msg = build_frameseq_messages(lay, frames, "q", policy, "window")[0]["content"]
        imgs = [c["image"].getpixel((0, 0))[0] for c in msg if c["type"] == "image"]
        assert imgs == list(range(31))
    msg = build_frameseq_messages(lay, frames, "q", "last", "frame")[0]["content"]
    assert [c["text"] for c in msg if c["type"] == "text"][:2] == ["<Frame 0>", "<Frame 1>"]


def test_grid_messages_labels_and_question():
    frames = [Image.new("RGB", (28, 28)) for _ in range(31)]
    lay, grids = build_grids(frames, 30.96, GridConfig(4, 3, 1.0, 7, (28, 28)))
    texts = [c["text"] for c in build_grid_messages(lay, grids, "a person sits")[0]["content"] if c["type"] == "text"]
    assert texts[0] == "<Frame 0 to Frame 11>" and texts[3] == "<Frame 21 to Frame 30>"
    assert texts[-1].startswith("This video has 30 frames in total.During which frames can we see a person sits?")


def test_time_prompt_labels_in_seconds():
    frames = [Image.new("RGB", (28, 28)) for _ in range(20)]
    lay, grids = build_grids(frames, 10.0, GridConfig(4, 3, 2.0, 7, (28, 28)))
    use_frame_index_labels(lay)
    content = build_grid_messages(lay, grids, "x", time_prompt=True)[0]["content"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert texts[0] == "<0.0s to 5.5s>" and texts[-2] == "<7.0s to 9.5s>"   # partial window: frames 14..19
    assert texts[-1].startswith("This video is 10.0 seconds long. To accurately pinpoint the event 'x'")


def test_seq_messages_validate_prompt():
    import pytest
    assert build_seq_messages("v.mp4", "q", 2.0, "mo")[0]["content"][0]["fps"] == 2.0
    with pytest.raises(ValueError):
        build_seq_messages("v.mp4", "q", 2.0, "numpro")
