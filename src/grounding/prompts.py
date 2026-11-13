"""Chat messages for temporal grounding (Qwen-VL message format, also consumed by the InternVL runner).

Three input representations of the same sampled frames:

* ``build_grid_messages``     one grid image per temporal window, each preceded by a text label
                              ``<Frame a to Frame b>`` (T2SGrid input);
* ``build_frameseq_messages`` every frame as its own image, each frame shown once, grouped under
                              window labels; the model's image wrapper tokens then act as
                              delimiters between consecutive frames;
* ``build_seq_messages``      native video input.

The question prompts follow the T2SGrid evaluation protocol so that results are comparable.
"""

from __future__ import annotations

from PIL import Image

from src.representations.grid import GridLayout

# Asks for frame indices ("from x to y").
NUMPRO_PROMPT = "During which frames can we see {}? Answer strictly in the format of 'from x to y'."

# Asks for start and end times in seconds.
MO_PROMPT = ("To accurately pinpoint the event '{}' in the video, determine the precise timeperiod of the event."
             " Provide the start and end times in the format: 'From x to y'.")


def _seconds(i: int, fps: float) -> str:
    return f"{i / fps:.1f}"


def build_grid_messages(layout: GridLayout, grids: list[Image.Image], query: str,
                        time_prompt: bool = False) -> list[dict]:
    """Grid input: one labelled grid image per window, then the question.

    time_prompt=False: frame-number labels and the frame-index question (T2SGrid).
    time_prompt=True:  the same layout, but labels give the window's frame times in seconds
    (``<0.0s to 5.5s>``, frame i at i / fps), the closing sentence gives the duration and the
    question asks for times. Requires unit-consistent labels (``use_frame_index_labels``).
    """
    contents = []
    k = layout.cfg.k
    fps = layout.cfg.fps
    end = None
    for w, img in zip(layout.windows, grids):
        start = w.index * layout.cfg.stride
        end = w.label_end if w.partial else start + k - 1
        if time_prompt:
            contents.append({"type": "text", "text": f"<{_seconds(start, fps)}s to {_seconds(end, fps)}s>"})
        elif k == 1:
            contents.append({"type": "text", "text": f"<Frame {start}>"})
        else:
            contents.append({"type": "text", "text": f"<Frame {start} to Frame {end}>"})
        contents.append({"type": "image", "image": img})
    if time_prompt:
        contents.append({"type": "text",
                         "text": f"This video is {layout.duration:.1f} seconds long. " + MO_PROMPT.format(query)})
    else:
        contents.append({"type": "text",
                         "text": f"This video has {end} frames in total." + NUMPRO_PROMPT.format(query)})
    return [{"role": "user", "content": contents}]


def build_frameseq_messages(layout: GridLayout, frames: list[Image.Image], query: str,
                            policy: str = "last", label: str = "window") -> list[dict]:
    """Per-frame sequence: every sampled frame is its own image and appears once, in chronological order.

    label="window": frames are grouped under the window labels of ``layout``; each frame is assigned
                    to its first or last window occurrence (``policy``). No padding cells.
    label="frame":  a ``<Frame i>`` label before every frame.
    The closing sentence and question are the same as for grids.
    """
    owner = layout.frame_owner(policy)
    k = layout.cfg.k
    contents, end = [], None
    for w in layout.windows:
        start = w.index * layout.cfg.stride
        end = w.label_end if w.partial else start + k - 1
        mine = [f for f in w.frames if owner[f] == w.index]
        if not mine:
            continue
        if label == "window":
            contents.append({"type": "text", "text": f"<Frame {start} to Frame {end}>"})
        for f in mine:
            if label == "frame":
                contents.append({"type": "text", "text": f"<Frame {f}>"})
            contents.append({"type": "image", "image": frames[f]})
    contents.append({"type": "text", "text": f"This video has {end} frames in total." + NUMPRO_PROMPT.format(query)})
    return [{"role": "user", "content": contents}]


def build_seq_messages(video_path: str, query: str, fps: float = 1, prompt: str = "numpro") -> list[dict]:
    """Native video input. ``numpro`` asks for frame indices, which equal seconds only at fps = 1;
    ``mo`` asks for times in seconds and is usable at any fps."""
    if prompt not in ("numpro", "mo") or (prompt == "numpro" and fps != 1):
        raise ValueError(f"prompt {prompt!r} is not valid at fps={fps}")
    text = (NUMPRO_PROMPT if prompt == "numpro" else MO_PROMPT).format(query)
    return [{"role": "user", "content": [
        {"type": "video", "video": str(video_path), "fps": fps},
        {"type": "text", "text": text},
    ]}]
