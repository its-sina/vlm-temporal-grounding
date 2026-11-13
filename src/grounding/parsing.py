"""Parsing model answers into temporal spans.

The patterns follow the T2SGrid evaluation protocol: the first matching pattern wins, endpoints are
ordered, and an unparseable answer becomes [0, 0] (IoU 0).
"""

from __future__ import annotations

import re

_PATTERNS = [
    # from frame 12 to frame 34 / from 12 to 34 / from 12s to 34s
    r"\bfrom\s*(?:frame[s]?\s*)?(\d+(?:\.\d+)?)(?:s)?\s*to\s*(?:frame[s]?\s*)?(\d+(?:\.\d+)?)(?:s)?\b",
    # frame 12 to frame 34 / frames 12 and 34
    r"\b(?:frame[s]?\s*)(\d+(?:\.\d+)?)(?:s)?\s*(?:to|and)\s*(?:frame[s]?\s*)?(\d+(?:\.\d+)?)(?:s)?\b",
    # between frames 12 and 34 / between frames 12-34
    r"\bbetween\s*frames?\s*(\d+(?:\.\d+)?)(?:s)?\s*(?:and|-)\s*(\d+(?:\.\d+)?)(?:s)?\b",
    # 12-34 / 12 - 34
    r"\b(\d+(?:\.\d+)?)(?:s)?\s*-\s*(\d+(?:\.\d+)?)(?:s)?\b",
    # 12 to 34 (general fallback)
    r"\b(\d+(?:\.\d+)?)(?:s)?\s*(?:to|and)\s*(\d+(?:\.\d+)?)(?:s)?\b",
]


def _match(text: str):
    for p in _PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
    return None


def parse_timestamps(text: str) -> list[int]:
    """Integer span (frame indices). Fractional values are truncated, as in the reference protocol."""
    m = _match(text)
    if m is None:
        return [0, 0]
    a, b = int(float(m[0])), int(float(m[1]))
    return [min(a, b), max(a, b)]


def parse_timestamps_float(text: str) -> list[float]:
    """Same patterns, keeping fractional values (e.g. '20.6 - 31.0 seconds').

    Used for answers given in decimal seconds (native video, time prompts), where integer
    truncation would cost up to 1 s per endpoint.
    """
    m = _match(text)
    if m is None:
        return [0.0, 0.0]
    a, b = float(m[0]), float(m[1])
    return [min(a, b), max(a, b)]


def frames_to_seconds(pred: list[int], fps: float) -> list[float]:
    """Frame-index answers to seconds (exact identity at fps = 1)."""
    return [p / fps for p in pred]


def unit_aware_span(pred_raw: list[int], duration: float, fps: float) -> list[float]:
    """Heuristic unit resolution for prompts with mixed units.

    With the T2SGrid labels, window labels are frame indices at ``fps`` but the last (partial)
    window label is ``int(duration)`` in seconds, and the model may answer in either unit. If an
    endpoint cannot be a frame index (> number of sampled frames + 1) the answer is taken as
    seconds, otherwise it is converted from frames to seconds.
    """
    a, b = pred_raw
    n_frames = int(duration * fps) + 1
    return [a, b] if b > n_frames + 1 else [a / fps, b / fps]
