"""Temporal windows and temporal-to-spatial grids.

A video sampled at ``fps`` is split into windows of ``k = cols * rows`` consecutive frames.
Windows start at 0, stride, 2*stride, ... while ``start + k <= n_frames``; consecutive windows
overlap when ``stride < k``. If frames remain after the last full window, one partial window
covers them and is padded with black cells. Each window is shown to the model either as one grid
image (frames in row-major order) or as a per-frame sequence (see ``src.grounding.prompts``).

The layout reproduces the T2SGrid preprocessing exactly, including its labelling of the partial
window (label end ``int(duration)``, in seconds); ``use_frame_index_labels`` gives unit-consistent
frame-index labels instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass(frozen=True)
class GridConfig:
    cols: int = 4
    rows: int = 3
    fps: float = 1.0
    stride: int = 7
    thumb: tuple[int, int] = (336, 336)

    @property
    def k(self) -> int:
        """Frames per window (= cells per grid)."""
        return self.cols * self.rows

    @property
    def name(self) -> str:
        """Compact identifier, e.g. ``g43_f1.0_s7_r1_l0_ts336x336`` (T2SGrid naming)."""
        return (f"g{self.cols}{self.rows}_f{self.fps}_s{self.stride}_r1_l0"
                f"_ts{self.thumb[0]}x{self.thumb[1]}")


@dataclass
class Window:
    """One window: the frame index in each cell (None = black padding) and its text label range."""
    index: int
    cells: list[int | None]
    label_start: int
    label_end: int
    partial: bool = False

    @property
    def frames(self) -> list[int]:
        return [c for c in self.cells if c is not None]


@dataclass
class GridLayout:
    cfg: GridConfig
    n_frames: int
    duration: float
    windows: list[Window] = field(default_factory=list)

    @property
    def total_frames_text(self) -> int:
        """Number stated in the prompt ("This video has N frames in total."): the label end of the
        last window, as in the T2SGrid prompt."""
        return self.windows[-1].label_end

    def frame_occurrences(self) -> dict[int, list[tuple[int, int]]]:
        """frame index -> list of (window index, cell index) where the frame appears."""
        occ: dict[int, list[tuple[int, int]]] = {}
        for w in self.windows:
            for ci, f in enumerate(w.cells):
                if f is not None:
                    occ.setdefault(f, []).append((w.index, ci))
        return occ

    def frame_owner(self, policy: str = "last") -> dict[int, int]:
        """frame index -> the window that keeps it under a first/last-occurrence policy."""
        occ = self.frame_occurrences()
        pick = (lambda lst: lst[-1]) if policy == "last" else (lambda lst: lst[0])
        return {f: pick(lst)[0] for f, lst in occ.items()}


def make_layout(n_frames: int, duration: float, cfg: GridConfig) -> GridLayout:
    k, s = cfg.k, cfg.stride
    layout = GridLayout(cfg=cfg, n_frames=n_frames, duration=duration)
    i = 0
    while i + k <= n_frames:
        layout.windows.append(Window(len(layout.windows), list(range(i, i + k)), i, i + k - 1))
        i += s
    if i < n_frames:
        cells: list[int | None] = list(range(i, n_frames))
        cells += [None] * (k - len(cells))
        layout.windows.append(Window(len(layout.windows), cells, i, int(duration), partial=True))
    return layout


def use_frame_index_labels(layout: GridLayout) -> GridLayout:
    """Unit-consistent labels: the partial window's label end becomes the last frame index
    (instead of ``int(duration)`` in seconds, which mixes units when fps != 1)."""
    for w in layout.windows:
        if w.partial:
            w.label_end = layout.n_frames - 1
    return layout


def compose(window: Window, frames: list[Image.Image], cfg: GridConfig) -> Image.Image:
    """Paste a window's frames into one grid image (row-major, black padding cells)."""
    w, h = cfg.thumb
    canvas = Image.new("RGB", (cfg.cols * w, cfg.rows * h))
    for ci, f in enumerate(window.cells):
        if f is None:
            continue
        canvas.paste(frames[f], ((ci % cfg.cols) * w, (ci // cfg.cols) * h))
    return canvas


def build_grids(frames: list[Image.Image], duration: float,
                cfg: GridConfig) -> tuple[GridLayout, list[Image.Image]]:
    layout = make_layout(len(frames), duration, cfg)
    return layout, [compose(w, frames, cfg) for w in layout.windows]
