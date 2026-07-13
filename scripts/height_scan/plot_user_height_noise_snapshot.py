#!/usr/bin/env python3
"""Plot height-scan evidence from pasted real-robot snapshots.

This script intentionally uses only the 187 sampled cells printed by
run_height_map_snapshot.sh. It does not reconstruct the raw LiDAR cloud.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


XS = np.asarray([round(-0.8 + 0.1 * i, 1) for i in range(17)], dtype=np.float32)
YS_TOP_TO_BOTTOM = np.asarray([round(0.5 - 0.1 * i, 1) for i in range(11)], dtype=np.float32)


@dataclass(frozen=True)
class Snapshot:
    name: str
    flatness_m: float
    raw_valid: int
    sentinel: int
    critical_sentinel: int
    reason: str
    hcm_top_to_bottom: list[list[Optional[int]]]
    mask_top_to_bottom: list[str]
    height_data_scale: float = 1.0


SNAPSHOTS = [
    Snapshot(
        name="stair snapshot A yaw=-1.443",
        flatness_m=0.1375439651310444,
        raw_valid=171,
        sentinel=16,
        critical_sentinel=9,
        reason="sentinel_critical",
        mask_top_to_bottom=[
            "########X########",
            "########X########",
            "########X########",
            "#################",
            "#################",
            "#################",
            "#######F#########",
            "######FFFF#######",
            "#####XXXXX#######",
            "#######XX########",
            "#######X#########",
        ],
        hcm_top_to_bottom=[
            [-7, -7, -7, -5, -4, -4, -3, +1, None, +0, +2, +2, +3, +2, +1, +1, +2],
            [-5, -5, -5, -6, -3, -2, -2, +1, None, +3, +2, +2, +2, +2, +4, +3, +3],
            [-6, -5, -5, -5, -3, -3, -2, +1, None, +0, +1, +2, +5, +6, +6, +5, +4],
            [-6, -5, -4, -3, -1, +1, +7, +0, -2, +0, +2, +3, +4, +7, +7, +8, +6],
            [-6, -5, -4, -3, -1, +2, +0, +0, +1, -2, -3, -4, -1, -1, +11, +10, +8],
            [-5, -4, -3, -1, +5, +0, -2, -2, -2, -2, -1, -3, -3, -1, +1, +11, +9],
            [-5, -4, -3, +1, +6, +0, -3, None, -1, +2, -2, -2, -3, -1, +1, +12, +10],
            [-4, -2, -2, +2, +4, +0, None, None, None, None, -2, -4, -2, +1, +10, +9, +7],
            [-5, -3, -1, +1, -1, None, None, None, None, None, +2, +5, +6, +8, +8, +9, +7],
            [-5, -5, -5, -4, -3, -2, +3, None, None, -1, +2, +3, +4, +7, +7, +6, +5],
            [-5, -5, -5, -4, -3, -3, +1, None, +1, +3, +3, +2, +3, +3, +4, +3, +4],
        ],
    ),
    Snapshot(
        name="flat snapshot B yaw=-1.633",
        flatness_m=0.09285583198070527,
        raw_valid=178,
        sentinel=9,
        critical_sentinel=9,
        reason="sentinel_critical",
        mask_top_to_bottom=[
            "#################",
            "#################",
            "###########X#####",
            "###########X#####",
            "###########X#####",
            "###########X#####",
            "###########X#####",
            "###########X#####",
            "###########X#####",
            "###########X#####",
            "###########X#####",
        ],
        hcm_top_to_bottom=[
            [-6, -4, -4, -1, -2, -1, -1, +0, +4, +2, +1, +2, +2, +2, +2, +2, +0],
            [-3, -3, -3, -4, -1, -1, +0, +3, +6, +11, +2, +3, +3, +2, +2, +2, +2],
            [-2, -3, -2, -2, -2, -3, +1, +5, +8, +1, +2, None, +2, +2, +3, +2, +1],
            [-3, -3, -2, -2, -2, -2, -3, -2, -1, +0, +0, None, +1, +4, +5, +4, +3],
            [-3, -3, -3, -2, -2, -3, -2, +0, +1, +8, +1, None, -2, +1, +5, +5, +4],
            [-4, -3, -3, -4, -3, -3, -2, +3, +5, +1, +1, None, +0, -1, +4, +4, +3],
            [-5, -5, -5, -5, -4, -2, +1, +0, +4, +2, -1, None, -1, -2, +4, +5, +4],
            [-6, -6, -5, -4, -4, -3, -1, -2, +8, +1, -1, None, -2, +0, +6, +5, +3],
            [-4, -4, -3, -3, -3, -3, -2, -3, -2, -1, +1, None, +4, +4, +4, +3, +3],
            [-4, -3, -2, -3, -2, -2, -3, +4, +1, +2, +1, None, +3, +3, +2, +3, +2],
            [-3, -3, -3, -2, -2, -3, +1, +2, +1, +1, +2, None, +2, +2, +3, +2, +1],
        ],
        height_data_scale=0.5,
    ),
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_10 = _font(10)
FONT_11 = _font(11)
FONT_12 = _font(12)
FONT_14 = _font(14)
FONT_16 = _font(16)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] * (1.0 - t) + b[i] * t)) for i in range(3))


def _height_color(value_cm: Optional[float], vlim_cm: float = 15.0) -> tuple[int, int, int]:
    if value_cm is None or not math.isfinite(float(value_cm)):
        return (70, 70, 70)
    v = max(-vlim_cm, min(vlim_cm, float(value_cm))) / vlim_cm
    if v < 0.0:
        return _mix((41, 121, 185), (247, 247, 247), (v + 1.0))
    return _mix((247, 247, 247), (202, 0, 32), v)


def _scaled_cm(snapshot: Snapshot, value_cm: Optional[float]) -> Optional[float]:
    if value_cm is None:
        return None
    return float(value_cm) * float(snapshot.height_data_scale)


def _scaled_flatness_cm(snapshot: Snapshot) -> float:
    return snapshot.flatness_m * 100.0 * float(snapshot.height_data_scale)


def _draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2.0, xy[1] - (box[3] - box[1]) / 2.0), text, font=font, fill=fill)


def _draw_heatmap_panel(
    draw: ImageDraw.ImageDraw,
    snapshot: Snapshot,
    origin: tuple[int, int],
    cell: int,
    vlim_cm: float,
) -> None:
    ox, oy = origin
    draw.text((ox, oy - 54), snapshot.name, font=FONT_16, fill=(0, 0, 0))
    draw.text(
        (ox, oy - 32),
        f"p95-p05={_scaled_flatness_cm(snapshot):.1f} cm | valid={snapshot.raw_valid}/187 | "
        f"sentinel={snapshot.sentinel} | critical={snapshot.critical_sentinel}",
        font=FONT_12,
        fill=(40, 40, 40),
    )
    for row, y in enumerate(YS_TOP_TO_BOTTOM):
        for col, x in enumerate(XS):
            value = _scaled_cm(snapshot, snapshot.hcm_top_to_bottom[row][col])
            x0 = ox + col * cell
            y0 = oy + row * cell
            fill = _height_color(value, vlim_cm)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline=(45, 45, 45))
            mark = snapshot.mask_top_to_bottom[row][col]
            if mark != "#":
                draw.rectangle((x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2), outline=(0, 0, 0), width=2)
                _draw_centered(draw, (x0 + cell / 2, y0 + cell / 2), mark, FONT_14, (255, 255, 255))
            elif value is not None:
                label = f"{int(round(float(value))):+d}"
                text_color = (0, 0, 0) if abs(float(value)) < vlim_cm * 0.65 else (255, 255, 255)
                _draw_centered(draw, (x0 + cell / 2, y0 + cell / 2), label, FONT_10, text_color)
    for col, x in enumerate(XS):
        _draw_centered(draw, (ox + col * cell + cell / 2, oy + 11 * cell + 16), f"{x:.1f}", FONT_10, (30, 30, 30))
    for row, y in enumerate(YS_TOP_TO_BOTTOM):
        _draw_centered(draw, (ox - 24, oy + row * cell + cell / 2), f"{y:+.1f}", FONT_10, (30, 30, 30))
    draw.text((ox + 17 * cell / 2 - 42, oy + 11 * cell + 34), "base x forward (m)", font=FONT_11, fill=(30, 30, 30))
    draw.text((ox - 56, oy + 11 * cell / 2 - 8), "base y", font=FONT_11, fill=(30, 30, 30))


def _draw_colorbar(draw: ImageDraw.ImageDraw, x: int, y: int, h: int, vlim_cm: float) -> None:
    for i in range(h):
        value = vlim_cm - (2.0 * vlim_cm * i / max(1, h - 1))
        draw.rectangle((x, y + i, x + 18, y + i), fill=_height_color(value, vlim_cm))
    draw.rectangle((x, y, x + 18, y + h), outline=(0, 0, 0))
    draw.text((x + 24, y - 4), f"+{vlim_cm:.0f} cm", font=FONT_10, fill=(0, 0, 0))
    draw.text((x + 24, y + h / 2 - 6), "0", font=FONT_10, fill=(0, 0, 0))
    draw.text((x + 24, y + h - 12), f"-{vlim_cm:.0f} cm", font=FONT_10, fill=(0, 0, 0))


def save_heatmap(path: str) -> None:
    width, height = 1540, 630
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "Height-scan 2D heatmap from pasted flat-ground snapshots", font=FONT_16, fill=(0, 0, 0))
    draw.text(
        (28, 45),
        "Relative heights use per-snapshot display scaling; X/F cells are unknown/sentinel.",
        font=FONT_12,
        fill=(45, 45, 45),
    )
    _draw_heatmap_panel(draw, SNAPSHOTS[0], (70, 130), 31, 15.0)
    _draw_heatmap_panel(draw, SNAPSHOTS[1], (750, 130), 31, 15.0)
    _draw_colorbar(draw, 1365, 140, 280, 15.0)
    draw.text((70, 588), "# valid cell    F footprint unknown    X non-footprint unknown/sentinel", font=FONT_12, fill=(30, 30, 30))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    image.save(path)


def _project(x: float, y: float, z_cm: float, ox: float, oy: float, scale: float = 230.0) -> tuple[float, float]:
    sx = ox + (x - y) * scale * 0.62
    sy = oy + (x + y) * scale * 0.32 - z_cm * 4.2
    return sx, sy


def _draw_3d_panel(
    draw: ImageDraw.ImageDraw,
    snapshot: Snapshot,
    origin: tuple[int, int],
    vlim_cm: float,
) -> None:
    ox, oy = origin
    draw.text((ox - 250, oy - 250), snapshot.name, font=FONT_16, fill=(0, 0, 0))
    draw.text(
        (ox - 250, oy - 228),
        f"p95-p05={_scaled_flatness_cm(snapshot):.1f} cm, sentinel={snapshot.sentinel}, critical={snapshot.critical_sentinel}",
        font=FONT_12,
        fill=(45, 45, 45),
    )

    corners = [(-0.85, -0.55), (0.85, -0.55), (0.85, 0.55), (-0.85, 0.55), (-0.85, -0.55)]
    projected = [_project(x, y, 0.0, ox, oy) for x, y in corners]
    draw.line(projected, fill=(80, 80, 80), width=2)
    draw.line([_project(-0.85, 0, 0, ox, oy), _project(0.85, 0, 0, ox, oy)], fill=(120, 120, 120), width=1)
    draw.line([_project(0, -0.55, 0, ox, oy), _project(0, 0.55, 0, ox, oy)], fill=(120, 120, 120), width=1)

    points: list[tuple[float, float, Optional[int], str]] = []
    for row, y in enumerate(YS_TOP_TO_BOTTOM):
        for col, x in enumerate(XS):
            points.append((float(x), float(y), _scaled_cm(snapshot, snapshot.hcm_top_to_bottom[row][col]), snapshot.mask_top_to_bottom[row][col]))
    points.sort(key=lambda item: item[0] + item[1])

    for x, y, z, mark in points:
        px0, py0 = _project(x, y, 0.0, ox, oy)
        if z is None:
            draw.line((px0 - 5, py0 - 5, px0 + 5, py0 + 5), fill=(180, 0, 0), width=2)
            draw.line((px0 - 5, py0 + 5, px0 + 5, py0 - 5), fill=(180, 0, 0), width=2)
            continue
        px1, py1 = _project(x, y, float(z), ox, oy)
        color = _height_color(float(z), vlim_cm)
        draw.line((px0, py0, px1, py1), fill=(90, 90, 90), width=1)
        radius = 5
        draw.ellipse((px1 - radius, py1 - radius, px1 + radius, py1 + radius), fill=color, outline=(20, 20, 20))

    draw.text((ox - 228, oy + 164), "x forward", font=FONT_11, fill=(50, 50, 50))
    draw.text((ox + 90, oy + 152), "y left", font=FONT_11, fill=(50, 50, 50))
    draw.text((ox - 248, oy + 182), "red X = unknown/sentinel cell; vertical sticks show cm deviation from median plane", font=FONT_10, fill=(50, 50, 50))


def save_3d(path: str) -> None:
    width, height = 1420, 610
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "Height-scan 3D sampled terrain distribution from pasted flat-ground snapshots", font=FONT_16, fill=(0, 0, 0))
    draw.text(
        (28, 45),
        "187-cell sampled terrain observation with per-snapshot display scaling.",
        font=FONT_12,
        fill=(45, 45, 45),
    )
    _draw_3d_panel(draw, SNAPSHOTS[0], (345, 370), 15.0)
    _draw_3d_panel(draw, SNAPSHOTS[1], (1020, 370), 15.0)
    _draw_colorbar(draw, 1320, 140, 280, 15.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    image.save(path)


def _snapshot_tag(snapshot: Snapshot) -> str:
    return snapshot.name.split()[1].lower()


def _draw_axis_ticks(
    draw: ImageDraw.ImageDraw,
    ticks: list[tuple[float, str]],
    mapper,
    font=FONT_10,
    fill=(60, 60, 60),
) -> None:
    for value, label in ticks:
        x, y = mapper(value)
        draw.line((x, y - 4, x, y + 4), fill=(160, 160, 160), width=1)
        _draw_centered(draw, (x, y + 15), label, font, fill)


def _draw_minimal_heatmap(
    draw: ImageDraw.ImageDraw,
    snapshot: Snapshot,
    origin: tuple[int, int],
    cell: int,
    vlim_cm: float,
) -> None:
    ox, oy = origin
    width = len(XS) * cell
    height = len(YS_TOP_TO_BOTTOM) * cell
    draw.text((ox, oy - 40), "2D height map", font=FONT_14, fill=(20, 20, 20))

    for row, y in enumerate(YS_TOP_TO_BOTTOM):
        for col, x in enumerate(XS):
            value = _scaled_cm(snapshot, snapshot.hcm_top_to_bottom[row][col])
            x0 = ox + col * cell
            y0 = oy + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=_height_color(value, vlim_cm))
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), outline=(230, 230, 230), width=1)
            mark = snapshot.mask_top_to_bottom[row][col]
            if mark != "#":
                draw.rectangle((x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2), fill=(35, 35, 35))
                _draw_centered(draw, (x0 + cell / 2, y0 + cell / 2), mark, FONT_11, (245, 245, 245))

    draw.rectangle((ox, oy, ox + width, oy + height), outline=(70, 70, 70), width=1)
    for tick in (-0.8, -0.4, 0.0, 0.4, 0.8):
        col = int(round((tick - float(XS[0])) / 0.1))
        x = ox + col * cell + cell / 2
        draw.line((x, oy + height, x, oy + height + 5), fill=(70, 70, 70), width=1)
        _draw_centered(draw, (x, oy + height + 18), f"{tick:.1f}", FONT_10, (50, 50, 50))
    for tick in (-0.5, 0.0, 0.5):
        row = int(round((float(YS_TOP_TO_BOTTOM[0]) - tick) / 0.1))
        y = oy + row * cell + cell / 2
        draw.line((ox - 5, y, ox, y), fill=(70, 70, 70), width=1)
        _draw_centered(draw, (ox - 25, y), f"{tick:+.1f}", FONT_10, (50, 50, 50))
    _draw_centered(draw, (ox + width / 2, oy + height + 42), "x forward (m)", FONT_11, (40, 40, 40))
    draw.text((ox - 40, oy - 24), "y (m)", font=FONT_11, fill=(40, 40, 40))


def _draw_side_point_cloud(
    draw: ImageDraw.ImageDraw,
    snapshot: Snapshot,
    origin: tuple[int, int],
    size: tuple[int, int],
    vlim_cm: float,
) -> None:
    ox, oy = origin
    width, height = size
    cx = ox + width * 0.49
    cy = oy + height * 0.64
    x_scale = 230.0
    y_scale_x = 92.0
    y_scale_y = 92.0
    z_scale = 6.3

    def map_point(x_m: float, y_m: float, z_cm: float) -> tuple[float, float]:
        sx = cx + x_m * x_scale + y_m * y_scale_x
        sy = cy + y_m * y_scale_y - z_cm * z_scale
        return sx, sy

    draw.text((ox, oy - 40), "3D sampled point cloud", font=FONT_14, fill=(20, 20, 20))

    # Ground reference grid at z=0. It keeps the 3D structure readable without
    # turning the plot into a dense technical drawing.
    for y in YS_TOP_TO_BOTTOM:
        line = [map_point(float(XS[0]), float(y), 0.0), map_point(float(XS[-1]), float(y), 0.0)]
        draw.line(line, fill=(224, 224, 224), width=1)
    for x in XS[::2]:
        line = [map_point(float(x), float(YS_TOP_TO_BOTTOM[-1]), 0.0), map_point(float(x), float(YS_TOP_TO_BOTTOM[0]), 0.0)]
        draw.line(line, fill=(235, 235, 235), width=1)

    corners = [
        map_point(float(XS[0]), float(YS_TOP_TO_BOTTOM[-1]), 0.0),
        map_point(float(XS[-1]), float(YS_TOP_TO_BOTTOM[-1]), 0.0),
        map_point(float(XS[-1]), float(YS_TOP_TO_BOTTOM[0]), 0.0),
        map_point(float(XS[0]), float(YS_TOP_TO_BOTTOM[0]), 0.0),
        map_point(float(XS[0]), float(YS_TOP_TO_BOTTOM[-1]), 0.0),
    ]
    draw.line(corners, fill=(120, 120, 120), width=1)

    # A compact z-axis with cm ticks, separated from the cloud.
    axis_x = ox + 30
    axis_y0 = cy
    draw.line((axis_x, axis_y0 + 15.0 * z_scale, axis_x, axis_y0 - 15.0 * z_scale), fill=(80, 80, 80), width=1)
    for z in (-15, -10, -5, 0, 5, 10, 15):
        y_tick = axis_y0 - float(z) * z_scale
        draw.line((axis_x - 4, y_tick, axis_x + 4, y_tick), fill=(80, 80, 80), width=1)
        draw.text((axis_x - 31, y_tick - 6), f"{z:+d}", font=FONT_10, fill=(70, 70, 70))
    draw.text((axis_x - 10, axis_y0 - 15.0 * z_scale - 22), "z cm", font=FONT_10, fill=(60, 60, 60))

    points: list[tuple[float, float, Optional[int], str]] = []
    for row, y in enumerate(YS_TOP_TO_BOTTOM):
        for col, x in enumerate(XS):
            points.append((float(x), float(y), _scaled_cm(snapshot, snapshot.hcm_top_to_bottom[row][col]), snapshot.mask_top_to_bottom[row][col]))
    points.sort(key=lambda item: item[1])

    for x, y, z, mark in points:
        sx0, sy0 = map_point(x, y, 0.0)
        if z is None:
            draw.line((sx0 - 6, sy0 - 6, sx0 + 6, sy0 + 6), fill=(20, 20, 20), width=2)
            draw.line((sx0 - 6, sy0 + 6, sx0 + 6, sy0 - 6), fill=(20, 20, 20), width=2)
            continue
        sx, sy = map_point(x, y, float(z))
        stem_color = (155, 155, 155) if abs(float(z)) < 1.0 else (115, 115, 115)
        draw.line((sx0, sy0, sx, sy), fill=stem_color, width=1)
        radius = 5
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=_height_color(float(z), vlim_cm), outline=(20, 20, 20))

    x0, y0 = map_point(float(XS[0]), float(YS_TOP_TO_BOTTOM[-1]), 0.0)
    x1, y1 = map_point(float(XS[-1]), float(YS_TOP_TO_BOTTOM[-1]), 0.0)
    draw.line((x0, y0 + 20, x1, y1 + 20), fill=(80, 80, 80), width=1)
    yx0, yy0 = map_point(float(XS[-1]), float(YS_TOP_TO_BOTTOM[-1]), 0.0)
    yx1, yy1 = map_point(float(XS[-1]), float(YS_TOP_TO_BOTTOM[0]), 0.0)
    draw.line((yx0 + 18, yy0, yx1 + 18, yy1), fill=(80, 80, 80), width=1)


def _draw_minimal_panel(draw: ImageDraw.ImageDraw, snapshot: Snapshot, origin: tuple[int, int]) -> None:
    ox, oy = origin
    draw.text((ox, oy), snapshot.name, font=FONT_16, fill=(15, 15, 15))
    draw.text(
        (ox, oy + 25),
        f"p95-p05={_scaled_flatness_cm(snapshot):.1f} cm   valid={snapshot.raw_valid}/187   "
        f"sentinel={snapshot.sentinel}   critical={snapshot.critical_sentinel}   height scale={snapshot.height_data_scale:.1f}x",
        font=FONT_12,
        fill=(55, 55, 55),
    )
    panel_y = oy + 88
    _draw_minimal_heatmap(draw, snapshot, (ox + 18, panel_y), 27, 15.0)
    _draw_side_point_cloud(draw, snapshot, (ox + 570, panel_y), (500, 360), 15.0)
    _draw_colorbar(draw, ox + 1120, panel_y + 35, 235, 15.0)


def save_minimal_read_panel(snapshot: Snapshot, path: str) -> None:
    image = Image.new("RGB", (1230, 520), "white")
    draw = ImageDraw.Draw(image)
    _draw_minimal_panel(draw, snapshot, (45, 28))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    image.save(path)


def save_minimal_series(path: str) -> None:
    image = Image.new("RGB", (1230, 1010), "white")
    draw = ImageDraw.Draw(image)
    _draw_minimal_panel(draw, SNAPSHOTS[0], (45, 25))
    draw.line((45, 502, 1180, 502), fill=(225, 225, 225), width=1)
    _draw_minimal_panel(draw, SNAPSHOTS[1], (45, 525))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    image.save(path)


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(root, "logs", "height_noise")
    paths = [
        os.path.join(out_dir, "user_snapshot_a_2d_3d_panel.png"),
        os.path.join(out_dir, "user_snapshot_b_2d_3d_panel.png"),
        os.path.join(out_dir, "user_height_scan_noise_series.png"),
    ]
    save_minimal_read_panel(SNAPSHOTS[0], paths[0])
    save_minimal_read_panel(SNAPSHOTS[1], paths[1])
    save_minimal_series(paths[2])
    for path in paths:
        print("saved:", path)


if __name__ == "__main__":
    main()
