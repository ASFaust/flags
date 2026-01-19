# seam_cost.py
from __future__ import annotations
import numpy as np

from typing import Iterable, Tuple


def srgb_to_lab(edge: np.ndarray) -> np.ndarray:
    """
    Convert sRGB edge strip to CIE Lab (D65).

    edge:
      shape [L, 3], dtype uint8 (or float 0-255)
    Returns:
      shape [L, 3], dtype float32
    """
    rgb = edge.astype(np.float32) / 255.0
    # inverse sRGB companding
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # sRGB to XYZ (D65)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = lin @ m.T

    # normalize by D65 reference white
    x = xyz[:, 0] / 0.95047
    y = xyz[:, 1] / 1.00000
    z = xyz[:, 2] / 1.08883

    epsilon = 216.0 / 24389.0  # (6/29)^3
    kappa = 24389.0 / 27.0
    fx = np.where(x > epsilon, x ** (1.0 / 3.0), (kappa * x + 16.0) / 116.0)
    fy = np.where(y > epsilon, y ** (1.0 / 3.0), (kappa * y + 16.0) / 116.0)
    fz = np.where(z > epsilon, z ** (1.0 / 3.0), (kappa * z + 16.0) / 116.0)

    lab = np.empty_like(xyz, dtype=np.float32)
    lab[:, 0] = 116.0 * fy - 16.0
    lab[:, 1] = 500.0 * (fx - fy)
    lab[:, 2] = 200.0 * (fy - fz)
    return lab


def seam_cost_lab(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    """
    Fast seam cost between two 1px edge strips.

    edge_a, edge_b:
      shape [L, 3], dtype uint8 (or float)
    Returns:
      scalar float cost (mean Delta E in Lab)
    """
    lab_a = srgb_to_lab(edge_a)
    lab_b = srgb_to_lab(edge_b)
    return seam_cost_lab_from_lab(lab_a, lab_b)


def seam_cost_lab_from_lab(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """
    Mean Delta E cost between two Lab edge strips.
    """
    delta = lab_a - lab_b
    de = np.sqrt(np.sum(delta * delta, axis=1))
    return float(np.mean(de))


def blank_edges(height: int, width: int, rgb=(255, 255, 255)) -> dict[str, np.ndarray]:
    """
    Creates 1px edges for a blank tile (constant color).
    """
    r, g, b = rgb
    left = np.full((height, 3), (r, g, b), dtype=np.uint8)
    right = np.full((height, 3), (r, g, b), dtype=np.uint8)
    top = np.full((width, 3), (r, g, b), dtype=np.uint8)
    bottom = np.full((width, 3), (r, g, b), dtype=np.uint8)
    return {"L": left, "R": right, "T": top, "B": bottom}


def build_seam_cost_tables(
    flags: Iterable,
    blank: dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute seam costs into lookup tables.

    Returns:
      (right_cost, down_cost), both shape [(N+1), (N+1)]
      where index N refers to the blank tile.
    """
    flags = list(flags)
    n = len(flags)

    left_edges = [f.edge_left for f in flags] + [blank["L"]]
    right_edges = [f.edge_right for f in flags] + [blank["R"]]
    top_edges = [f.edge_top for f in flags] + [blank["T"]]
    bottom_edges = [f.edge_bottom for f in flags] + [blank["B"]]

    left_lab = [srgb_to_lab(e) for e in left_edges]
    right_lab = [srgb_to_lab(e) for e in right_edges]
    top_lab = [srgb_to_lab(e) for e in top_edges]
    bottom_lab = [srgb_to_lab(e) for e in bottom_edges]

    right_cost = np.zeros((n + 1, n + 1), dtype=np.float32)
    down_cost = np.zeros((n + 1, n + 1), dtype=np.float32)

    for i in range(n + 1):
        for j in range(n + 1):
            right_cost[i, j] = seam_cost_lab_from_lab(right_lab[i], left_lab[j])
            down_cost[i, j] = seam_cost_lab_from_lab(bottom_lab[i], top_lab[j])

    return right_cost, down_cost
