# flag.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Flag:
    """
    Represents a single normalized flag image.

    Image:
      rgb: uint8, shape [H, W, 3]

    Precomputed edge strips (1px thick) for fast seam comparisons:
      edge_left:   uint8, shape [H, 3]
      edge_right:  uint8, shape [H, 3]
      edge_top:    uint8, shape [W, 3]
      edge_bottom: uint8, shape [W, 3]
    """
    idx: int
    iso2: str
    rgb: np.ndarray

    edge_left: np.ndarray
    edge_right: np.ndarray
    edge_top: np.ndarray
    edge_bottom: np.ndarray

    meta: dict

    @property
    def name(self) -> Optional[str]:
        return self.meta.get("name", None)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    def copy_rgb(self) -> np.ndarray:
        return np.array(self.rgb, copy=True)

    def to_float01(self) -> np.ndarray:
        return self.rgb.astype(np.float32) / 255.0

    # ----------------------------
    # Internal: edge extraction
    # ----------------------------

    @staticmethod
    def _compute_edges(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 1-pixel edge strips.

        Returns:
          (left, right, top, bottom)
        """
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected rgb shape [H,W,3], got {rgb.shape}")

        # NOTE: these are views (no copies) as long as rgb is contiguous.
        left = rgb[:, 0, :]      # [H, 3]
        right = rgb[:, -1, :]    # [H, 3]
        top = rgb[0, :, :]       # [W, 3]
        bottom = rgb[-1, :, :]   # [W, 3]

        return left, right, top, bottom

    # ----------------------------
    # Loading
    # ----------------------------

    @staticmethod
    def load_flags(
        npy_path: str | Path = "flags_2x3_uint8.npy",
        meta_path: str | Path = "flags_2x3_meta.json",
        mmap: bool = True,
        copy_edges: bool = True,
    ) -> List["Flag"]:
        """
        Loads all normalized flags from the shared numpy file + metadata JSON.

        For computational efficiency, edge strips are precomputed on load.

        Parameters
        ----------
        npy_path:
            Path to flags_2x3_uint8.npy
        meta_path:
            Path to flags_2x3_meta.json
        mmap:
            If True, memory-map the numpy array (good for large arrays).
        copy_edges:
            If True, edges are copied into separate small arrays (recommended).
            If False, edges are views into the shared rgb array.
        """
        npy_path = Path(npy_path)
        meta_path = Path(meta_path)

        if not npy_path.exists():
            raise FileNotFoundError(f"Missing flags npy: {npy_path.resolve()}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing flags meta json: {meta_path.resolve()}")

        arr = np.load(npy_path, mmap_mode="r" if mmap else None)  # [N, H, W, 3]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Expected array shape [N,H,W,3], got {arr.shape}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta_list = json.load(f)

        if len(meta_list) != arr.shape[0]:
            raise ValueError(
                f"Metadata length mismatch: meta={len(meta_list)} vs npy N={arr.shape[0]}"
            )

        flags: List[Flag] = []

        for i, meta in enumerate(meta_list):
            iso2 = meta.get("iso2", None)
            if not iso2:
                raise ValueError(f"Missing iso2 in meta entry {i}: {meta}")

            rgb = arr[i]  # view into shared array (no copy)

            left, right, top, bottom = Flag._compute_edges(rgb)

            if copy_edges:
                # Make tiny contiguous copies so comparisons are fast and cache-friendly.
                left = np.ascontiguousarray(left)
                right = np.ascontiguousarray(right)
                top = np.ascontiguousarray(top)
                bottom = np.ascontiguousarray(bottom)

            flags.append(
                Flag(
                    idx=i,
                    iso2=str(iso2),
                    rgb=rgb,
                    edge_left=left,
                    edge_right=right,
                    edge_top=top,
                    edge_bottom=bottom,
                    meta=meta,
                )
            )

        return flags

    @staticmethod
    def load_flag_map(
        npy_path: str | Path = "flags_2x3_uint8.npy",
        meta_path: str | Path = "flags_2x3_meta.json",
        mmap: bool = True,
        copy_edges: bool = True,
    ) -> dict[str, "Flag"]:
        """
        Convenience: returns dict mapping iso2 -> Flag.
        """
        flags = Flag.load_flags(npy_path=npy_path, meta_path=meta_path, mmap=mmap, copy_edges=copy_edges)
        return {f.iso2: f for f in flags}

if __name__ == "__main__":
    # Simple test: load flags and print some info.
    flags = Flag.load_flags()
    print(f"Loaded {len(flags)} flags.")
    for f in flags[:5]:
        print(f"Flag {f.idx}: iso2={f.iso2}, name={f.name}, size=({f.width}x{f.height})")