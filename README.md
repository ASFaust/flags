# Flags Mosaic Optimizer

This project downloads national flags, normalizes them to a common 2:3 aspect ratio,
and searches for a low-seam-cost grid arrangement. It includes a greedy frontier
initializer and a multi-tile hill-climbing optimizer that renders improvement frames as PNGs.

## What is in here

- `dl_flags.py` downloads flag PNGs from flagcdn.com into `flags_png/`.
- `normalize_flags.py` crops transparency, stretches to 2:3, and saves:
  - `flags_2x3_uint8.npy` (N x 60 x 90 x 3 uint8)
  - `flags_2x3_meta.json` (per-flag metadata)
  - `flags_norm_2x3/` (normalized PNGs)
- `flag.py` loads the normalized flags and precomputes 1px edges.
- `greedy_init.py` is a thin wrapper around the greedy frontier initializer in `hillclimb.py`.
- `hillclimb.py` builds a greedy initial grid, then repeatedly picks 4 random positions,
  tests all permutations of those tiles, and applies the best improvement before rendering frames to `frames/`.
- `qap_opt.py` runs a simulated-annealing solver over the same seam-cost objective,
  with optional rendering of the best assignment.
- `seam_cost.py` builds seam-cost lookup tables for fast right/down edge scoring.

## Quick start

Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pillow requests pycountry matplotlib
```

Download and normalize flags:

```bash
python dl_flags.py
python normalize_flags.py
```

Generate a greedy initial grid and run hill-climbing permutation swaps (single entrypoint):

```bash
python hillclimb.py
```

Run the quadratic assignment solver (simulated annealing) and render the best grid:

```bash
python qap_opt.py --auto-temp --render-dir frames
```

The optimizer writes frames to `frames/` every N improvements (default is every 100, scale 4x).
You can stitch them into a video using ffmpeg:

```bash
ffmpeg -framerate 30 -i frames/frame_%06d.png -pix_fmt yuv420p out.mp4
```

## Notes

- The normalizer expects RGBA flags with transparent padding from `dl_flags.py`.
- Blank tiles are treated as white in seam scoring and rendering.
- The hill climber samples 4 positions per step and tries all 24 permutations, accepting only improvements.
- The default grid size is 16x16 in `greedy_init.py` and `hillclimb.py`.

## File layout

- `flags_png/`: downloaded square PNGs with transparent padding
- `flags_norm_2x3/`: normalized PNGs for inspection
- `flags_2x3_uint8.npy`: normalized RGB array
- `flags_2x3_meta.json`: metadata for each flag
- `frames/`: hill-climb render output
