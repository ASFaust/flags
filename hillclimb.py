#!/usr/bin/env python3
# hillclimb.py
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import itertools
from PIL import Image
import matplotlib.pyplot as plt

from flag import Flag
from seam_cost import blank_edges, build_seam_cost_tables


@dataclass
class PlacedGrid:
    grid: np.ndarray  # shape [GH, GW], int32, flag index or -1 for blank


def neighbors4(r: int, c: int, GH: int, GW: int):
    if r > 0:
        yield (r - 1, c, "T")  # neighbor is above -> we match our TOP with their BOTTOM
    if r + 1 < GH:
        yield (r + 1, c, "B")
    if c > 0:
        yield (r, c - 1, "L")
    if c + 1 < GW:
        yield (r, c + 1, "R")


def placement_cost_at(
    grid: np.ndarray,
    r: int,
    c: int,
    flag_idx: int,
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    blank_idx: int,
) -> float:
    """
    Cost of placing flag_idx at (r,c) given current partial grid.
    Only sums seams to already-placed neighbors.
    """
    GH, GW = grid.shape
    cost = 0.0
    for rr, cc, side in neighbors4(r, c, GH, GW):
        nb = grid[rr, cc]
        if nb == -2:  # empty marker
            continue

        nb_idx = blank_idx if nb == -1 else nb

        # side describes where neighbor is relative to us
        if side == "L":
            cost += float(right_cost[nb_idx, flag_idx])
        elif side == "R":
            cost += float(right_cost[flag_idx, nb_idx])
        elif side == "T":
            cost += float(down_cost[nb_idx, flag_idx])
        elif side == "B":
            cost += float(down_cost[flag_idx, nb_idx])

    return cost


def greedy_frontier_init(
    flags: List[Flag],
    GH: int = 16,
    GW: int = 16,
    seed: int = 0,
    topk: int = 5,
) -> PlacedGrid:
    """
    Greedy frontier growth:
    - start with 1 random flag at random location
    - expand into frontier cells
    - choose most constrained frontier cell first (max placed neighbors)
    - choose among top-k flags randomly (greedy-but-randomized)

    Uses:
      -2 = empty
      -1 = blank (only used at end)
      0..N-1 = flags
    """
    rng = random.Random(seed)
    N = len(flags)

    H, W, _ = flags[0].rgb.shape
    blank = blank_edges(height=H, width=W, rgb=(255, 255, 255))
    right_cost, down_cost = build_seam_cost_tables(flags, blank)
    blank_idx = len(flags)

    grid = np.full((GH, GW), -2, dtype=np.int32)
    remaining = set(range(N))

    # random seed placement
    start_flag = rng.choice(list(remaining))
    start_r = rng.randrange(GH)
    start_c = rng.randrange(GW)

    grid[start_r, start_c] = start_flag
    remaining.remove(start_flag)

    # frontier: empty cells adjacent to placed tiles
    frontier = set()
    for rr, cc, _ in neighbors4(start_r, start_c, GH, GW):
        if grid[rr, cc] == -2:
            frontier.add((rr, cc))

    while remaining and frontier:
        # choose most constrained frontier cell first: max number of placed neighbors
        frontier_list = list(frontier)

        def constraint_score(cell):
            r, c = cell
            cnt = 0
            for rr, cc, _ in neighbors4(r, c, GH, GW):
                if grid[rr, cc] != -2:
                    cnt += 1
            return cnt

        max_constr = max(constraint_score(cell) for cell in frontier_list)
        constrained_cells = [cell for cell in frontier_list if constraint_score(cell) == max_constr]

        # tie-break randomly among equally constrained
        r, c = rng.choice(constrained_cells)

        # score all remaining flags for this cell
        scored = []
        for fi in remaining:
            cost = placement_cost_at(grid, r, c, fi, right_cost, down_cost, blank_idx)
            scored.append((cost, fi))

        scored.sort(key=lambda x: x[0])

        # choose among top-k randomly (but biased towards best)
        k = min(topk, len(scored))
        candidates = scored[:k]

        # simple biased sampling: weights = 1/(rank+1)
        weights = np.array([1.0 / (i + 1) for i in range(k)], dtype=np.float64)
        weights /= weights.sum()
        chosen_i = int(np.random.choice(np.arange(k), p=weights))
        chosen_flag = candidates[chosen_i][1]

        # place it
        grid[r, c] = chosen_flag
        remaining.remove(chosen_flag)
        frontier.remove((r, c))

        # update frontier around (r,c)
        for rr, cc, _ in neighbors4(r, c, GH, GW):
            if grid[rr, cc] == -2:
                frontier.add((rr, cc))

    # fill any leftover empties with blanks
    grid[grid == -2] = -1
    return PlacedGrid(grid=grid)


def total_grid_cost(
    grid: np.ndarray,
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    blank_idx: int,
) -> float:
    """
    Total seam cost of the whole grid (sum over right + bottom seams).
    """
    GH, GW = grid.shape
    cost = 0.0

    for r in range(GH):
        for c in range(GW):
            a = blank_idx if grid[r, c] == -1 else grid[r, c]

            if c + 1 < GW:
                b = blank_idx if grid[r, c + 1] == -1 else grid[r, c + 1]
                cost += float(right_cost[a, b])

            if r + 1 < GH:
                b = blank_idx if grid[r + 1, c] == -1 else grid[r + 1, c]
                cost += float(down_cost[a, b])

    return float(cost)


def render_grid_png(
    flags: List[Flag],
    grid: np.ndarray,
    out_path: Path,
    blank_rgb=(255, 255, 255),
    scale: int = 1,
):
    """
    Renders the grid to a PNG.
    """
    GH, GW = grid.shape
    H, W, _ = flags[0].rgb.shape

    canvas = np.zeros((GH * H, GW * W, 3), dtype=np.uint8)
    canvas[:] = np.array(blank_rgb, dtype=np.uint8)

    for r in range(GH):
        for c in range(GW):
            idx = grid[r, c]
            if idx >= 0:
                tile = flags[idx].rgb
            else:
                tile = np.full((H, W, 3), blank_rgb, dtype=np.uint8)

            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = tile

    img = Image.fromarray(canvas, mode="RGB")
    if scale != 1:
        img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.Resampling.NEAREST)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)


def swap_delta_cost_local(
    grid: np.ndarray,
    pos_a: Tuple[int, int],
    pos_b: Tuple[int, int],
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    blank_idx: int,
) -> float:
    """
    Compute delta cost for swapping two tiles, by only re-evaluating seams
    touching the swapped positions.

    Returns:
      new_cost - old_cost  (negative is improvement)
    """
    GH, GW = grid.shape

    def seam_cost_between(pos1, pos2) -> float:
        r1, c1 = pos1
        r2, c2 = pos2
        t1 = blank_idx if grid[r1, c1] == -1 else grid[r1, c1]
        t2 = blank_idx if grid[r2, c2] == -1 else grid[r2, c2]

        if r1 == r2 and c2 == c1 + 1:  # pos2 is right of pos1
            return float(right_cost[t1, t2])
        if r1 == r2 and c2 == c1 - 1:  # pos2 is left
            return float(right_cost[t2, t1])
        if c1 == c2 and r2 == r1 + 1:  # pos2 below
            return float(down_cost[t1, t2])
        if c1 == c2 and r2 == r1 - 1:  # pos2 above
            return float(down_cost[t2, t1])
        raise ValueError("positions not adjacent")

    # collect affected undirected edges (as ordered adjacent pairs)
    affected = set()

    def add_edges_around(r, c):
        if c + 1 < GW:
            affected.add(((r, c), (r, c + 1)))
        if c - 1 >= 0:
            affected.add(((r, c - 1), (r, c)))
        if r + 1 < GH:
            affected.add(((r, c), (r + 1, c)))
        if r - 1 >= 0:
            affected.add(((r - 1, c), (r, c)))

    ra, ca = pos_a
    rb, cb = pos_b
    add_edges_around(ra, ca)
    add_edges_around(rb, cb)

    # compute old cost for affected seams
    old = 0.0
    for p, q in affected:
        old += seam_cost_between(p, q)

    # do swap in-place
    grid[ra, ca], grid[rb, cb] = grid[rb, cb], grid[ra, ca]

    # compute new cost
    new = 0.0
    for p, q in affected:
        new += seam_cost_between(p, q)

    # swap back
    grid[ra, ca], grid[rb, cb] = grid[rb, cb], grid[ra, ca]

    return float(new - old)


def hillclimb_swaps(
    flags: List[Flag],
    grid: np.ndarray,
    steps: int = 200_000,
    seed: int = 0,
    render_dir: str = "frames",
    render_scale: int = 4,
    render_every: int = 1,
):
    rng = random.Random(seed)

    H, W, _ = flags[0].rgb.shape
    blank = blank_edges(height=H, width=W, rgb=(255, 255, 255))
    right_cost, down_cost = build_seam_cost_tables(flags, blank)
    blank_idx = len(flags)

    GH, GW = grid.shape

    # initial cost
    cur_cost = total_grid_cost(grid, right_cost, down_cost, blank_idx)
    history = [cur_cost]

    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    # save initial
    render_grid_png(flags, grid, render_dir / f"frame_{0:06d}.png", scale=render_scale)

    improvements = 0
    frame_id = 1

    HOR = 0
    VER = 1

    def edge_cost_sum(affected, tiles) -> float:
        total = 0.0
        for orient, r1, c1, r2, c2, idx1, idx2 in affected:
            t1 = tiles[idx1] if idx1 >= 0 else grid[r1, c1]
            t2 = tiles[idx2] if idx2 >= 0 else grid[r2, c2]
            if t1 == -1:
                t1 = blank_idx
            if t2 == -1:
                t2 = blank_idx
            if orient == HOR:
                total += right_cost[t1, t2]
            else:
                total += down_cost[t1, t2]
        return float(total)

    for t in range(steps):
        if t % 1000 == 0:
            print(f"\rstep={t:8d} cost={cur_cost:.2f}", end="", flush=True)

        # sample four distinct positions
        pos_set = set()
        while len(pos_set) < 4:
            pos_set.add((rng.randrange(GH), rng.randrange(GW)))
        positions = list(pos_set)

        old_tiles = [grid[r, c] for r, c in positions]
        pos_index = {pos: i for i, pos in enumerate(positions)}

        edge_keys = set()
        affected = []

        def add_edge(r1, c1, r2, c2):
            if r1 == r2:
                if c1 < c2:
                    a = (r1, c1)
                    b = (r2, c2)
                else:
                    a = (r2, c2)
                    b = (r1, c1)
                key = (a[0], a[1], b[0], b[1])
                if key in edge_keys:
                    return
                edge_keys.add(key)
                idx1 = pos_index.get(a, -1)
                idx2 = pos_index.get(b, -1)
                affected.append((HOR, a[0], a[1], b[0], b[1], idx1, idx2))
            else:
                if r1 < r2:
                    a = (r1, c1)
                    b = (r2, c2)
                else:
                    a = (r2, c2)
                    b = (r1, c1)
                key = (a[0], a[1], b[0], b[1])
                if key in edge_keys:
                    return
                edge_keys.add(key)
                idx1 = pos_index.get(a, -1)
                idx2 = pos_index.get(b, -1)
                affected.append((VER, a[0], a[1], b[0], b[1], idx1, idx2))

        for r, c in positions:
            if c + 1 < GW:
                add_edge(r, c, r, c + 1)
            if c - 1 >= 0:
                add_edge(r, c - 1, r, c)
            if r + 1 < GH:
                add_edge(r, c, r + 1, c)
            if r - 1 >= 0:
                add_edge(r - 1, c, r, c)

        old_cost = edge_cost_sum(affected, old_tiles)
        best_delta = 0.0
        best_perm = old_tiles

        # try all permutations of the 4 tiles
        for perm in itertools.permutations(old_tiles, 4):
            new_cost = edge_cost_sum(affected, perm)
            delta = new_cost - old_cost
            if delta < best_delta:
                best_delta = delta
                best_perm = list(perm)

        if best_delta < 0.0:
            # apply best permutation
            for (r, c), v in zip(positions, best_perm):
                grid[r, c] = v

            cur_cost += best_delta
            history.append(cur_cost)

            improvements += 1

            # render EVERY improvement
            if render_every > 0 and improvements % render_every == 0:
                render_grid_png(
                    flags,
                    grid,
                    render_dir / f"frame_{frame_id:06d}.png",
                    scale=render_scale,
                )
                frame_id += 1

            print(
                f"[improve {improvements:6d}] step={t:8d} cost={cur_cost:.2f} delta={best_delta:.2f}"
            )

    # always save final frame, independent of render_every
    render_grid_png(
        flags,
        grid,
        render_dir / f"frame_{frame_id:06d}.png",
        scale=render_scale,
    )

    return history


def plot_cost(history: List[float], out_path: str = "cost.png"):
    plt.figure()
    plt.plot(history)
    plt.xlabel("improvement step")
    plt.ylabel("total seam cost")
    plt.title("Hillclimb cost over improvements")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    flags = Flag.load_flags(mmap=True, copy_edges=True)

    init = greedy_frontier_init(flags, GH=16, GW=16, seed=0, topk=5)
    grid = init.grid

    history = hillclimb_swaps(
        flags=flags,
        grid=grid,
        steps=1_000_000,
        seed=1,
        render_dir="frames",
        render_scale=4,
        render_every=100,
    )

    plot_cost(history, out_path="cost.png")
    print("Saved cost plot to cost.png")


if __name__ == "__main__":
    main()
