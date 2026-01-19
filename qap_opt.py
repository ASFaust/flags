#!/usr/bin/env python3
# qap_opt.py
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from flag import Flag
from hillclimb import render_grid_png
from seam_cost import blank_edges, build_seam_cost_tables


def build_tile_list(
    num_flags: int,
    grid_size: int,
    blank_idx: int,
    seed: int,
) -> List[int]:
    """
    Return a list of length grid_size containing flag indices or blank_idx.
    If there are more flags than slots, sample a subset.
    If there are fewer, pad with blanks (distinct tiles with identical costs).
    """
    rng = random.Random(seed)
    if num_flags >= grid_size:
        chosen = rng.sample(range(num_flags), grid_size)
        return chosen

    tiles = list(range(num_flags))
    tiles.extend([blank_idx] * (grid_size - num_flags))
    return tiles


def assignment_cost(
    assignment: np.ndarray,
    tile_cost_idx: List[int],
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    GH: int,
    GW: int,
) -> float:
    """
    QAP objective: sum of seam costs over right and down adjacencies.
    """
    cost = 0.0
    for r in range(GH):
        for c in range(GW):
            idx = assignment[r * GW + c]
            a = tile_cost_idx[idx]
            if c + 1 < GW:
                idx_r = assignment[r * GW + (c + 1)]
                b = tile_cost_idx[idx_r]
                cost += float(right_cost[a, b])
            if r + 1 < GH:
                idx_d = assignment[(r + 1) * GW + c]
                b = tile_cost_idx[idx_d]
                cost += float(down_cost[a, b])
    return float(cost)


def swap_delta_cost(
    assignment: np.ndarray,
    a: Tuple[int, int],
    b: Tuple[int, int],
    tile_cost_idx: List[int],
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    GH: int,
    GW: int,
) -> float:
    """
    Compute delta cost for swapping two positions by re-evaluating local edges.
    Returns new_cost - old_cost.
    """
    (ra, ca) = a
    (rb, cb) = b
    pos_a = ra * GW + ca
    pos_b = rb * GW + cb

    def tile_at(pos: Tuple[int, int]) -> int:
        r, c = pos
        return assignment[r * GW + c]

    def edge_cost(p: Tuple[int, int], q: Tuple[int, int]) -> float:
        r1, c1 = p
        r2, c2 = q
        idx1 = tile_cost_idx[tile_at(p)]
        idx2 = tile_cost_idx[tile_at(q)]
        if r1 == r2 and c2 == c1 + 1:
            return float(right_cost[idx1, idx2])
        if r1 == r2 and c2 == c1 - 1:
            return float(right_cost[idx2, idx1])
        if c1 == c2 and r2 == r1 + 1:
            return float(down_cost[idx1, idx2])
        if c1 == c2 and r2 == r1 - 1:
            return float(down_cost[idx2, idx1])
        raise ValueError("positions not adjacent")

    affected = set()

    def add_edges_around(r: int, c: int):
        if c + 1 < GW:
            affected.add(((r, c), (r, c + 1)))
        if c - 1 >= 0:
            affected.add(((r, c - 1), (r, c)))
        if r + 1 < GH:
            affected.add(((r, c), (r + 1, c)))
        if r - 1 >= 0:
            affected.add(((r - 1, c), (r, c)))

    add_edges_around(ra, ca)
    add_edges_around(rb, cb)

    old_cost = 0.0
    for p, q in affected:
        old_cost += edge_cost(p, q)

    assignment[pos_a], assignment[pos_b] = assignment[pos_b], assignment[pos_a]

    new_cost = 0.0
    for p, q in affected:
        new_cost += edge_cost(p, q)

    assignment[pos_a], assignment[pos_b] = assignment[pos_b], assignment[pos_a]
    return float(new_cost - old_cost)


def assignment_to_grid(
    assignment: np.ndarray,
    tile_cost_idx: List[int],
    blank_idx: int,
    GH: int,
    GW: int,
) -> np.ndarray:
    grid = np.full((GH, GW), -1, dtype=np.int32)
    for r in range(GH):
        for c in range(GW):
            tid = assignment[r * GW + c]
            v = tile_cost_idx[tid]
            grid[r, c] = -1 if v == blank_idx else v
    return grid


def simulated_annealing_qap(
    tile_cost_idx: List[int],
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    GH: int,
    GW: int,
    steps: int,
    seed: int,
    temp_start: float,
    temp_end: float,
    log_every: int,
):
    rng = random.Random(seed)
    P = GH * GW

    assignment = np.array(rng.sample(range(P), P), dtype=np.int32)
    cur_cost = assignment_cost(assignment, tile_cost_idx, right_cost, down_cost, GH, GW)

    best_cost = cur_cost
    best_assignment = assignment.copy()

    if steps <= 1:
        decay = 1.0
    else:
        ratio = temp_end / temp_start
        decay = ratio ** (1.0 / (steps - 1))

    for t in range(steps):
        if log_every > 0 and t % log_every == 0:
            print(f"\rstep={t:8d} cost={cur_cost:.2f}", end="", flush=True)

        a = rng.randrange(GH), rng.randrange(GW)
        b = rng.randrange(GH), rng.randrange(GW)
        while b == a:
            b = rng.randrange(GH), rng.randrange(GW)

        delta = swap_delta_cost(
            assignment, a, b, tile_cost_idx, right_cost, down_cost, GH, GW
        )

        accept = False
        if delta < 0:
            accept = True
        else:
            temp = temp_start * (decay ** t)
            if temp > 0.0:
                prob = math.exp(-delta / temp)
                accept = rng.random() < prob

        if accept:
            ia = a[0] * GW + a[1]
            ib = b[0] * GW + b[1]
            assignment[ia], assignment[ib] = assignment[ib], assignment[ia]
            cur_cost += delta
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_assignment = assignment.copy()

    print(f"\nfinal cost={cur_cost:.2f} best cost={best_cost:.2f}")
    return best_assignment, best_cost


def estimate_median_positive_delta(
    tile_cost_idx: List[int],
    right_cost: np.ndarray,
    down_cost: np.ndarray,
    GH: int,
    GW: int,
    samples: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    P = GH * GW
    assignment = np.array(rng.sample(range(P), P), dtype=np.int32)

    deltas = []
    for _ in range(samples):
        a = rng.randrange(GH), rng.randrange(GW)
        b = rng.randrange(GH), rng.randrange(GW)
        while b == a:
            b = rng.randrange(GH), rng.randrange(GW)
        delta = swap_delta_cost(
            assignment, a, b, tile_cost_idx, right_cost, down_cost, GH, GW
        )
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 0.0

    deltas.sort()
    mid = len(deltas) // 2
    if len(deltas) % 2 == 1:
        return float(deltas[mid])
    return float(0.5 * (deltas[mid - 1] + deltas[mid]))


def main():
    parser = argparse.ArgumentParser(
        description="Solve the flag mosaic as a quadratic assignment problem."
    )
    parser.add_argument("--gh", type=int, default=16, help="grid height")
    parser.add_argument("--gw", type=int, default=16, help="grid width")
    parser.add_argument("--steps", type=int, default=10_000_000, help="SA iterations")
    parser.add_argument("--seed", type=int, default=1125, help="random seed")
    parser.add_argument("--temp-start", type=float, default=27.44, help="start temperature")
    parser.add_argument("--temp-end", type=float, default=4.0, help="end temperature")
    parser.add_argument(
        "--auto-temp",
        action="store_true",
        help="set temp-start from median positive swap delta",
    )
    parser.add_argument(
        "--temp-samples",
        type=int,
        default=1000,
        help="random swap samples for auto-temp",
    )
    parser.add_argument("--log-every", type=int, default=1000, help="log cadence")
    parser.add_argument("--render-dir", type=str, default="", help="optional render dir")
    parser.add_argument("--render-scale", type=int, default=4, help="render scale")
    args = parser.parse_args()

    flags = Flag.load_flags(mmap=True, copy_edges=True)
    H, W, _ = flags[0].rgb.shape
    blank = blank_edges(height=H, width=W, rgb=(255, 255, 255))
    right_cost, down_cost = build_seam_cost_tables(flags, blank)
    blank_idx = len(flags)

    P = args.gh * args.gw
    tile_cost_idx = build_tile_list(len(flags), P, blank_idx, seed=args.seed)

    temp_start = args.temp_start
    if args.auto_temp:
        median_delta = estimate_median_positive_delta(
            tile_cost_idx=tile_cost_idx,
            right_cost=right_cost,
            down_cost=down_cost,
            GH=args.gh,
            GW=args.gw,
            samples=args.temp_samples,
            seed=args.seed,
        )
        if median_delta > 0:
            temp_start = median_delta / math.log(10.0)
            print(
                f"auto temp-start={temp_start:.2f} "
                f"(median positive delta={median_delta:.2f})"
            )
        else:
            print("auto temp-start skipped (no positive deltas found)")

    best_assignment, _ = simulated_annealing_qap(
        tile_cost_idx=tile_cost_idx,
        right_cost=right_cost,
        down_cost=down_cost,
        GH=args.gh,
        GW=args.gw,
        steps=args.steps,
        seed=args.seed,
        temp_start=temp_start,
        temp_end=args.temp_end,
        log_every=args.log_every,
    )

    if args.render_dir:
        grid = assignment_to_grid(best_assignment, tile_cost_idx, blank_idx, args.gh, args.gw)
        render_dir = Path(args.render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)
        render_grid_png(
            flags,
            grid,
            render_dir / "frame_qap_best.png",
            scale=args.render_scale,
        )
        print(f"Saved render to {render_dir / 'frame_qap_best.png'}")


if __name__ == "__main__":
    main()
