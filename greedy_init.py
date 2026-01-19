#!/usr/bin/env python3
# greedy_init.py
from __future__ import annotations

import numpy as np

from flag import Flag
from hillclimb import greedy_frontier_init


if __name__ == "__main__":
    flags = Flag.load_flags()
    pg = greedy_frontier_init(flags, GH=16, GW=16, seed=0, topk=5)
    print(pg.grid.shape, pg.grid.dtype)
    print("flags used:", int(np.sum(pg.grid >= 0)))
    print("blanks:", int(np.sum(pg.grid == -1)))
