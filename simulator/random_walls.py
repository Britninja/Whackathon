"""Génère des murs aléatoires (JSON) pour tester la perception : python random_walls.py
Puis :  uv run --with opencv-python python wall_holds.py --sim random_walls/r100_3.json --move 0.1 --seed 1
Les murs 'c' ont des prises groupées (le cas le plus dur)."""
import pathlib

import numpy as np
from climbing_sim import Wall


def gen(n, seed, cluster=False):
    rng = np.random.default_rng(seed)
    w = Wall(name=f"r{n}_{seed}{'c' if cluster else ''}")
    ctr, tries = None, 0
    while len(w.holds) < n and tries < 20000:
        tries += 1
        if cluster:
            if ctr is None or rng.random() < 0.02:
                ctr = (rng.integers(5, 45), rng.integers(8, 85))
            c, r = int(ctr[0] + rng.normal(0, 5)), int(ctr[1] + rng.normal(0, 6))
        else:
            c, r = int(rng.integers(0, 50)), int(rng.integers(0, 93))
        if 0 <= c <= 49 and 0 <= r <= 92:
            try:
                w.add(c, r)                       # refuse les prises qui se chevauchent ou couvrent un marqueur de coin
            except ValueError:
                pass
    return w


if __name__ == "__main__":
    out = pathlib.Path("random_walls"); out.mkdir(exist_ok=True)
    for n, seed, cl in [(30, 1, 0), (60, 2, 0), (100, 3, 0), (150, 4, 0), (60, 5, 1), (100, 6, 1), (150, 7, 1)]:
        w = gen(n, seed, bool(cl)); w.to_json(out / f"{w.name}.json"); print(out / f"{w.name}.json", len(w.holds), "prises")