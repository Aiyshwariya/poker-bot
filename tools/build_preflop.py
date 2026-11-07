"""Builds preflop_table.json.

For each starting hand (up to suit relabeling): equity vs a random hand,
vs the top 40%, vs the top 15%, and the hand's percentile.

    python tools/build_preflop.py [samples_per_hand]
"""

import json
import os
import random
import sys
from itertools import combinations, permutations
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import get_table  # noqa: E402

PERMS = list(permutations(range(4)))
_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))


def canon(hand):
    """Canonical form of a 5-card hand under suit relabeling."""
    best = None
    for p in PERMS:
        t = tuple(sorted(((c >> 2) << 2) | p[c & 3] for c in hand))
        if best is None or t < best:
            best = t
    return best


def score(table, hole, board):
    hb = [1 << c for c in hole]
    bb = [1 << c for c in board]
    hm = [hb[i] | hb[j] for i, j in _PAIRS]
    bm = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]
    best = 0
    for h in hm:
        for b in bm:
            s = table[h | b]
            if s > best:
                best = s
    return best


G = {}


def _init(opp_pools):
    G["table"] = get_table()
    G["pools"] = opp_pools


def _work(args):
    cls, n = args
    table = G["table"]
    rng = random.Random(hash(cls) & 0xFFFFFFFF)
    hero = list(cls)
    hero_set = set(hero)
    deck = [c for c in range(36) if c not in hero_set]
    out = []
    for pool in G["pools"]:
        tot = 0.0
        k = 0
        while k < n:
            if pool is None:
                opp = rng.sample(deck, 5)
            else:
                opp = pool[rng.randrange(len(pool))]
                if hero_set.intersection(opp):
                    continue
            used = hero_set.union(opp)
            rest = [c for c in deck if c not in used]
            board = rng.sample(rest, 5)
            a = score(table, hero, board)
            b = score(table, opp, board)
            tot += 1.0 if a > b else (0.5 if a == b else 0.0)
            k += 1
        out.append(tot / n)
    return cls, out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    get_table()
    print("enumerating classes...")
    hand_class = {}
    for hand in combinations(range(36), 5):
        hand_class[hand] = canon(hand)
    classes = sorted(set(hand_class.values()))
    print(len(classes), "classes")

    with Pool(initializer=_init, initargs=([None],)) as pool:
        res = dict(pool.map(_work, [(c, n) for c in classes], chunksize=64))
    eq0 = {c: v[0] for c, v in res.items()}

    hands_sorted = sorted(hand_class, key=lambda h: eq0[hand_class[h]], reverse=True)
    top40 = hands_sorted[: int(0.40 * len(hands_sorted))]
    top15 = hands_sorted[: int(0.15 * len(hands_sorted))]
    print("pass 2...")
    with Pool(initializer=_init, initargs=([top40, top15],)) as pool:
        res2 = dict(pool.map(_work, [(c, n) for c in classes], chunksize=64))

    total = len(hands_sorted)
    pct = {}
    for i, h in enumerate(hands_sorted):
        c = hand_class[h]
        if c not in pct:
            pct[c] = 1.0 - i / total
    data = {
        ",".join(map(str, c)): [round(eq0[c], 4), round(res2[c][0], 4), round(res2[c][1], 4), round(pct[c], 4)]
        for c in classes
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preflop_table.json")
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", out, len(data))


if __name__ == "__main__":
    main()
