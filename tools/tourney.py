"""Parallel sim matches: python tools/tourney.py BOT [BOT ...] [--opps ...] [--seeds 2]"""
import argparse, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sim

DEFAULT_OPPS = [
    "baseline-bots/baseline_random.py",
    "baseline-bots/baseline_calling_station.py",
    "baseline-bots/baseline_always_pot.py",
    "tools/refbots/guide_example.py",
    "tools/refbots/simple_equity.py",
    "strat:VB=0.45,SB_OPEN_PCT=0.3,BB_RAISE_PCT=0.4,RAISE_EQ=0.52",
    "strat:VB=0.66,SB_OPEN_PCT=0.85,SB_FOLD_PCT=0.3,PF_BONUS=-0.04",
    "strat",
]

def run(args):
    a, b, seed, deals = args
    return a, b, seed, sim.match(a, b, deals, seed, quiet=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a", nargs="+")
    p.add_argument("--opps", nargs="*", default=DEFAULT_OPPS)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--deals", type=int, default=1000)
    x = p.parse_args()
    jobs = [(a, o, f"s{s}", x.deals) for a in x.a for o in x.opps for s in range(x.seeds)]
    with Pool(min(8, len(jobs))) as pool:
        res = pool.map(run, jobs, chunksize=1)
    for a in x.a:
        print(f"== {a}")
        tot = 0
        for o in x.opps:
            rs = [r[3] for r in res if r[0] == a and r[1] == o]
            m = sum(r[0] for r in rs) / len(rs)
            tot += m
            v = "".join("W" if r[1] > 0 else "L" if r[2] < 0 else "c" for r in rs)
            print(f"   {m:+7.3f}  {v:4s} vs {o}")
        print(f"   avg {tot/len(x.opps):+.3f}")
