"""In-process duplicate match, no HTTP.

    python tools/sim.py A B [--deals 1000] [--seed 1]

A bot is a file path (agent.py, baseline-bots/...) or strat[:NAME=VAL,...]
for strategy.py with some constants overridden.
"""

import argparse
import importlib.util
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import (Action, AgentConfig, Game, button_for_deal, default_action,  # noqa: E402
                  make_deal)

_counter = [0]


def _load_file(path):
    _counter[0] += 1
    spec = importlib.util.spec_from_file_location(f"bot{_counter[0]}", os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Bot:
    def __init__(self, spec):
        self.spec = spec
        if spec.startswith("strat"):
            mod = _load_file("strategy.py")
            if ":" in spec:
                for kv in spec.split(":", 1)[1].split(","):
                    k, v = kv.split("=")
                    setattr(mod, k, eval(v))
            self.new_half = lambda cfg, mem: mod.new_half(mem, cfg.num_hands)
            self.choose_action = mod.choose
            self.on_hand_end = mod.on_hand_end
        else:
            mod = _load_file(spec)
            self.new_half = mod.new_half
            self.choose_action = mod.choose_action
            self.on_hand_end = mod.on_hand_end


def play_half(bots, seed, deals, half):
    """Returns seat 0's net per deal, plus time and error counts per seat."""
    mems = [{}, {}]
    cfg = AgentConfig(num_hands=deals)
    times = [0.0, 0.0]
    errors = [0, 0]
    for s in (0, 1):
        bots[s].new_half(cfg, mems[s])
    nets = []
    for d in range(deals):
        deal = make_deal(seed, d)
        g = Game(deal, button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            s = g.to_act
            obs = g.observation(s)
            t = time.perf_counter()
            try:
                a = bots[s].choose_action(obs, mems[s])
            except Exception:
                a = None
                errors[s] += 1
            times[s] += time.perf_counter() - t
            if a not in obs.legal_actions:
                if a is not None:
                    errors[s] += 1
                a = default_action(obs.to_call)
            g.apply(a)
        for s in (0, 1):
            t = time.perf_counter()
            try:
                bots[s].on_hand_end(g.hand_result(s), mems[s])
            except Exception:
                errors[s] += 1
            times[s] += time.perf_counter() - t
        nets.append(g.result().nets[0])
    return nets, times, errors


def match(a_spec, b_spec, deals, seed, quiet=False):
    A, B = Bot(a_spec), Bot(b_spec)
    n1, t1, e1 = play_half([A, B], seed, deals, 1)
    n2, t2, e2 = play_half([B, A], seed, deals, 2)
    per = [(n1[i] - n2[i]) / 2.0 for i in range(deals)]
    mean = sum(per) / deals
    var = sum((x - mean) ** 2 for x in per) / (deals - 1)
    se = math.sqrt(var / deals)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    verdict = "A_WIN" if lo > 0 else ("B_WIN" if hi < 0 else "CHOP")
    if not quiet:
        print(f"{a_spec} vs {b_spec}: {mean:+.3f} bb/deal  CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
        print(f"   A time {t1[0]:.2f}s/{t2[1]:.2f}s errors {e1[0]}/{e2[1]} | "
              f"B time {t1[1]:.2f}s/{t2[0]:.2f}s errors {e1[1]}/{e2[0]}")
    return mean, lo, hi


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--seed", default="1")
    args = p.parse_args()
    match(args.a, args.b, args.deals, args.seed)
