"""Strategy for heads-up short deck PLO5.

Preflop uses a precomputed equity table (preflop_table.json). Postflop uses
Monte Carlo equity against the top fraction of opponent hands, where that
fraction shrinks as the opponent bets and raises.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from itertools import combinations, permutations

from evaluator import get_table
from game import Action, Street

TABLE = get_table()
_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))
_PERMS = tuple(permutations(range(4)))
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

TIME_BUDGET = 3.0       # seconds of compute per half
MAX_DECISION = 0.006
MIN_DECISION = 0.0004

SB_OPEN_PCT = 0.88
SB_FOLD_PCT = 0.06
BB_RAISE_PCT = 0.68
PF_RERAISE_EQ = 0.58
PF_BONUS = 0.04
VB = 0.52
BIG_BET_EQ = 0.70
RAISE_EQ = 0.68
K_POST_A = 0.25
K_POST_B = 1.2
FLOP_MARGIN = 0.03


# Preflop

PRE: dict[tuple[int, ...], list[float]] = {}
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_table.json")) as _f:
        PRE = {tuple(int(x) for x in k.split(",")): v for k, v in json.load(_f).items()}
except Exception:
    PRE = {}


def canon(hand) -> tuple[int, ...]:
    """Hand with suits relabeled to a canonical order."""
    best = None
    for p in _PERMS:
        t = tuple(sorted(((c >> 2) << 2) | p[c & 3] for c in hand))
        if best is None or t < best:
            best = t
    return best


def _heuristic_pre(hole) -> list[float]:
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    suits = [c & 3 for c in hole]
    s = sum(ranks) / 400.0
    if len(set(ranks)) < 5:
        s += 0.03
    suit_counts = sorted((suits.count(x) for x in range(4)), reverse=True)
    if suit_counts[0] >= 2:
        s += 0.03
    if suit_counts[1] >= 2:
        s += 0.02
    uniq = sorted(set(ranks))
    if len(uniq) >= 4 and uniq[-1] - uniq[0] <= 5:
        s += 0.03
    eq0 = 0.42 + s
    return [eq0, eq0 - 0.04, eq0 - 0.07, min(1.0, max(0.0, (eq0 - 0.40) / 0.2))]


def preflop_info(hole) -> list[float]:
    """[equity vs random, vs top 40%, vs top 15%, percentile]."""
    v = PRE.get(canon(hole))
    return v if v is not None else _heuristic_pre(hole)


def pre_eq_vs(info, f: float) -> float:
    """Equity vs the top-f range, interpolated in log(f)."""
    x = math.log(min(1.0, max(0.06, f)))
    pts = ((0.0, info[0]), (math.log(0.40), info[1]), (math.log(0.15), info[2]))
    (x0, y0), (x1, y1) = (pts[1], pts[0]) if x >= pts[1][0] else (pts[2], pts[1])
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


# Postflop sampling

def _best(hole_masks, board_masks, table=TABLE):
    best = 0
    for h in hole_masks:
        for b in board_masks:
            s = table[h | b]
            if s > best:
                best = s
    return best


def _hole_masks(hole):
    hb = [1 << c for c in hole]
    return [hb[i] | hb[j] for i, j in _PAIRS]


def _board_masks(board):
    bb = [1 << c for c in board]
    return [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]


def sample_postflop(hole, board, deadline: float, rng: random.Random,
                    min_samples: int = 24, max_samples: int = 600):
    """List of (my_score, opp_score) samples, collected until the deadline."""
    table = TABLE
    known = set(hole) | set(board)
    deck = [c for c in range(36) if c not in known]
    need = 5 - len(board)
    out = []
    perf = time.perf_counter

    if need == 0:
        # Best score for each unseen 2-card combo, so an opponent hand is 10 lookups.
        bm = _board_masks(board)
        my = _best(_hole_masks(hole), bm)
        pair_best = {}
        for a, b in combinations(deck, 2):
            m = (1 << a) | (1 << b)
            best = 0
            for x in bm:
                s = table[m | x]
                if s > best:
                    best = s
            pair_best[m] = best
        n = 0
        while n < max_samples:
            for _ in range(16):
                ob = [1 << c for c in rng.sample(deck, 5)]
                best = 0
                for i, j in _PAIRS:
                    s = pair_best[ob[i] | ob[j]]
                    if s > best:
                        best = s
                out.append((my, best))
            n += 16
            if n >= min_samples and perf() > deadline:
                break
        return out

    hm = _hole_masks(hole)
    per_runout = 3
    n = 0
    while n < max_samples:
        cards = rng.sample(deck, need + 5 * per_runout)
        bm = _board_masks(list(board) + cards[:need])
        my = _best(hm, bm)
        for k in range(per_runout):
            o = cards[need + 5 * k: need + 5 * k + 5]
            out.append((my, _best(_hole_masks(o), bm)))
        n += per_runout
        if n >= min_samples and perf() > deadline:
            break
    return out


def equity_vs_top(samples_sorted, f: float) -> float:
    """Equity against the strongest fraction f of samples (sorted by opp score, desc)."""
    n = len(samples_sorted)
    if n == 0:
        return 0.5
    k = max(1, min(n, int(math.ceil(f * n))))
    tot = 0.0
    for my, op in samples_sorted[:k]:
        if my > op:
            tot += 1.0
        elif my == op:
            tot += 0.5
    return tot / k


# Opponent model

def new_half(memory: dict, num_hands: int = 1000) -> None:
    memory.clear()
    memory.update(
        rng=random.Random(0x5EED),
        num_hands=num_hands,
        t_used=0.0,
        hand=-1,
        seen=0,
        pending_bet=False,
        opp_pf_acts=0,
        opp_pf_raises=0,
        opp_post_acts=0,
        opp_post_raises=0,
        opp_faced=0,    # times the opponent responded to our bet
        opp_folds=0,
    )


def _rate(num, den, prior, weight):
    return (num + prior * weight) / (den + weight)


def opp_pf_raise_rate(m):
    return _rate(m["opp_pf_raises"], m["opp_pf_acts"], 0.35, 12)


def opp_post_raise_rate(m):
    return _rate(m["opp_post_raises"], m["opp_post_acts"], 0.30, 12)


def opp_fold_rate(m):
    return _rate(m["opp_folds"], m["opp_faced"], 0.35, 10)


def _process_history(obs, m):
    if m["hand"] != obs.hand_index:
        m["hand"] = obs.hand_index
        m["seen"] = 0
        m["pending_bet"] = False
    hist = obs.action_history
    for a in hist[m["seen"]:]:
        if a.is_me:
            continue
        is_raise = a.action in RAISES
        if a.street == Street.PREFLOP:
            m["opp_pf_acts"] += 1
            m["opp_pf_raises"] += is_raise
        else:
            m["opp_post_acts"] += 1
            m["opp_post_raises"] += is_raise
        if m["pending_bet"]:
            m["opp_faced"] += 1
            m["pending_bet"] = False
    m["seen"] = len(hist)


def on_hand_end(result, m: dict) -> None:
    if result.winner == "me" and not result.went_to_showdown:
        if len(result.board) == 0:
            m["opp_pf_acts"] += 1
        else:
            m["opp_post_acts"] += 1
        if m.get("pending_bet") and m.get("hand") == result.hand_index:
            m["opp_faced"] += 1
            m["opp_folds"] += 1
    m["pending_bet"] = False


def _range_factor(obs, m) -> float:
    """Fraction of opponent holdings still considered, given their actions this hand."""
    k_pf = min(0.95, max(0.45, 0.25 + 1.1 * opp_pf_raise_rate(m)))
    k_post = min(0.92, max(0.30, K_POST_A + K_POST_B * opp_post_raise_rate(m)))
    f = 1.0
    for a in obs.action_history:
        if a.is_me:
            continue
        if a.action in RAISES:
            if a.street == Street.PREFLOP:
                f *= math.sqrt(k_pf)
            else:
                f *= k_post
                if a.action == Action.RAISE_POT:
                    f *= 0.92
        elif a.action == Action.CHECK_CALL and a.street != Street.PREFLOP and a.amount > 0:
            f *= 1.0 - 0.45 * opp_fold_rate(m)
    return max(0.05, f)


# Decisions

def choose(obs, m: dict) -> Action:
    t0 = time.perf_counter()
    try:
        _process_history(obs, m)
        if obs.street == Street.PREFLOP:
            act = _preflop(obs, m)
        else:
            act = _postflop(obs, m, t0)
    finally:
        m["t_used"] += time.perf_counter() - t0
    if act in RAISES:
        m["pending_bet"] = True
    return act


def _pick_raise(obs, big: bool) -> Action:
    legal = obs.legal_actions
    if big and Action.RAISE_POT in legal:
        return Action.RAISE_POT
    if Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    return Action.RAISE_POT


def _preflop(obs, m) -> Action:
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    info = preflop_info(obs.hole_cards)
    pct = info[3]
    rng = m["rng"]
    C, P = obs.to_call, obs.pot
    opp_raises = sum(1 for a in obs.action_history if not a.is_me and a.action in RAISES)

    if opp_raises == 0:
        if C > 0:  # button, first to act
            if can_raise and (pct >= SB_OPEN_PCT or (pct >= 0.45 and rng.random() < 0.25)):
                return Action.RAISE_POT
            if pct < SB_FOLD_PCT and Action.FOLD in legal:
                return Action.FOLD
            return Action.CHECK_CALL
        if can_raise and (pct >= BB_RAISE_PCT or (pct >= 0.50 and rng.random() < 0.15)):
            return Action.RAISE_POT
        return Action.CHECK_CALL

    f = min(1.0, max(0.08, opp_pf_raise_rate(m) * 1.1)) * (0.55 ** (opp_raises - 1))
    e = pre_eq_vs(info, f)
    if C == 0:
        return Action.CHECK_CALL
    need = C / (P + C)
    spr = (obs.my_stack - C) / max(1, P + C)
    bonus = PF_BONUS if spr > 1.5 else 0.01
    if obs.is_button:
        bonus += 0.01

    if can_raise and e >= PF_RERAISE_EQ and obs.raises_this_street < 4:
        return Action.RAISE_POT
    if e + bonus >= need:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def _deadline(obs, m, t0):
    remaining = TIME_BUDGET - m["t_used"]
    if remaining < 0.3:
        return t0 + MIN_DECISION
    hands_left = max(1, m["num_hands"] - obs.hand_index)
    per = remaining / (hands_left * 1.6)
    return t0 + min(MAX_DECISION, max(MIN_DECISION, per))


def _postflop(obs, m, t0) -> Action:
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    rng = m["rng"]
    C, P = obs.to_call, obs.pot
    street = obs.street

    samples = sample_postflop(obs.hole_cards, obs.board, _deadline(obs, m, t0), rng)
    samples.sort(key=lambda s: s[1], reverse=True)

    soften = {Street.FLOP: 0.65, Street.TURN: 0.8, Street.RIVER: 1.0}[street]
    f = _range_factor(obs, m) ** soften
    e = equity_vs_top(samples, f)
    fr = opp_fold_rate(m)

    if C == 0:
        e_call = equity_vs_top(samples, f * max(0.15, 1.0 - fr) ** soften)
        if can_raise:
            vb = VB + {Street.FLOP: 0.0, Street.TURN: 0.01, Street.RIVER: 0.02}[street]
            if e_call >= vb:
                return _pick_raise(obs, big=e_call >= BIG_BET_EQ)
            if street == Street.RIVER:
                if fr > 0.42 and e < 0.35 and rng.random() < min(0.6, (fr - 0.3) * 1.5):
                    return _pick_raise(obs, big=False)
            else:
                if fr > 0.33 and e >= 0.40 and rng.random() < min(0.5, fr):
                    return _pick_raise(obs, big=False)
                if fr > 0.50 and rng.random() < 0.35:
                    return _pick_raise(obs, big=False)
        return Action.CHECK_CALL

    need = C / (P + C)
    margin = {Street.FLOP: FLOP_MARGIN, Street.TURN: FLOP_MARGIN * 0.67, Street.RIVER: 0.0}[street]
    if can_raise:
        e_re = equity_vs_top(samples, f * max(0.2, 1.0 - fr * 0.7) ** soften)
        if e_re >= RAISE_EQ:
            return _pick_raise(obs, big=True)
    if e >= need + margin:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
