"""Monte Carlo equity vs a random hand and pot odds. No opponent model."""
import random, time
import strategy as S
from game import Action, Street


def new_half(config, memory):
    memory["rng"] = random.Random(7)


def choose_action(obs, memory):
    legal = obs.legal_actions
    rng = memory["rng"]
    if obs.street == Street.PREFLOP:
        e = S.preflop_info(obs.hole_cards)[0]
    else:
        smp = S.sample_postflop(obs.hole_cards, obs.board, time.perf_counter() + 0.0015, rng)
        e = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in smp) / len(smp)
    C, P = obs.to_call, obs.pot
    if e > 0.62 and Action.RAISE_POT in legal:
        return Action.RAISE_POT if e > 0.72 else Action.RAISE_HALF
    if C == 0:
        return Action.CHECK_CALL
    if e >= C / (P + C) + 0.05:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def on_hand_end(result, memory):
    pass
