"""
Bayesian Bot for Omaha Hi-Lo 8-or-Better
=========================================

Two Bayesian pieces, both trained from self-play data (see
`train_bayesian_model`):

1. A naive-Bayes preflop classifier (`NaiveBayesPreflop`) that predicts
   P(win) from discrete hand features, replacing ABCBot's fixed-weight
   preflop heuristic with one fit to observed outcomes.

2. Opponent range modeling (`ActionModel` + `OpponentBelief`): a
   likelihood table P(action | true hand-strength bucket, street) learned
   from watching other bots act with known hole cards, used to run a
   live Bayesian update on each opponent's hidden hand strength as they
   act during a hand. The bot then plays relative to the strongest
   estimated opponent range, not just its own hand in isolation.

Training data comes from self-play: `RecordingBot` wraps any bot and,
using the real hole/board cards it's handed each decision (ground truth,
since we control the simulation), logs (street, true_bucket, action)
triples plus (preflop_features, won) pairs from the hand's outcome.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field

from abc_omaha_hilo import (
    ABCBot,
    CallerBot,
    RandomBot,
    _card_rank,
    _card_suit,
    hi_strength,
    lo_strength,
    play_one_hand,
    preflop_score,
    run_session,
)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bayesian_model.json")

N_BUCKETS = 5
ACTIONS = ("fold", "call", "raise")

HI_WEIGHT = 0.55
LO_WEIGHT = 0.45


def composite_score(hi: float, lo: float) -> float:
    """Own-hand hi/lo blend — same weighting ABCBot uses postflop."""
    if lo > 0:
        return HI_WEIGHT * hi + LO_WEIGHT * lo
    return hi * 0.55


def strength_bucket(score: float, n_buckets: int = N_BUCKETS) -> int:
    return max(0, min(n_buckets - 1, int(score * n_buckets)))


# ---------------------------------------------------------------------------
# Preflop features (discrete, for naive Bayes)
# ---------------------------------------------------------------------------

def preflop_features(hole: list) -> dict[str, int]:
    ranks = [_card_rank(c) for c in hole]
    suits = [_card_suit(c) for c in hole]

    low_cards = sum(1 for r in ranks if r <= 3 or r == 12)  # 2-5 or Ace
    aces = sum(1 for r in ranks if r == 12)

    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suited = max(suit_counts.values())
    suited_bucket = 0 if max_suited < 2 else (1 if max_suited in (2, 3) else 2)

    rank_counts: dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    pairs = min(2, sum(1 for v in rank_counts.values() if v >= 2))

    sorted_ranks = sorted(set(ranks))
    gaps = sum(
        1 for i in range(len(sorted_ranks) - 1)
        if sorted_ranks[i + 1] - sorted_ranks[i] <= 2
    )
    connect_bucket = min(2, gaps)

    high_cards = min(4, sum(1 for r in ranks if r >= 10))  # J, Q, K, A

    return {
        "low": low_cards,
        "aces": aces,
        "suited": suited_bucket,
        "pairs": pairs,
        "connect": connect_bucket,
        "high": high_cards,
    }


# ---------------------------------------------------------------------------
# Naive Bayes preflop classifier
# ---------------------------------------------------------------------------

@dataclass
class NaiveBayesPreflop:
    """P(win | features) via naive Bayes over independent discrete features."""

    win_total: int = 0
    lose_total: int = 0
    # feature_name -> value -> [win_count, lose_count]
    value_counts: dict[str, dict[int, list[int]]] = field(default_factory=dict)
    smoothing: float = 1.0

    def fit(self, records: list[tuple[dict[str, int], bool]]) -> None:
        for features, won in records:
            if won:
                self.win_total += 1
            else:
                self.lose_total += 1
            for name, value in features.items():
                vc = self.value_counts.setdefault(name, {})
                counts = vc.setdefault(value, [0, 0])
                counts[0 if won else 1] += 1

    def _feature_ratio(self, name: str, value: int) -> float:
        counts = self.value_counts.get(name, {}).get(value)
        n_values = max(1, len(self.value_counts.get(name, {})))
        s = self.smoothing
        if counts is None:
            counts = [0, 0]
        p_win = (counts[0] + s) / (self.win_total + s * n_values)
        p_lose = (counts[1] + s) / (self.lose_total + s * n_values)
        return math.log(p_win) - math.log(p_lose)

    def predict_proba(self, features: dict[str, int]) -> float:
        if self.win_total == 0 or self.lose_total == 0:
            return 0.5
        log_odds = math.log(self.win_total) - math.log(self.lose_total)
        for name, value in features.items():
            log_odds += self._feature_ratio(name, value)
        log_odds = max(-20.0, min(20.0, log_odds))
        return 1.0 / (1.0 + math.exp(-log_odds))

    def to_dict(self) -> dict:
        return {
            "win_total": self.win_total,
            "lose_total": self.lose_total,
            "value_counts": self.value_counts,
            "smoothing": self.smoothing,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NaiveBayesPreflop":
        obj = cls(win_total=d["win_total"], lose_total=d["lose_total"], smoothing=d.get("smoothing", 1.0))
        obj.value_counts = {
            name: {int(v): counts for v, counts in vc.items()}
            for name, vc in d["value_counts"].items()
        }
        return obj


# ---------------------------------------------------------------------------
# Opponent action-likelihood model:  P(action | street, true strength bucket)
# ---------------------------------------------------------------------------

@dataclass
class ActionModel:
    # (street, bucket) -> {action: count}
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    smoothing: float = 1.0

    @staticmethod
    def _key(street: str, bucket: int) -> str:
        return f"{street}:{bucket}"

    def fit(self, records: list[tuple[str, int, str]]) -> None:
        for street, bucket, action in records:
            k = self._key(street, bucket)
            row = self.counts.setdefault(k, {a: 0 for a in ACTIONS})
            row[action] = row.get(action, 0) + 1

    def likelihood(self, street: str, bucket: int, action: str) -> float:
        row = self.counts.get(self._key(street, bucket))
        s = self.smoothing
        if row is None:
            return 1.0 / len(ACTIONS)
        total = sum(row.values())
        return (row.get(action, 0) + s) / (total + s * len(ACTIONS))

    def to_dict(self) -> dict:
        return {"counts": self.counts, "smoothing": self.smoothing}

    @classmethod
    def from_dict(cls, d: dict) -> "ActionModel":
        return cls(counts=d["counts"], smoothing=d.get("smoothing", 1.0))


class OpponentBelief:
    """Posterior over an opponent's true hand-strength bucket, updated live
    from their observed actions during a hand."""

    def __init__(self, n_buckets: int = N_BUCKETS) -> None:
        self.n_buckets = n_buckets
        self.posterior = [1.0 / n_buckets] * n_buckets

    def update(self, street: str, action: str, model: ActionModel) -> None:
        weighted = [
            self.posterior[b] * model.likelihood(street, b, action)
            for b in range(self.n_buckets)
        ]
        total = sum(weighted)
        if total > 0:
            self.posterior = [w / total for w in weighted]

    def mean_value(self) -> float:
        return sum(
            self.posterior[b] * ((b + 0.5) / self.n_buckets)
            for b in range(self.n_buckets)
        )


# ---------------------------------------------------------------------------
# Bayesian Bot
# ---------------------------------------------------------------------------

class BayesianBot:
    """Blends a naive-Bayes preflop win estimate (own hand) with a live
    Bayesian read on opponents' hand strength (from their actions this
    hand) to decide fold/call/raise relative to the toughest opponent
    range, not just absolute hand strength."""

    def __init__(self, name: str = "Bayesian", model_path: str = DEFAULT_MODEL_PATH,
                 fold_margin: float = 0.05, raise_margin: float = 0.10,
                 preflop_model: "NaiveBayesPreflop | None" = None,
                 action_model: "ActionModel | None" = None) -> None:
        self.name = name
        self.fold_margin = fold_margin
        self.raise_margin = raise_margin
        if preflop_model is not None and action_model is not None:
            self.preflop_model, self.action_model = preflop_model, action_model
        else:
            self.preflop_model, self.action_model = load_model(model_path)

    def _own_score(self, hole, board, street: str) -> float:
        if street == "preflop":
            return self.preflop_model.predict_proba(preflop_features(hole))
        return composite_score(hi_strength(hole, board), lo_strength(hole, board))

    def _opponent_estimate(self, history: list[dict] | None, hero_index: int | None) -> float:
        if not history:
            return 0.5
        beliefs: dict[int, OpponentBelief] = {}
        last_action: dict[int, str] = {}
        for entry in history:
            actor = entry["actor"]
            if actor == hero_index:
                continue
            belief = beliefs.setdefault(actor, OpponentBelief())
            belief.update(entry["street"], entry["action"], self.action_model)
            last_action[actor] = entry["action"]
        active = [a for a, act in last_action.items() if act != "fold"]
        if not active:
            return 0.5
        return max(beliefs[a].mean_value() for a in active)

    def decide(
        self,
        hole_cards,
        board,
        pot: float,
        to_call: float,
        can_raise: bool,
        street: str,
        history: list[dict] | None = None,
        hero_index: int | None = None,
    ) -> str:
        own_score = self._own_score(hole_cards, board, street)
        opponent_est = self._opponent_estimate(history, hero_index)
        edge = own_score - opponent_est

        # Raw pot odds: do we have enough equity to continue on price alone?
        required_equity = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        pot_odds_ok = own_score >= required_equity

        # Fold only when the price is bad *and* we look behind the toughest
        # active opponent's estimated range — good odds alone (cheap call)
        # or a favorable read alone (bad odds but likely ahead) both keep us in.
        if not pot_odds_ok and edge < -self.fold_margin:
            return "fold"
        if can_raise and edge > self.raise_margin:
            return "raise"
        return "call"


# ---------------------------------------------------------------------------
# Training-data recorder
# ---------------------------------------------------------------------------

class RecordingBot:
    """Wraps any bot; logs (street, true_bucket, action) for every decision
    it makes, using the real hole/board cards it's handed (ground truth
    during self-play), then delegates to the wrapped bot's real policy."""

    def __init__(self, wrapped, action_records: list[tuple[str, int, str]]) -> None:
        self._wrapped = wrapped
        self._records = action_records
        self.name = wrapped.name

    def decide(self, hole_cards, board, pot, to_call, can_raise, street,
               history=None, hero_index=None) -> str:
        if street == "preflop":
            score = preflop_score(hole_cards)
        else:
            score = composite_score(hi_strength(hole_cards, board), lo_strength(hole_cards, board))
        bucket = strength_bucket(score)

        action = self._wrapped.decide(
            hole_cards, board, pot, to_call, can_raise, street,
            history=history, hero_index=hero_index,
        )
        self._records.append((street, bucket, action))
        return action


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(preflop_model: NaiveBayesPreflop, action_model: ActionModel, path: str = DEFAULT_MODEL_PATH) -> None:
    with open(path, "w") as f:
        json.dump({"preflop": preflop_model.to_dict(), "action": action_model.to_dict()}, f)


def load_model(path: str = DEFAULT_MODEL_PATH) -> tuple[NaiveBayesPreflop, ActionModel]:
    if not os.path.exists(path):
        return NaiveBayesPreflop(), ActionModel()
    with open(path) as f:
        d = json.load(f)
    return NaiveBayesPreflop.from_dict(d["preflop"]), ActionModel.from_dict(d["action"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bayesian_model(num_hands: int = 3000, seed: int = 7,
                          path: str = DEFAULT_MODEL_PATH, verbose: bool = True,
                          ) -> tuple[NaiveBayesPreflop, ActionModel]:
    """Self-play `num_hands` hands among a diverse mix of existing bot
    policies (Random / Caller / ABC, at several ABC threshold settings)
    so the learned action->strength likelihoods generalize reasonably
    across playing styles. Every decision's true hand strength is known
    (we control the simulation), giving fully labeled training data."""
    random.seed(seed)

    base_bots = [
        RandomBot(name="Random"),
        CallerBot(name="Caller"),
        ABCBot(name="ABC-tight", preflop_fold=0.30, postflop_fold=0.25),
        ABCBot(name="ABC-loose", preflop_fold=0.12, postflop_fold=0.10, preflop_raise=0.40, postflop_raise=0.35),
    ]
    action_records: list[tuple[str, int, str]] = []
    recorders = [RecordingBot(b, action_records) for b in base_bots]

    preflop_records: list[tuple[dict[str, int], bool]] = []

    for hand_num in range(num_hands):
        rotation = hand_num % len(recorders)
        rotated = recorders[rotation:] + recorders[:rotation]
        result = play_one_hand(rotated, verbose=False)
        for seat in range(len(rotated)):
            feats = preflop_features(result.hole_cards[seat])
            preflop_records.append((feats, result.payoffs[seat] > 0))

        if verbose and (hand_num + 1) % 500 == 0:
            print(f"  ...trained on {hand_num + 1}/{num_hands} hands "
                  f"({len(action_records)} action samples)")

    preflop_model = NaiveBayesPreflop()
    preflop_model.fit(preflop_records)

    action_model = ActionModel()
    action_model.fit(action_records)

    save_model(preflop_model, action_model, path)

    if verbose:
        print(f"\n  Trained on {num_hands} hands "
              f"({len(action_records)} action samples, {len(preflop_records)} preflop samples)")
        print(f"  Saved model → {path}\n")

    return preflop_model, action_model


# ---------------------------------------------------------------------------
# Self-play refinement — iterate BayesianBot vs itself, retraining from its
# own games each round, until the model stops changing (fixed-point / crude
# fictitious-play style self-improvement).
# ---------------------------------------------------------------------------

_PREFLOP_GRID = [
    {"low": lo, "aces": ac, "suited": su, "pairs": pa, "connect": co, "high": hi}
    for lo in range(5) for ac in range(3) for su in range(3)
    for pa in range(3) for co in range(3) for hi in range(5)
][::37]  # thinned sample of the feature space, enough to gauge drift


def _model_distance(old: tuple[NaiveBayesPreflop, ActionModel],
                     new: tuple[NaiveBayesPreflop, ActionModel]) -> float:
    """Mean absolute change between two (preflop, action) model pairs:
    preflop win-prob predictions over a fixed feature grid, and action
    likelihoods over every trained (street, bucket, action) cell."""
    old_pf, old_am = old
    new_pf, new_am = new

    pf_diffs = [
        abs(old_pf.predict_proba(f) - new_pf.predict_proba(f)) for f in _PREFLOP_GRID
    ]

    keys = set(old_am.counts) | set(new_am.counts)
    am_diffs = []
    for k in keys:
        street, bucket = k.split(":")
        bucket = int(bucket)
        for action in ACTIONS:
            am_diffs.append(abs(
                old_am.likelihood(street, bucket, action)
                - new_am.likelihood(street, bucket, action)
            ))

    diffs = pf_diffs + am_diffs
    return sum(diffs) / len(diffs) if diffs else 0.0


def self_play_refine(preflop_model: NaiveBayesPreflop, action_model: ActionModel,
                      hands_per_round: int = 2000, max_rounds: int = 15,
                      tol: float = 0.01, n_players: int = 4, seed: int = 99,
                      path: str = DEFAULT_MODEL_PATH, verbose: bool = True,
                      ) -> tuple[NaiveBayesPreflop, ActionModel]:
    """Refine the model via self-play, seeded from `preflop_model`/`action_model`
    (typically an already-validated baseline).

    Two things that plain self-play gets wrong, both fixed here:

    - **Accumulation**: each round's data is merged into the running model
      (via `fit`, which only adds counts) rather than discarding prior
      rounds and refitting from scratch. Without this, round-over-round
      "drift" never shrinks — it's just fresh sampling noise every time,
      since nothing pools. With accumulation, each new round's data is a
      smaller fraction of the growing total, so drift decays and the
      process actually has a fixed point to converge to.
    - **Opponent mix**: half the table each round is Random/Caller/ABC,
      not just copies of the current Bayesian model. Pure mirror-self-play
      lets the action-likelihood model specialize to mirror-match dynamics
      and forget how to read genuinely different opponents (e.g. ABCBot),
      which is what caused the earlier regression.
    """
    random.seed(seed)
    # Start the accumulator from the seed model's own counts, so refinement
    # builds on the validated baseline instead of discarding it.
    accum_pf = NaiveBayesPreflop.from_dict(preflop_model.to_dict())
    accum_am = ActionModel.from_dict(action_model.to_dict())

    opponent_pool = [
        RandomBot(name="Random"),
        CallerBot(name="Caller"),
        ABCBot(name="ABC-tight", preflop_fold=0.30, postflop_fold=0.25),
        ABCBot(name="ABC-loose", preflop_fold=0.12, postflop_fold=0.10, preflop_raise=0.40, postflop_raise=0.35),
    ]
    n_bayes_seats = max(1, n_players // 2)
    n_opp_seats = n_players - n_bayes_seats

    dist = float("inf")
    for round_num in range(1, max_rounds + 1):
        # Snapshot pre-round state (cheap dict round-trip) to measure drift.
        snapshot = (
            NaiveBayesPreflop.from_dict(accum_pf.to_dict()),
            ActionModel.from_dict(accum_am.to_dict()),
        )

        bayes_bots = [
            BayesianBot(f"Bayes{i}", preflop_model=accum_pf, action_model=accum_am)
            for i in range(n_bayes_seats)
        ]
        opp_bots = [random.choice(opponent_pool) for _ in range(n_opp_seats)]
        table = bayes_bots + opp_bots

        action_records: list[tuple[str, int, str]] = []
        recorders = [RecordingBot(b, action_records) for b in table]
        preflop_records: list[tuple[dict[str, int], bool]] = []

        for hand_num in range(hands_per_round):
            rotation = hand_num % len(recorders)
            rotated = recorders[rotation:] + recorders[:rotation]
            result = play_one_hand(rotated, verbose=False)
            for seat in range(len(rotated)):
                feats = preflop_features(result.hole_cards[seat])
                preflop_records.append((feats, result.payoffs[seat] > 0))

        accum_pf.fit(preflop_records)
        accum_am.fit(action_records)

        dist = _model_distance(snapshot, (accum_pf, accum_am))
        if verbose:
            print(f"  round {round_num:2d}: {hands_per_round} hands "
                  f"({n_bayes_seats} Bayesian + {n_opp_seats} baseline seats), "
                  f"model drift = {dist:.5f} (tol {tol})")

        if dist < tol:
            if verbose:
                print(f"  converged after round {round_num} (drift {dist:.5f} < {tol})\n")
            break
    else:
        if verbose:
            print(f"  stopped at max_rounds={max_rounds} without converging (last drift {dist:.5f})\n")

    save_model(accum_pf, accum_am, path)
    current = (accum_pf, accum_am)
    return current


# ---------------------------------------------------------------------------
# Main — train then evaluate BayesianBot vs the baselines
# ---------------------------------------------------------------------------

def _print_profits(profits: dict[str, float], num_hands: int) -> None:
    print("-" * 40)
    print(f"  {'Bot':<12} {'Profit':>10} {'bb/hand':>10}")
    print("-" * 40)
    for name, profit in sorted(profits.items(), key=lambda x: -x[1]):
        bb_per_hand = profit / num_hands / 2
        print(f"  {name:<12} {profit:>+10.1f} {bb_per_hand:>+10.3f}")
    print("-" * 40)


def eval_vs_baselines(preflop_model: NaiveBayesPreflop, action_model: ActionModel,
                       num_hands: int = 2000, seed: int = 123) -> dict[str, float]:
    random.seed(seed)
    bots = [
        BayesianBot(preflop_model=preflop_model, action_model=action_model),
        ABCBot(), RandomBot(), CallerBot(),
    ]
    profits = run_session(bots, num_hands=num_hands)
    _print_profits(profits, num_hands)
    return profits


def main():
    print("=" * 56)
    print("  Training BayesianBot from self-play")
    print("=" * 56)
    preflop_model, action_model = train_bayesian_model(num_hands=10000)

    print("=" * 56)
    print("  Evaluating BayesianBot vs ABC / Random / Caller")
    print("=" * 56)
    profits = eval_vs_baselines(preflop_model, action_model)

    if profits["Bayesian"] <= profits["ABC"]:
        print(f"\n  Bayesian ({profits['Bayesian']:+.1f}) hasn't beaten "
              f"ABC ({profits['ABC']:+.1f}) yet — skipping self-play refinement.\n")
        return

    print("=" * 56)
    print("  Bayesian beat ABC — refining via self-play until convergence")
    print("=" * 56)
    preflop_model, action_model = self_play_refine(preflop_model, action_model)

    print("=" * 56)
    print("  Evaluating refined BayesianBot vs ABC / Random / Caller")
    print("=" * 56)
    eval_vs_baselines(preflop_model, action_model)


if __name__ == "__main__":
    main()
