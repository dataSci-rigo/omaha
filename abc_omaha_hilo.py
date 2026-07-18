"""
ABC (Absolute Hand Strength) Bot for Omaha Hi-Lo 8-or-Better
=============================================================

A naive, rule-based poker bot that evaluates its own hand strength
in isolation and maps it to fold/call/raise via fixed thresholds.
No opponent modeling, no range tracking — pure hand quality.

Requires: pip install pokerkit
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

from pokerkit import (
    Automation,
    Card,
    FixedLimitOmahaHoldemHighLowSplitEightOrBetter,
    OmahaEightOrBetterLowHand,
    OmahaHoldemHand,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HI_INDEX_MAX = 7460  # approx max entry.index for hi hands (straight flush)
LO_INDEX_MAX = 56    # approx max entry.index for lo hands (worst qualifying)

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}

SUITS = "cdhs"
ALL_CARDS = [f"{r}{s}" for r in RANKS for s in SUITS]


# ---------------------------------------------------------------------------
# Bot protocol — any bot just needs a `decide` method
# ---------------------------------------------------------------------------

class Bot(Protocol):
    name: str

    def decide(
        self,
        hole_cards: list[Card],
        board: list[Card],
        pot: float,
        to_call: float,
        can_raise: bool,
        street: str,
    ) -> str:
        """Return 'fold', 'call', or 'raise'."""
        ...


# ---------------------------------------------------------------------------
# Hand‑strength helpers
# ---------------------------------------------------------------------------

def hi_strength(hole: list[Card], board: list[Card]) -> float:
    """
    Evaluate the high hand.  Returns 0.0–1.0  (1.0 = nuts).
    Uses PokerKit's Omaha hand evaluator which enforces the
    "exactly 2 hole + 3 board" rule automatically.
    """
    try:
        hand = OmahaHoldemHand.from_game(hole, board)
        return min(hand.entry.index / HI_INDEX_MAX, 1.0)
    except (ValueError, KeyError):
        return 0.0


def lo_strength(hole: list[Card], board: list[Card]) -> float:
    """
    Evaluate the low hand.  Returns 0.0–1.0  (1.0 = nut low A‑2‑3‑4‑5).
    Returns 0.0 when no 8‑or‑better low qualifies.
    """
    hand = OmahaEightOrBetterLowHand.from_game_or_none(hole, board)
    if hand is None:
        return 0.0
    # index 0 = best low, ~56 = worst qualifying → invert
    return max(1.0 - hand.entry.index / LO_INDEX_MAX, 0.0)


def _card_rank(c: Card) -> int:
    return RANK_VALUE.get(str(c)[0], 0)


def _card_suit(c: Card) -> str:
    return str(c)[1].lower()


# ---------------------------------------------------------------------------
# Pre‑flop heuristics  (no board yet → can't use evaluator)
# ---------------------------------------------------------------------------

def preflop_score(hole: list[Card]) -> float:
    """
    Rate a 4‑card Omaha Hi‑Lo starting hand on a 0.0–1.0 scale.

    Rewards:
      • Low cards  (A through 5)  — needed to compete for the low pot
      • Aces       — strong for both hi and lo
      • Suited‑ness— flush potential
      • Pairs      — set potential for hi
      • Connectedness — straight potential
    """
    ranks = [_card_rank(c) for c in hole]
    suits = [_card_suit(c) for c in hole]

    score = 0.0

    # --- Low potential (0–0.35) ---
    low_cards = sum(1 for r in ranks if r <= 3 or r == 12)  # 2-5 or Ace
    score += low_cards * 0.088  # max 0.35 with 4 low cards

    # --- Ace bonus (0–0.15) ---
    aces = sum(1 for r in ranks if r == 12)
    score += aces * 0.15

    # --- Suited bonus (0–0.15) ---
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suited = max(suit_counts.values())
    if max_suited >= 2:
        score += 0.10
    if max_suited >= 3:
        score += 0.05

    # --- Pair bonus (0–0.10) ---
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    pairs = sum(1 for v in rank_counts.values() if v >= 2)
    score += pairs * 0.05

    # --- Connectedness (0–0.10) ---
    sorted_ranks = sorted(set(ranks))
    gaps = 0
    for i in range(len(sorted_ranks) - 1):
        diff = sorted_ranks[i + 1] - sorted_ranks[i]
        if diff <= 2:
            gaps += 1
    score += min(gaps * 0.035, 0.10)

    # --- High card bonus (0–0.10) ---
    high_cards = sum(1 for r in ranks if r >= 10)  # J, Q, K, A
    score += high_cards * 0.025

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# ABC Bot
# ---------------------------------------------------------------------------

@dataclass
class ABCBot:
    """
    Absolute hand strength bot.  Evaluates its own hand, ignores opponents.

    Thresholds are tuned per‑street and split between hi and lo components.
    The composite score weights hi at 55% and lo at 45% to reflect the
    split‑pot structure (scooping both halves is the real goal).
    """

    name: str = "ABC"

    # Composite thresholds  (0.0–1.0)
    preflop_fold: float = 0.22
    preflop_raise: float = 0.50

    postflop_fold: float = 0.18
    postflop_raise: float = 0.45

    hi_weight: float = 0.55
    lo_weight: float = 0.45

    def _composite(self, hi: float, lo: float) -> float:
        if lo > 0:
            # We can compete for both pots — scoop potential
            return self.hi_weight * hi + self.lo_weight * lo
        else:
            # No low — we're only playing for half the pot at best
            return hi * 0.55

    def decide(
        self,
        hole_cards: list[Card],
        board: list[Card],
        pot: float,
        to_call: float,
        can_raise: bool,
        street: str,
    ) -> str:
        if street == "preflop":
            score = preflop_score(hole_cards)
            if score < self.preflop_fold:
                return "fold"
            if score >= self.preflop_raise and can_raise:
                return "raise"
            return "call"

        # Post‑flop: use actual hand evaluation
        hi = hi_strength(hole_cards, board)
        lo = lo_strength(hole_cards, board)
        score = self._composite(hi, lo)

        if score < self.postflop_fold:
            return "fold"
        if score >= self.postflop_raise and can_raise:
            return "raise"
        return "call"


# ---------------------------------------------------------------------------
# Random Bot  (baseline comparison)
# ---------------------------------------------------------------------------

@dataclass
class RandomBot:
    """Uniformly random legal actions — the weakest possible opponent."""

    name: str = "Random"

    def decide(self, hole_cards, board, pot, to_call, can_raise, street) -> str:
        choices = ["fold", "call"]
        if can_raise:
            choices.append("raise")
        return random.choice(choices)


# ---------------------------------------------------------------------------
# Caller Bot  (always calls — another baseline)
# ---------------------------------------------------------------------------

@dataclass
class CallerBot:
    """Always calls, never folds, never raises."""

    name: str = "Caller"

    def decide(self, hole_cards, board, pot, to_call, can_raise, street) -> str:
        return "call"


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_DEALING,
    Automation.BOARD_DEALING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


def _get_street(board_len: int) -> str:
    if board_len == 0:
        return "preflop"
    elif board_len <= 3:
        return "flop"
    elif board_len == 4:
        return "turn"
    else:
        return "river"


@dataclass
class HandResult:
    payoffs: list[float]
    board: list[str]
    hole_cards: list[list[str]]


def play_one_hand(
    bots: list,
    starting_stack: float = 100.0,
    small_bet: float = 2.0,
    big_bet: float = 4.0,
    blinds: tuple[float, float] = (1, 2),
    verbose: bool = False,
) -> HandResult:
    """Play a single hand of fixed‑limit Omaha Hi‑Lo 8/b."""

    n = len(bots)
    state = FixedLimitOmahaHoldemHighLowSplitEightOrBetter.create_state(
        AUTOMATIONS,
        True,                         # uniform antes
        0,                            # antes
        blinds,
        small_bet,
        big_bet,
        (starting_stack,) * n,
        n,
    )

    # Save hole cards immediately (they get cleared at showdown)
    saved_holes = [list(h) for h in state.hole_cards]

    if verbose:
        for i, bot in enumerate(bots):
            cards_str = " ".join(
                str(c).split("(")[1].rstrip(")") if "(" in str(c) else str(c)
                for c in saved_holes[i]
            )
            print(f"  {bot.name} [{i}]: {cards_str}")

    # Play until the hand ends
    actions_taken = 0
    max_actions = 200  # safety valve

    while state.status and actions_taken < max_actions:
        actor = state.actor_index
        if actor is None:
            break

        bot = bots[actor]
        board = [c for b in state.board_cards for c in b]
        street = _get_street(len(board))

        # Figure out cost to call
        # In fixed‑limit, the call amount is the difference between the
        # current bet and what we've already put in this round.
        to_call = 0.0
        can_raise = state.can_complete_bet_or_raise_to()

        decision = bot.decide(
            saved_holes[actor],
            board,
            state.total_pot_amount,
            to_call,
            can_raise,
            street,
        )

        actual = decision
        try:
            if decision == "fold" and state.can_fold():
                state.fold()
            elif decision == "raise" and can_raise:
                state.complete_bet_or_raise_to()
            else:
                actual = "check/call"
                state.check_or_call()
        except (ValueError, IndexError):
            actual = "check/call*"
            try:
                state.check_or_call()
            except Exception:
                break

        if verbose:
            print(f"    {street:8s} | {bot.name} [{actor}] → {actual}")

        actions_taken += 1

    board = [c for b in state.board_cards for c in b]

    result = HandResult(
        payoffs=list(state.payoffs),
        board=[str(c).split("(")[1].rstrip(")") if "(" in str(c) else str(c) for c in board],
        hole_cards=[[str(c).split("(")[1].rstrip(")") if "(" in str(c) else str(c) for c in h] for h in saved_holes],
    )

    if verbose:
        print(f"  Board: {' '.join(result.board)}")
        print(f"  Payoffs: {result.payoffs}")

    return result


def run_session(
    bots: list,
    num_hands: int = 1000,
    rotate_button: bool = True,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Run many hands and return cumulative profit for each bot.

    When rotate_button is True, the bot list is rotated each hand so
    every bot gets equal time in each seat position.
    """
    profits = {bot.name: 0.0 for bot in bots}

    for hand_num in range(num_hands):
        # Rotate seats
        if rotate_button:
            rotation = hand_num % len(bots)
            rotated = bots[rotation:] + bots[:rotation]
        else:
            rotated = bots

        if verbose and hand_num < 5:
            print(f"\n--- Hand {hand_num + 1} ---")

        result = play_one_hand(rotated, verbose=(verbose and hand_num < 5))

        for i, bot in enumerate(rotated):
            profits[bot.name] += result.payoffs[i]

    return profits


# ---------------------------------------------------------------------------
# Main — run ABC vs Random vs Caller
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    bots = [ABCBot(), RandomBot(), CallerBot()]
    num_hands = 1000

    print("=" * 56)
    print("  Omaha Hi‑Lo 8/b  ·  ABC Bot Simulation")
    print("=" * 56)
    print(f"  Players:  {', '.join(b.name for b in bots)}")
    print(f"  Hands:    {num_hands:,}")
    print(f"  Blinds:   1/2 fixed‑limit (2/4 bets)")
    print("=" * 56)

    # Show first 3 hands in detail
    print("\n▸ Sample hands:\n")
    sample_profits = run_session(bots, num_hands=3, verbose=True)

    # Run full session
    print("\n▸ Running full session...\n")
    profits = run_session(bots, num_hands=num_hands)

    print("-" * 40)
    print(f"  {'Bot':<12} {'Profit':>10} {'bb/hand':>10}")
    print("-" * 40)
    for name, profit in sorted(profits.items(), key=lambda x: -x[1]):
        bb_per_hand = profit / num_hands / 2  # big blind = 2
        print(f"  {name:<12} {profit:>+10.1f} {bb_per_hand:>+10.3f}")
    print("-" * 40)

    winner = max(profits, key=profits.get)
    print(f"\n  Winner: {winner} 🏆\n")


if __name__ == "__main__":
    main()
