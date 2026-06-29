# Omaha Hi-Lo Poker Game — Development Plan

## Refined Prompt (Phase 1: Game Engine + Text UI)

Use this prompt to build the core game. Changes from your original are noted in comments.

---

> Build a text-based **Pot-Limit Omaha Hi-Lo (8-or-better)** game in Python using PokerKit, structured as a Jupyter notebook called `omaha_player.ipynb`. Include a config flag to switch between **pot-limit** and **fixed-limit** modes.
>
> **Architecture — keep these layers separate:**
>
> 1. **GameEngine** — wraps PokerKit. Manages the deck, board, pot(s) including side pots, blinds, dealing, hand evaluation (hi and lo with 8-or-better qualifier), and showdown logic (scoop vs split). Exposes a clean interface: `get_game_state()` returns the public game state; `submit_action(seat, action)` applies an action. This class should know nothing about display or input — it just runs the game.
> 2. **Bot base class (AbstractBot)** — defines `decide(game_state) -> Action`. Each bot instance has a `name`, `seat`, `stack`, and `bot_type` label. New bot types are created by subclassing. The constructor takes `starting_stack` as a parameter.
> 3. **HumanPlayer** — prompts for input via `input()` for now (will be swapped for WebSocket later).
> 4. **TableDisplay** — renders the text UI. By seat number, print each player's action as it happens. Below the seats, show the community cards in a text table. For human players, show their 4 hole cards. For bots, show "[hidden]" unless it's showdown (two or more players remain after all betting on the river) or the bot is all-in and called.
>
> **Table setup:**
> - Configurable: 2–6 seats total, with 0–2 human players and 1–4 bots.
> - For the initial build: 1 human + 1 RandomBot.
> - Each player/bot is initialized with a configurable starting stack.
> - Small blind and big blind amounts are configurable (default 1/2).
> - Blinds rotate each hand. Display who is SB, BB, and dealer.
>
> **Legal actions:**
> - Fold, Check, Call, Raise (to any legal amount up to all-in).
> - In pot-limit mode, max raise = current pot size. In fixed-limit mode, use standard fixed increments.
> - All-in is always legal if a player doesn't have enough to call/raise.
>
> **Showdown rules (important for Omaha Hi-Lo):**
> - Players must use exactly 2 of their 4 hole cards + 3 of the 5 board cards.
> - The pot is split between the best high hand and the best qualifying low hand (8-or-better). If no low qualifies, the high hand scoops.
> - Side pots are handled correctly when players are all-in for different amounts.
>
> **Bust-out:** If a player's stack hits 0 after a hand, they are eliminated. Announce it.
>
> **Seat rotation:** Every 10 hands, prompt the human: "Rotate seats? (y/n)". If yes, shuffle all remaining players into random seats.
>
> **RandomBot implementation:** Chooses uniformly at random from legal actions. When raising, picks a random legal raise amount.
>
> **Test mode:** Include a cell that runs 2,000 hands with 2 RandomBots (no human), logging per-hand results to a list of dicts. Print summary stats at the end: total hands, each bot's final stack, win count, and average pot size. Save the hand history log as JSON for future ML training data.
>
> **Code quality:** Type hints, docstrings, and clear separation so the GameEngine can later be wrapped in a web server without changes.

---

## Phase 2: Hand Strength Bot

### Concept

A rules-based bot that estimates its equity (probability of winning the hi pot, the lo pot, or both) and maps that to actions using pot odds.

### How It Works

```
┌──────────────────────────────────────────────┐
│              HandStrengthBot                 │
│                                              │
│  1. Receive game_state                       │
│  2. Run Monte Carlo equity estimation:       │
│     - Sample N random runouts (1000–3000)    │
│     - For each sample:                       │
│       • Deal remaining board cards randomly  │
│       • Deal random hands to opponents       │
│       • Evaluate hi + lo using PokerKit      │
│       • Track: win_hi, win_lo, scoop, split  │
│  3. Compute:                                 │
│     - equity_hi  = wins_hi / N               │
│     - equity_lo  = wins_lo / N               │
│     - equity_total = expected_share_of_pot    │
│  4. Compare equity_total to pot odds         │
│  5. Decision matrix:                         │
│     - equity < fold_threshold     → fold     │
│     - equity < call_threshold     → call     │
│     - equity >= raise_threshold   → raise    │
│     - Raise sizing proportional to equity    │
│                                              │
│  Tunable params: fold/call/raise thresholds, │
│  aggression factor, bluff frequency,         │
│  number of Monte Carlo samples               │
└──────────────────────────────────────────────┘
```

### Key Design Decisions

- **Speed vs accuracy tradeoff:** Omaha Hi-Lo equity calculation is expensive because each player has 4 hole cards (more combinations) and you must evaluate both hi and lo. Start with 500 samples per decision; tune upward if the bot has time budget.
- **Opponent modeling (optional upgrade):** Track opponent tendencies (VPIP, aggression %) over the session and weight the Monte Carlo sampling toward hands they're more likely to hold.
- **Preflop hand rankings:** Use a preflop lookup table for Omaha Hi-Lo starting hand strength (hands like A-2-3-x suited are premium). This avoids running Monte Carlo preflop where the full board is unknown and computation is heaviest.

### Implementation Outline

```python
class HandStrengthBot(AbstractBot):
    def __init__(self, starting_stack, simulations=1000,
                 fold_threshold=0.3, raise_threshold=0.55,
                 aggression=0.5):
        ...

    def estimate_equity(self, game_state) -> dict:
        """Monte Carlo equity for hi, lo, and combined."""
        ...

    def decide(self, game_state) -> Action:
        equity = self.estimate_equity(game_state)
        pot_odds = game_state.call_amount / game_state.pot_total
        # decision logic here
        ...
```

### Open-Source Resources

- **PokerKit** handles the hand evaluation natively for Omaha Hi-Lo (hi + lo with 8-or-better). You don't need a separate evaluator.
- **Treys** (Python poker hand evaluator) is fast but only does 5-card hands — not ideal for Omaha. Stick with PokerKit.
- For the preflop lookup table, ProPokerTools or published Omaha Hi-Lo starting hand charts can be encoded as a dict.

---

## Phase 3: ML Bot (Deep CFR)

Since you're comfortable with PyTorch, the strongest approach is **Deep CFR** (Deep Counterfactual Regret Minimization). This is the algorithm family behind Facebook AI's poker work and is a natural fit.

### Why Deep CFR for Omaha Hi-Lo

Regular tabular CFR can't handle Omaha Hi-Lo — the information set space is too large (4 hole cards × board × betting history). Deep CFR replaces the regret tables with neural networks that generalize across similar situations.

### Architecture

```
┌───────────────────────────────────────────────────────┐
│                    Deep CFR Pipeline                  │
│                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │  Self-Play   │───▶│ Reservoir    │───▶│ Train    │ │
│  │  Traversals  │    │ Sampling     │    │ Networks │ │
│  │ (game tree)  │    │ (memory buf) │    │ (PyTorch)│ │
│  └─────────────┘    └──────────────┘    └──────────┘ │
│         │                                     │       │
│         ▼                                     ▼       │
│  ┌─────────────┐                      ┌──────────┐   │
│  │ Advantage   │◀─────────────────────│ Value    │   │
│  │ Network     │                      │ Network  │   │
│  │ π(a|info)   │                      │ V(info)  │   │
│  └─────────────┘                      └──────────┘   │
│                                                       │
│  At play time:                                        │
│  game_state → encode → advantage_net → action probs   │
└───────────────────────────────────────────────────────┘
```

### State Representation (input to the networks)

Encode the game state as a fixed-size feature vector:

| Feature Group         | Encoding                              | Size  |
|-----------------------|---------------------------------------|-------|
| Hole cards            | 4 cards × 52 one-hot (or rank+suit)   | ~52   |
| Board cards           | 5 slots × 52 one-hot                  | ~52   |
| Betting round         | One-hot (preflop/flop/turn/river)     | 4     |
| Pot size (normalized) | Float                                 | 1     |
| Stack sizes (norm.)   | Per-player floats                     | 6     |
| Betting history       | Action sequence encoding              | ~50   |
| Position              | One-hot                               | 6     |
| Hand strength features| Hi equity est., lo equity est.        | 2     |

Total: ~170–200 features. A 3-layer MLP (256→256→num_actions) is a reasonable starting point.

### Training Loop (pseudocode)

```
for iteration in range(T):
    # 1. External sampling CFR traversal
    for each player p:
        traverse(root, p, advantage_memories, strategy_memories)

    # 2. Train advantage network on collected samples
    advantage_net[p].train(advantage_memories[p])

    # 3. Every K iterations, train value network
    if iteration % K == 0:
        value_net.train(strategy_memories)
```

### Open-Source Starting Points

1. **OpenSpiel** (google-deepmind/open_spiel)
   - Has CFR, Deep CFR, and NFSP implementations
   - Supports custom game definitions via Python or C++
   - You'd define Omaha Hi-Lo as a custom game conforming to their `Game` interface
   - Strongest option, most community support
   - License: Apache 2.0

2. **RLCard** (datamllab/rlcard)
   - RL toolkit specifically for card games
   - Has DQN, NFSP agents ready to go
   - Doesn't have Omaha Hi-Lo, but the environment API is clean and you can add new games
   - Easier to get started than OpenSpiel
   - License: MIT

3. **Pluribus reimplementations**
   - Several open-source attempts exist on GitHub (search "pluribus poker")
   - Quality varies; most target No-Limit Hold'em
   - Can adapt the abstraction and search ideas

### Recommended Path

```
Week 1-2:  Define Omaha Hi-Lo as an OpenSpiel game
           (or wrap your existing GameEngine in their interface)

Week 3-4:  Run tabular CFR on a heavily abstracted version
           (bucket hands into ~200 categories)
           This gives you a baseline strategy

Week 5-6:  Implement Deep CFR with PyTorch
           - Advantage network: 3-layer MLP
           - Train via self-play traversals
           - Use reservoir sampling for memory efficiency

Week 7-8:  Evaluate against HandStrengthBot and RandomBot
           Tune hyperparameters
           Add real-time search (optional, for stronger play)
```

### Hand Abstraction (Critical for Omaha Hi-Lo)

Omaha has far more hand combinations than Hold'em. You need to bucket similar hands:

- **Preflop:** Cluster starting hands by features (hi potential, lo potential, suitedness, connectedness). ~200 buckets.
- **Postflop:** Use equity-based bucketing — run fast equity calculations and group hands with similar equity into ~500 buckets per street.
- **Earth Mover's Distance (EMD)** or **k-means on equity distributions** are standard approaches.

---

## Phase 4: Multiplayer Web Server (Future)

### Architecture for Internet Play

```
┌──────────┐     WebSocket      ┌──────────────┐
│ Browser  │◀──────────────────▶│ Game Server  │
│ (phone)  │                    │ (FastAPI +   │
└──────────┘                    │  WebSocket)  │
                                │              │
┌──────────┐     WebSocket      │  GameEngine  │
│ Browser  │◀──────────────────▶│  (unchanged) │
│ (desktop)│                    │              │
└──────────┘                    │  Bot threads │
                                └──────────────┘
```

- **Server:** FastAPI + `websockets` library. Your GameEngine stays exactly as-is; the server just translates between WebSocket messages and `submit_action()` calls.
- **Client:** A single HTML page with JS. For text mode, literally just styled `<pre>` blocks. For pixel art mode, an HTML5 Canvas.
- **Hosting:** For internet access, use **ngrok** (free, instant tunnel) during development, then a $5/mo VPS (DigitalOcean/Vultr) for persistent hosting.
- **Auth:** Simple room codes (generate a 4-digit code, friend enters it) — no accounts needed.

---

## Phase 5: Pixel Art UI (Future)

- **Pyxel** (Python retro game engine) if you want to stay pure Python
- **HTML5 Canvas + sprite sheets** if you want it browser-based (recommended since you're already serving a web client)
- Asset sources: OpenGameArt.org has free card sprites and table assets
- The GameEngine and Bot classes don't change at all — you just swap the rendering layer

---

## Development Order Summary

| Phase | What                          | Depends On | Est. Effort |
|-------|-------------------------------|------------|-------------|
| 1     | GameEngine + TextUI + RandomBot | —         | 1–2 weeks   |
| 2     | HandStrengthBot               | Phase 1    | 1 week      |
| 3     | ML Bot (Deep CFR)             | Phase 1    | 4–6 weeks   |
| 4     | Web server + mobile client    | Phase 1    | 1–2 weeks   |
| 5     | Pixel art UI                  | Phase 4    | 2–3 weeks   |

Phases 2, 3, and 4 can be done in parallel once Phase 1 is solid.
