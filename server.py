#!/usr/bin/env python3
"""Omaha Hi-Lo 8-or-Better — browser web server.

Run:  conda run -n omaha python server.py
Then open http://localhost:5000 in any browser or phone (same network).

Actions and results are appended to game_log.json.
"""
from __future__ import annotations

import json, os, random, signal, threading, time
from dataclasses import dataclass
from typing import Optional

from flask import Flask, jsonify, render_template_string, request
from pokerkit import (
    Automation, FixedLimitOmahaHoldemHighLowSplitEightOrBetter,
    OmahaEightOrBetterLowHand, OmahaHoldemHand, PotLimitOmahaHoldem,
)

# ── Game engine ────────────────────────────────────────────────────────────────

class PotLimitOmahaHoldemHighLowSplitEightOrBetter(PotLimitOmahaHoldem):
    hand_types = (OmahaHoldemHand, OmahaEightOrBetterLowHand)


@dataclass
class Action:
    action_type: str
    amount: Optional[int] = None


class GameEngine:
    _AUTOMATIONS = (
        Automation.ANTE_POSTING, Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING, Automation.HOLE_DEALING,
        Automation.BOARD_DEALING, Automation.CARD_BURNING,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
    )
    _STREET = {0: 'preflop', 3: 'flop', 4: 'turn', 5: 'river'}

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._state = None
        self._saved_holes: list = []
        self._player_names: list = []

    def start_hand(self, player_names: list, stacks: list) -> dict:
        n = len(player_names)
        self._player_names = list(player_names)
        sb, bb = self._cfg['small_blind'], self._cfg['big_blind']
        # PokerKit reverses blinds for 2-player heads-up
        blinds = (bb, sb) if n == 2 else tuple(
            sb if i == 0 else bb if i == 1 else 0 for i in range(n)
        )
        if self._cfg['game_type'] == 'fixed_limit':
            self._state = FixedLimitOmahaHoldemHighLowSplitEightOrBetter.create_state(
                self._AUTOMATIONS, True, 0, blinds,
                self._cfg['fixed_small_bet'], self._cfg['fixed_big_bet'],
                tuple(stacks), n,
            )
        else:
            self._state = PotLimitOmahaHoldemHighLowSplitEightOrBetter.create_state(
                self._AUTOMATIONS, True, 0, blinds, bb, tuple(stacks), n,
            )
        self._saved_holes = [[repr(c) for c in h] for h in self._state.hole_cards]
        return self.get_game_state()

    def get_game_state(self) -> dict:
        st = self._state
        board = [repr(c) for grp in st.board_cards for c in grp]
        n = len(self._player_names)
        actor = st.actor_index
        hand_over = not st.status
        legal: dict = {}
        if actor is not None and not hand_over:
            call_amt = st.checking_or_calling_amount or 0
            can_coc  = st.can_check_or_call()
            can_raise = st.can_complete_bet_or_raise_to()
            legal = {
                'can_fold':    st.can_fold(),
                'can_check':   can_coc and call_amt == 0,
                'can_call':    can_coc and call_amt > 0,
                'call_amount': call_amt,
                'can_raise':   can_raise,
                'min_raise':   st.min_completion_betting_or_raising_to_amount,
                'max_raise':   st.max_completion_betting_or_raising_to_amount,
                'pot_raise':   st.pot_completion_betting_or_raising_to_amount,
            }
        return {
            'stacks':        list(st.stacks),
            'bets':          list(st.bets),
            'pot':           st.total_pot_amount,
            'board':         board,
            'hole_cards':    self._saved_holes,
            'actor_index':   actor,
            'legal_actions': legal,
            'street':        self._STREET.get(len(board), f'street_{len(board)}'),
            'player_names':  list(self._player_names),
            'sb_pos':        0,
            'bb_pos':        1,
            'dealer_pos':    0 if n == 2 else n - 1,
            'hand_over':     hand_over,
            'payoffs':       list(st.payoffs) if hand_over else None,
            'player_count':  n,
        }

    def submit_action(self, actor_pos: int, action: Action) -> dict:
        st = self._state
        assert st.actor_index == actor_pos
        t = action.action_type
        if t == 'fold':
            st.fold()
        elif t in ('check', 'call'):
            st.check_or_call()
        elif t == 'raise':
            amt = action.amount or st.min_completion_betting_or_raising_to_amount
            st.complete_bet_or_raise_to(amt)
        return self.get_game_state()


class RandomBot:
    _MAX_RAISE = 400

    def __init__(self, name: str, starting_stack: int) -> None:
        self.name = name
        self.starting_stack = starting_stack

    def decide(self, gs: dict, my_pos: int) -> Action:
        legal = gs['legal_actions']
        pool: list[str] = []
        if legal.get('can_fold'):  pool.append('fold')
        if legal.get('can_check'): pool.append('check')
        if legal.get('can_call'):  pool.append('call')
        if legal.get('can_raise'): pool.append('raise')
        if not pool: pool = ['check']
        choice = random.choice(pool)
        if choice == 'raise':
            lo = int(legal['min_raise'])
            hi = max(min(int(legal['max_raise']), self._MAX_RAISE), lo)
            amt = random.randint(lo, hi) if lo != hi else lo
            return Action('raise', amt)
        return Action(choice)


# ── Flask server ───────────────────────────────────────────────────────────────

app = Flask(__name__)
_DIR     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(_DIR, 'game_log.json')
_lock    = threading.Lock()
_S: dict = {}   # single-user session


def _log(entry: dict) -> None:
    with _lock:
        try:
            with open(LOG_FILE) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append({'ts': time.time(), **entry})
        with open(LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)


def _active() -> list[int]:
    return [i for i, s in enumerate(_S['stacks']) if s > 0]


def _human_pk() -> int:
    return _S['rotation'].index(0)   # abs_seat 0 = human


def _build_rotation() -> list[int]:
    active = _active()
    n = len(active)
    if n == 0:
        return []
    ds  = _S['dealer_seat']
    btn = active.index(ds) if ds in active else 0
    return [active[(btn + 1 + i) % n] for i in range(n)]


def _advance_dealer() -> None:
    active = _active()
    if not active:
        return
    ds  = _S['dealer_seat']
    cur = active.index(ds) if ds in active else 0
    _S['dealer_seat'] = active[(cur + 1) % len(active)]


def _run_bot(gs: dict, one_street: bool = False) -> dict:
    """Run bot turns until it's the human's turn, the hand ends, or (when
    one_street=True) the board gains new cards (a new street was dealt).
    one_street mode lets the client animate each street reveal step-by-step."""
    hpk = _human_pk()
    initial_board = len(gs['board'])
    while not gs['hand_over'] and gs['actor_index'] is not None:
        if gs['actor_index'] == hpk:
            break
        if one_street and len(gs['board']) != initial_board:
            break   # new street dealt — pause so client can show the board
        bot_pos  = gs['actor_index']
        abs_seat = _S['rotation'][bot_pos]
        bot      = _S['bots'][abs_seat - 1]   # bots indexed from abs_seat 1
        action   = bot.decide(gs, bot_pos)
        _log({'event': 'bot_action', 'hand': _S['hand_count'],
              'actor': gs['player_names'][bot_pos],
              'action': action.action_type, 'amount': action.amount,
              'street': gs['street'], 'pot': gs['pot']})
        gs = _S['engine'].submit_action(bot_pos, action)
    return gs


def _finalise_hand(gs: dict) -> None:
    payoffs = gs['payoffs'] or [0] * len(_S['rotation'])
    for pk_pos, abs_seat in enumerate(_S['rotation']):
        _S['stacks'][abs_seat] += payoffs[pk_pos]
    _log({'event': 'hand_over', 'hand': _S['hand_count'],
          'board': gs['board'],
          'hole_cards': {gs['player_names'][i]: gs['hole_cards'][i]
                         for i in range(gs['player_count'])},
          'payoffs':    {gs['player_names'][i]: payoffs[i]
                         for i in range(gs['player_count'])},
          'stacks': _S['stacks'][:]})
    _advance_dealer()


def _resp(gs: dict) -> dict:
    hpk      = _human_pk()
    rotation = _S['rotation']
    players  = [{'pk_pos': pk, 'abs_seat': seat, 'is_human': seat == 0,
                  'global_stack': _S['stacks'][seat]}
                for pk, seat in enumerate(rotation)]
    return {
        'gs':            gs,
        'human_pk_pos':  hpk,
        'players':       players,
        'global_stacks': _S['stacks'][:],
        'hand_count':    _S['hand_count'],
        'num_bots':      len(_S.get('bots', [])),
        'max_bots':      5,
        'human_folded':  _S.get('human_folded', False),
        'game_type':     _S['cfg']['game_type'],
    }


def _start_hand():
    rotation = _build_rotation()
    names, stacks = [], []
    for seat in rotation:
        names.append('You' if seat == 0 else _S['bots'][seat - 1].name)
        stacks.append(_S['stacks'][seat])
    _S['rotation'] = rotation
    _S['human_folded'] = False
    _S['hand_count'] += 1
    gs = _S['engine'].start_hand(names, stacks)
    _log({'event': 'new_hand', 'hand': _S['hand_count'],
          'dealer_seat': _S['dealer_seat'], 'stacks': _S['stacks'][:]})
    gs = _run_bot(gs)
    _S['current_gs'] = gs
    if gs['hand_over']:
        _finalise_hand(gs)
    return jsonify(_resp(gs))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/new_game', methods=['POST'])
def api_new_game():
    body  = request.json or {}
    start = int(body.get('starting_stack', 200))
    gtype = body.get('game_type', 'pot_limit')
    nbots = max(1, min(5, int(body.get('num_bots', 1))))
    cfg   = {'game_type': gtype, 'fixed_small_bet': 2, 'fixed_big_bet': 4,
             'small_blind': 1, 'big_blind': 2}
    _S.clear()
    _S.update({
        'engine':          GameEngine(cfg),
        'bots':            [RandomBot(f'Bot {i+1}', start) for i in range(nbots)],
        'stacks':          [start] * (nbots + 1),
        'dealer_seat':     0,
        'hand_count':      0,
        'starting_stack':  start,
        'cfg':             cfg,
    })
    _log({'event': 'new_game', 'starting_stack': start,
          'game_type': gtype, 'num_bots': nbots})
    return _start_hand()


@app.route('/api/next_hand', methods=['POST'])
def api_next_hand():
    if not _S:
        return jsonify({'error': 'No active game'}), 400
    if not _S.get('current_gs', {}).get('hand_over'):
        return jsonify({'error': 'Hand not over'}), 400
    active = _active()
    if _S['stacks'][0] <= 0 or len(active) < 2:
        return jsonify({'game_over': True, 'stacks': _S['stacks']}), 200
    return _start_hand()


@app.route('/api/action', methods=['POST'])
def api_action():
    if not _S:
        return jsonify({'error': 'No active game'}), 400
    body = request.json or {}
    gs   = _S.get('current_gs')
    if not gs or gs['hand_over']:
        return jsonify({'error': 'No hand in progress'}), 400
    hpk = _human_pk()
    if gs['actor_index'] != hpk:
        return jsonify({'error': 'Not your turn'}), 400
    action = Action(body.get('action'), body.get('amount'))
    _log({'event': 'human_action', 'hand': _S['hand_count'],
          'action': action.action_type, 'amount': action.amount,
          'street': gs['street'], 'pot': gs['pot'],
          'hole_cards': gs['hole_cards'][hpk], 'board': gs['board']})
    gs = _S['engine'].submit_action(hpk, action)
    # If human folded or went all-in, pause at street boundaries so the
    # client can reveal each board card with a delay via /api/bot_step.
    human_out = (action.action_type == 'fold') or (
        not gs['hand_over'] and gs['stacks'][hpk] == 0
    )
    if human_out:
        _S['human_folded'] = (action.action_type == 'fold')
    gs = _run_bot(gs, one_street=human_out)
    _S['current_gs'] = gs
    if gs['hand_over']:
        _finalise_hand(gs)
    return jsonify(_resp(gs))


@app.route('/api/bot_step', methods=['POST'])
def api_bot_step():
    """Advance one street of bot-only play. Called by the client when the
    human is out of the hand (folded / all-in) and bots are still playing."""
    if not _S:
        return jsonify({'error': 'No active game'}), 400
    gs = _S.get('current_gs')
    if not gs:
        return jsonify({'error': 'No hand in progress'}), 400
    if gs['hand_over']:
        return jsonify(_resp(gs))
    hpk = _human_pk()
    if gs['actor_index'] == hpk:
        return jsonify(_resp(gs))   # shouldn't happen; human's turn
    gs = _run_bot(gs, one_street=True)
    _S['current_gs'] = gs
    if gs['hand_over']:
        _finalise_hand(gs)
    return jsonify(_resp(gs))


@app.route('/api/add_bot', methods=['POST'])
def api_add_bot():
    if not _S:
        return jsonify({'error': 'No active game'}), 400
    if len(_S['bots']) >= 5:
        return jsonify({'error': 'Maximum 5 bots reached'}), 400
    gs = _S.get('current_gs', {})
    if gs and not gs.get('hand_over', True):
        return jsonify({'error': 'Wait for the hand to end'}), 400
    name  = f'Bot {len(_S["bots"]) + 1}'
    start = _S.get('starting_stack', 200)
    _S['bots'].append(RandomBot(name, start))
    _S['stacks'].append(start)
    _log({'event': 'add_bot', 'name': name, 'stack': start})
    return jsonify({'ok': True, 'name': name,
                    'num_bots': len(_S['bots']), 'stacks': _S['stacks']})


@app.route('/api/state')
def api_state():
    if not _S or 'current_gs' not in _S:
        return jsonify({'error': 'No active game'}), 404
    return jsonify(_resp(_S['current_gs']))


@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    _log({'event': 'shutdown'})
    threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    return jsonify({'ok': True})


# ── Embedded HTML / CSS / JS ───────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Omaha Hi-Lo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#1a5c2a;color:#fff;font-family:system-ui,-apple-system,sans-serif;-webkit-tap-highlight-color:transparent}
body{display:flex;flex-direction:column;align-items:center;padding:8px;max-width:520px;margin:0 auto;gap:5px}

/* ── Header ── */
.hdr{width:100%;display:flex;justify-content:space-between;align-items:center;padding:7px 10px;background:rgba(0,0,0,.38);border-radius:8px}
.hdr h1{font-size:1rem;font-weight:800;letter-spacing:2px;color:#a5d6a7}
.hdr-right{display:flex;align-items:center;gap:8px}
.hinfo{font-size:.72rem;opacity:.75;text-align:right;line-height:1.5}
.end-btn{background:#7f1d1d;border:none;border-radius:6px;color:#fca5a5;font-size:.72rem;font-weight:700;padding:5px 9px;cursor:pointer;touch-action:manipulation;white-space:nowrap}
.end-btn:active{opacity:.7}

/* ── Player boxes ── */
.pbox{width:100%;background:rgba(0,0,0,.22);border-radius:10px;padding:10px 12px;border:2px solid transparent;transition:border-color .2s}
.pbox.actor{border-color:#ffeb3b}
.pbox.human{background:rgba(0,0,0,.32)}
.pbox.compact{padding:7px 10px}
.ph{display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-wrap:wrap}
.pbox.compact .ph{margin-bottom:4px}
.pname{font-weight:700;font-size:.92rem;flex:1}
.pstack{font-size:.85rem;color:#ffd54f;font-weight:600}
.pstack.busted{color:#f87171}
.plabel{font-size:.67rem;color:#a5d6a7;background:rgba(255,255,255,.12);padding:2px 6px;border-radius:4px;white-space:nowrap}

/* ── Cards ── */
.cards{display:flex;gap:5px;flex-wrap:wrap;min-height:60px;align-items:center}
.pbox.compact .cards{min-height:50px;gap:4px}
.card{width:44px;height:60px;background:#fff;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:1rem;font-weight:800;line-height:1.1;box-shadow:0 2px 6px rgba(0,0,0,.5);color:#111;user-select:none;flex-shrink:0}
.pbox.compact .card{width:36px;height:50px;font-size:.85rem}
.card .s{font-size:1rem;line-height:1}
.pbox.compact .card .s{font-size:.85rem}
.card.r{color:#c62828}
.card.hid{background:#1a3a6b;background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.07) 0,rgba(255,255,255,.07) 1px,transparent 1px,transparent 7px)}
.pbet{font-size:.75rem;color:#ff8a65;margin-top:3px}

/* ── Board ── */
.board{width:100%;background:rgba(0,0,0,.18);border-radius:10px;padding:10px 14px;text-align:center}
.pot{font-size:1.05rem;font-weight:700;color:#ffd54f;margin-bottom:5px}
.blbl{font-size:.67rem;color:#a5d6a7;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
.bcards{display:flex;gap:5px;justify-content:center;flex-wrap:wrap;min-height:60px;align-items:center}
.sbdg{font-size:.66rem;background:rgba(255,255,255,.15);padding:2px 7px;border-radius:10px;margin-left:6px;font-weight:600;vertical-align:middle}

/* ── Action area ── */
.actarea{width:100%;display:flex;flex-direction:column;gap:7px}
.btnrow{display:flex;gap:7px}
.btn{border:none;border-radius:8px;padding:15px 10px;font-size:.92rem;font-weight:700;cursor:pointer;color:#fff;touch-action:manipulation;flex:1;transition:opacity .1s,transform .08s}
.btn:active{opacity:.72;transform:scale(.96)}
.fold{background:#b71c1c}.chk{background:#1565c0}.call{background:#1565c0}
.bnew{background:#2e7d32;width:100%;padding:16px}
.bnxt{background:#4a148c;width:100%;padding:16px}
.badd{background:#0d47a1;width:100%;padding:12px;font-size:.85rem}

/* ── Raise controls ── */
.rctrl{background:rgba(0,0,0,.22);border-radius:10px;padding:10px 12px}
.rctrl label{font-size:.74rem;color:#a5d6a7;display:block;margin-bottom:6px}
.rrow{display:flex;gap:5px;align-items:center;margin-bottom:5px}
.rinput{flex:1;padding:10px 6px;border-radius:6px;border:none;font-size:1rem;font-weight:700;text-align:center;background:rgba(255,255,255,.92);color:#111;min-width:0}
.sm{padding:10px 8px;border:none;border-radius:6px;font-size:.74rem;font-weight:700;cursor:pointer;color:#fff;background:rgba(255,255,255,.2);touch-action:manipulation;white-space:nowrap;flex-shrink:0}
.sm:active{opacity:.7}
.rbtn{flex:1.4;padding:13px;background:#e65100;border:none;border-radius:8px;font-size:.9rem;font-weight:700;color:#fff;cursor:pointer;touch-action:manipulation}
.rbtn:active{opacity:.7}

/* ── Result ── */
.res{width:100%;background:rgba(0,0,0,.38);border-radius:10px;padding:14px;text-align:center}
.res h2{font-size:1.05rem;margin-bottom:10px;color:#ffd54f}
.rline{display:flex;justify-content:space-between;padding:5px 0;font-size:.85rem;border-bottom:1px solid rgba(255,255,255,.1)}
.rline.win{color:#86efac}.rline.lose{color:#fca5a5}

/* ── Start screen ── */
.start{width:100%;display:flex;flex-direction:column;gap:14px;padding:16px 0}
.start h2{text-align:center;font-size:1.15rem;color:#a5d6a7}
.fg{display:flex;flex-direction:column;gap:5px}
.fg label{font-size:.82rem;color:#a5d6a7}
.fg select,.fg input[type=number]{padding:12px;border-radius:7px;border:none;font-size:1rem;background:rgba(255,255,255,.92);color:#111}
.bot-row{display:flex;align-items:center;gap:8px}
.bot-row span{font-size:.9rem;color:#a5d6a7;min-width:60px}
.bot-count{width:60px;padding:10px;border-radius:7px;border:none;font-size:1rem;font-weight:700;text-align:center;background:rgba(255,255,255,.92);color:#111}
.bot-adj{padding:10px 16px;border:none;border-radius:7px;font-size:1.1rem;font-weight:700;color:#fff;background:rgba(255,255,255,.2);cursor:pointer;touch-action:manipulation}
.bot-adj:active{opacity:.7}

.thinking{text-align:center;padding:14px;opacity:.65;font-size:.88rem}
.folded-out{opacity:.7}

/* ── Toast ── */
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.85);color:#fff;padding:10px 20px;border-radius:20px;font-size:.85rem;pointer-events:none;opacity:0;transition:opacity .3s;z-index:99}
.toast.show{opacity:1}
</style>
</head>
<body>

<div class="hdr">
  <h1>OMAHA HI-LO</h1>
  <div class="hdr-right">
    <div class="hinfo" id="hinfo">8-or-Better</div>
    <button class="end-btn" onclick="shutdown()">✕ End</button>
  </div>
</div>

<div id="app" style="width:100%;display:flex;flex-direction:column;gap:5px"></div>
<div class="toast" id="toast"></div>

<script>
// ── Audio ─────────────────────────────────────────────────────────────────────
let _ac = null;
function ac() { return _ac || (_ac = new (window.AudioContext||window.webkitAudioContext)()); }
function tone(freq, dur, type, vol, when) {
  const c=ac(), t=c.currentTime+(when||0);
  const o=c.createOscillator(), g=c.createGain();
  o.connect(g); g.connect(c.destination);
  o.type=type||'sine'; o.frequency.value=freq;
  g.gain.setValueAtTime(0,t);
  g.gain.linearRampToValueAtTime(vol||0.2, t+0.012);
  g.gain.exponentialRampToValueAtTime(0.001, t+dur);
  o.start(t); o.stop(t+dur+0.02);
}
const SFX = {
  deal:    ()=>{ tone(900,.07,'triangle',.18); tone(1100,.07,'triangle',.18,.07); tone(900,.07,'triangle',.15,.14); tone(1100,.07,'triangle',.15,.21); },
  check:   ()=>tone(700,.06,'square',.09),
  call:    ()=>{ tone(440,.1,'sine',.2); tone(550,.1,'sine',.18,.09); },
  raise:   ()=>{ tone(550,.08,'triangle',.2); tone(700,.08,'triangle',.2,.08); tone(900,.1,'triangle',.22,.16); },
  fold:    ()=>{ tone(280,.22,'sawtooth',.15); tone(200,.28,'sawtooth',.1,.18); },
  win:     ()=>{ [523,659,784,1047].forEach((f,i)=>tone(f,.38,'sine',.32,i*.13)); },
  lose:    ()=>{ [380,300,240,180].forEach((f,i)=>tone(f,.3,'sawtooth',.15,i*.15)); },
  split:   ()=>{ tone(440,.12,'sine',.2); tone(440,.12,'sine',.2,.25); },
  flop:    ()=>{ [380,480,580].forEach((f,i)=>tone(f,.12,'triangle',.18,i*.1)); },
  turn:    ()=>{ tone(420,.12,'triangle',.18); tone(560,.12,'triangle',.18,.12); },
  river:   ()=>{ tone(460,.12,'triangle',.18); tone(680,.18,'triangle',.22,.13); },
  newHand: ()=>tone(440,.15,'sine',.2),
  addBot:  ()=>{ tone(520,.1,'sine',.2); tone(740,.12,'sine',.2,.1); },
  shutdown:()=>{ [380,280,180].forEach((f,i)=>tone(f,.28,'sawtooth',.2,i*.13)); },
  error:   ()=>tone(180,.35,'sawtooth',.2),
};

// ── State ─────────────────────────────────────────────────────────────────────
let G=null, RA=0, _prevStreet=null, _prevHandOver=false, _prevHandCount=0;
let _stepTimer=null;
const $=id=>document.getElementById(id);
const app=$('app');

function needsBotStep(d) {
  if (!d||!d.gs) return false;
  const gs=d.gs;
  return !gs.hand_over && gs.actor_index!==null && gs.actor_index!==d.human_pk_pos;
}
function cancelStep() { clearTimeout(_stepTimer); _stepTimer=null; }
async function botStep() {
  G=await post('/api/bot_step');
  render();
  if (needsBotStep(G)) _stepTimer=setTimeout(botStep, 1100);
}
function scheduleStep() { cancelStep(); _stepTimer=setTimeout(botStep, 900); }

function toast(msg, ms=2200) {
  const t=$('toast'); t.textContent=msg; t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), ms);
}

// ── API ───────────────────────────────────────────────────────────────────────
async function post(url, body) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body||{})});
  return r.json();
}

async function newGame() {
  cancelStep();
  const start = parseInt($('ss')?.value||'200');
  const type  = $('gt')?.value||'pot_limit';
  const nbots = parseInt($('nb')?.value||'1');
  _prevStreet=null; _prevHandOver=false; _prevHandCount=0;
  G = await post('/api/new_game', {starting_stack:start, game_type:type, num_bots:nbots});
  SFX.deal(); render();
}

async function nextHand() {
  cancelStep();
  const d = await post('/api/next_hand');
  if (d.game_over) { SFX.lose(); gameOver(d.stacks); return; }
  _prevStreet=null; _prevHandOver=false;
  G=d; SFX.newHand(); render();
}

async function act(type, amount) {
  SFX[type] && SFX[type]();
  const body={action:type}; if(amount!==undefined) body.amount=amount;
  G = await post('/api/action', body);
  render();
  // Human folded or all-in: drive bot play street-by-street with delays
  if (needsBotStep(G)) scheduleStep();
}

async function addBot() {
  const d = await post('/api/add_bot');
  if (d.error) { SFX.error(); toast(d.error); return; }
  SFX.addBot();
  if (G) { G.num_bots=d.num_bots; G.global_stacks=d.stacks; }
  toast(`${d.name} joined with $${d.stacks[d.stacks.length-1]}`);
  render();
}

async function shutdown() {
  cancelStep();
  SFX.shutdown();
  if (!confirm('Shut down the server and end the session?')) return;
  try { await post('/api/shutdown'); } catch(e) {}
  document.body.innerHTML='<div style="text-align:center;padding:60px 20px;color:#a5d6a7"><h2 style="font-size:1.3rem;margin-bottom:12px">Server stopped.</h2><p style="opacity:.7">You can close this tab.</p></div>';
}

// ── Cards ─────────────────────────────────────────────────────────────────────
const SUIT={s:'♠',h:'♥',d:'♦',c:'♣'}, RED={h:1,d:1};
function card(c) {
  const rank=c.slice(0,-1).toUpperCase(), suit=c.slice(-1).toLowerCase();
  return `<div class="card${RED[suit]?' r':''}">${rank}<span class="s">${SUIT[suit]||suit}</span></div>`;
}
function hidCard() { return '<div class="card hid"></div>'; }
function renderCards(arr,hidden) { return hidden?arr.map(()=>hidCard()).join(''):arr.map(card).join(''); }

// ── Position labels ───────────────────────────────────────────────────────────
function posTag(pk, gs) {
  const n=gs.player_count, t=[];
  if(pk===gs.dealer_pos) t.push(n===2?'BTN/SB':'BTN');
  if(pk===gs.sb_pos&&n>2) t.push('SB');
  if(pk===gs.bb_pos) t.push('BB');
  return t.join(' ');
}

// ── Player box HTML ───────────────────────────────────────────────────────────
function playerBox(p, gs, compact, forceReveal) {
  const pk=p.pk_pos, isOver=gs.hand_over;
  const isActor=!isOver&&gs.actor_index===pk;
  const hidden=!p.is_human&&!isOver&&!forceReveal;
  const bet=gs.bets[pk]||0;
  const stack=isOver?p.global_stack:gs.stacks[pk];
  const busted=p.global_stack===0;
  const tag=posTag(pk,gs);
  const name=gs.player_names[pk];
  const humanFoldedNow = p.is_human && G.human_folded && !isOver;
  return `<div class="pbox${isActor?' actor':''}${p.is_human?' human':''}${compact?' compact':''}${humanFoldedNow?' folded-out':''}">
    <div class="ph">
      <span class="pname">${name}</span>
      <span class="pstack${busted&&isOver?' busted':''}">${busted&&isOver?'BUSTED':'$'+stack}</span>
      ${tag?`<span class="plabel">${tag}</span>`:''}
      ${humanFoldedNow?`<span class="plabel" style="color:#fca5a5;background:rgba(220,38,38,.25)">FOLDED</span>`:''}
    </div>
    <div class="cards">${renderCards(gs.hole_cards[pk],hidden)}</div>
    ${bet?`<div class="pbet">Bet: $${bet}</div>`:''}
  </div>`;
}

// ── Main render ───────────────────────────────────────────────────────────────
function render() {
  if (!G||!G.gs) { renderStart(); return; }
  const gs=G.gs, hpk=G.human_pk_pos;
  const isOver=gs.hand_over;

  // ── Sound triggers ────────────────────────────────────────────────────────
  if (G.hand_count !== _prevHandCount) {
    _prevHandCount=G.hand_count; _prevStreet=null; _prevHandOver=false;
  }
  if (_prevStreet!==null && _prevStreet!==gs.street && !isOver) {
    (SFX[gs.street]||SFX.newHand)();
  }
  _prevStreet = gs.street;
  if (isOver && !_prevHandOver) {
    _prevHandOver=true;
    const hp=(gs.payoffs||[])[hpk]||0;
    setTimeout(()=>{ if(hp>0) SFX.win(); else if(hp<0) SFX.lose(); else SFX.split(); }, 250);
  }

  // ── Header ────────────────────────────────────────────────────────────────
  const gtLabel = G.game_type==='fixed_limit' ? 'FL $2/$4' : 'PL';
  $('hinfo').textContent = `${gtLabel}  Hand #${G.hand_count}  ${gs.street.toUpperCase()}`;

  const bots    = G.players.filter(p=>!p.is_human);
  const human   = G.players.find(p=>p.is_human);
  const compact = bots.length > 1;

  let h = '';

  // ── Bot rows (above board) ────────────────────────────────────────────────
  bots.forEach(p => { h += playerBox(p, gs, compact, false); });

  // ── Board ─────────────────────────────────────────────────────────────────
  const potLine = isOver && gs.payoffs
    ? gs.payoffs.map((v,i)=>`${gs.player_names[i]}: ${v>=0?'+':''}${v}`).join(' &nbsp;|&nbsp; ')
    : `Pot: $${gs.pot}`;
  h += `<div class="board">
    <div class="pot">${potLine}</div>
    <div class="blbl">Board<span class="sbdg">${gs.street}</span></div>
    <div class="bcards">${gs.board.length?gs.board.map(card).join(''):'<span style="opacity:.35">—</span>'}</div>
  </div>`;

  // ── Human row ─────────────────────────────────────────────────────────────
  h += playerBox(human, gs, false, false);

  // ── Actions ───────────────────────────────────────────────────────────────
  h += '<div class="actarea">';

  if (isOver) {
    const pay=gs.payoffs||[], hp=pay[hpk]||0;
    const emoji=hp>0?'🎉':hp<0?'':hp===0?'🤝':'';
    const msg=hp>0?`You won $${hp}! ${emoji}`:hp<0?`You lost $${Math.abs(hp)}.`:`Split pot. ${emoji}`;
    const canGo=G.global_stacks[0]>0&&G.global_stacks.slice(1).some(s=>s>0);

    // Result box with per-player breakdown
    let rows='';
    G.players.forEach(p=>{
      const pk=p.pk_pos, pv=pay[pk]||0;
      const cls=pv>0?'win':pv<0?'lose':'';
      rows+=`<div class="rline ${cls}"><span>${gs.player_names[pk]}</span><span>${pv>=0?'+':''}${pv} ($${p.global_stack})</span></div>`;
    });
    h+=`<div class="res"><h2>${msg}</h2>${rows}</div>`;

    if (canGo) {
      h+=`<button class="btn bnxt" onclick="nextHand()">Next Hand ▶</button>`;
      if (G.num_bots < G.max_bots) {
        h+=`<button class="btn badd" onclick="addBot()">＋ Add Bot Player</button>`;
      }
    } else {
      h+=`<button class="btn bnew" onclick="renderStart()">New Game</button>`;
    }

  } else if (gs.actor_index===hpk) {
    const L=gs.legal_actions;
    const mn=L.min_raise||0, mx=L.max_raise||0, pt=L.pot_raise||mn;
    if(RA<mn||RA>mx) RA=mn;

    const btns=[];
    if(L.can_fold)  btns.push(`<button class="btn fold" onclick="act('fold')">Fold</button>`);
    if(L.can_check) btns.push(`<button class="btn chk"  onclick="act('check')">Check</button>`);
    if(L.can_call)  btns.push(`<button class="btn call" onclick="act('call')">Call $${L.call_amount}</button>`);
    if(L.can_raise && G.game_type==='fixed_limit') {
      const betLabel = L.call_amount===0 ? `Bet $${mn}` : `Raise $${mn}`;
      btns.push(`<button class="btn rbtn" onclick="act('raise',${mn})">${betLabel}</button>`);
    }
    if(btns.length) h+=`<div class="btnrow">${btns.join('')}</div>`;

    if(L.can_raise && G.game_type!=='fixed_limit') {
      h+=`<div class="rctrl">
        <label>Raise to: $<span id="rd">${RA||mn}</span>&nbsp;&nbsp;(min $${mn} &ndash; max $${mx})</label>
        <div class="rrow">
          <button class="sm" onclick="adj(-10)">-10</button>
          <button class="sm" onclick="adj(-1)">-1</button>
          <input class="rinput" id="ri" type="number" value="${RA||mn}" min="${mn}" max="${mx}" inputmode="numeric" oninput="sync()">
          <button class="sm" onclick="adj(1)">+1</button>
          <button class="sm" onclick="adj(10)">+10</button>
        </div>
        <div class="rrow">
          <button class="sm" onclick="sr(${mn})">Min</button>
          <button class="sm" onclick="sr(${Math.round(pt)})">Pot</button>
          <button class="sm" onclick="sr(${mx})">Max</button>
          <button class="rbtn" onclick="doRaise()">Raise ▶</button>
        </div>
      </div>`;
    }
  } else {
    const watching = G.human_folded
      ? `You folded — watching the bots play (${gs.street.toUpperCase()})…`
      : 'Bot is thinking…';
    h+=`<div class="thinking">${watching}</div>`;
  }

  h+='</div>';
  app.innerHTML=h;
}

// ── Start screen ──────────────────────────────────────────────────────────────
let _nbots=1;
function setBots(d) {
  _nbots=Math.max(1,Math.min(5,_nbots+d));
  const el=$('nb'); if(el) el.value=_nbots;
}
function renderStart() {
  app.innerHTML=`<div class="start">
    <h2>New Game</h2>
    <div class="fg"><label>Starting Stack ($)</label>
      <input type="number" id="ss" value="200" min="10" max="1000000" inputmode="numeric"></div>
    <div class="fg"><label>Game Type</label>
      <select id="gt">
        <option value="pot_limit">Pot Limit — max bet = pot size, no raise cap</option>
        <option value="fixed_limit">Fixed Limit $2/$4 — fixed bets, max 4 raises/street</option>
      </select></div>
    <div class="fg"><label>Number of Bots</label>
      <div class="bot-row">
        <button class="bot-adj" onclick="setBots(-1)">−</button>
        <input class="bot-count" id="nb" type="number" value="${_nbots}" min="1" max="5" readonly>
        <button class="bot-adj" onclick="setBots(1)">＋</button>
        <span>(1 – 5)</span>
      </div></div>
    <button class="btn bnew" onclick="newGame()">Start Game</button>
  </div>`;
}

function gameOver(stacks) {
  const youWin=stacks[0]>0;
  const names=['You',...stacks.slice(1).map((_,i)=>`Bot ${i+1}`)];
  let rows=stacks.map((s,i)=>`<div class="rline"><span>${names[i]}</span><span>$${s}</span></div>`).join('');
  app.innerHTML=`<div class="res" style="margin-top:16px">
    <h2>${youWin?'You Win! 🏆':'Game Over 💀'}</h2>${rows}
  </div>
  <button class="btn bnew" onclick="renderStart()">Play Again</button>`;
}

// ── Raise helpers ─────────────────────────────────────────────────────────────
function adj(d){ const i=$('ri');if(!i)return; const mn=+i.min,mx=+i.max; RA=Math.max(mn,Math.min(mx,(RA||mn)+d)); i.value=RA; $('rd').textContent=RA; }
function sr(v){ const i=$('ri');if(!i)return; RA=v;i.value=v;$('rd').textContent=v; }
function sync(){ const i=$('ri');if(!i)return; RA=parseInt(i.value)||0;$('rd').textContent=RA; }
function doRaise(){ const i=$('ri'); act('raise',i?parseInt(i.value):RA); }

renderStart();
</script>
</body>
</html>"""


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Omaha Hi-Lo web server')
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--host', default='0.0.0.0')
    args = p.parse_args()
    print(f'\n  Omaha Hi-Lo server')
    print(f'  Local:   http://localhost:{args.port}')
    print(f'  Network: http://<your-ip>:{args.port}')
    print(f'  Log:     {LOG_FILE}\n')
    app.run(host=args.host, port=args.port, debug=False)
