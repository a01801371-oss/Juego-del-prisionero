"""
╔══════════════════════════════════════════════════════════════════╗
║  IPD Research Dashboard — Axelrod & Hamilton (1981)             ║
║  Iterated Prisoner's Dilemma · Option 1 · 15 Strategies         ║
║  Streamlit Application · app.py                                  ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    pip install streamlit numpy pandas scipy plotly
    streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────
import io
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────
import numpy as np
from numpy.random import Generator, PCG64
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ═════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IPD Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark, minimal, academic
# ═════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=DM+Sans:wght@300;400;500;700&display=swap');

  /* ── Root palette ── */
  :root {
    --bg:        #080b12;
    --surface:   #0e1420;
    --surface2:  #141b2d;
    --border:    rgba(56, 189, 248, 0.12);
    --accent:    #38bdf8;
    --accent2:   #34d399;
    --warn:      #fbbf24;
    --danger:    #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'DM Sans', sans-serif;
  }

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── App container ── */
  .block-container {
    padding: 1.5rem 2.5rem 3rem;
    max-width: 1600px;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface2);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border-bottom: none;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: var(--muted);
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 500;
    padding: 8px 20px;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
  }
  [data-testid="metric-container"] label {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    color: var(--accent) !important;
    font-size: 1.6rem !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: var(--accent);
    color: #080b12;
    border: none;
    border-radius: 7px;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 13px;
    padding: 0.55rem 1.4rem;
    letter-spacing: 0.03em;
    transition: all 0.15s ease;
    width: 100%;
  }
  .stButton > button:hover {
    background: #7dd3fc;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(56,189,248,0.3);
  }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: transparent;
    color: var(--accent2);
    border: 1px solid var(--accent2);
    border-radius: 7px;
    font-family: var(--mono);
    font-size: 12px;
    padding: 0.45rem 1rem;
    width: 100%;
  }
  .stDownloadButton > button:hover {
    background: rgba(52, 211, 153, 0.1);
  }

  /* ── Sliders ── */
  [data-testid="stSlider"] label { font-size: 12px; color: var(--muted); font-family: var(--mono); }

  /* ── Selectbox / Multiselect ── */
  [data-baseweb="select"] {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
  }

  /* ── Number input ── */
  [data-testid="stNumberInput"] input {
    background: var(--surface2);
    border-color: var(--border);
    font-family: var(--mono);
    font-size: 13px;
  }

  /* ── Info / success boxes ── */
  .stAlert { border-radius: 8px; font-size: 13px; }

  /* ── Section headers ── */
  .section-header {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }

  /* ── Badge ── */
  .badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    color: var(--accent);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 7px;
    margin-right: 4px;
    vertical-align: middle;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .badge-green {
    background: rgba(52,211,153,0.12);
    color: var(--accent2);
    border-color: rgba(52,211,153,0.25);
  }
  .badge-warn {
    background: rgba(251,191,36,0.12);
    color: var(--warn);
    border-color: rgba(251,191,36,0.25);
  }

  /* ── App title ── */
  .app-title {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
    margin: 0 0 2px 0;
  }
  .app-subtitle {
    font-family: var(--sans);
    font-size: 13px;
    color: var(--muted);
    margin: 0;
  }
  .app-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  1. REPRODUCIBLE RNG
# ═════════════════════════════════════════════════════════════════

class ReproducibleRNG:
    """
    Reproducible random number generator using numpy PCG64.

    Wraps numpy.random.Generator(PCG64(seed)) to provide a stable,
    reproducible interface. Does NOT use numpy.random.seed() or the
    standard library random module.

    Args:
        seed: Integer seed for PCG64. Default: 20260211 (YYYYMMDD).
    """

    def __init__(self, seed: int = 20260211):
        self.seed = seed
        self.rng  = Generator(PCG64(seed))

    def uniform(self, low: float = 0.0, high: float = 1.0,
                size=None) -> np.ndarray:
        """Uniform samples in [low, high)."""
        return self.rng.uniform(low, high, size)

    def choice(self, a, size=None, replace: bool = True,
               p=None) -> np.ndarray:
        """Random choice from array a."""
        return self.rng.choice(a, size, replace, p)

    def integers(self, low: int, high: int = None,
                 size=None) -> np.ndarray:
        """Random integers in [low, high)."""
        return self.rng.integers(low, high, size)

    def standard_normal(self, size=None) -> np.ndarray:
        """Standard normal samples."""
        return self.rng.standard_normal(size)

    def reset(self) -> None:
        """Reinitialize with the same seed (full reproducibility)."""
        self.rng = Generator(PCG64(self.seed))

    def draw_sample(self, n: int = 10_000) -> np.ndarray:
        """Generate n uniform samples for QC tests."""
        return self.rng.uniform(0.0, 1.0, n)


# ═════════════════════════════════════════════════════════════════
#  2. PAYOFF MATRIX
# ═════════════════════════════════════════════════════════════════

C, D = "C", "D"   # Action constants

class PayoffMatrix:
    """
    Standard Prisoner's Dilemma payoff matrix with strict validation.

    Enforces T > R > P > S. Raises ValueError on violation.

    Args:
        T: Temptation payoff (defect vs cooperate)
        R: Reward for mutual cooperation
        P: Punishment for mutual defection
        S: Sucker's payoff (cooperate vs defect)
    """

    def __init__(self, T: float = 5.0, R: float = 3.0,
                 P: float = 1.0, S: float = 0.0):
        if not (T > R > P > S):
            raise ValueError(
                f"Payoff condition T > R > P > S violated: "
                f"T={T}, R={R}, P={P}, S={S}"
            )
        self.T, self.R, self.P, self.S = T, R, P, S
        self._lut: Dict[Tuple[str, str], Tuple[float, float]] = {
            (C, C): (R, R),
            (C, D): (S, T),
            (D, C): (T, S),
            (D, D): (P, P),
        }

    def payoffs(self, a_i: str, a_j: str) -> Tuple[float, float]:
        """Return (payoff_i, payoff_j) for actions a_i, a_j."""
        return self._lut[(a_i, a_j)]

    def as_dict(self) -> Dict[str, float]:
        return dict(T=self.T, R=self.R, P=self.P, S=self.S)


# ═════════════════════════════════════════════════════════════════
#  3. STRATEGY BASE CLASS + 15 IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════

class Strategy(ABC):
    """Abstract base class for all IPD strategies."""

    def __init__(self, rng: ReproducibleRNG = None):
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        """Clear history for a fresh match."""
        self.my_history:  List[str] = []
        self.opp_history: List[str] = []

    def record(self, my_a: str, opp_a: str) -> None:
        """Record this round's actions."""
        self.my_history.append(my_a)
        self.opp_history.append(opp_a)

    @abstractmethod
    def action(self) -> str:
        """Return 'C' or 'D'."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name."""
        ...


# ── 1. ALL-C ──────────────────────────────────────────────────
class AllC(Strategy):
    """Always cooperate — unconditional baseline."""
    @property
    def name(self): return "ALL-C"
    def action(self): return C


# ── 2. ALL-D ──────────────────────────────────────────────────
class AllD(Strategy):
    """Always defect — unconditional baseline."""
    @property
    def name(self): return "ALL-D"
    def action(self): return D


# ── 3. TIT FOR TAT ────────────────────────────────────────────
class TitForTat(Strategy):
    """
    Cooperate on round 1; thereafter mirror the opponent's last move.
    Winner of Axelrod's 1980 tournament (Rapoport, 1980).
    """
    @property
    def name(self): return "TIT FOR TAT"
    def action(self):
        return C if not self.opp_history else self.opp_history[-1]


# ── 4. TIT FOR TWO TATS ───────────────────────────────────────
class TitForTwoTats(Strategy):
    """Defect only after two consecutive defections by opponent."""
    @property
    def name(self): return "TIT FOR TWO TATS"
    def action(self):
        if len(self.opp_history) < 2:
            return C
        return D if self.opp_history[-1] == D and self.opp_history[-2] == D else C


# ── 5. GRIM ───────────────────────────────────────────────────
class Grim(Strategy):
    """Cooperate until first defection; defect forever after."""
    @property
    def name(self): return "GRIM"
    def action(self):
        return D if D in self.opp_history else C


# ── 6. FRIEDMAN ───────────────────────────────────────────────
class Friedman(Strategy):
    """
    Permanent punishment trigger — identical to GRIM in mechanics
    but named after James Friedman (1971) in the game theory literature.
    """
    @property
    def name(self): return "FRIEDMAN"
    def action(self):
        return D if D in self.opp_history else C


# ── 7. PAVLOV ─────────────────────────────────────────────────
class Pavlov(Strategy):
    """
    Win-Stay, Lose-Shift (Nowak & Sigmund, 1993).
    Repeat last action if previous outcome was R or T; switch otherwise.
    """
    @property
    def name(self): return "PAVLOV"
    def action(self):
        if not self.my_history:
            return C
        # Win = opponent cooperated last round
        return self.my_history[-1] if self.opp_history[-1] == C \
               else (C if self.my_history[-1] == D else D)


# ── 8. RANDOM ─────────────────────────────────────────────────
class RandomStrategy(Strategy):
    """50/50 random using the project's PCG64 RNG exclusively."""
    @property
    def name(self): return "RANDOM"
    def action(self):
        if self.rng is None:
            raise RuntimeError("RANDOM requires a ReproducibleRNG instance.")
        return C if self.rng.uniform() < 0.5 else D


# ── 9. JOSS ───────────────────────────────────────────────────
class Joss(Strategy):
    """
    TIT FOR TAT with a 10% backstab rate: even when opponent cooperated,
    defect with probability 0.10.
    """
    @property
    def name(self): return "JOSS"
    def action(self):
        if not self.opp_history:
            return C
        if self.opp_history[-1] == D:
            return D
        return D if (self.rng and self.rng.uniform() < 0.10) else C


# ── 10. GRADUAL ───────────────────────────────────────────────
class Gradual(Strategy):
    """
    Counts opponent's total defections (n_d). Responds with n_d consecutive
    defections followed by 2 cooperative rounds before forgiving.
    """
    @property
    def name(self): return "GRADUAL"

    def reset(self):
        super().reset()
        self._punish_left = 0
        self._calm_left   = 0
        self._n_def       = 0

    def action(self):
        if not self.opp_history:
            return C
        # New defection trigger
        if self.opp_history[-1] == D and self._punish_left == 0 and self._calm_left == 0:
            self._n_def      += 1
            self._punish_left = self._n_def
        if self._punish_left > 0:
            self._punish_left -= 1
            if self._punish_left == 0:
                self._calm_left = 2
            return D
        if self._calm_left > 0:
            self._calm_left -= 1
            return C
        return C


# ── 11. TESTER ────────────────────────────────────────────────
class Tester(Strategy):
    """
    Probes opponent on round 1 (defect). If opponent retaliates → TFT.
    If opponent forgives → exploit with alternating C/D.
    """
    @property
    def name(self): return "TESTER"

    def reset(self):
        super().reset()
        self._mode  = "probe"
        self._round = 0

    def action(self):
        self._round += 1
        if self._round == 1:
            return D
        if self._round == 2:
            self._mode = "tft" if self.opp_history[0] == D else "exploit"
        if self._mode == "tft":
            return self.opp_history[-1]
        # exploit: alternate C / D
        return C if self._round % 2 == 0 else D


# ── 12. ADAPTIVE (Bayesian Updating) ─────────────────────────
class Adaptive(Strategy):
    """
    Bayesian adaptive strategy.

    Maintains two Beta(α,β) posteriors:
        P(opp_C | my_C)  updated via conjugate Beta–Bernoulli model.
        P(opp_C | my_D)

    Selects the action with higher expected payoff under the
    current posterior mean estimates.
    """
    @property
    def name(self): return "ADAPTIVE"

    def reset(self):
        super().reset()
        # Beta(1,1) = uniform prior
        self._ac, self._bc = 1.0, 1.0   # opp cooperated | I cooperated
        self._ad, self._bd = 1.0, 1.0   # opp cooperated | I defected

    def _bayesian_update(self):
        """Update posteriors with most recent observation."""
        if not self.my_history:
            return
        my_a, opp_a = self.my_history[-1], self.opp_history[-1]
        if my_a == C:
            if opp_a == C: self._ac += 1
            else:          self._bc += 1
        else:
            if opp_a == C: self._ad += 1
            else:          self._bd += 1

    def action(self) -> str:
        self._bayesian_update()
        p_c_given_C = self._ac / (self._ac + self._bc)   # posterior mean
        p_c_given_D = self._ad / (self._ad + self._bd)

        # Payoffs: R=3, S=0, T=5, P=1 (hardcoded defaults)
        ev_cooperate  = p_c_given_C * 3 + (1 - p_c_given_C) * 0
        ev_defect     = p_c_given_D * 5 + (1 - p_c_given_D) * 1
        return C if ev_cooperate >= ev_defect else D


# ── 13. EVOLVED-NN (NumPy-only 2-layer MLP) ──────────────────
class EvolvedNN(Strategy):
    """
    Two-layer feedforward neural network (NumPy only, no frameworks).

    Architecture:
        Input  : 6 units  — last 3 self-actions + last 3 opponent-actions (binary)
        Hidden : 8 units  — tanh activation
        Output : 1 unit   — sigmoid → C if > 0.5

    Weights are fixed at initialization (seed=1337) to represent a
    pre-evolved cooperative agent consistent with Axelrod tournament results.
    """
    @property
    def name(self): return "EVOLVED-NN"

    def reset(self):
        super().reset()
        _rng = Generator(PCG64(1337))
        # He initialization
        self._W1 = _rng.standard_normal((8, 6)) * np.sqrt(2.0 / 6)
        self._b1 = np.zeros(8)
        self._W2 = _rng.standard_normal((1, 8)) * np.sqrt(2.0 / 8)
        self._b2 = np.zeros(1)

    def _encode(self) -> np.ndarray:
        def _enc(hist: List[str], n: int = 3) -> List[float]:
            raw = [1.0 if a == C else 0.0 for a in hist[-n:]]
            return [0.0] * (n - len(raw)) + raw
        return np.array(_enc(self.my_history) + _enc(self.opp_history))

    def _forward(self, x: np.ndarray) -> float:
        h = np.tanh(self._W1 @ x + self._b1)
        return float(1.0 / (1.0 + np.exp(-(self._W2 @ h + self._b2)[0])))

    def action(self) -> str:
        return C if self._forward(self._encode()) > 0.5 else D


# ── 14. PSO-PLAYER ────────────────────────────────────────────
class PSOPlayer(Strategy):
    """
    Particle Swarm Optimization inspired strategy.

    Maintains a cooperation probability p as a 1-D particle.
    Velocity update: v = w·v + c1·r1·(p_best - p) + c2·r2·(social - p)
    where social target is 1.0 if opponent outscored self, else 0.0.
    """
    @property
    def name(self): return "PSO-PLAYER"

    def reset(self):
        super().reset()
        self._p      = 0.70
        self._p_best = 0.70
        self._v      = 0.0
        self._score  = 0.0
        self._opp_sc = 0.0

    def _update(self):
        w_in, c1, c2 = 0.5, 0.3, 0.3
        r1 = self.rng.uniform() if self.rng else 0.5
        r2 = self.rng.uniform() if self.rng else 0.5
        social = 1.0 if self._opp_sc > self._score else 0.0
        self._v = (w_in * self._v
                   + c1 * r1 * (self._p_best - self._p)
                   + c2 * r2 * (social - self._p))
        self._p = float(np.clip(self._p + self._v, 0.05, 0.95))
        if self._score >= self._p_best:
            self._p_best = self._p

    def action(self) -> str:
        self._update()
        r = self.rng.uniform() if self.rng else 0.5
        return C if r < self._p else D

    def record(self, my_a: str, opp_a: str) -> None:
        _lut = {(C,C):(3,3),(C,D):(0,5),(D,C):(5,0),(D,D):(1,1)}
        pa, pb = _lut[(my_a, opp_a)]
        self._score  = 0.9 * self._score  + 0.1 * pa
        self._opp_sc = 0.9 * self._opp_sc + 0.1 * pb
        super().record(my_a, opp_a)


# ── 15. MEMORY-3 ──────────────────────────────────────────────
class Memory3(Strategy):
    """
    Lookup table strategy with 3-round memory.

    State = 6 bits (last 3 self-actions + last 3 opponent-actions).
    64-entry lookup table generated once with a fixed seed (mostly cooperative).
    Cooperates unconditionally for the first 3 rounds.
    """
    @property
    def name(self): return "MEMORY-3"

    def reset(self):
        super().reset()
        _rng = Generator(PCG64(999))
        self._table = (_rng.uniform(size=64) > 0.35).astype(int)

    def _key(self) -> int:
        def _enc(h): return [1 if a == C else 0 for a in h[-3:]]
        bits = ([0]*3 + _enc(self.my_history))[-3:] + ([0]*3 + _enc(self.opp_history))[-3:]
        return int("".join(map(str, bits)), 2)

    def action(self) -> str:
        if len(self.my_history) < 3:
            return C
        return C if self._table[self._key()] else D


# Registry
STRATEGY_REGISTRY: Dict[str, type] = {
    "ALL-C":            AllC,
    "ALL-D":            AllD,
    "TIT FOR TAT":      TitForTat,
    "TIT FOR TWO TATS": TitForTwoTats,
    "GRIM":             Grim,
    "FRIEDMAN":         Friedman,
    "PAVLOV":           Pavlov,
    "RANDOM":           RandomStrategy,
    "JOSS":             Joss,
    "GRADUAL":          Gradual,
    "TESTER":           Tester,
    "ADAPTIVE":         Adaptive,
    "EVOLVED-NN":       EvolvedNN,
    "PSO-PLAYER":       PSOPlayer,
    "MEMORY-3":         Memory3,
}


# ═════════════════════════════════════════════════════════════════
#  4. SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════

def play_match(
    s_a:    Strategy,
    s_b:    Strategy,
    payoff: PayoffMatrix,
    rng:    ReproducibleRNG,
    w:      float = 0.995,
) -> Tuple[float, float, float, float, int]:
    """
    Play one match with stochastic stopping criterion.

    Each round continues with probability w; stops with prob (1-w).
    Expected duration ≈ 1/(1-w).

    Returns:
        (score_a, score_b, coop_rate_a, coop_rate_b, n_rounds)
    """
    s_a.reset(); s_b.reset()
    sc_a = sc_b = 0.0
    co_a = co_b = n = 0

    while True:
        a_i, a_j = s_a.action(), s_b.action()
        p_a, p_b = payoff.payoffs(a_i, a_j)
        sc_a += p_a; sc_b += p_b
        co_a += a_i == C; co_b += a_j == C
        n += 1
        s_a.record(a_i, a_j)
        s_b.record(a_j, a_i)
        if rng.uniform() > w:
            break

    return sc_a, sc_b, co_a / n, co_b / n, n


def run_tournament(
    strategies: List[Strategy],
    payoff:     PayoffMatrix,
    rng:        ReproducibleRNG,
    w:          float = 0.995,
    n_games:    int   = 5,
) -> Dict:
    """
    Round-robin tournament: every pair plays n_games matches (including self-play).

    Returns a results dict containing:
        names, scores, coop_rates, score_matrix, coop_matrix,
        avg_rounds, per_game_scores (for ANOVA).
    """
    n     = len(strategies)
    names = [s.name for s in strategies]

    score_matrix = np.zeros((n, n))
    coop_matrix  = np.zeros((n, n))
    total_scores: Dict[str, float]      = {nm: 0.0 for nm in names}
    total_coop:   Dict[str, float]      = {nm: 0.0 for nm in names}
    match_count:  Dict[str, int]        = {nm: 0   for nm in names}
    total_rounds: Dict[str, int]        = {nm: 0   for nm in names}
    per_game:     Dict[str, List[float]]= {nm: []  for nm in names}

    for i in range(n):
        for j in range(n):
            game_sc = []
            for _ in range(n_games):
                sa, sb, ca, cb, nr = play_match(
                    strategies[i], strategies[j], payoff, rng, w
                )
                game_sc.append(sa)
                score_matrix[i, j] += sa
                coop_matrix[i, j]  += ca
                total_scores[names[i]] += sa
                total_coop[names[i]]   += ca
                match_count[names[i]]  += 1
                total_rounds[names[i]] += nr
            per_game[names[i]].extend(game_sc)
        # Average over n_games for heatmap cells
        score_matrix[i] /= n_games
        coop_matrix[i]  /= n_games

    avg_coop   = {k: total_coop[k] / max(match_count[k], 1)   for k in names}
    avg_rounds = {k: total_rounds[k] / max(match_count[k], 1) for k in names}

    return dict(
        names=names, scores=total_scores,
        coop_rates=avg_coop, score_matrix=score_matrix,
        coop_matrix=coop_matrix, avg_rounds=avg_rounds,
        per_game_scores=per_game,
    )


# ═════════════════════════════════════════════════════════════════
#  5. STATISTICAL ANALYSIS — ANOVA
# ═════════════════════════════════════════════════════════════════

def compute_anova(per_game_scores: Dict[str, List[float]]) -> Dict:
    """
    One-way ANOVA across all strategies' per-game score distributions.

    Returns a dict with the full ANOVA table, descriptive stats DataFrame,
    F-statistic, p-value, and plain-language interpretation.
    """
    groups = list(per_game_scores.values())
    names  = list(per_game_scores.keys())
    F, p   = stats.f_oneway(*groups)

    all_data   = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    N, k       = len(all_data), len(groups)

    SS_b = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    SS_w = sum(np.sum((np.array(g) - np.mean(g))**2)  for g in groups)
    SS_t = SS_b + SS_w
    df_b, df_w = k - 1, N - k
    MS_b, MS_w = SS_b / df_b, SS_w / df_w

    anova_df = pd.DataFrame([
        {"Source": "Between groups", "SS": round(SS_b, 2), "df": df_b,
         "MS": round(MS_b, 2), "F": round(F, 4), "p-value": round(p, 6)},
        {"Source": "Within groups",  "SS": round(SS_w, 2), "df": df_w,
         "MS": round(MS_w, 2), "F": "—", "p-value": "—"},
        {"Source": "Total",          "SS": round(SS_t, 2), "df": N - 1,
         "MS": "—", "F": "—", "p-value": "—"},
    ])

    desc_rows = []
    for nm, g in zip(names, groups):
        a = np.array(g)
        desc_rows.append({
            "Strategy": nm, "N": len(a),
            "Mean": round(np.mean(a), 2),
            "Std":  round(np.std(a, ddof=1), 2) if len(a) > 1 else 0.0,
            "Min":  round(np.min(a), 2),
            "Max":  round(np.max(a), 2),
        })
    desc_df = pd.DataFrame(desc_rows).sort_values("Mean", ascending=False).reset_index(drop=True)
    desc_df.index += 1

    interp = (
        "✅ Statistically significant differences detected (p < 0.05). "
        "Strategy performance is not homogeneous."
        if p < 0.05 else
        "⚠️ No significant differences detected (p ≥ 0.05). "
        "Strategy scores may be statistically equivalent."
    )

    return dict(F=F, p=p, anova_df=anova_df, desc_df=desc_df, interpretation=interp)


# ═════════════════════════════════════════════════════════════════
#  6. RNG QUALITY TESTS
# ═════════════════════════════════════════════════════════════════

def rng_quality_tests(rng: ReproducibleRNG, n: int = 10_000) -> Dict:
    """
    Statistical quality assessment of the PCG64 RNG.

    Tests:
        1. Kolmogorov-Smirnov test for U[0,1] uniformity.
        2. Lag-1 scatter plot data for serial independence.
        3. Autocorrelation for lags 1-20.

    Returns a dict with sample, KS stats, and autocorrelations.
    """
    sample       = rng.draw_sample(n)
    ks_stat, ks_p = stats.kstest(sample, "uniform")
    lags         = np.arange(1, 21)
    autocorrs    = [float(np.corrcoef(sample[:-lag], sample[lag:])[0, 1]) for lag in lags]

    return dict(
        sample=sample, ks_stat=ks_stat, ks_p=ks_p,
        lags=lags, autocorrs=autocorrs, n=n,
        passed=ks_p > 0.05,
    )


# ═════════════════════════════════════════════════════════════════
#  7. PLOTLY FIGURES
# ═════════════════════════════════════════════════════════════════

_DARK = "plotly_dark"
_BG   = "rgba(0,0,0,0)"
_GRID = "rgba(255,255,255,0.04)"
_PAL  = [
    "#38bdf8","#34d399","#fbbf24","#f87171","#a78bfa",
    "#fb923c","#4ade80","#60a5fa","#f472b6","#94a3b8",
    "#2dd4bf","#facc15","#c084fc","#86efac","#fdba74",
]


def _base_layout(**kwargs) -> dict:
    return dict(
        template=_DARK, paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="JetBrains Mono, monospace", size=11, color="#94a3b8"),
        margin=dict(l=16, r=16, t=48, b=16),
        **kwargs,
    )


def fig_ranking(results: Dict) -> go.Figure:
    """Horizontal bar chart — ranked by total score."""
    names  = results["names"]
    scores = [results["scores"][n]         for n in names]
    coops  = [results["coop_rates"][n]*100 for n in names]
    order  = np.argsort(scores)[::-1]

    sn = [names[i]  for i in order]
    ss = [scores[i] for i in order]
    sc = [coops[i]  for i in order]
    cl = [_PAL[i % len(_PAL)] for i in range(len(order))]

    fig = go.Figure(go.Bar(
        y=sn, x=ss, orientation="h",
        marker=dict(color=cl, line=dict(color="rgba(255,255,255,0.06)", width=1)),
        text=[f" {s:.0f} pts · {c:.1f}% coop" for s, c in zip(ss, sc)],
        textposition="outside",
        textfont=dict(size=10, color="#64748b"),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Tournament Ranking", font=dict(size=13, color="#e2e8f0"), x=0),
            height=max(340, len(names) * 40 + 80),
            xaxis=dict(title="Cumulative Score", gridcolor=_GRID, zeroline=False),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            showlegend=False,
        )
    )
    return fig


def fig_heatmap(results: Dict) -> go.Figure:
    """Score heatmap (row strategy vs column opponent, averaged over games)."""
    names  = results["names"]
    matrix = results["score_matrix"]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names,
        colorscale=[[0,"#0f172a"],[0.35,"#0c4a6e"],[0.7,"#0ea5e9"],[1,"#bae6fd"]],
        text=np.round(matrix).astype(int),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Score: %{z:.1f}<extra></extra>",
        colorbar=dict(title="Score", thickness=12,
                      tickfont=dict(size=9), titlefont=dict(size=10)),
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Match Score Heatmap · avg / game", font=dict(size=13, color="#e2e8f0"), x=0),
            height=max(400, len(names) * 38 + 120),
            xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(l=16, r=20, t=48, b=100),
        )
    )
    return fig


def fig_cooperation(results: Dict) -> go.Figure:
    """Bar chart — cooperation rate per strategy."""
    names  = results["names"]
    coops  = [results["coop_rates"][n]*100 for n in names]
    avg    = float(np.mean(coops))
    order  = np.argsort(coops)[::-1]

    sn = [names[i]  for i in order]
    sc = [coops[i]  for i in order]

    fig = go.Figure(go.Bar(
        x=sn, y=sc,
        marker=dict(
            color=sc,
            colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
            showscale=True,
            colorbar=dict(title="%", thickness=12,
                          tickfont=dict(size=9), titlefont=dict(size=9)),
            line=dict(color="rgba(255,255,255,0.05)", width=1),
        ),
        text=[f"{v:.1f}%" for v in sc],
        textposition="outside", textfont=dict(size=9, color="#64748b"),
        hovertemplate="<b>%{x}</b><br>Cooperation: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=avg, line_dash="dash", line_color="#fbbf24", line_width=1,
                  annotation_text=f"μ = {avg:.1f}%",
                  annotation_font=dict(color="#fbbf24", size=10))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Cooperation Rate by Strategy", font=dict(size=13, color="#e2e8f0"), x=0),
            height=360,
            xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
            yaxis=dict(title="% Cooperation", range=[0, 115], gridcolor=_GRID),
            showlegend=False,
        )
    )
    return fig


def fig_anova_table(anova: Dict) -> go.Figure:
    """ANOVA table as Plotly Table."""
    df = anova["anova_df"]
    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df.columns],
            fill_color="rgba(56,189,248,0.12)",
            font=dict(color="#38bdf8", size=11),
            align="center", height=32,
            line_color="rgba(56,189,248,0.2)",
        ),
        cells=dict(
            values=[df[c].tolist() for c in df.columns],
            fill_color="rgba(14,20,45,0.8)",
            font=dict(color="#e2e8f0", size=11),
            align=["left","center","center","center","center","center"],
            height=28,
            line_color="rgba(255,255,255,0.04)",
        )
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="One-Way ANOVA Table", font=dict(size=13, color="#e2e8f0"), x=0),
            height=210, margin=dict(l=0, r=0, t=44, b=0),
        )
    )
    return fig


def fig_ks_test(qc: Dict) -> go.Figure:
    """KS test: histogram + ECDF vs theoretical uniform CDF."""
    sample = qc["sample"]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Density Histogram vs U[0,1]", "ECDF vs Theoretical CDF"],
        horizontal_spacing=0.08,
    )
    # Histogram
    fig.add_trace(go.Histogram(
        x=sample, nbinsx=60, histnorm="probability density",
        marker_color="rgba(56,189,248,0.55)", name="PCG64 samples",
        showlegend=True,
    ), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#fbbf24", line_width=1,
                  row=1, col=1)
    # ECDF
    sx   = np.sort(sample)
    ecdf = np.arange(1, len(sx)+1) / len(sx)
    fig.add_trace(go.Scatter(x=sx, y=ecdf, mode="lines",
                             line=dict(color="#38bdf8", width=1.5), name="Empirical CDF"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(color="#fbbf24", dash="dash", width=1), name="U[0,1] CDF"), row=1, col=2)
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Kolmogorov–Smirnov Test · D={qc['ks_stat']:.5f} · "
                     f"p={qc['ks_p']:.6f} · n={qc['n']:,}",
                font=dict(size=12, color="#e2e8f0"), x=0,
            ),
            height=350, legend=dict(font=dict(size=10)),
        )
    )
    fig.update_xaxes(gridcolor=_GRID); fig.update_yaxes(gridcolor=_GRID)
    return fig


def fig_lag_plot(qc: Dict) -> go.Figure:
    """Lag plot + autocorrelation bar chart."""
    sample = qc["sample"][:5000]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Lag-1 Scatter (x[i] vs x[i+1])", "Autocorrelation · Lags 1–20"],
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Scattergl(
        x=sample[:-1], y=sample[1:], mode="markers",
        marker=dict(size=2, color="rgba(56,189,248,0.25)"),
        name="Lag-1",
    ), row=1, col=1)
    ac  = qc["autocorrs"]
    clr = ["#34d399" if abs(v) < 0.02 else "#f87171" for v in ac]
    fig.add_trace(go.Bar(
        x=qc["lags"], y=ac,
        marker_color=clr, name="Autocorr",
    ), row=1, col=2)
    for lv in [0.02, -0.02]:
        fig.add_hline(y=lv, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=2)
    fig.update_layout(
        **_base_layout(
            title=dict(text="Serial Independence Analysis", font=dict(size=12, color="#e2e8f0"), x=0),
            height=350, showlegend=False,
        )
    )
    fig.update_xaxes(gridcolor=_GRID); fig.update_yaxes(gridcolor=_GRID)
    return fig


# ═════════════════════════════════════════════════════════════════
#  8. STREAMLIT UI
# ═════════════════════════════════════════════════════════════════

def sidebar_controls() -> Dict:
    """Render sidebar and return configuration dict."""
    with st.sidebar:
        st.markdown("""
        <div style="padding-bottom:1rem;border-bottom:1px solid rgba(56,189,248,0.12);margin-bottom:1rem;">
          <p style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#38bdf8;letter-spacing:0.1em;margin:0;text-transform:uppercase;">
            IPD · Config
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">Payoff Matrix</p>', unsafe_allow_html=True)
        T = st.number_input("T — Temptation",  value=5.0, step=0.5, format="%.1f")
        R = st.number_input("R — Reward",       value=3.0, step=0.5, format="%.1f")
        P = st.number_input("P — Punishment",   value=1.0, step=0.5, format="%.1f")
        S = st.number_input("S — Sucker",       value=0.0, step=0.5, format="%.1f")

        st.markdown('<p class="section-header">Simulation</p>', unsafe_allow_html=True)
        w       = st.slider("w  (continuation prob)", 0.90, 0.999, 0.995, 0.001, format="%.3f")
        n_games = st.slider("Games per pair",          1, 20, 5)
        seed    = st.number_input("RNG Seed", value=20260211, step=1)

        st.markdown('<p class="section-header">Strategies</p>', unsafe_allow_html=True)
        all_names = list(STRATEGY_REGISTRY.keys())
        selected  = st.multiselect(
            "Select strategies",
            options=all_names,
            default=all_names,
            label_visibility="collapsed",
        )

        st.markdown('<p class="section-header">RNG Tests</p>', unsafe_allow_html=True)
        rng_n = st.slider("Sample size (KS test)", 1000, 100_000, 10_000, 1000)

        st.markdown("---")
        run_btn = st.button("▶  Run Tournament", use_container_width=True)
        rng_btn = st.button("🔬  Run RNG Tests",  use_container_width=True)

    return dict(T=T, R=R, P=P, S=S, w=w, n_games=n_games, seed=int(seed),
                selected=selected, rng_n=rng_n, run=run_btn, rng_test=rng_btn)


def render_header():
    st.markdown("""
    <div class="app-header">
      <p class="app-title">🧬 IPD Research Dashboard</p>
      <p class="app-subtitle">
        Axelrod &amp; Hamilton (1981) · Iterated Prisoner's Dilemma · Option 1 · 15 Strategies
        &nbsp;·&nbsp;
        <span class="badge">PCG64</span>
        <span class="badge">w = 0.995</span>
        <span class="badge">ANOVA</span>
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(results: Dict, cfg: Dict):
    names   = results["names"]
    scores  = results["scores"]
    winner  = max(scores, key=scores.get)
    avg_c   = np.mean(list(results["coop_rates"].values())) * 100
    exp_dur = round(1 / (1 - cfg["w"]), 1)
    n_strat = len(names)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Winner",        winner)
    c2.metric("⚡ Top Score",      f"{scores[winner]:.0f}")
    c3.metric("🤝 Avg Cooperation", f"{avg_c:.1f}%")
    c4.metric("⏱ E[Rounds / game]", f"~{exp_dur:.0f}")


def tab_simulation(cfg: Dict):
    """Tab 1: Simulation results."""
    if "results" not in st.session_state:
        st.info("Configure parameters in the sidebar and press **▶ Run Tournament**.")
        return

    results = st.session_state["results"]
    render_metrics(results, st.session_state["last_cfg"])

    st.markdown("---")
    st.plotly_chart(fig_ranking(results),     use_container_width=True)
    st.plotly_chart(fig_heatmap(results),     use_container_width=True)
    st.plotly_chart(fig_cooperation(results), use_container_width=True)

    # ── Download CSV ─────────────────────────────────────────
    names  = results["names"]
    df_dl  = pd.DataFrame({
        "Strategy":         names,
        "Total Score":      [results["scores"][n]         for n in names],
        "Cooperation Rate": [round(results["coop_rates"][n]*100, 2) for n in names],
        "Avg Rounds/Match": [round(results["avg_rounds"][n], 1)     for n in names],
    }).sort_values("Total Score", ascending=False).reset_index(drop=True)
    df_dl.index += 1

    csv_buf = io.StringIO()
    df_dl.to_csv(csv_buf)
    st.download_button(
        label="⬇  Download Ranking (CSV)",
        data=csv_buf.getvalue(),
        file_name="ipd_tournament_ranking.csv",
        mime="text/csv",
    )


def tab_statistics():
    """Tab 2: ANOVA + descriptive statistics."""
    if "anova" not in st.session_state:
        st.info("Run the tournament first (Tab 1).")
        return

    anova = st.session_state["anova"]
    cfg   = st.session_state["last_cfg"]

    # ANOVA summary line
    p_col = "#34d399" if anova["p"] < 0.05 else "#fbbf24"
    sig   = "Significant ✅" if anova["p"] < 0.05 else "Not significant ⚠️"
    st.markdown(f"""
    <div style="background:rgba(14,20,45,0.6);border:1px solid rgba(56,189,248,0.15);
                border-radius:8px;padding:14px 18px;margin-bottom:1rem;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;
                   text-transform:uppercase;letter-spacing:0.1em;">One-Way ANOVA</span><br>
      <span style="font-size:13px;color:#e2e8f0;">
        F&thinsp;=&thinsp;<b>{anova['F']:.4f}</b>&emsp;
        p&thinsp;=&thinsp;<b>{anova['p']:.6f}</b>&emsp;
        <span style="color:{p_col};">{sig}</span>
      </span><br>
      <span style="font-size:12px;color:#64748b;">{anova['interpretation']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig_anova_table(anova), use_container_width=True)

    st.markdown("#### Descriptive Statistics by Strategy")
    st.dataframe(
        anova["desc_df"].style
        .background_gradient(subset=["Mean"], cmap="Blues")
        .format({"Mean": "{:.2f}", "Std": "{:.2f}", "Min": "{:.2f}", "Max": "{:.2f}"}),
        use_container_width=True,
        height=min(600, len(anova["desc_df"]) * 38 + 60),
    )


def tab_rng_annex():
    """Tab 3: Technical RNG annex."""
    if "rng_qc" not in st.session_state:
        st.info("Press **🔬 Run RNG Tests** in the sidebar.")
        return

    qc = st.session_state["rng_qc"]

    # KS result banner
    passed = qc["passed"]
    bc = "rgba(52,211,153,0.1)" if passed else "rgba(248,113,113,0.1)"
    bc2 = "#34d399" if passed else "#f87171"
    verdict = "H₀ not rejected — distribution is uniform" if passed else \
              "H₀ rejected — uniformity concern"
    st.markdown(f"""
    <div style="background:{bc};border:1px solid {bc2}33;border-radius:8px;
                padding:12px 18px;margin-bottom:1rem;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                   color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">
        KS Test Result
      </span><br>
      <span style="font-size:13px;color:#e2e8f0;">
        D&thinsp;=&thinsp;<b>{qc['ks_stat']:.6f}</b>&emsp;
        p&thinsp;=&thinsp;<b>{qc['ks_p']:.6f}</b>&emsp;
        <span style="color:{bc2};">{verdict}</span>
      </span><br>
      <span style="font-size:12px;color:#64748b;">
        n = {qc['n']:,} samples · Generator: PCG64 · Threshold: p &gt; 0.05
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig_ks_test(qc),  use_container_width=True)
    st.plotly_chart(fig_lag_plot(qc), use_container_width=True)

    # Autocorrelation table
    ac_df = pd.DataFrame({
        "Lag":           qc["lags"],
        "Autocorrelation": [round(v, 6) for v in qc["autocorrs"]],
        "Status":        ["✅ OK" if abs(v) < 0.02 else "⚠️ Inspect" for v in qc["autocorrs"]],
    })
    st.markdown("#### Autocorrelation Summary · Lags 1–20")
    st.dataframe(ac_df, use_container_width=True, height=280, hide_index=True)


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  CASE STUDY — RUSSIA–EU ENERGY CRISIS (2021–2026)
#  ─────────────────────────────────────────────────────────────────────────
#  RESTRICTION: This module does NOT modify any tournament logic, strategy
#  classes, or payoff matrix classes defined above.  It reads the three
#  repository CSV/XLSX data files, applies Prisoner's Dilemma framing, and
#  renders a dedicated Streamlit tab.
# ═══════════════════════════════════════════════════════════════════════════

import os as _os

# ── Economic translation of PD payoffs ─────────────────────────────────────

ECONOMIC_OUTCOMES = {
    ("C", "C"): {
        "label":     "Mutual Cooperation — Stable supply",
        "gdp_ru":    +2.0,   # % GDP change
        "gdp_eu":    +2.0,
        "inf_eu":    +2.5,   # % inflation
        "gas_price": "Normal market",
        "color":     "#34d399",
        "score_ru":  3,
        "score_eu":  3,
    },
    ("D", "C"): {
        "label":     "Russia defects — High-price extraction",
        "gdp_ru":    +4.5,
        "gdp_eu":    -4.0,
        "inf_eu":    +15.0,
        "gas_price": "Price spike >€200/MWh",
        "color":     "#f87171",
        "score_ru":  5,
        "score_eu":  0,
    },
    ("C", "D"): {
        "label":     "EU sanctions / LNG pivot",
        "gdp_ru":    -3.5,
        "gdp_eu":    +0.5,
        "inf_eu":    +6.0,
        "gas_price": "Spot market premium",
        "color":     "#fbbf24",
        "score_ru":  0,
        "score_eu":  5,
    },
    ("D", "D"): {
        "label":     "Mutual punishment — Energy crisis",
        "gdp_ru":    -2.5,
        "gdp_eu":    -2.5,
        "inf_eu":    +8.0,
        "gas_price": "Market disruption",
        "color":     "#a78bfa",
        "score_ru":  1,
        "score_eu":  1,
    },
}

_FLOW_THRESHOLD   = 300   # mcm/day — below = Defect for Russia
_STORAGE_DROP_PCT = 5.0   # % weekly drop — above = Defect for EU


# ── Data loading (cached) ───────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading energy data…")
def load_daily_data() -> pd.DataFrame:
    """
    Load daily gas flow data from 'daily_data_*.csv' in the repo root.
    Returns a DataFrame with columns: Date, Russia (mcm/day), and derived
    move_russia ('C' / 'D').
    Falls back to synthetic demo data if the file is absent.
    """
    candidates = sorted(
        [f for f in _os.listdir(".") if f.startswith("daily_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        df = pd.read_csv(candidates[0], parse_dates=True)
        # Normalise column names: keep first date-like col as Date
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        # Identify Russia flow column (case-insensitive)
        russia_col = next(
            (c for c in df.columns if "russia" in c.lower() or "ru" == c.lower()),
            None,
        )
        if russia_col and russia_col != "Russia":
            df = df.rename(columns={russia_col: "Russia"})
        if "Russia" not in df.columns:
            # Use first numeric column as proxy
            num_cols = df.select_dtypes("number").columns.tolist()
            if num_cols:
                df["Russia"] = df[num_cols[0]]
    else:
        # ── Synthetic fallback (reproducible) ──────────────────────
        _rng_demo = Generator(PCG64(2026))
        dates = pd.date_range("2021-01-01", "2026-03-12", freq="D")
        base  = 350 + _rng_demo.normal(0, 80, len(dates)).cumsum() * 0.05
        # Inject geopolitical shocks
        base[365:730]  -= 60   # 2022 supply reduction
        base[730:910]  -= 150  # 2022 war shock
        base[910:1100] += 30   # partial recovery
        base[1100:]    -= 100  # 2023-24 further cuts
        df = pd.DataFrame({"Date": dates, "Russia": np.clip(base, 0, 600)})

    df["move_russia"] = df["Russia"].apply(
        lambda x: "C" if (pd.notna(x) and x > _FLOW_THRESHOLD) else "D"
    )
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading storage data…")
def load_storage_data() -> pd.DataFrame:
    """
    Load weekly EU + UA storage levels from the XLSX file.
    Returns DataFrame with: Week (date), Storage_pct, move_eu ('C'/'D').
    Falls back to synthetic data if file is absent.
    """
    candidates = sorted(
        [f for f in _os.listdir(".") if "storage" in f.lower() and f.endswith(".xlsx")],
        reverse=True,
    )
    if candidates:
        xl = pd.read_excel(candidates[0])
        date_col = next((c for c in xl.columns if "date" in c.lower() or "week" in c.lower()), xl.columns[0])
        xl = xl.rename(columns={date_col: "Week"})
        xl["Week"] = pd.to_datetime(xl["Week"], errors="coerce")
        xl = xl.dropna(subset=["Week"]).sort_values("Week")
        # Find storage percentage column
        stor_col = next(
            (c for c in xl.columns
             if any(k in c.lower() for k in ["storage", "stor", "filling", "pct", "%"])),
            None,
        )
        if stor_col and stor_col != "Storage_pct":
            xl = xl.rename(columns={stor_col: "Storage_pct"})
        if "Storage_pct" not in xl.columns:
            num_cols = xl.select_dtypes("number").columns.tolist()
            if num_cols:
                xl["Storage_pct"] = xl[num_cols[0]]
    else:
        _rng_dem2 = Generator(PCG64(2027))
        weeks = pd.date_range("2021-01-04", "2026-03-10", freq="W-MON")
        pct   = 60 + _rng_dem2.normal(0, 3, len(weeks)).cumsum() * 0.4
        pct   = np.clip(pct, 5, 100)
        xl    = pd.DataFrame({"Week": weeks, "Storage_pct": pct})

    # EU move: D if storage dropped > 5% vs previous week
    xl = xl.reset_index(drop=True)
    xl["Storage_change_pct"] = xl["Storage_pct"].pct_change() * 100
    xl["move_eu"] = xl["Storage_change_pct"].apply(
        lambda x: "D" if (pd.notna(x) and x < -_STORAGE_DROP_PCT) else "C"
    )
    return xl


@st.cache_data(show_spinner="Loading route data…")
def load_route_data() -> pd.DataFrame:
    """
    Load pipeline route utilisation data from 'route_data_*.csv'.
    Used for supplementary pipeline capacity chart.
    """
    candidates = sorted(
        [f for f in _os.listdir(".") if f.startswith("route_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        df = pd.read_csv(candidates[0], parse_dates=True)
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    # Synthetic fallback
    _rng_r = Generator(PCG64(2028))
    dates  = pd.date_range("2021-01-01", "2026-03-12", freq="D")
    routes = ["Nord Stream 1", "Yamal-Europe", "Ukrainian GTS", "TurkStream"]
    rows   = []
    for r in routes:
        base = _rng_r.uniform(40, 90) + _rng_r.normal(0, 5, len(dates)).cumsum() * 0.03
        base = np.clip(base, 0, 100)
        for d, v in zip(dates, base):
            rows.append({"Date": d, "Route": r, "Utilisation_pct": round(v, 1)})
    return pd.DataFrame(rows)


# ── Simulation: merge daily moves into scored timeline ──────────────────────

def build_timeline(daily: pd.DataFrame, storage: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily Russia moves with weekly EU storage moves, assign PD outcomes,
    and compute cumulative scores for both players.

    Returns a daily-resolution DataFrame ready for charting.
    """
    # Forward-fill weekly EU storage move to daily frequency
    stor_daily = storage[["Week", "move_eu", "Storage_pct", "Storage_change_pct"]].copy()
    stor_daily = stor_daily.rename(columns={"Week": "Date"})
    # Merge with tolerance: match each day to the most recent week
    df = pd.merge_asof(
        daily.sort_values("Date"),
        stor_daily.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    df["move_eu"] = df["move_eu"].fillna("C")

    # Assign PD outcome per day
    outcomes, scores_ru, scores_eu = [], [], []
    cum_ru = cum_eu = 0.0
    cum_scores_ru, cum_scores_eu = [], []
    gdp_ru_list, gdp_eu_list, inf_eu_list = [], [], []

    for _, row in df.iterrows():
        key = (row["move_russia"], row["move_eu"])
        out = ECONOMIC_OUTCOMES.get(key, ECONOMIC_OUTCOMES[("C","C")])
        outcomes.append(out["label"])
        scores_ru.append(out["score_ru"])
        scores_eu.append(out["score_eu"])
        cum_ru += out["score_ru"]
        cum_eu += out["score_eu"]
        cum_scores_ru.append(cum_ru)
        cum_scores_eu.append(cum_eu)
        gdp_ru_list.append(out["gdp_ru"])
        gdp_eu_list.append(out["gdp_eu"])
        inf_eu_list.append(out["inf_eu"])

    df["outcome"]       = outcomes
    df["score_ru"]      = scores_ru
    df["score_eu"]      = scores_eu
    df["cum_score_ru"]  = cum_scores_ru
    df["cum_score_eu"]  = cum_scores_eu
    df["gdp_impact_ru"] = gdp_ru_list
    df["gdp_impact_eu"] = gdp_eu_list
    df["inf_impact_eu"] = inf_eu_list
    return df


# ── Plotly figures for energy tab ──────────────────────────────────────────

_EC_BG    = "#080b12"
_EC_SRF   = "#0e1420"
_EC_GRID  = "rgba(255,255,255,0.04)"
_EC_TEXT  = "#e2e8f0"
_EC_MUTED = "#64748b"
_EC_RU    = "#f87171"   # Russia — warm red
_EC_EU    = "#38bdf8"   # EU — blue
_EC_GAS   = "#fbbf24"   # gas flow — amber
_EC_STOR  = "#34d399"   # storage — green

_EC_LAYOUT = dict(
    paper_bgcolor=_EC_BG,
    plot_bgcolor=_EC_SRF,
    font=dict(family="JetBrains Mono, monospace", color=_EC_TEXT, size=10),
    legend=dict(bgcolor="rgba(0,0,0,0.35)", borderwidth=0,
                font=dict(size=9, color=_EC_MUTED)),
    xaxis=dict(gridcolor=_EC_GRID, tickfont=dict(size=9, color=_EC_MUTED),
               showspikes=True, spikecolor=_EC_MUTED, spikethickness=1),
    yaxis=dict(gridcolor=_EC_GRID, tickfont=dict(size=9, color=_EC_MUTED)),
    margin=dict(l=16, r=16, t=52, b=16),
    hovermode="x unified",
)


def fig_gas_vs_scores(df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis chart: Gas flow (mcm/day) + Cumulative PD scores for Russia and EU.
    Annotates key geopolitical events on the timeline.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.06,
        subplot_titles=[
            "GAS FLOW vs CUMULATIVE PRISONER'S DILEMMA SCORES",
            "DAILY MOVES: RUSSIA (R) · EU (E)",
        ],
    )

    # ── Row 1: Gas flow (bar) + cumulative scores (lines) ──────
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Russia"],
        name="Gas Flow (mcm/day)",
        marker=dict(color=_EC_GAS, opacity=0.45, line=dict(width=0)),
        hovertemplate="%{y:.0f} mcm<extra>Gas Flow</extra>",
        yaxis="y1",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_score_ru"],
        name="Russia cumulative score",
        line=dict(color=_EC_RU, width=1.8),
        hovertemplate="%{y:,.0f}<extra>Russia score</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_score_eu"],
        name="EU cumulative score",
        line=dict(color=_EC_EU, width=1.8),
        hovertemplate="%{y:,.0f}<extra>EU score</extra>",
    ), row=1, col=1)

    # Threshold line at 300 mcm
    fig.add_hline(
        y=_FLOW_THRESHOLD, line_dash="dot", line_color="rgba(251,191,36,0.4)",
        line_width=1, row=1, col=1,
        annotation_text=f"Cooperation threshold {_FLOW_THRESHOLD} mcm",
        annotation_font=dict(size=8, color=_EC_MUTED),
        annotation_position="top left",
    )

    # ── Row 2: Move encoding as coloured scatter ────────────────
    move_colors_ru = [_EC_EU if m == "C" else _EC_RU for m in df["move_russia"]]
    move_colors_eu = [_EC_EU if m == "C" else _EC_RU for m in df["move_eu"]]

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=[1.2] * len(df),
        mode="markers",
        marker=dict(color=move_colors_ru, size=3, symbol="square"),
        name="Russia move (C=blue / D=red)",
        hovertemplate="Russia: %{customdata}<extra></extra>",
        customdata=df["move_russia"],
        showlegend=True,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=[0.5] * len(df),
        mode="markers",
        marker=dict(color=move_colors_eu, size=3, symbol="square"),
        name="EU move (C=blue / D=red)",
        hovertemplate="EU: %{customdata}<extra>Outcome: </extra>",
        customdata=df["move_eu"],
        showlegend=True,
    ), row=2, col=1)

    # ── Key event annotations ───────────────────────────────────
    events = [
        ("2021-10-01", "Gas price spike"),
        ("2022-02-24", "Invasion begins"),
        ("2022-06-15", "NS1 cuts -60%"),
        ("2022-09-26", "NS1 sabotage"),
        ("2023-01-01", "UA transit uncertain"),
        ("2024-01-01", "UA contract expires"),
    ]
    for date_str, label in events:
        try:
            ev_date = pd.Timestamp(date_str)
            if df["Date"].min() <= ev_date <= df["Date"].max():
                fig.add_vline(
                    x=ev_date.timestamp() * 1000,
                    line_dash="dot", line_color="rgba(167,139,250,0.35)",
                    line_width=1, row="all", col=1,
                )
                fig.add_annotation(
                    x=ev_date, y=1,
                    xref="x", yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(size=8, color="rgba(167,139,250,0.7)"),
                    textangle=-90,
                    xanchor="left",
                )
        except Exception:
            pass

    fig.update_layout(
        **_EC_LAYOUT,
        height=560,
        title_text="RUSSIA vs EU — GAS FLOW & PRISONER'S DILEMMA TIMELINE",
        title_font=dict(size=12),
    )
    fig.update_yaxes(title_text="mcm/day  |  PD Score", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Move", row=2, col=1, tickvals=[0.5, 1.2],
                     ticktext=["EU", "Russia"], range=[0, 1.7])
    fig.update_annotations(font=dict(size=8, color=_EC_MUTED))
    return fig


def fig_storage_and_eu_move(df_stor: pd.DataFrame) -> go.Figure:
    """
    Weekly EU storage level (%) with EU move encoded as fill colour.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_stor["Week"],
        y=df_stor["Storage_pct"],
        mode="lines",
        line=dict(color=_EC_STOR, width=2),
        fill="tozeroy",
        fillcolor="rgba(52,211,153,0.08)",
        name="EU Storage (%)",
        hovertemplate="Week %{x|%Y-%m-%d}<br>Storage: %{y:.1f}%<extra></extra>",
    ))
    # Scatter overlay coloured by EU move
    move_colors = [_EC_EU if m == "C" else _EC_RU for m in df_stor["move_eu"]]
    fig.add_trace(go.Scatter(
        x=df_stor["Week"],
        y=df_stor["Storage_pct"],
        mode="markers",
        marker=dict(color=move_colors, size=6, opacity=0.8),
        name="EU move (C=blue / D=red)",
        hovertemplate=(
            "Week %{x|%Y-%m-%d}<br>"
            "Storage: %{y:.1f}%<br>"
            "Change: %{customdata:.1f}%<extra></extra>"
        ),
        customdata=df_stor["Storage_change_pct"].round(2),
    ))
    fig.add_hline(
        y=df_stor["Storage_pct"].mean(), line_dash="dot",
        line_color="rgba(251,191,36,0.4)", line_width=1,
        annotation_text=f"Mean {df_stor['Storage_pct'].mean():.1f}%",
        annotation_font=dict(size=8, color=_EC_MUTED),
    )
    fig.update_layout(
        **_EC_LAYOUT,
        height=320,
        title_text="EU + UA WEEKLY STORAGE LEVEL (%) — EU MOVE ENCODING",
        title_font=dict(size=12),
    )
    return fig


def fig_route_utilisation(df_route: pd.DataFrame) -> go.Figure:
    """
    Stacked-area chart of pipeline route utilisation percentages.
    """
    if "Route" not in df_route.columns:
        return go.Figure()

    fig = go.Figure()
    route_colors = {
        "Nord Stream 1":  _EC_RU,
        "Yamal-Europe":   _EC_GAS,
        "Ukrainian GTS":  _EC_EU,
        "TurkStream":     "#a78bfa",
    }
    for route in df_route["Route"].unique():
        sub = df_route[df_route["Route"] == route].sort_values("Date")
        col = route_colors.get(route, "#94a3b8")
        fig.add_trace(go.Scatter(
            x=sub["Date"],
            y=sub["Utilisation_pct"],
            name=route,
            mode="lines",
            line=dict(color=col, width=1.5),
            fill="tonexty" if route != df_route["Route"].unique()[0] else "tozeroy",
            fillcolor=col.replace(")", ",0.12)").replace("rgb", "rgba")
                       if col.startswith("rgb") else col + "20",
            hovertemplate=f"{route}<br>%{{y:.1f}}% utilisation<extra></extra>",
        ))
    fig.update_layout(
        **_EC_LAYOUT,
        height=320,
        title_text="PIPELINE ROUTE UTILISATION (%) — KEY CORRIDORS",
        title_font=dict(size=12),
    )
    return fig


def fig_economic_impact(df: pd.DataFrame) -> go.Figure:
    """
    30-day rolling mean of GDP impact and EU inflation impact derived from daily PD outcomes.
    """
    df = df.copy()
    df["gdp_eu_30d"]  = df["gdp_impact_eu"].rolling(30, min_periods=1).mean()
    df["gdp_ru_30d"]  = df["gdp_impact_ru"].rolling(30, min_periods=1).mean()
    df["inf_eu_30d"]  = df["inf_impact_eu"].rolling(30, min_periods=1).mean()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["GDP IMPACT (30d rolling avg, %)", "EU INFLATION IMPACT (30d rolling avg, %)"],
    )
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["gdp_eu_30d"],
        line=dict(color=_EC_EU, width=1.5), name="EU GDP",
        hovertemplate="EU GDP: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["gdp_ru_30d"],
        line=dict(color=_EC_RU, width=1.5), name="Russia GDP",
        hovertemplate="Russia GDP: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                  line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["inf_eu_30d"],
        line=dict(color=_EC_GAS, width=1.5), name="EU Inflation",
        hovertemplate="EU Inflation: %{y:.1f}%<extra></extra>",
    ), row=2 if False else 1, col=2)
    fig.add_hline(y=2.0, line_dash="dot", line_color="rgba(52,211,153,0.3)",
                  line_width=1, row=1, col=2,
                  annotation_text="ECB 2% target",
                  annotation_font=dict(size=8, color=_EC_MUTED))

    fig.update_layout(
        **_EC_LAYOUT,
        height=340,
        title_text="ESTIMATED ECONOMIC IMPACT FROM DAILY PD OUTCOMES",
        title_font=dict(size=12),
        showlegend=True,
    )
    fig.update_xaxes(gridcolor=_EC_GRID, tickfont=dict(size=9, color=_EC_MUTED))
    fig.update_yaxes(gridcolor=_EC_GRID, tickfont=dict(size=9, color=_EC_MUTED))
    return fig


def fig_outcome_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie + bar breakdown of daily PD outcome frequency."""
    counts = df["outcome"].value_counts()
    colors = []
    for lbl in counts.index:
        for v in ECONOMIC_OUTCOMES.values():
            if v["label"] == lbl:
                colors.append(v["color"])
                break
        else:
            colors.append("#64748b")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"pie"}, {"type":"bar"}]],
        subplot_titles=["OUTCOME DISTRIBUTION", "DAYS BY OUTCOME"],
    )
    fig.add_trace(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors, line=dict(color=_EC_BG, width=2)),
        textfont=dict(size=9, color=_EC_TEXT),
        hole=0.42,
        hovertemplate="%{label}<br>%{value} days (%{percent})<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=counts.index,
        y=counts.values,
        marker=dict(color=colors, opacity=0.85),
        hovertemplate="%{x}<br>%{y} days<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(
        **_EC_LAYOUT,
        height=340,
        title_text="PRISONER'S DILEMMA OUTCOME FREQUENCY — HISTORICAL DATA",
        title_font=dict(size=12),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-18, tickfont=dict(size=8), row=1, col=2)
    return fig


# ── Summary statistics ──────────────────────────────────────────────────────

def compute_case_summary(df: pd.DataFrame, df_stor: pd.DataFrame) -> Dict:
    """Compute headline statistics for the case-study metrics row."""
    coop_russia = (df["move_russia"] == "C").mean() * 100
    coop_eu     = (df["move_eu"]     == "C").mean() * 100
    final_sc_ru = df["cum_score_ru"].iloc[-1]
    final_sc_eu = df["cum_score_eu"].iloc[-1]
    most_common = df["outcome"].value_counts().idxmax()
    total_days  = len(df)
    cc_days     = (df["move_russia"]=="C") & (df["move_eu"]=="C")
    dd_days     = (df["move_russia"]=="D") & (df["move_eu"]=="D")
    avg_stor    = df_stor["Storage_pct"].mean()
    return dict(
        coop_russia=coop_russia,
        coop_eu=coop_eu,
        final_sc_ru=final_sc_ru,
        final_sc_eu=final_sc_eu,
        most_common=most_common,
        total_days=total_days,
        cc_days=int(cc_days.sum()),
        dd_days=int(dd_days.sum()),
        avg_stor=avg_stor,
    )


# ── Main tab renderer ───────────────────────────────────────────────────────

def tab_energy_crisis() -> None:
    """
    Streamlit tab: Russia–EU Energy Crisis (2021–2026) as an IPD case study.
    Reads data files from the repo root, applies PD framing, and renders
    interactive Plotly charts.  Does NOT modify any tournament logic.
    """
    # ── Header ─────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(248,113,113,0.08) 0%,
                rgba(56,189,248,0.06) 60%,rgba(52,211,153,0.04) 100%);
                border:1px solid rgba(248,113,113,0.2);border-radius:6px;
                padding:20px 24px;margin-bottom:20px;">
      <p style="font-family:'JetBrains Mono',monospace;font-size:9px;
                letter-spacing:0.18em;color:#f87171;text-transform:uppercase;margin:0 0 6px 0;">
        ● CASE STUDY — ITERATED PRISONER'S DILEMMA IN THE REAL WORLD
      </p>
      <p style="font-family:'DM Sans',sans-serif;font-size:18px;font-weight:600;
                color:#e2e8f0;margin:0 0 4px 0;">
        Russia–EU Energy Crisis (2021–2026)
      </p>
      <p style="font-family:'DM Sans',sans-serif;font-size:13px;color:#64748b;margin:0;">
        Historical gas flow data mapped to Prisoner's Dilemma moves.
        Russia: cooperates when flow&nbsp;>&nbsp;300&nbsp;mcm/day.
        EU: cooperates when weekly storage does not drop&nbsp;>&nbsp;5%.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ───────────────────────────────────────────────
    with st.spinner("Loading data files…"):
        try:
            df_daily = load_daily_data()
            df_stor  = load_storage_data()
            df_route = load_route_data()
        except Exception as exc:
            st.error(f"⚠ Error loading data files: {exc}")
            return

    # ── Date range filter ───────────────────────────────────────
    min_d = df_daily["Date"].min().date()
    max_d = df_daily["Date"].max().date()
    col_l, col_r = st.columns(2)
    start_d = col_l.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d)
    end_d   = col_r.date_input("End date",   value=max_d, min_value=min_d, max_value=max_d)

    if start_d > end_d:
        st.error("Start date must be before end date.")
        return

    df_daily = df_daily[
        (df_daily["Date"] >= pd.Timestamp(start_d)) &
        (df_daily["Date"] <= pd.Timestamp(end_d))
    ].copy()
    df_stor_f = df_stor[
        (df_stor["Week"] >= pd.Timestamp(start_d)) &
        (df_stor["Week"] <= pd.Timestamp(end_d))
    ].copy()

    if df_daily.empty:
        st.warning("No data in the selected date range.")
        return

    # ── Build timeline ──────────────────────────────────────────
    timeline = build_timeline(df_daily, df_stor_f)
    summary  = compute_case_summary(timeline, df_stor_f)

    # ── Metrics row ─────────────────────────────────────────────
    st.markdown("---")
    mc = st.columns(5)
    mc[0].metric("🇷🇺 Russia coop. rate",  f"{summary['coop_russia']:.1f}%")
    mc[1].metric("🇪🇺 EU coop. rate",      f"{summary['coop_eu']:.1f}%")
    mc[2].metric("🇷🇺 Russia total score", f"{summary['final_sc_ru']:,.0f}")
    mc[3].metric("🇪🇺 EU total score",     f"{summary['final_sc_eu']:,.0f}")
    mc[4].metric("🏭 Avg EU storage",      f"{summary['avg_stor']:.1f}%")

    col2a, col2b, col2c = st.columns(3)
    col2a.metric("📅 Days analysed",      f"{summary['total_days']:,}")
    col2b.metric("🤝 Mutual coop. days",  f"{summary['cc_days']:,}")
    col2c.metric("💥 Mutual defect days", f"{summary['dd_days']:,}")

    # ── PD payoff matrix reference ──────────────────────────────
    st.markdown("---")
    with st.expander("📋 Economic Translation of Payoff Matrix  (T=5, R=3, P=1, S=0)", expanded=False):
        rows = []
        for (mr, me), v in ECONOMIC_OUTCOMES.items():
            rows.append({
                "Russia move": mr, "EU move": me,
                "Outcome": v["label"],
                "Score Russia": v["score_ru"], "Score EU": v["score_eu"],
                "Russia GDP %": v["gdp_ru"], "EU GDP %": v["gdp_eu"],
                "EU Inflation %": v["inf_eu"],
                "Gas Market": v["gas_price"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Charts ──────────────────────────────────────────────────
    st.markdown("#### Gas Flow & Cumulative PD Scores")
    st.plotly_chart(fig_gas_vs_scores(timeline),         use_container_width=True)

    st.markdown("#### EU Weekly Storage & Response")
    st.plotly_chart(fig_storage_and_eu_move(df_stor_f),  use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Outcome Distribution")
        st.plotly_chart(fig_outcome_distribution(timeline), use_container_width=True)
    with c2:
        st.markdown("#### Economic Impact")
        st.plotly_chart(fig_economic_impact(timeline),      use_container_width=True)

    st.markdown("#### Pipeline Route Utilisation")
    st.plotly_chart(fig_route_utilisation(df_route),     use_container_width=True)

    # ── Download enriched CSV ───────────────────────────────────
    st.markdown("---")
    dl_cols = ["Date", "Russia", "move_russia", "move_eu",
               "outcome", "score_ru", "score_eu",
               "cum_score_ru", "cum_score_eu",
               "gdp_impact_ru", "gdp_impact_eu", "inf_impact_eu"]
    dl_df   = timeline[[c for c in dl_cols if c in timeline.columns]]
    csv_buf = io.StringIO()
    dl_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇  Download Enriched Timeline (CSV)",
        data=csv_buf.getvalue(),
        file_name="russia_eu_ipd_timeline.csv",
        mime="text/csv",
    )

    # ── Interpretation footnote ────────────────────────────────
    st.markdown("""
    <div style="margin-top:24px;padding:14px 18px;
                border:1px solid rgba(255,255,255,0.06);border-radius:4px;
                background:rgba(14,20,45,0.5);">
      <p style="font-family:'JetBrains Mono',monospace;font-size:9px;
                letter-spacing:0.12em;color:#38bdf8;text-transform:uppercase;margin:0 0 6px 0;">
        Methodology note
      </p>
      <p style="font-family:'DM Sans',sans-serif;font-size:12px;color:#64748b;margin:0;line-height:1.7;">
        <b style="color:#94a3b8">Russia move:</b>
        Cooperate (C) if daily flow &gt; 300 mcm; Defect (D) otherwise.<br>
        <b style="color:#94a3b8">EU move:</b>
        Cooperate (C) unless weekly storage drops &gt;5% (sanctions / LNG pivot = D).<br>
        <b style="color:#94a3b8">Scores:</b>
        T=5 (unilateral defection win) · R=3 (mutual cooperation) ·
        P=1 (mutual punishment) · S=0 (exploited cooperator).<br>
        <b style="color:#94a3b8">GDP/Inflation impacts</b>
        are illustrative proxies derived from the PD payoff hierarchy, not
        econometric estimates. Sources: ENTSOG, GIE, Bruegel, Eurostat.
      </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    render_header()
    cfg = sidebar_controls()

    # ── Run Tournament ─────────────────────────────────────────
    if cfg["run"]:
        if len(cfg["selected"]) < 2:
            st.error("Select at least **2 strategies** to run the tournament.")
            st.stop()
        try:
            payoff = PayoffMatrix(T=cfg["T"], R=cfg["R"], P=cfg["P"], S=cfg["S"])
        except ValueError as e:
            st.error(f"**Payoff matrix error:** {e}")
            st.stop()

        rng   = ReproducibleRNG(seed=cfg["seed"])
        strats = [STRATEGY_REGISTRY[nm](rng=rng) for nm in cfg["selected"]]

        with st.spinner("Running tournament…"):
            results = run_tournament(strats, payoff, rng, w=cfg["w"], n_games=cfg["n_games"])
            anova   = compute_anova(results["per_game_scores"])

        st.session_state["results"]  = results
        st.session_state["anova"]    = anova
        st.session_state["last_cfg"] = cfg
        st.success(f"Tournament complete · {len(cfg['selected'])} strategies · "
                   f"seed={cfg['seed']} · w={cfg['w']}")

    # ── Run RNG Tests ──────────────────────────────────────────
    if cfg["rng_test"]:
        rng = ReproducibleRNG(seed=cfg["seed"])
        with st.spinner("Running RNG quality tests…"):
            qc = rng_quality_tests(rng, n=cfg["rng_n"])
        st.session_state["rng_qc"] = qc
        verdict = "passed ✅" if qc["passed"] else "failed ⚠️"
        st.success(f"RNG tests complete · KS p={qc['ks_p']:.4f} · {verdict}")

    # ── Tabs ───────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "📊 Simulation",
        "🧪 Statistical Analysis",
        "🛡️ RNG Technical Annex",
        "⚡ Case Study: Russia–EU Energy Crisis",
    ])

    with t1: tab_simulation(cfg)
    with t2: tab_statistics()
    with t3: tab_rng_annex()
    with t4: tab_energy_crisis()


if __name__ == "__main__":
    main()
