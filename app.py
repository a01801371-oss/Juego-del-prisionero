"""
Prisoner's Dilemma Tournament Dashboard
Round-Robin tournament with 15 strategies
+ Case Study: Russia–EU Energy Crisis (2021–2026)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')
from scipy import stats as _scipy_stats

# ─────────────────────────────────────────────
# RNG global con semilla fija (PCG64, seed=42)
# ─────────────────────────────────────────────
RNG = np.random.default_rng(np.random.PCG64(42))

# ─────────────────────────────────────────────
# Clase base para estrategias
# ─────────────────────────────────────────────
class Strategy:
    """Clase base que gestiona el historial de movimientos."""
    name: str = "Base"

    def __init__(self):
        self.my_history: List[str] = []
        self.opp_history: List[str] = []

    def reset(self):
        self.my_history = []
        self.opp_history = []

    def move(self) -> str:
        raise NotImplementedError

    def update(self, my_move: str, opp_move: str):
        self.my_history.append(my_move)
        self.opp_history.append(opp_move)

    def __repr__(self):
        return self.name


# ─────────────────────────────────────────────
# 15 Estrategias
# ─────────────────────────────────────────────

class TitForTat(Strategy):
    name = "TIT FOR TAT"
    def move(self):
        return self.opp_history[-1] if self.opp_history else 'C'


class Grim(Strategy):
    name = "GRIM"
    def __init__(self):
        super().__init__()
        self._triggered = False
    def reset(self):
        super().reset()
        self._triggered = False
    def move(self):
        if not self._triggered and 'D' in self.opp_history:
            self._triggered = True
        return 'D' if self._triggered else 'C'


class Pavlov(Strategy):
    name = "PAVLOV"
    def move(self):
        if not self.my_history:
            return 'C'
        last_my  = self.my_history[-1]
        last_opp = self.opp_history[-1]
        if (last_my == 'C' and last_opp == 'C') or (last_my == 'D' and last_opp == 'D'):
            return last_my
        return 'D' if last_my == 'C' else 'C'


class AllD(Strategy):
    name = "ALL-D"
    def move(self):
        return 'D'


class AllC(Strategy):
    name = "ALL-C"
    def move(self):
        return 'C'


class TitForTwoTats(Strategy):
    name = "TIT FOR TWO TATS"
    def move(self):
        if len(self.opp_history) >= 2 and self.opp_history[-1] == 'D' and self.opp_history[-2] == 'D':
            return 'D'
        return 'C'


class Random(Strategy):
    name = "RANDOM"
    def move(self):
        return 'C' if RNG.random() < 0.5 else 'D'


class Joss(Strategy):
    """TIT FOR TAT pero traiciona con prob 0.1 tras cooperación del oponente."""
    name = "JOSS"
    def move(self):
        if not self.opp_history:
            return 'C'
        if self.opp_history[-1] == 'D':
            return 'D'
        return 'D' if RNG.random() < 0.1 else 'C'


class Gradual(Strategy):
    """Coopera hasta que el oponente traiciona; entonces traiciona N veces, luego dos C."""
    name = "GRADUAL"
    def __init__(self):
        super().__init__()
        self._punish_remaining = 0
        self._calm_remaining   = 0
    def reset(self):
        super().reset()
        self._punish_remaining = 0
        self._calm_remaining   = 0
    def move(self):
        if self._punish_remaining > 0:
            self._punish_remaining -= 1
            return 'D'
        if self._calm_remaining > 0:
            self._calm_remaining -= 1
            return 'C'
        if self.opp_history and self.opp_history[-1] == 'D':
            n = self.opp_history.count('D')
            self._punish_remaining = n - 1
            self._calm_remaining   = 2
            return 'D'
        return 'C'


def _get_payoff_from_session(my_move, opp_move):
    """Helper para Adaptive usando valores de sesión."""
    T = st.session_state.get('T', 5)
    R = st.session_state.get('R', 3)
    P = st.session_state.get('P', 1)
    S = st.session_state.get('S', 0)
    if my_move == 'C' and opp_move == 'C': return R
    if my_move == 'C' and opp_move == 'D': return S
    if my_move == 'D' and opp_move == 'C': return T
    return P


class Adaptive(Strategy):
    """Empieza con CCCCCDDDDD, luego adapta según qué acción dio mejores resultados."""
    name = "ADAPTIVE"
    _init_seq = list("CCCCCDDDDD")

    def move(self):
        t = len(self.my_history)
        if t < 10:
            return self._init_seq[t]
        c_payoffs, d_payoffs = [], []
        for m, o in zip(self.my_history, self.opp_history):
            score = _get_payoff_from_session(m, o)
            (c_payoffs if m == 'C' else d_payoffs).append(score)
        avg_c = np.mean(c_payoffs) if c_payoffs else 0
        avg_d = np.mean(d_payoffs) if d_payoffs else 0
        return 'C' if avg_c >= avg_d else 'D'


class EvolvedNN(Strategy):
    """Red Neuronal de 2 capas entrenada con ejemplos evolutivos simples."""
    name = "EVOLVED-NN"
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(np.random.PCG64(99))
        self.W1 = rng.standard_normal((8, 6)) * 0.5
        self.b1 = rng.standard_normal(8) * 0.1
        self.W2 = rng.standard_normal((1, 8)) * 0.5
        self.b2 = rng.standard_normal(1) * 0.1

    def _features(self):
        def enc(h, n=3):
            out = []
            for i in range(n):
                idx = -(i + 1)
                out.append(1.0 if (abs(idx) <= len(h) and h[idx] == 'C') else
                           (-1.0 if abs(idx) <= len(h) else 0.0))
            return out
        return np.array(enc(self.my_history) + enc(self.opp_history))

    def move(self):
        x   = self._features()
        h   = np.tanh(self.W1 @ x + self.b1)
        out = np.tanh(self.W2 @ h + self.b2)[0]
        return 'C' if out > 0 else 'D'


class PSOPlayer(Strategy):
    """Estrategia optimizada por PSO: usa una tabla de decisión con probabilidades."""
    name = "PSO-PLAYER"
    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(np.random.PCG64(7))
        self._weights = rng.dirichlet(np.ones(4))  # [CC,CD,DC,DD] → prob de C

    def move(self):
        if not self.my_history:
            return 'C'
        last_pair = self.my_history[-1] + self.opp_history[-1]
        idx = {'CC': 0, 'CD': 1, 'DC': 2, 'DD': 3}[last_pair]
        return 'C' if RNG.random() < self._weights[idx] else 'D'


class Memory3(Strategy):
    """Usa los últimos 3 movimientos del oponente para decidir."""
    name = "MEMORY-3"
    def move(self):
        if len(self.opp_history) < 3:
            return 'C'
        return 'D' if self.opp_history[-3:].count('D') >= 2 else 'C'


class Friedman(Strategy):
    """Alias de GRIM: traición permanente tras primer D del oponente."""
    name = "FRIEDMAN"
    def __init__(self):
        super().__init__()
        self._triggered = False
    def reset(self):
        super().reset()
        self._triggered = False
    def move(self):
        if not self._triggered and 'D' in self.opp_history:
            self._triggered = True
        return 'D' if self._triggered else 'C'


class Tester(Strategy):
    """Traiciona en ronda 1; si el oponente perdona, lo explota; si no, usa TFT."""
    name = "TESTER"
    def __init__(self):
        super().__init__()
        self._exploit   = False
        self._tft_mode  = False
    def reset(self):
        super().reset()
        self._exploit  = False
        self._tft_mode = False
    def move(self):
        t = len(self.my_history)
        if t == 0:
            return 'D'
        if t == 1:
            if self.opp_history and self.opp_history[-1] == 'D':
                self._tft_mode = True
            else:
                self._exploit = True
            return 'C'
        if self._exploit:
            return 'D'
        return self.opp_history[-1] if self.opp_history else 'C'


# ─────────────────────────────────────────────
# Registro
# ─────────────────────────────────────────────
ALL_STRATEGIES = [
    TitForTat, Grim, Pavlov, AllD, AllC,
    TitForTwoTats, Random, Joss, Gradual, Adaptive,
    EvolvedNN, PSOPlayer, Memory3, Friedman, Tester,
]
STRATEGY_NAMES = [s.name for s in ALL_STRATEGIES]


# ─────────────────────────────────────────────
# Motor del torneo
# ─────────────────────────────────────────────

def get_payoff(my_move, opp_move, T, R, P, S):
    if my_move == 'C' and opp_move == 'C': return R, R
    if my_move == 'C' and opp_move == 'D': return S, T
    if my_move == 'D' and opp_move == 'C': return T, S
    return P, P


def play_game(s1, s2, rounds, T, R, P, S, w):
    s1.reset(); s2.reset()
    score1 = score2 = 0.0
    h1, h2 = [], []
    for r in range(rounds):
        if r > 0 and RNG.random() > w:
            break
        m1 = s1.move(); m2 = s2.move()
        p1, p2 = get_payoff(m1, m2, T, R, P, S)
        s1.update(m1, m2); s2.update(m2, m1)
        score1 += p1; score2 += p2
        h1.append(m1); h2.append(m2)
    return score1, score2, h1, h2


def run_tournament(selected_names, T, R, P, S, w, games=5, rounds=200):
    name_to_class  = {s.name: s for s in ALL_STRATEGIES}
    strategies     = {n: name_to_class[n] for n in selected_names}
    n              = len(selected_names)
    payoff_matrix  = pd.DataFrame(np.zeros((n, n)),
                                  index=selected_names, columns=selected_names)
    total_scores   = {name: 0.0 for name in selected_names}
    h2h_data       = {}

    for name1, name2 in itertools.combinations(selected_names, 2):
        cls1 = strategies[name1]; cls2 = strategies[name2]
        gs1, gs2, lh1, lh2 = [], [], [], []
        for _ in range(games):
            sc1, sc2, h1, h2 = play_game(cls1(), cls2(), rounds, T, R, P, S, w)
            rlen = max(len(h1), 1)
            gs1.append(sc1 / rlen); gs2.append(sc2 / rlen)
            lh1, lh2 = h1, h2
        m1, m2 = np.mean(gs1), np.mean(gs2)
        payoff_matrix.loc[name1, name2] = m1
        payoff_matrix.loc[name2, name1] = m2
        total_scores[name1] += m1; total_scores[name2] += m2
        h2h_data[(name1, name2)] = (lh1, lh2)

    for name in selected_names:
        cls    = strategies[name]
        sc_list = []
        for _ in range(games):
            sc1, sc2, _, _ = play_game(cls(), cls(), rounds, T, R, P, S, w)
            rlen = max(1, rounds)
            sc_list.append((sc1 + sc2) / 2 / rlen)
        payoff_matrix.loc[name, name] = np.mean(sc_list)

    ranking = (pd.DataFrame({'Estrategia': list(total_scores.keys()),
                              'Score Total': list(total_scores.values())})
               .sort_values('Score Total', ascending=False)
               .reset_index(drop=True))
    ranking.index += 1
    return payoff_matrix, ranking, h2h_data



# ═══════════════════════════════════════════════════════════════════════════
#  CASE STUDY — RUSSIA–EU ENERGY CRISIS (2021–2026)
#  Simulador de Teoría de Juegos Geopolítica
#  ─────────────────────────────────────────────────────────────────────────
#  Jugador A (Rusia):  D si flujo < 250 mcm  O caída mensual ≥ 20 %
#  Jugador B (Europa): D si post-Feb-2022 + LNG sube ≥ 15 %  O 2025/26 activo
#  RESTRICCIÓN: No toca el motor del torneo ni las clases de estrategia.
# ═══════════════════════════════════════════════════════════════════════════

import os as _os

# ── Constantes base ────────────────────────────────────────────────────────
_FLOW_HARD_THRESH  = 250      # mcm/día — umbral absoluto defección Rusia
_FLOW_DROP_PCT     = 20.0     # % caída mensual → Rusia defecta
_LNG_SURGE_PCT     = 15.0     # % subida LNG → UE "traiciona" contrato ruso
_SANCTIONS_DATE    = pd.Timestamp("2022-02-24")
_INDEPENDENCE_DATE = pd.Timestamp("2025-01-01")   # REPowerEU / independence regs

# Eventos históricos anotados en gráficos
HISTORICAL_EVENTS = [
    ("2021-10-01", "Crisis precio gas",            "rgba(251,191,36,0.55)"),
    ("2022-02-24", "Invasión Ucrania\n(Sanciones)","rgba(248,113,113,0.70)"),
    ("2022-06-15", "NS1 recorte −60 %",            "rgba(248,113,113,0.50)"),
    ("2022-09-26", "Sabotaje Nord Stream",          "rgba(167,139,250,0.70)"),
    ("2022-12-05", "Techo precio petróleo G7",      "rgba(251,191,36,0.45)"),
    ("2024-01-01", "Tránsito UA expira",            "rgba(248,113,113,0.50)"),
    ("2025-01-01", "Independencia energética UE",   "rgba(52,211,153,0.55)"),
]

# Paleta temática oscura
_BG  = "#080b12"
_SRF = "#0e1420"
_GRD = "rgba(255,255,255,0.045)"
_RU, _EU, _GAS, _GRN = "#f87171", "#38bdf8", "#fbbf24", "#34d399"

# ══════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS (cached, con fallback sintético reproducible)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_daily_data() -> pd.DataFrame:
    """
    Lee daily_data_*.csv de la raíz del repositorio.
    Columnas esperadas: Date, Russia (mcm/día), LNG (mcm/día, opcional).
    Manejo de errores: si el archivo no existe genera datos sintéticos.
    """
    candidates = sorted(
        [f for f in _os.listdir(".") if f.startswith("daily_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        try:
            df = pd.read_csv(candidates[0])
            # Normalizar fecha
            dcol = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            df = df.rename(columns={dcol: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            # Normalizar Russia
            rcol = next((c for c in df.columns
                         if "russia" in c.lower() or c.strip().lower() == "ru"), None)
            if rcol and rcol != "Russia":
                df = df.rename(columns={rcol: "Russia"})
            if "Russia" not in df.columns:
                nums = df.select_dtypes("number").columns.tolist()
                df["Russia"] = df[nums[0]] if nums else 300.0
            # Normalizar LNG (puede no existir)
            lcol = next((c for c in df.columns if "lng" in c.lower()), None)
            if lcol and lcol != "LNG":
                df = df.rename(columns={lcol: "LNG"})
            if "LNG" not in df.columns:
                # Generar LNG proxy: incremento real desde 2022
                _r = np.random.default_rng(np.random.PCG64(3001))
                base_lng = np.full(len(df), 30.0)
                for i, d in enumerate(df["Date"]):
                    if d >= _SANCTIONS_DATE:
                        base_lng[i] = 30 + (d - _SANCTIONS_DATE).days * 0.05
                    if d >= _INDEPENDENCE_DATE:
                        base_lng[i] *= 1.3
                df["LNG"] = np.clip(base_lng + _r.normal(0, 3, len(df)), 0, 200)
            df["Russia"] = pd.to_numeric(df["Russia"], errors="coerce").fillna(300)
            df["LNG"]    = pd.to_numeric(df["LNG"],    errors="coerce").fillna(30)
            return df
        except Exception as e:
            pass  # fall through to synthetic

    # ── Datos sintéticos reproducibles ─────────────────────────────
    _r = np.random.default_rng(np.random.PCG64(2026))
    dates = pd.date_range("2021-01-01", "2026-03-12", freq="D")
    n = len(dates)
    base_ru = np.full(n, 380.0)
    base_ru[365:730]  -= 80    # reducción 2022
    base_ru[730:910]  -= 200   # shock post-invasión
    base_ru[910:1100] += 40    # recuperación parcial
    base_ru[1100:]    -= 130   # cortes 2023–24
    russia = np.clip(base_ru + _r.normal(0, 22, n), 0, 600)

    base_lng = np.full(n, 28.0)
    for i, d in enumerate(dates):
        if d >= _SANCTIONS_DATE:
            base_lng[i] = 28 + (d - _SANCTIONS_DATE).days * 0.055
        if d >= _INDEPENDENCE_DATE:
            base_lng[i] *= 1.35
    lng = np.clip(base_lng + _r.normal(0, 4, n), 0, 200)

    return pd.DataFrame({"Date": dates, "Russia": russia, "LNG": lng})


@st.cache_data(show_spinner=False)
def load_storage_data() -> pd.DataFrame:
    """
    Lee Weekly Storage EU & UA *.xlsx de la raíz.
    Fallback a datos sintéticos si no existe.
    """
    candidates = sorted(
        [f for f in _os.listdir(".") if "storage" in f.lower() and f.endswith(".xlsx")],
        reverse=True,
    )
    if candidates:
        try:
            xl = pd.read_excel(candidates[0])
            dcol = next((c for c in xl.columns
                         if any(k in c.lower() for k in ["date","week","fecha"])),
                        xl.columns[0])
            xl = xl.rename(columns={dcol: "Week"})
            xl["Week"] = pd.to_datetime(xl["Week"], errors="coerce")
            xl = xl.dropna(subset=["Week"]).sort_values("Week").reset_index(drop=True)
            scol = next((c for c in xl.columns
                         if any(k in c.lower() for k in ["storage","stor","fill","pct","%","level"])),
                        None)
            if scol and scol != "Storage_pct":
                xl = xl.rename(columns={scol: "Storage_pct"})
            if "Storage_pct" not in xl.columns:
                nums = xl.select_dtypes("number").columns.tolist()
                xl["Storage_pct"] = xl[nums[0]] if nums else 50.0
            xl["Storage_pct"] = pd.to_numeric(xl["Storage_pct"], errors="coerce").fillna(50)
            return xl.reset_index(drop=True)
        except Exception:
            pass

    _r = np.random.default_rng(np.random.PCG64(2027))
    weeks = pd.date_range("2021-01-04", "2026-03-10", freq="W-MON")
    pct   = np.clip(55 + _r.normal(0, 3, len(weeks)).cumsum() * 0.35, 5, 100)
    return pd.DataFrame({"Week": weeks, "Storage_pct": pct})


@st.cache_data(show_spinner=False)
def load_route_data() -> pd.DataFrame:
    """Lee route_data_*.csv de la raíz. Fallback a datos sintéticos."""
    candidates = sorted(
        [f for f in _os.listdir(".") if f.startswith("route_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        try:
            df = pd.read_csv(candidates[0])
            dcol = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
            df = df.rename(columns={dcol: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        except Exception:
            pass

    _r = np.random.default_rng(np.random.PCG64(2028))
    dates = pd.date_range("2021-01-01", "2026-03-12", freq="D")
    profiles = {
        "Nord Stream 1": (80, "2022-06-15", 0),
        "Yamal-Europe":  (60, "2022-04-27", 5),
        "Ukrainian GTS": (70, "2024-01-01", 15),
        "TurkStream":    (52, "2021-01-01", 44),
    }
    rows = []
    for route, (base_val, cut_date, floor) in profiles.items():
        cut_ts = pd.Timestamp(cut_date)
        for d in dates:
            v = (base_val + _r.normal(0, 4) if d < cut_ts
                 else max(floor, base_val - (d - cut_ts).days * 0.25 + _r.normal(0, 3)))
            rows.append({"Date": d, "Route": route,
                         "Utilisation_pct": round(float(np.clip(v, 0, 100)), 1)})
    return pd.DataFrame(rows)



@st.cache_data(show_spinner=False)
def load_macro_data() -> pd.DataFrame:
    """
    Datos macroeconómicos reales interpolados a frecuencia diaria.

    Fuentes verificadas:
      GDP_EU  — Eurostat namq_10_gdp Sheet 4 (PIB EU27 interanual %, trimestral)
                Valores: -13.1% Q2-2020 (COVID), +14.7% Q2-2021 (rebote),
                         +5.6% Q1-2022, +0.1% Q3-2023, +1.5% Q4-2024
      GDP_RU  — Banco Mundial NY.GDP.MKTP.KD.ZG (anual)
                Valores: +5.87% 2021, -1.44% 2022, +4.08% 2023, +4.34% 2024
                         2025-2026: proyección FMI WEO
      INFL_EU — Eurostat HICP prc_hicp_minr (tasa YoY mensual)
                Valores: 1.2% ene-2021, 11.5% oct-2022 (pico real),
                         6.15% jul-2023, 2.8% jul-2024, 2.4% dic-2024
      INFL_RU — Banco Mundial FP.CPI.TOTL.ZG (anual)
                Valores: 6.7% 2021, 13.7% 2022, 5.9% 2023, 8.4% 2024
                         2025-2026: proyección CBR
    """
    daily_idx = pd.date_range("2021-01-01", "2026-03-12", freq="D")

    def _interp(anchor_dict):
        s = pd.Series({pd.Timestamp(k): v for k, v in anchor_dict.items()})
        return (s.reindex(s.index.union(daily_idx))
                 .interpolate("time")
                 .reindex(daily_idx)
                 .values.round(3))

    # ── GDP UE trimestral interanual % (Eurostat) ──────────────
    gdp_eu = _interp({
        "2020-01-01": -2.2,  "2020-04-01": -13.1, "2020-07-01": -3.9,  "2020-10-01": -3.5,
        "2021-01-01":  0.3,  "2021-04-01":  14.7, "2021-07-01":  5.3,  "2021-10-01":  5.8,
        "2022-01-01":  5.6,  "2022-04-01":   4.2, "2022-07-01":  2.9,  "2022-10-01":  1.8,
        "2023-01-01":  1.2,  "2023-04-01":   0.4, "2023-07-01":  0.1,  "2023-10-01":  0.4,
        "2024-01-01":  0.6,  "2024-04-01":   0.8, "2024-07-01":  1.1,  "2024-10-01":  1.5,
        "2025-01-01":  1.7,  "2025-04-01":   1.7, "2025-07-01":  1.7,  "2025-10-01":  1.4,
    })

    # ── GDP Rusia anual % (Banco Mundial) ──────────────────────
    gdp_ru = _interp({
        "2019-07-01":  2.20, "2020-07-01": -2.65,
        "2021-07-01":  5.87, "2022-07-01": -1.44,
        "2023-07-01":  4.08, "2024-07-01":  4.34,
        "2025-07-01":  3.50, "2026-01-01":  2.50,
    })

    # ── Inflación UE mensual YoY % (Eurostat HICP) ─────────────
    infl_eu = _interp({
        "2020-01-01":  1.67, "2020-04-01":  0.69,
        "2020-07-01":  0.81, "2020-10-01":  0.25,
        "2021-01-01":  1.21, "2021-04-01":  2.03,
        "2021-07-01":  2.47, "2021-10-01":  4.37,
        "2022-01-01":  5.58, "2022-04-01":  8.10,
        "2022-07-01":  9.78, "2022-10-01": 11.50,   # pico real oct 2022
        "2023-01-01":  9.97, "2023-04-01":  8.11,
        "2023-07-01":  6.15, "2023-10-01":  3.63,
        "2024-01-01":  3.09, "2024-04-01":  2.57,
        "2024-07-01":  2.80, "2024-10-01":  2.32,
        "2025-01-01":  2.81, "2025-04-01":  2.41,
        "2025-07-01":  2.36, "2025-10-01":  2.48,
        "2026-01-01":  2.06,
    })

    # ── Inflación Rusia anual % (Banco Mundial) ─────────────────
    infl_ru = _interp({
        "2019-07-01":  4.47, "2020-07-01":  3.38,
        "2021-07-01":  6.69, "2022-07-01": 13.74,
        "2023-07-01":  5.87, "2024-07-01":  8.44,
        "2025-07-01":  9.00, "2026-01-01":  8.00,
    })

    return pd.DataFrame({
        "Date":    daily_idx,
        "GDP_EU":  gdp_eu,
        "GDP_RU":  gdp_ru,
        "INFL_EU": infl_eu,
        "INFL_RU": infl_ru,
    })


# ══════════════════════════════════════════════════════════════════════════
#  LÓGICA DE MOVIDAS — JUGADORES A y B
# ══════════════════════════════════════════════════════════════════════════

def compute_russia_moves(df: pd.DataFrame, flow_thresh: float = 250) -> pd.Series:
    """
    Jugador A — Rusia.
    D (Defect) si:
      • Flujo < flow_thresh mcm/día  (corte absoluto)
      • O caída ≥ 20 % respecto a la media del mes anterior  (corte relativo)
    """
    df = df.copy()
    df["_russia_ma30"] = df["Russia"].rolling(30, min_periods=1).mean().shift(1)
    df["_drop_pct"] = (df["Russia"] - df["_russia_ma30"]) / df["_russia_ma30"].replace(0, np.nan) * 100

    def _decide(row):
        if pd.isna(row["Russia"]):
            return "D"
        if float(row["Russia"]) < flow_thresh:
            return "D"
        if pd.notna(row["_drop_pct"]) and row["_drop_pct"] <= -_FLOW_DROP_PCT:
            return "D"
        return "C"

    return df.apply(_decide, axis=1)


def compute_eu_moves_realistic(df: pd.DataFrame, stor: pd.DataFrame) -> pd.Series:
    """
    Jugador B — Europa, modo 'Realista/Sanciones'.
    D (Defect) si CUALQUIERA de:
      a) Fecha > 24-Feb-2022  (régimen de sanciones activo)
         Y  LNG sube ≥ 15 % respecto a media 30d previa
      b) Fecha ≥ 01-Ene-2025  (independencia energética activada)
    C en cualquier otro caso.
    """
    df = df.copy()
    df["_lng_ma30"]   = df["LNG"].rolling(30, min_periods=1).mean().shift(1)
    df["_lng_surge"]  = (df["LNG"] - df["_lng_ma30"]) / df["_lng_ma30"].replace(0, np.nan) * 100

    def _decide(row):
        d = row["Date"]
        # Condición (b): independencia energética
        if d >= _INDEPENDENCE_DATE:
            return "D"
        # Condición (a): sanciones + pivot GNL
        if d > _SANCTIONS_DATE:
            if pd.notna(row["_lng_surge"]) and row["_lng_surge"] >= _LNG_SURGE_PCT:
                return "D"
        return "C"

    return df.apply(_decide, axis=1)


def compute_eu_moves_strategy(
    eu_strategy: str,
    df: pd.DataFrame,
    stor: pd.DataFrame,
    russia_moves: pd.Series,
) -> pd.Series:
    """
    Calcula las movidas de la UE según la estrategia elegida por el usuario.
    - 'Realista/Sanciones': lógica geopolítica (compute_eu_moves_realistic)
    - 'Pacificadora':       siempre C
    - 'Tit-for-Tat':        C en ronda 1; luego copia la movida anterior de Rusia
    - Cualquier estrategia del catálogo (15): usa la clase correspondiente
    """
    n = len(df)
    if eu_strategy == "Realista/Sanciones":
        return compute_eu_moves_realistic(df, stor)
    if eu_strategy == "Pacificadora":
        return pd.Series(["C"] * n, index=df.index)
    if eu_strategy == "Tit-for-Tat (clásico)":
        moves = ["C"]
        for i in range(1, n):
            moves.append(russia_moves.iloc[i - 1])
        return pd.Series(moves, index=df.index)

    # Estrategias del catálogo
    name_to_cls = {s.name: s for s in ALL_STRATEGIES}
    if eu_strategy in name_to_cls:
        inst = name_to_cls[eu_strategy]()
        eu_moves = []
        for ru_m in russia_moves:
            eu_m = inst.move()
            eu_moves.append(eu_m)
            inst.update(eu_m, ru_m)   # UE observa su propia movida y la de Rusia
        return pd.Series(eu_moves, index=df.index)

    return pd.Series(["C"] * n, index=df.index)


def compute_russia_strategy_moves(
    ru_strategy: str,
    df: pd.DataFrame,
    eu_moves_hist: pd.Series,
    flow_thresh: float = 250,
) -> pd.Series:
    """
    Movidas de Rusia según estrategia elegida por el usuario.
    - 'Datos Históricos':  lógica real del CSV
    - 'Tit-for-Tat':       copia movida previa de la UE
    - 'Always Defect':     siempre D
    - 'Bully':             defecta primero, explota si la UE cede (Tester)
    """
    n = len(df)
    if ru_strategy == "Datos Históricos":
        return compute_russia_moves(df, flow_thresh)
    if ru_strategy == "Always Defect":
        return pd.Series(["D"] * n, index=df.index)
    if ru_strategy == "Tit-for-Tat":
        moves = ["C"]
        for i in range(1, n):
            moves.append(eu_moves_hist.iloc[i - 1])
        return pd.Series(moves, index=df.index)
    if ru_strategy == "Bully":
        inst = Tester()
        ru_moves = []
        for eu_m in eu_moves_hist:
            ru_m = inst.move()
            ru_moves.append(ru_m)
            inst.update(ru_m, eu_m)
        return pd.Series(ru_moves, index=df.index)
    return compute_russia_moves(df, flow_thresh)


# ══════════════════════════════════════════════════════════════════════════
#  MOTOR DE SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════

def simulate_ipd(
    df_daily:    pd.DataFrame,
    df_stor:     pd.DataFrame,
    T: float, R: float, P: float, S: float,
    flow_thresh: float,
    eu_strategy: str,
    date_start:  pd.Timestamp,
    date_end:    pd.Timestamp,
    ru_strategy: str = "Datos Históricos",
) -> pd.DataFrame:
    """
    Simulador completo del Dilema del Prisionero Iterado Rusia–UE.

    Genera una fila por día con:
      • Movidas (move_russia, move_eu)
      • Scores DP (score_ru, score_eu, cum_ru, cum_eu)
      • Impactos económicos con volatilidad estocástica en DD
      • Indicadores macroeconómicos (30d rolling)
    """
    # Filtrar período
    df = df_daily[
        (df_daily["Date"] >= date_start) &
        (df_daily["Date"] <= date_end)
    ].copy().reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # ── Movidas históricas de referencia ──────────────────────────
    df["move_russia_hist"] = compute_russia_moves(df, flow_thresh)

    stor_f = df_stor[
        (df_stor["Week"] >= date_start) &
        (df_stor["Week"] <= date_end)
    ].copy().reset_index(drop=True)

    # ── Movida UE histórica/realista (señal observable para Rusia sim.) ──
    eu_hist = compute_eu_moves_strategy(
        "Realista/Sanciones", df, stor_f, df["move_russia_hist"]
    )

    # ── Movida Rusia final (histórica o simulada) ──────────────────
    df["move_russia"] = compute_russia_strategy_moves(
        ru_strategy, df, eu_hist, flow_thresh
    )

    # ── Movida UE final ───────────────────────────────────────────
    df["move_eu"] = compute_eu_moves_strategy(
        eu_strategy, df, stor_f, df["move_russia"]
    )

    # ── Merge almacenamiento semanal → diario ─────────────────────
    if not stor_f.empty:
        stor_daily = stor_f[["Week","Storage_pct"]].rename(columns={"Week":"Date"})
        df = pd.merge_asof(
            df.sort_values("Date"),
            stor_daily.sort_values("Date"),
            on="Date", direction="backward",
        ).reset_index(drop=True)
        df["Storage_pct"] = df["Storage_pct"].ffill().fillna(50)
    else:
        df["Storage_pct"] = 50.0

    # ── Payoff map base ────────────────────────────────────────────
    _base_payoffs = {
        ("C","C"): (R, R),
        ("D","C"): (T, S),
        ("C","D"): (S, T),
        ("D","D"): (P, P),
    }
    _outcome_labels = {
        ("C","C"): "Cooperación Mutua",
        ("D","C"): "Rusia defecta — precio alto",
        ("C","D"): "UE responde — sanciones/GNL",
        ("D","D"): "Castigo mutuo — crisis total",
    }
    _outcome_colors = {
        "Cooperación Mutua":          _GRN,
        "Rusia defecta — precio alto":_RU,
        "UE responde — sanciones/GNL":_GAS,
        "Castigo mutuo — crisis total":"#a78bfa",
    }

    # ── Parámetros base de impacto económico ──────────────────────
    # Escalados por la magnitud de T (cuanto más alto T, más severos los efectos)
    # ── Datos macroeconómicos reales ───────────────────────────────────
    _macro_df  = load_macro_data()
    _has_real  = not _macro_df.empty
    if _has_real:
        _macro_f = (
            _macro_df[(_macro_df["Date"] >= date_start) & (_macro_df["Date"] <= date_end)]
            .set_index("Date")
            .reindex(pd.date_range(date_start, date_end, freq="D"))
            .interpolate("time")
            .reset_index()
            .rename(columns={"index": "Date"})
        )
    else:
        _macro_f = pd.DataFrame()

    _t_factor = T / 5.0   # para proxies de fallback

    _base_econ = {
        ("C","C"): (2.0,  2.0,  2.5),    # (gdp_ru, gdp_eu, inf_eu)
        ("D","C"): (4.5 * _t_factor, -4.0 * _t_factor, 15.0 * _t_factor),
        ("C","D"): (-3.5, 0.5, 6.0),
        ("D","D"): (-2.5 * _t_factor, -2.5 * _t_factor, 8.0 * _t_factor),
    }

    # ── RNG para volatilidad estocástica en DD ─────────────────────
    _rng_noise = np.random.default_rng(np.random.PCG64(42))

    records = []
    cum_ru = cum_eu = 0.0
    _loop_idx = 0

    for _, row in df.iterrows():
        key = (row["move_russia"], row["move_eu"])
        pr, pe    = _base_payoffs.get(key, (R, R))
        gdp_ru_b, gdp_eu_b, inf_eu_b = _base_econ.get(key, (2.0, 2.0, 2.5))

        # ── Impacto económico: datos reales de Eurostat/BM ─────────────────
        # Con datos reales, la inflación y el PIB ya reflejan la realidad
        # geopolítica — no necesitan ajuste adicional por outcome del DP.
        # Solo añadimos ruido estocástico pequeño en DD para la banda de volatilidad.
        if _has_real and _loop_idx < len(_macro_f):
            _mr = _macro_f.iloc[_loop_idx]
            _gdp_eu_r  = float(_mr.get("GDP_EU",  gdp_eu_b))
            _gdp_ru_r  = float(_mr.get("GDP_RU",  gdp_ru_b))
            _infl_eu_r = float(_mr.get("INFL_EU", inf_eu_b))
            _infl_ru_r = float(_mr.get("INFL_RU", 7.0))
            if key == ("D","D"):
                _n1 = float(_rng_noise.normal(0, 0.25))
                _n2 = float(abs(_rng_noise.normal(0, 0.5)))
                gdp_eu = _gdp_eu_r + _n1
                gdp_ru = _gdp_ru_r + _n1 * 0.5
                inf_eu = _infl_eu_r + _n2
                inf_ru = _infl_ru_r + abs(_n2) * 0.5
            else:
                gdp_eu = _gdp_eu_r + float(_rng_noise.normal(0, 0.08))
                gdp_ru = _gdp_ru_r + float(_rng_noise.normal(0, 0.08))
                inf_eu = _infl_eu_r
                inf_ru = _infl_ru_r
        else:
            if key == ("D","D"):
                _noise_gdp = float(_rng_noise.normal(0, 1.2))
                _noise_inf = float(abs(_rng_noise.normal(0, 3.5)))
                gdp_ru  = gdp_ru_b + _noise_gdp
                gdp_eu  = gdp_eu_b - abs(_noise_gdp) * 0.8
                inf_eu  = inf_eu_b + _noise_inf
                inf_ru  = 7.0 + _noise_inf * 0.8
            else:
                _noise  = float(_rng_noise.normal(0, 0.3))
                gdp_ru  = gdp_ru_b + _noise
                gdp_eu  = gdp_eu_b + _noise * 0.5
                inf_eu  = inf_eu_b + abs(_noise) * 0.4
                inf_ru  = 7.0 + abs(_noise) * 0.3

        cum_ru += pr; cum_eu += pe
        label = _outcome_labels.get(key, "Cooperación Mutua")
        _loop_idx += 1
        records.append({
            "score_ru":  pr,    "score_eu":  pe,
            "cum_ru":    cum_ru,"cum_eu":    cum_eu,
            "gdp_ru":    round(gdp_ru, 3),
            "gdp_eu":    round(gdp_eu, 3),
            "inf_eu":    round(inf_eu, 3),
            "inf_ru":    round(inf_ru, 3),
            "outcome":   label,
            "color":     _outcome_colors.get(label, "#64748b"),
        })

    result = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(records),
    ], axis=1)

    # ── Rolling indicators (30d) ──────────────────────────────────
    result["gdp_eu_30d"]  = result["gdp_eu"].rolling(30, min_periods=1).mean()
    result["gdp_ru_30d"]  = result["gdp_ru"].rolling(30, min_periods=1).mean()
    result["inf_eu_30d"]  = result["inf_eu"].rolling(30, min_periods=1).mean()
    result["inf_eu_7d"]   = result["inf_eu"].rolling(7,  min_periods=1).std()
    result["inf_ru_30d"]  = result["inf_ru"].rolling(30, min_periods=1).mean()
    result["inf_ru_7d"]   = result["inf_ru"].rolling(7,  min_periods=1).std()
    result["coop_30d"]    = (
        ((result["move_russia"]=="C") & (result["move_eu"]=="C"))
        .rolling(30, min_periods=1).mean() * 100
    )
    result["dd_30d"] = (
        ((result["move_russia"]=="D") & (result["move_eu"]=="D"))
        .rolling(30, min_periods=1).mean() * 100
    )
    return result


# ══════════════════════════════════════════════════════════════════════════
#  IDENTIFICADOR DE ESTRATEGIA PREDOMINANTE
# ══════════════════════════════════════════════════════════════════════════

def identify_strategy(
    df_daily: pd.DataFrame,
    df_stor:  pd.DataFrame,
    flow_thresh: float = 250,
    date_start:  pd.Timestamp = None,
    date_end:    pd.Timestamp = None,
) -> dict:
    """
    Compara la secuencia histórica de Rusia con las 15 estrategias del catálogo.
    Retorna % de coincidencia de movidas para cada una.
    """
    dd = df_daily.copy()
    if date_start is not None: dd = dd[dd["Date"] >= date_start]
    if date_end   is not None: dd = dd[dd["Date"] <= date_end]
    dd = dd.reset_index(drop=True)
    if dd.empty:
        return {"scores": {}, "best": "—", "best_pct": 0.0, "top3": [], "n_days": 0}

    hist_russia = compute_russia_moves(dd, flow_thresh).tolist()
    n = len(hist_russia)

    # Señal UE observable para que las estrategias puedan reaccionar
    sf = df_stor.copy()
    if date_start is not None: sf = sf[sf["Week"] >= date_start]
    if date_end   is not None: sf = sf[sf["Week"] <= date_end]
    if not sf.empty:
        sf = sf.reset_index(drop=True)
        sf["_chg"] = sf["Storage_pct"].pct_change() * 100
        sf["_eu"]  = sf["_chg"].apply(
            lambda x: "D" if (pd.notna(x) and x < -5.0) else "C")
        sig = sf[["Week","_eu"]].rename(columns={"Week":"Date"})
        eu_sig = pd.merge_asof(
            dd[["Date"]].sort_values("Date"),
            sig.sort_values("Date"),
            on="Date", direction="backward",
        )["_eu"].fillna("C").tolist()
    else:
        eu_sig = ["C"] * n

    scores = {}
    for cls in ALL_STRATEGIES:
        inst = cls(); m2 = 0
        for actual, eu_s in zip(hist_russia, eu_sig):
            pred = inst.move()
            if pred == actual: m2 += 1
            inst.update(actual, eu_s)
        scores[cls.name] = round(m2 / n * 100, 1)

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {"scores": scores, "best": top[0][0], "best_pct": top[0][1],
            "top3": top[:3], "n_days": n}


# ══════════════════════════════════════════════════════════════════════════
#  FIGURAS PLOTLY
# ══════════════════════════════════════════════════════════════════════════

_LAYOUT = dict(
    paper_bgcolor=_BG, plot_bgcolor=_SRF,
    font=dict(family="monospace", color="#e2e8f0", size=10),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", borderwidth=0, font=dict(size=9)),
    hovermode="x unified",
    margin=dict(l=12, r=12, t=52, b=12),
)


def _add_event_lines(fig, df, row="all"):
    """Añade vlines anotadas para eventos históricos clave."""
    for ds, label, color in HISTORICAL_EVENTS:
        ts = pd.Timestamp(ds)
        if not (df["Date"].min() <= ts <= df["Date"].max()):
            continue
        kw = dict(row=row, col=1) if row != "all" else {}
        fig.add_vline(
            x=ts.timestamp() * 1000,
            line_dash="dot", line_color=color, line_width=1.2,
            **kw,
        )
        fig.add_annotation(
            x=ts, y=1, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(size=7, color=color),
            textangle=-90, xanchor="left",
        )


def fig_gas_flow(df: pd.DataFrame, flow_thresh: float) -> go.Figure:
    """
    Gráfico principal de flujo de gas con:
    - Barras coloreadas por movida de Rusia (verde=C / rojo=D)
    - Línea LNG superpuesta (eje secundario)
    - Umbral de cooperación
    - Eventos históricos anotados
    - Rangeslider para zoom
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ru_colors = [_GRN if m == "C" else _RU for m in df["move_russia"]]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Russia"],
        name="Flujo Rusia (mcm/día)",
        marker=dict(color=ru_colors, opacity=0.75, line=dict(width=0)),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.0f} mcm/día</b><extra>Rusia</extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["LNG"],
        name="Flujo LNG (mcm/día)",
        mode="lines", line=dict(color="#a78bfa", width=1.8),
        hovertemplate="%{y:.1f} mcm<extra>LNG</extra>",
    ), secondary_y=True)

    # Umbral cooperación Rusia
    fig.add_hline(
        y=flow_thresh, line_dash="dot",
        line_color="rgba(251,191,36,0.55)", line_width=1.5,
        annotation_text=f"Umbral C/D Rusia ({flow_thresh} mcm)",
        annotation_font=dict(size=8, color=_GAS),
        annotation_position="top left",
        secondary_y=False,
    )
    _add_event_lines(fig, df)

    fig.update_layout(
        **_LAYOUT, height=380,
        title_text="FLUJO DE GAS RUSIA → EUROPA  ·  Verde=Coopera | Rojo=Defecta",
        title_font=dict(size=11),
        xaxis=dict(
            gridcolor=_GRD, tickfont=dict(size=9),
            rangeslider=dict(visible=True, thickness=0.04),
        ),
    )
    fig.update_yaxes(title_text="mcm/día (Gas)", secondary_y=False,
                     gridcolor=_GRD, tickfont=dict(size=9))
    fig.update_yaxes(title_text="mcm/día (LNG)", secondary_y=True,
                     gridcolor=_GRD, tickfont=dict(size=9), showgrid=False)
    return fig


def fig_welfare_accumulation(df: pd.DataFrame) -> go.Figure:
    """
    Curvas de bienestar acumulado con fill entre ellas para visualizar
    quién va ganando. Incluye eventos históricos.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_eu"],
        name="Bienestar UE", mode="lines",
        line=dict(color=_EU, width=2.5),
        hovertemplate="UE: %{y:,.0f} pts<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_ru"],
        name="Bienestar Rusia", mode="lines",
        line=dict(color=_RU, width=2.5),
        fill="tonexty",
        fillcolor="rgba(248,113,113,0.07)",
        hovertemplate="Rusia: %{y:,.0f} pts<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.18)", line_width=1)

    # Anotar punto de máxima divergencia
    df2 = df.dropna(subset=["cum_ru","cum_eu"])
    if not df2.empty:
        diff = (df2["cum_ru"] - df2["cum_eu"]).abs()
        idx  = diff.idxmax()
        row  = df2.loc[idx]
        fig.add_annotation(
            x=row["Date"],
            y=max(float(row["cum_ru"]), float(row["cum_eu"])),
            text=f"Δ máx = {diff[idx]:,.0f}",
            showarrow=True, arrowhead=2,
            font=dict(size=8, color=_GAS),
            arrowcolor=_GAS, bgcolor="rgba(0,0,0,0.55)",
        )
    _add_event_lines(fig, df)
    fig.update_layout(
        **_LAYOUT, height=320,
        title_text="ACUMULACIÓN DE BIENESTAR — SCORES DP",
        title_font=dict(size=11),
    )
    fig.update_xaxes(gridcolor=_GRD, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRD, tickfont=dict(size=9),
                     title="Score acumulado")
    return fig


def fig_economic_impact(df: pd.DataFrame, T: float) -> go.Figure:
    """
    Solo PIB — UE y Rusia (media móvil 30d).
    La inflación se muestra en fig_inflation_comparison por separado.
    Fuentes: Eurostat namq_10_gdp (UE) · Banco Mundial NY.GDP.MKTP.KD.ZG (Rusia).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["gdp_eu_30d"],
        name="PIB UE — Eurostat (30d)", mode="lines",
        line=dict(color=_EU, width=2.2),
        hovertemplate="PIB UE: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["gdp_ru_30d"],
        name="PIB Rusia — Banco Mundial (30d)", mode="lines",
        line=dict(color=_RU, width=2.2),
        hovertemplate="PIB Rusia: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.18)", line_width=1.2)

    # Sombrear períodos DD
    for ds, de in _get_dd_periods(df):
        fig.add_vrect(x0=ds, x1=de,
                      fillcolor="rgba(167,139,250,0.07)", line_width=0)

    _add_event_lines(fig, df)
    fig.update_layout(
        **_LAYOUT, height=320,
        title_text=(
            f"PIB UE vs RUSIA — Crecimiento interanual % (media móvil 30d)  ·  "
            f"Fuentes: Eurostat · Banco Mundial  ·  T={T}"
        ),
        title_font=dict(size=11),
        yaxis=dict(gridcolor=_GRD, tickfont=dict(size=9),
                   title="Crecimiento YoY (%)", ticksuffix="%"),
        xaxis=dict(gridcolor=_GRD, tickfont=dict(size=9)),
    )
    return fig


def _get_dd_periods(df: pd.DataFrame, min_run: int = 3):
    """Extrae períodos continuos de DD (≥ min_run días) para sombrear."""
    periods = []
    in_dd = False; start_d = None
    for _, row in df.iterrows():
        if row["move_russia"] == "D" and row["move_eu"] == "D":
            if not in_dd:
                in_dd = True; start_d = row["Date"]
        else:
            if in_dd:
                dur = (row["Date"] - start_d).days
                if dur >= min_run:
                    periods.append((start_d, row["Date"]))
                in_dd = False
    if in_dd and start_d is not None:
        periods.append((start_d, df["Date"].iloc[-1]))
    return periods


def fig_cooperation_rate(df: pd.DataFrame) -> go.Figure:
    """Tasa CC y DD en ventana 30d, con eventos."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["coop_30d"],
        name="% CC (30d)", mode="lines",
        line=dict(color=_GRN, width=2),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        hovertemplate="CC: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["dd_30d"],
        name="% DD / Crisis (30d)", mode="lines",
        line=dict(color="#a78bfa", width=1.5, dash="dot"),
        hovertemplate="DD: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dot",
                  line_color="rgba(255,255,255,0.18)", line_width=1)
    _add_event_lines(fig, df)
    fig.update_layout(
        **_LAYOUT, height=270,
        title_text="TASA DE COOPERACIÓN MUTUA (CC) vs CRISIS TOTAL (DD) — 30d rolling",
        title_font=dict(size=11),
    )
    fig.update_xaxes(gridcolor=_GRD, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRD, tickfont=dict(size=9),
                     title="% días", range=[0, 105])
    return fig


def fig_moves_timeline(df: pd.DataFrame) -> go.Figure:
    """Movidas diarias codificadas por color (Rusia arriba, UE abajo)."""
    fig = go.Figure()
    for move_col, y_val, label in [
        ("move_russia", 1.3, "Rusia"),
        ("move_eu",     0.5, "Europa"),
    ]:
        colors = [_GRN if m == "C" else _RU for m in df[move_col]]
        fig.add_trace(go.Scatter(
            x=df["Date"], y=[y_val] * len(df),
            mode="markers",
            marker=dict(color=colors, size=3.5, symbol="square", opacity=0.85),
            name=f"Movida {label}  (🟢=C  🔴=D)",
            hovertemplate=f"{label}: %{{customdata}}<br>%{{x|%d %b %Y}}<extra></extra>",
            customdata=df[move_col],
        ))
    _add_event_lines(fig, df)
    fig.update_layout(
        **_LAYOUT, height=200,
        title_text="MOVIDAS DIARIAS — RUSIA (arriba) · EUROPA (abajo)",
        title_font=dict(size=11),
    )
    fig.update_xaxes(gridcolor=_GRD, tickfont=dict(size=9))
    fig.update_yaxes(tickvals=[0.5, 1.3], ticktext=["Europa", "Rusia"],
                     range=[0.1, 1.7], gridcolor=_GRD, tickfont=dict(size=9))
    return fig


def fig_strategy_detection(detection: dict) -> go.Figure:
    """Barras horizontales de similitud con las 15 estrategias."""
    if not detection["scores"]:
        return go.Figure()
    items  = sorted(detection["scores"].items(), key=lambda x: x[1], reverse=True)
    names  = [x[0] for x in items]
    values = [x[1] for x in items]
    top3   = {x[0] for x in detection["top3"]}
    colors = [
        _GAS      if names[0] == n else
        _EU       if n in top3     else
        "rgba(100,116,139,0.45)"
        for n in names
    ]
    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.05)", width=0.4)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=9, color="#94a3b8"),
        hovertemplate="<b>%{y}</b><br>Similitud: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dot",
                  line_color="rgba(255,255,255,0.18)", line_width=1,
                  annotation_text="Azar 50%",
                  annotation_font=dict(size=8, color="#64748b"))
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_SRF,
        font=dict(family="monospace", color="#e2e8f0", size=10),
        height=max(300, len(names) * 28 + 80),
        title=dict(
            text="SIMILITUD DEL COMPORTAMIENTO HISTÓRICO DE RUSIA CON LAS 15 ESTRATEGIAS",
            font=dict(size=11), x=0,
        ),
        xaxis=dict(range=[0, 115], gridcolor=_GRD, tickfont=dict(size=9),
                   title="% coincidencia de movidas"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        margin=dict(l=12, r=85, t=48, b=12),
    )
    return fig


def fig_axelrod_comparison(df_hist: pd.DataFrame, df_sim: pd.DataFrame,
                            label: str) -> go.Figure:
    """Comparativa bienestar acumulado: Comportamiento Real vs Estrategia simulada."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "BIENESTAR RUSIA — Real vs Simulado",
            "BIENESTAR EUROPA — Real vs Simulado",
        ],
    )
    for ci, (player, col, ch, cw) in enumerate([
        ("Rusia",  "cum_ru", _RU,  "#fca5a5"),
        ("Europa", "cum_eu", _EU,  "#7dd3fc"),
    ], 1):
        fig.add_trace(go.Scatter(
            x=df_hist["Date"], y=df_hist[col],
            name=f"{player} — Comportamiento Real",
            mode="lines", line=dict(color=ch, width=2.2),
            hovertemplate=f"{player} real: %{{y:,.0f}}<extra></extra>",
        ), row=1, col=ci)
        fig.add_trace(go.Scatter(
            x=df_sim["Date"], y=df_sim[col],
            name=f"{player} — {label}",
            mode="lines", line=dict(color=cw, width=2.2, dash="dash"),
            fill="tonexty",
            fillcolor=f"rgba({'248,113,113' if ci==1 else '56,189,248'},0.07)",
            hovertemplate=f"{player} {label}: %{{y:,.0f}}<extra></extra>",
        ), row=1, col=ci)
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_SRF,
        font=dict(family="monospace", color="#e2e8f0", size=10),
        height=320, hovermode="x unified",
        title=dict(
            text=f"ANÁLISIS AXELROD — Comportamiento Real vs «{label}»",
            font=dict(size=11), x=0,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", borderwidth=0, font=dict(size=9)),
        margin=dict(l=12, r=12, t=48, b=12),
    )
    fig.update_xaxes(gridcolor=_GRD, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRD, tickfont=dict(size=9))
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  TAB PRINCIPAL — render_energy_crisis_tab()
# ══════════════════════════════════════════════════════════════════════════


def fig_inflation_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Inflación YoY% directa (sin rolling, sin ajuste DP).
    UE: Eurostat HICP prc_hicp_minr — interpolado mensual → diario
    Rusia: Banco Mundial FP.CPI.TOTL.ZG — interpolado anual → diario
    2025-2026: proyecciones FMI WEO / CBR.
    """
    # Cargar los puntos de anclaje reales (mismos que load_macro_data)
    macro = load_macro_data()

    fig = go.Figure()

    # ── Inflación UE — datos directos del CSV macro ───────────
    fig.add_trace(go.Scatter(
        x=macro["Date"], y=macro["INFL_EU"],
        name="Inflación UE (Eurostat HICP)", mode="lines",
        line=dict(color=_EU, width=2.2),
        hovertemplate="%{x|%b %Y}<br>UE: <b>%{y:.1f}%</b><extra></extra>",
    ))

    # ── Inflación Rusia — datos directos del CSV macro ────────
    fig.add_trace(go.Scatter(
        x=macro["Date"], y=macro["INFL_RU"],
        name="Inflación Rusia (Banco Mundial)", mode="lines",
        line=dict(color=_RU, width=2.2),
        hovertemplate="%{x|%b %Y}<br>Rusia: <b>%{y:.1f}%</b><extra></extra>",
    ))

    # ── Objetivo BCE ──────────────────────────────────────────
    fig.add_hline(y=2.0, line_dash="dot",
                  line_color="rgba(52,211,153,0.5)", line_width=1.5,
                  annotation_text="Objetivo BCE 2%",
                  annotation_font=dict(size=8, color=_GRN),
                  annotation_position="bottom right")

    # ── Anotación pico UE ────────────────────────────────────
    peak_eu_idx = macro["INFL_EU"].idxmax()
    peak_eu = macro.loc[peak_eu_idx]
    fig.add_annotation(
        x=peak_eu["Date"], y=float(peak_eu["INFL_EU"]),
        text=f"Pico UE: {peak_eu['INFL_EU']:.1f}% (oct 2022)",
        showarrow=True, arrowhead=2, ay=-32,
        font=dict(size=8, color=_EU),
        arrowcolor=_EU, bgcolor="rgba(0,0,0,0.65)",
    )

    # ── Anotación pico Rusia ─────────────────────────────────
    peak_ru_idx = macro["INFL_RU"].idxmax()
    peak_ru = macro.loc[peak_ru_idx]
    fig.add_annotation(
        x=peak_ru["Date"], y=float(peak_ru["INFL_RU"]),
        text=f"Pico Rusia: {peak_ru['INFL_RU']:.1f}% (2022)",
        showarrow=True, arrowhead=2, ay=-32,
        font=dict(size=8, color=_RU),
        arrowcolor=_RU, bgcolor="rgba(0,0,0,0.65)",
    )

    # ── Sombrear zona de convergencia ────────────────────────
    fig.add_vrect(
        x0=pd.Timestamp("2022-08-01"), x1=pd.Timestamp("2023-02-01"),
        fillcolor="rgba(167,139,250,0.07)", line_width=0,
        annotation_text="Convergencia", annotation_position="top left",
        annotation_font=dict(size=7, color="rgba(167,139,250,0.6)"),
    )

    _add_event_lines(fig, macro)

    fig.update_layout(
        **_LAYOUT, height=360,
        title_text=(
            "INFLACIÓN UE vs RUSIA — Tasa anual YoY %  ·  "
            "Fuentes: Eurostat HICP prc_hicp_minr · Banco Mundial FP.CPI.TOTL.ZG"
        ),
        title_font=dict(size=11),
        yaxis=dict(
            gridcolor=_GRD, tickfont=dict(size=9),
            title="Inflación YoY (%)", ticksuffix="%",
            rangemode="tozero",
        ),
        xaxis=dict(gridcolor=_GRD, tickfont=dict(size=9)),
    )
    return fig



def fig_tft_vs_historical(df_hist: pd.DataFrame, df_tft: pd.DataFrame) -> go.Figure:
    """
    Comparativa PIB e Inflación: comportamiento histórico vs Rusia TFT puro.
    Valida empíricamente la tesis de Axelrod & Hamilton (1981):
    TFT maximiza el bienestar mutuo en horizontes largos.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "BIENESTAR ACUMULADO — Rusia",
            "BIENESTAR ACUMULADO — UE",
            "Δ BIENESTAR: TFT vs Histórico",
            "% COOPERACIÓN MUTUA (30d)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Fila 1: Bienestar acumulado ──────────────────────────
    for ci, (player, col, ch, ct) in enumerate([
        ("Rusia", "cum_ru", _RU, "#fca5a5"),
        ("UE",    "cum_eu", _EU, "#7dd3fc"),
    ], 1):
        fig.add_trace(go.Scatter(
            x=df_hist["Date"], y=df_hist[col],
            name=f"{player} — Histórico",
            mode="lines", line=dict(color=ch, width=2.2),
            hovertemplate=f"{player} hist: %{{y:,.0f}}<extra></extra>",
            legendgroup="hist", showlegend=(ci==1),
        ), row=1, col=ci)
        fig.add_trace(go.Scatter(
            x=df_tft["Date"], y=df_tft[col],
            name=f"{player} — Rusia TFT",
            mode="lines", line=dict(color=ct, width=2.2, dash="dash"),
            fill="tonexty",
            fillcolor=f"rgba({'248,113,113' if ci==1 else '56,189,248'},0.08)",
            hovertemplate=f"{player} TFT: %{{y:,.0f}}<extra></extra>",
            legendgroup="tft", showlegend=(ci==1),
        ), row=1, col=ci)

    # ── Fila 2 izq: Score diario (flujo de pagos) ───────────
    # El score diario SÍ cambia: DC da T=5 (histórico) vs CC da R=3 (TFT)
    # Mostrar la diferencia diaria acumulada revela el trade-off
    df_hist_copy = df_hist.copy()
    df_tft_copy  = df_tft.copy()
    diff_ru = df_tft_copy["cum_ru"] - df_hist_copy["cum_ru"]
    diff_eu = df_tft_copy["cum_eu"] - df_hist_copy["cum_eu"]

    fig.add_trace(go.Scatter(
        x=df_hist["Date"], y=diff_ru,
        name="Δ Bienestar Rusia (TFT − Histórico)",
        mode="lines", line=dict(color=_RU, width=2),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.08)",
        hovertemplate="Δ Rusia: %{y:+,.0f} pts<extra></extra>",
        legendgroup="diff", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_hist["Date"], y=diff_eu,
        name="Δ Bienestar UE (TFT − Histórico)",
        mode="lines", line=dict(color=_EU, width=2),
        hovertemplate="Δ UE: %{y:+,.0f} pts<extra></extra>",
        legendgroup="diff", showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.25)", line_width=1.5,
                  row=2, col=1)
    fig.add_annotation(
        x=0.01, y=0.18, xref="paper", yref="paper",
        text="🔴 Rusia pierde · 🔵 UE gana",
        font=dict(size=8, color="#94a3b8"),
        showarrow=False, bgcolor="rgba(0,0,0,0.4)",
    )

    # ── Fila 2 der: % cooperación mutua — colores distintos ──
    fig.add_trace(go.Scatter(
        x=df_hist["Date"], y=df_hist["coop_30d"],
        name="CC% — Histórico",
        mode="lines", line=dict(color=_RU, width=2),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.07)",
        hovertemplate="CC histórico: %{y:.1f}%<extra></extra>",
        legendgroup="hist", showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=df_tft["Date"], y=df_tft["coop_30d"],
        name="CC% — Rusia TFT",
        mode="lines", line=dict(color=_GRN, width=2),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        hovertemplate="CC con TFT: %{y:.1f}%<extra></extra>",
        legendgroup="tft", showlegend=False,
    ), row=2, col=2)
    fig.add_hline(y=50, line_dash="dot",
                  line_color="rgba(255,255,255,0.18)", row=2, col=2)
    fig.add_annotation(
        x=0.99, y=0.18, xref="paper", yref="paper",
        text="🔴 Histórico · 🟢 Con TFT",
        font=dict(size=8, color="#94a3b8"),
        showarrow=False, bgcolor="rgba(0,0,0,0.4)",
        xanchor="right",
    )

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_SRF,
        font=dict(family="monospace", color="#e2e8f0", size=10),
        height=500,
        title=dict(
            text="VALIDACIÓN AXELROD — Comportamiento Histórico vs TFT Puro",
            font=dict(size=12), x=0,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)", borderwidth=0,
            font=dict(size=9), orientation="h",
            y=-0.08, x=0,
        ),
        hovermode="x unified",
        margin=dict(l=12, r=12, t=52, b=60),
    )
    fig.update_xaxes(gridcolor=_GRD, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRD, tickfont=dict(size=9))
    return fig


def render_energy_crisis_tab():
    """
    Dashboard interactivo de Teoría de Juegos Geopolítica.
    Crisis Energética Rusia–Europa (2021–2026).
    """
    # ════════════════════════════════════════════════════════
    # SIDEBAR — todos los controles de simulación
    # ════════════════════════════════════════════════════════
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### ⚡ Crisis Energética\n"
        "<span style='font-size:10px;color:#64748b;letter-spacing:.1em;'>"
        "CONTROLES DE SIMULACIÓN</span>",
        unsafe_allow_html=True,
    )

    # Cargar datos
    with st.spinner("Cargando datos de energía…"):
        df_daily_full = load_daily_data()
        df_stor_full  = load_storage_data()
        df_route      = load_route_data()

    min_date = df_daily_full["Date"].min().date()
    max_date = df_daily_full["Date"].max().date()

    # 1. Rango de fechas
    st.sidebar.markdown("**📅 Período de simulación**")
    date_range = st.sidebar.slider(
        "Rango", min_value=min_date, max_value=max_date,
        value=(min_date, max_date), format="MMM YYYY", key="ec_date_range",
    )
    date_start = pd.Timestamp(date_range[0])
    date_end   = pd.Timestamp(date_range[1])

    # 2. Matriz de pagos — valores canonicos Axelrod (1981), fijos
    ec_T, ec_R, ec_P, ec_S = 5.0, 3.0, 1.0, 0.0
    valid = True
    st.sidebar.markdown("**💰 Matriz de Pagos — Axelrod (1981)**")
    st.sidebar.info(
        "**T=5 · R=3 · P=1 · S=0** (fijos) \n\n"
        "Valores canónicos del paper original. Garantizan T > R > P ≥ S "
        "y w crítico = 0.5, condición de estabilidad evolutiva de TFT. \n"
        "Axelrod & Hamilton (1981). Science, 211(4489), 1390-1396."
    )

    # 3. Umbral fijo 250 mcm — justificación empírica
    flow_thresh = 250
    st.sidebar.markdown("**🎚️ Umbral de cooperación Rusia**")
    st.sidebar.info(
        "**Fijo: 250 mcm/día** \n\n"
        "El flujo mínimo histórico durante el período de cooperación normal "
        "(2021 – feb 2022) fue exactamente 250.4 mcm/día — piso observado antes "
        "de cualquier corte deliberado. Todo valor por debajo representa una "
        "reducción fuera del rango contractual habitual. "
        "Fuente: ENTSOG / daily\_data\_2026-03-12.csv"
    )

    # 4. Estrategia de Europa
    st.sidebar.markdown("**🇪🇺 Estrategia Europa**")
    eu_options = [
        "Realista/Sanciones",
        "Pacificadora",
        "Tit-for-Tat (clásico)",
    ] + STRATEGY_NAMES
    eu_strategy = st.sidebar.selectbox(
        "Comportamiento de Europa",
        eu_options, index=0, key="ec_eu_strat",
        help=(
            "Realista/Sanciones: D tras Feb-2022 si LNG sube ≥15 %, D desde 2025.\n"
            "Pacificadora: siempre C.\n"
            "Tit-for-Tat: copia la última movida de Rusia."
        ),
    )
    _eu_desc = {
        "Realista/Sanciones":   "Sanciones + pivot GNL + independencia 2025",
        "Pacificadora":         "Coopera siempre — sin sanciones",
        "Tit-for-Tat (clásico)":"Copia la última movida de Rusia",
    }
    if eu_strategy in _eu_desc:
        st.sidebar.caption(_eu_desc[eu_strategy])

    # 5. Estrategia de Rusia (simulación)
    st.sidebar.markdown("**🇷🇺 Estrategia Rusia**")
    ru_strategy = st.sidebar.selectbox(
        "Comportamiento de Rusia",
        ["Datos Históricos", "Tit-for-Tat", "Always Defect", "Bully"],
        index=0, key="ec_ru_strat",
    )
    _ru_desc = {
        "Tit-for-Tat":   "Coopera primero · copia la UE",
        "Always Defect": "Defecta siempre",
        "Bully":         "Defecta primero · explota si la UE cede",
    }
    if ru_strategy in _ru_desc:
        st.sidebar.caption(_ru_desc[ru_strategy])

    # ════════════════════════════════════════════════════════
    # HEADER DEL TAB
    # ════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(248,113,113,0.07),
                rgba(56,189,248,0.05));border:1px solid rgba(248,113,113,0.18);
                border-radius:6px;padding:16px 20px;margin-bottom:14px;">
      <p style="font-size:10px;letter-spacing:.15em;color:#f87171;
                text-transform:uppercase;margin:0 0 3px 0;">
        ● SIMULADOR DE TEORÍA DE JUEGOS GEOPOLÍTICA
      </p>
      <p style="font-size:18px;font-weight:700;color:#e2e8f0;margin:0 0 2px 0;">
        Crisis Energética Rusia–Europa (2021–2026)
      </p>
      <p style="font-size:12px;color:#64748b;margin:0;">
        Rusia: D si flujo &lt; {thresh} mcm/día o caída ≥20 % mensual.
        Europa: D si post-Feb-2022 + GNL sube ≥15 %  |  D desde Ene-2025.
        Mueve los sliders del sidebar — el simulador se recalcula al instante.
      </p>
    </div>
    """.format(thresh=flow_thresh), unsafe_allow_html=True)

    if not valid:
        st.error("Corrige la matriz de pagos.")
        return

    # ════════════════════════════════════════════════════════
    # SIMULACIÓN
    # ════════════════════════════════════════════════════════
    with st.spinner("Simulando…"):
        df_sim = simulate_ipd(
            df_daily_full, df_stor_full,
            ec_T, ec_R, ec_P, ec_S,
            flow_thresh, eu_strategy,
            date_start, date_end,
            ru_strategy=ru_strategy,
        )
        # Baseline histórico para comparaciones
        df_hist = simulate_ipd(
            df_daily_full, df_stor_full,
            ec_T, ec_R, ec_P, ec_S,
            flow_thresh, "Realista/Sanciones",
            date_start, date_end,
            ru_strategy="Datos Históricos",
        )

    if df_sim.empty:
        st.warning("Sin datos para el período seleccionado.")
        return

    # ════════════════════════════════════════════════════════
    # SECCIÓN A: w CRÍTICO
    # ════════════════════════════════════════════════════════
    w_critical = (ec_T - ec_R) / (ec_T - ec_P) if (ec_T - ec_P) != 0 else 0.5
    if not df_stor_full.empty:
        _stor_f = df_stor_full.copy()
        _stor_f["_chg"] = _stor_f["Storage_pct"].pct_change() * 100
        vol    = _stor_f["_chg"].std()
        w_real = float(np.clip(1 - vol / 100, 0.5, 0.999))
    else:
        w_real = 0.97

    with st.expander("🎯 Parámetro w — Sombra del Futuro (Axelrod)", expanded=False):
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("w crítico (Axelrod)",
                   f"{w_critical:.4f}",
                   help="(T−R)/(T−P)")
        wc2.metric("w real (volatilidad almac.)",
                   f"{w_real:.4f}",
                   delta=f"{'✅ >' if w_real >= w_critical else '❌ <'} w crítico")
        wc3.markdown(
            f"**Fórmula:** `w > (T−R)/(T−P) = {w_critical:.4f}`  \n"
            f"{'🟢 Cooperación evolutivamente estable' if w_real >= w_critical else '🔴 Cooperación inestable'}"
        )

    # ════════════════════════════════════════════════════════
    # SECCIÓN B: MÉTRICAS DE BIENESTAR ACUMULADO (st.columns)
    # ════════════════════════════════════════════════════════
    st.markdown("---")
    _hp = [f"### 📊 {date_range[0].strftime('%b %Y')} → {date_range[1].strftime('%b %Y')}"]
    if ru_strategy != "Datos Históricos":
        _hp.append(f"  ·  🇷🇺 **{ru_strategy}**")
    if eu_strategy != "Realista/Sanciones":
        _hp.append(f"  ·  🇪🇺 **{eu_strategy}**")
    st.markdown("".join(_hp))

    coop_ru  = (df_sim["move_russia"] == "C").mean() * 100
    coop_eu  = (df_sim["move_eu"]     == "C").mean() * 100
    cc_pct   = ((df_sim["move_russia"]=="C") & (df_sim["move_eu"]=="C")).mean() * 100
    dd_pct   = ((df_sim["move_russia"]=="D") & (df_sim["move_eu"]=="D")).mean() * 100
    final_ru = df_sim["cum_ru"].iloc[-1]
    final_eu = df_sim["cum_eu"].iloc[-1]
    avg_inf  = df_sim["inf_eu_30d"].mean()

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("🇷🇺 Bienestar Rusia",     f"{final_ru:,.0f} pts")
    mc2.metric("🇪🇺 Bienestar Europa",    f"{final_eu:,.0f} pts")
    mc3.metric("🤝 Coop. mutua (CC)",     f"{cc_pct:.1f}%")
    mc4.metric("💥 Crisis total (DD)",    f"{dd_pct:.1f}%")
    mc5.metric("🇷🇺 % días Rusia coop.", f"{coop_ru:.1f}%")
    mc6.metric("📈 Inflación media UE",   f"{avg_inf:.1f}%")

    # Deltas vs baseline si hay simulación alternativa
    if ru_strategy != "Datos Históricos" or eu_strategy != "Realista/Sanciones":
        d_ru = final_ru - df_hist["cum_ru"].iloc[-1]
        d_eu = final_eu - df_hist["cum_eu"].iloc[-1]
        da1, da2 = st.columns(2)
        da1.metric("Δ Bienestar Rusia vs baseline",
                   f"{d_ru:+,.0f} pts",
                   delta_color="normal" if d_ru >= 0 else "inverse")
        da2.metric("Δ Bienestar Europa vs baseline",
                   f"{d_eu:+,.0f} pts",
                   delta_color="normal" if d_eu >= 0 else "inverse")

    # ════════════════════════════════════════════════════════
    # SECCIÓN C: GRÁFICOS INTERACTIVOS
    # ════════════════════════════════════════════════════════
    st.markdown("---")

    # Gas flow
    st.plotly_chart(fig_gas_flow(df_sim, flow_thresh), use_container_width=True)

    # Movidas diarias
    st.plotly_chart(fig_moves_timeline(df_sim), use_container_width=True)

    # Bienestar + cooperación
    col_w, col_c = st.columns([1.6, 1.0])
    with col_w:
        st.plotly_chart(fig_welfare_accumulation(df_sim), use_container_width=True)
    with col_c:
        st.plotly_chart(fig_cooperation_rate(df_sim), use_container_width=True)

    # Impacto económico (con volatilidad DD)
    st.plotly_chart(fig_economic_impact(df_sim, ec_T), use_container_width=True)

    # Inflación comparada UE vs Rusia
    st.plotly_chart(fig_inflation_comparison(df_sim), use_container_width=True)

    # Comparación Axelrod si hay simulación alternativa
    if ru_strategy != "Datos Históricos" or eu_strategy != "Realista/Sanciones":
        _label = f"Rusia:{ru_strategy} / UE:{eu_strategy}"
        st.plotly_chart(
            fig_axelrod_comparison(df_hist, df_sim, _label),
            use_container_width=True,
        )

    # ════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════
    # SECCIÓN AXELROD-HAMILTON: VALIDACIÓN EMPÍRICA
    # ════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📐 Validación Axelrod & Hamilton (1981) — ¿Qué hubiera pasado con TFT?")

    # Banner de la tesis central
    st.markdown("""
    <div style="background:rgba(14,20,45,0.8);border:1px solid rgba(56,189,248,0.20);
                border-radius:6px;padding:14px 20px;margin-bottom:14px;">
      <p style="font-size:9px;letter-spacing:.14em;color:#475569;
                text-transform:uppercase;margin:0 0 6px 0;">
        Axelrod & Hamilton · <em>The Evolution of Cooperation</em> · Science, 1981
      </p>
      <p style="font-size:13px;color:#e2e8f0;margin:0 0 6px 0;">
        <b>"En juegos repetidos con horizonte suficientemente largo (w alto),
        TIT FOR TAT es la estrategia evolutivamente estable:
        cooperar primero, castigar la traición inmediatamente,
        perdonar tras la represalia."</b>
      </p>
      <p style="font-size:11px;color:#64748b;margin:0;">
        Si Rusia hubiera seguido TFT desde 2021 — cooperando inicialmente
        y respondiendo de forma proporcional a las sanciones en lugar de
        escalar unilateralmente los cortes — ¿habrían sido mejores los
        resultados económicos para ambas partes?
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Simulando escenario TFT…"):
        df_tft = simulate_ipd(
            df_daily_full, df_stor_full,
            ec_T, ec_R, ec_P, ec_S,
            flow_thresh, eu_strategy,
            date_start, date_end,
            ru_strategy="Tit-for-Tat",
        )

    if not df_tft.empty:
        # Métricas comparativas
        tft_cc  = ((df_tft["move_russia"]=="C") & (df_tft["move_eu"]=="C")).mean() * 100
        hist_cc = ((df_sim["move_russia"]=="C") & (df_sim["move_eu"]=="C")).mean() * 100
        tft_ru  = df_tft["cum_ru"].iloc[-1]
        hist_ru = df_sim["cum_ru"].iloc[-1]
        tft_eu  = df_tft["cum_eu"].iloc[-1]
        hist_eu = df_sim["cum_eu"].iloc[-1]

        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("🤝 CC% con TFT",
                   f"{tft_cc:.1f}%",
                   delta=f"{tft_cc - hist_cc:+.1f}pp vs histórico",
                   delta_color="normal")
        vm2.metric("🇷🇺 Bienestar Rusia TFT",
                   f"{tft_ru:,.0f} pts",
                   delta=f"{tft_ru - hist_ru:+,.0f} vs histórico",
                   delta_color="normal")
        vm3.metric("🇪🇺 Bienestar UE TFT",
                   f"{tft_eu:,.0f} pts",
                   delta=f"{tft_eu - hist_eu:+,.0f} vs histórico",
                   delta_color="normal")
        vm4.metric("📊 Bienestar total TFT",
                   f"{tft_ru + tft_eu:,.0f} pts",
                   delta=f"{(tft_ru+tft_eu)-(hist_ru+hist_eu):+,.0f} vs histórico",
                   delta_color="normal")

        st.plotly_chart(
            fig_tft_vs_historical(df_sim, df_tft),
            use_container_width=True,
        )

        # Interpretación académica
        total_gain = (tft_ru + tft_eu) - (hist_ru + hist_eu)
        winner = "ambas partes" if (tft_ru > hist_ru and tft_eu > hist_eu)                  else ("Rusia" if tft_ru > hist_ru else "la UE")
        with st.expander("📖 Interpretación — conexión con Axelrod & Hamilton (1981)", expanded=True):
            st.markdown(f"""
            **¿Qué dice el modelo?**

            Con TFT, la tasa de cooperación mutua (CC) sería de **{tft_cc:.1f}%**
            frente al **{hist_cc:.1f}%** histórico — una diferencia de
            **{tft_cc - hist_cc:+.1f} puntos porcentuales**.
            El bienestar conjunto {'**aumentaría**' if total_gain > 0 else '**disminuiría**'}
            en **{abs(total_gain):,.0f} puntos**, beneficiando principalmente a {winner}.

            **Conexión con Axelrod & Hamilton (1981):**

            El artículo demuestra que TFT gana los torneos de DP iterado por tres razones:
            1. **Amabilidad:** coopera primero — evita conflictos innecesarios
            2. **Provocabilidad:** castiga de inmediato cualquier defección
            3. **Perdón:** vuelve a cooperar tras castigar — no escala indefinidamente

            El comportamiento histórico de Rusia ({'se acerca' if hist_cc > 20 else 'se aleja'}
            del TFT según el identificador de estrategia) muestra que la escalada unilateral
            de cortes — sin señales de disposición a cooperar — viola el principio 3
            (perdón), llevando al sistema hacia el equilibrio de Nash subóptimo (DD)
            en lugar del óptimo de Pareto (CC).

            **Implicación de política:** si los actores geopolíticos siguieran TFT,
            el modelo predice {'mayor estabilidad energética y mejor desempeño económico' 
            if total_gain > 0 else 'resultados similares a los históricos'},
            lo que es consistente con la predicción central de Axelrod & Hamilton (1981)
            sobre la evolución de la cooperación en sistemas con sombra del futuro alta.
            """)

    # SECCIÓN D: IDENTIFICADOR DE ESTRATEGIA + ANÁLISIS AXELROD
    # ════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 Análisis Axelrod — Comportamiento Real vs Estrategia más parecida")
    st.caption(
        "Compara la secuencia histórica de movidas de Rusia con las 15 estrategias "
        "del catálogo y muestra cuál se parece más."
    )

    with st.spinner("Analizando…"):
        detection = identify_strategy(
            df_daily_full, df_stor_full,
            flow_thresh=flow_thresh,
            date_start=date_start,
            date_end=date_end,
        )

    if detection["n_days"] > 0:
        det1, det2, det3 = st.columns(3)
        for _col, (_sn, _pct), _lbl in zip(
            [det1, det2, det3],
            detection["top3"],
            ["🥇 Más parecida", "🥈 Segunda", "🥉 Tercera"],
        ):
            _col.metric(_lbl, _sn, f"{_pct:.1f}% coincidencia")

        _bn, _bp = detection["best"], detection["best_pct"]
        _bc   = _GAS if _bp >= 70 else (_EU if _bp >= 55 else "#94a3b8")
        _conf = "Alta ✅" if _bp >= 70 else ("Media ⚠️" if _bp >= 55 else "Baja ❌")

        # Cuadro comparativo Comportamiento Real vs Axelrod
        st.markdown(f"""
        <div style="background:rgba(251,191,36,0.07);
                    border:1px solid rgba(251,191,36,0.22);
                    border-radius:6px;padding:14px 20px;margin:10px 0;">
          <div style="display:flex;gap:40px;align-items:center;">
            <div>
              <p style="font-size:9px;letter-spacing:.12em;color:#64748b;
                        text-transform:uppercase;margin:0 0 4px 0;">Comportamiento Real</p>
              <p style="font-size:15px;font-weight:700;color:#e2e8f0;margin:0;">
                Rusia — datos CSV
              </p>
              <p style="font-size:11px;color:#64748b;margin:3px 0 0 0;">
                {coop_ru:.1f}% cooperación · {detection["n_days"]:,} días analizados
              </p>
            </div>
            <div style="font-size:20px;color:#64748b;">≈</div>
            <div>
              <p style="font-size:9px;letter-spacing:.12em;color:#64748b;
                        text-transform:uppercase;margin:0 0 4px 0;">
                Estrategia Axelrod más parecida
              </p>
              <p style="font-size:15px;font-weight:700;color:{_bc};margin:0;">
                {_bn}
              </p>
              <p style="font-size:11px;color:#64748b;margin:3px 0 0 0;">
                {_bp:.1f}% coincidencia · Confianza: {_conf}
              </p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_strategy_detection(detection), use_container_width=True)

        with st.expander("ℹ️ Metodología de detección", expanded=False):
            st.markdown("""
            1. Se extrae la secuencia histórica de Rusia: **C** si flujo > umbral y sin caída ≥20 %, **D** en caso contrario.
            2. Cada una de las 15 estrategias del catálogo juega contra la señal histórica de la UE (almacenamiento semanal).
            3. Se calcula qué porcentaje de días la estrategia habría tomado la **misma decisión** que Rusia.
            4. Alta confianza ≥ 70 % · Media 55–70 % · Baja < 55 %.
            > Similitud de movidas ≠ intención estratégica. Alta coincidencia con ALL-D en períodos de cortes puede reflejar restricciones técnicas o políticas, no una estrategia deliberada.
            """)

    # ════════════════════════════════════════════════════════
    # SECCIÓN E: DESCARGA
    # ════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("📋 Matriz de impacto económico actual", expanded=False):
        _rows = []
        _t_f  = ec_T / 5.0
        for (mr, me), (lbl) in [
            (("C","C"), "Cooperación Mutua"),
            (("D","C"), "Rusia defecta — precio alto"),
            (("C","D"), "UE responde — sanciones/GNL"),
            (("D","D"), "Castigo mutuo — crisis total"),
        ]:
            _pr = {"CC":ec_R,"DC":ec_T,"CD":ec_S,"DD":ec_P}[mr+me]
            _pe = {"CC":ec_R,"DC":ec_S,"CD":ec_T,"DD":ec_P}[mr+me]
            _rows.append({"Rusia":mr,"Europa":me,"Resultado":lbl,
                          "Score Ru":_pr,"Score EU":_pe,
                          "PIB Ru %": round( 2.0 if mr+me=="CC" else
                                             4.5*_t_f if mr+me=="DC" else
                                             -3.5 if mr+me=="CD" else -2.5*_t_f, 1),
                          "PIB UE %": round( 2.0 if mr+me=="CC" else
                                             -4.0*_t_f if mr+me=="DC" else
                                             0.5 if mr+me=="CD" else -2.5*_t_f, 1),
                          "Inflac. UE %": round(2.5 if mr+me=="CC" else
                                                15*_t_f if mr+me=="DC" else
                                                6.0 if mr+me=="CD" else 8*_t_f, 1)})
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

    import io as _io
    _buf = _io.StringIO()
    _dl_cols = ["Date","Russia","LNG","move_russia","move_eu","outcome",
                "score_ru","score_eu","cum_ru","cum_eu",
                "gdp_eu_30d","gdp_ru_30d","inf_eu_30d","inf_eu_7d","inf_ru_30d","coop_30d","dd_30d"]
    df_sim[[c for c in _dl_cols if c in df_sim.columns]].to_csv(_buf, index=False)
    st.download_button(
        "⬇ Descargar simulación (CSV)",
        data=_buf.getvalue(),
        file_name=f"ipd_energia_{date_range[0]}_{date_range[1]}.csv",
        mime="text/csv",
    )

    st.markdown("""
    <p style="font-size:11px;color:#475569;margin-top:14px;line-height:1.6;">
    <b style="color:#94a3b8">Fuentes macro:</b>
    PIB UE: Eurostat namq_10_gdp (trimestral interanual %) · dato real.<br>
    PIB Rusia: Banco Mundial NY.GDP.MKTP.KD.ZG (anual) · 2025-26: FMI WEO.<br>
    Inflación UE: Eurostat HICP prc_hicp_minr (mensual YoY %) · dato real.<br>
    Inflación Rusia: Banco Mundial FP.CPI.TOTL.ZG (anual) · 2025-26: CBR.<br>
    <b style="color:#94a3b8">Fuentes de gas:</b> ENTSOG, GIE, Bruegel, EC REPowerEU.<br>
    <b style="color:#94a3b8">Nota:</b> Impactos económicos son proxies ilustrativos escalados por la
    magnitud de T. No son estimaciones econométricas.
    </p>
    """, unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════
#  ANÁLISIS ESTADÍSTICO — ANOVA DE UNA VÍA
# ══════════════════════════════════════════════════════════════════════════

def compute_anova(per_game_scores: dict) -> dict:
    """
    ANOVA de una vía sobre los scores individuales por partida.

    H₀: Las medias de puntaje de todas las estrategias son iguales.
    H₁: Al menos una media difiere significativamente.

    Args:
        per_game_scores: dict {nombre_estrategia: [score_partida_1, ...]}

    Returns:
        dict con F, p_value, tabla ANOVA, resumen descriptivo e interpretación.
    """
    groups = list(per_game_scores.values())
    names  = list(per_game_scores.keys())

    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {"error": "Se necesitan al menos 2 estrategias con ≥2 partidas cada una."}

    F, p = _scipy_stats.f_oneway(*groups)

    all_data   = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    k = len(groups)
    N = len(all_data)

    SS_b = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups))
    SS_w = float(sum(np.sum((np.array(g) - np.mean(g)) ** 2) for g in groups))
    SS_t = SS_b + SS_w
    df_b, df_w = k - 1, N - k
    MS_b = SS_b / df_b if df_b > 0 else 0
    MS_w = SS_w / df_w if df_w > 0 else 0

    anova_table = pd.DataFrame([
        {"Fuente":    "Entre grupos (estrategias)",
         "SS":        round(SS_b, 2),
         "df":        df_b,
         "MS":        round(MS_b, 4),
         "F":         round(float(F), 4),
         "p-valor":   f"{float(p):.2e}"},
        {"Fuente":    "Dentro de grupos (error)",
         "SS":        round(SS_w, 2),
         "df":        df_w,
         "MS":        round(MS_w, 4),
         "F":         "—",
         "p-valor":   "—"},
        {"Fuente":    "Total",
         "SS":        round(SS_t, 2),
         "df":        N - 1,
         "MS":        "—",
         "F":         "—",
         "p-valor":   "—"},
    ])

    desc_rows = []
    for nm, g in zip(names, groups):
        a = np.array(g)
        desc_rows.append({
            "Estrategia": nm,
            "N":          len(a),
            "Media":      round(float(np.mean(a)), 3),
            "Desv. Std.": round(float(np.std(a, ddof=1)), 3),
            "Mín.":       round(float(np.min(a)), 3),
            "Máx.":       round(float(np.max(a)), 3),
        })
    desc_df = (pd.DataFrame(desc_rows)
               .sort_values("Media", ascending=False)
               .reset_index(drop=True))
    desc_df.index = desc_df.index + 1

    sig = bool(p < 0.05)
    return {
        "F":           float(F),
        "p":           float(p),
        "sig":         sig,
        "anova_table": anova_table,
        "desc_df":     desc_df,
        "interp": (
            "✅ Existen diferencias estadísticamente significativas entre las estrategias "
            f"(F = {F:.4f}, p = {p:.2e} < 0.05). Se rechaza H₀ con 95 % de confianza."
            if sig else
            "⚠️ No se detectan diferencias significativas entre las estrategias "
            f"(F = {F:.4f}, p = {p:.2e} ≥ 0.05). No hay evidencia suficiente para rechazar H₀."
        ),
        "n_strategies": k,
        "n_games_total": N,
    }


def run_tournament_with_anova(selected_names, T, R, P, S, w, games=5, rounds=200):
    """
    Extiende run_tournament() para capturar scores individuales por partida,
    necesarios para el análisis ANOVA.

    Returns:
        (payoff_matrix, ranking, h2h_data, per_game_scores)
    """
    name_to_class = {s.name: s for s in ALL_STRATEGIES}
    strategies    = {n: name_to_class[n] for n in selected_names}
    n             = len(selected_names)
    payoff_matrix = pd.DataFrame(np.zeros((n, n)),
                                 index=selected_names, columns=selected_names)
    total_scores  = {name: 0.0 for name in selected_names}
    h2h_data      = {}
    per_game      = {name: [] for name in selected_names}   # ← para ANOVA

    for name1, name2 in itertools.combinations(selected_names, 2):
        cls1 = strategies[name1]; cls2 = strategies[name2]
        gs1, gs2, lh1, lh2 = [], [], [], []
        for _ in range(games):
            sc1, sc2, h1, h2 = play_game(cls1(), cls2(), rounds, T, R, P, S, w)
            rlen = max(len(h1), 1)
            avg1 = sc1 / rlen; avg2 = sc2 / rlen
            gs1.append(avg1); gs2.append(avg2)
            per_game[name1].append(avg1)
            per_game[name2].append(avg2)
            lh1, lh2 = h1, h2
        m1, m2 = float(np.mean(gs1)), float(np.mean(gs2))
        payoff_matrix.loc[name1, name2] = m1
        payoff_matrix.loc[name2, name1] = m2
        total_scores[name1] += m1; total_scores[name2] += m2
        h2h_data[(name1, name2)] = (lh1, lh2)

    for name in selected_names:
        cls = strategies[name]
        sc_list = []
        for _ in range(games):
            sc1, sc2, _, _ = play_game(cls(), cls(), rounds, T, R, P, S, w)
            rlen = max(1, rounds)
            sc_list.append((sc1 + sc2) / 2 / rlen)
        payoff_matrix.loc[name, name] = float(np.mean(sc_list))

    ranking = (pd.DataFrame({"Estrategia": list(total_scores.keys()),
                              "Score Total": list(total_scores.values())})
               .sort_values("Score Total", ascending=False)
               .reset_index(drop=True))
    ranking.index += 1
    return payoff_matrix, ranking, h2h_data, per_game


def fig_anova_table(anova: dict) -> go.Figure:
    """Tabla ANOVA formateada como figura Plotly (tema oscuro)."""
    df = anova["anova_table"]
    sig_color = "#34d399" if anova["sig"] else "#fbbf24"

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df.columns],
            fill_color="rgba(14,20,45,0.95)",
            font=dict(color="#e2e8f0", size=11, family="monospace"),
            align="center", height=34,
            line=dict(color="rgba(56,189,248,0.18)", width=1),
        ),
        cells=dict(
            values=[df[c].tolist() for c in df.columns],
            fill_color="rgba(8,11,18,0.92)",
            font=dict(
                color=["#e2e8f0", "#e2e8f0", "#e2e8f0",
                       "#e2e8f0", sig_color, sig_color],
                size=11, family="monospace",
            ),
            align=["left", "right", "right", "right", "right", "right"],
            height=30,
            line=dict(color="rgba(255,255,255,0.05)", width=1),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#080b12",
        title=dict(
            text=(f"ANOVA DE UNA VÍA  ·  F = {anova['F']:.4f}  ·  "
                  f"p = {anova['p']:.2e}  ·  "
                  f"{'Significativo ✅' if anova['sig'] else 'No significativo ⚠️'}"),
            font=dict(size=12, color="#e2e8f0", family="monospace"),
            x=0,
        ),
        margin=dict(l=8, r=8, t=56, b=8),
        height=210,
    )
    return fig


def fig_desc_table(anova: dict) -> go.Figure:
    """Tabla descriptiva por estrategia (Media, Std, N) con ranking visual."""
    df = anova["desc_df"].copy()
    n  = len(df)
    row_fills = []
    for i in range(n):
        if i == 0:   row_fills.append("rgba(255,215,0,0.14)")
        elif i == 1: row_fills.append("rgba(192,192,192,0.10)")
        elif i == 2: row_fills.append("rgba(205,127,50,0.10)")
        else:        row_fills.append("rgba(8,11,18,0.88)")

    ranks = [f"#{i}" for i in df.index]
    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Rank</b>"] + [f"<b>{c}</b>" for c in df.columns],
            fill_color="rgba(14,20,45,0.95)",
            font=dict(color="#e2e8f0", size=11, family="monospace"),
            align="center", height=34,
            line=dict(color="rgba(56,189,248,0.18)", width=1),
        ),
        cells=dict(
            values=[ranks] + [df[c].tolist() for c in df.columns],
            fill_color=[row_fills] * (len(df.columns) + 1),
            font=dict(color="#e2e8f0", size=11, family="monospace"),
            align=["center", "left", "center", "center", "center", "center", "center"],
            height=28,
            line=dict(color="rgba(255,255,255,0.04)", width=1),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#080b12",
        title=dict(
            text="ESTADÍSTICAS DESCRIPTIVAS POR ESTRATEGIA (ordenadas por media)",
            font=dict(size=12, color="#e2e8f0", family="monospace"),
            x=0,
        ),
        margin=dict(l=8, r=8, t=52, b=8),
        height=max(260, n * 32 + 110),
    )
    return fig


def fig_scores_distribution(per_game: dict, selected: list) -> go.Figure:
    """Box plot de distribución de scores por partida para cada estrategia."""
    # Ordenar por mediana descendente
    order = sorted(selected, key=lambda n: np.median(per_game.get(n, [0])), reverse=True)
    colors = [
        "#fbbf24","#38bdf8","#34d399","#f87171","#a78bfa",
        "#2dd4bf","#f97316","#e879f9","#86efac","#fde68a",
        "#7dd3fc","#f9a8d4","#c4b5fd","#6ee7b7","#fed7aa",
    ]
    fig = go.Figure()
    for i, name in enumerate(order):
        vals = per_game.get(name, [])
        if not vals:
            continue
        fig.add_trace(go.Box(
            y=vals,
            name=name,
            marker_color=colors[i % len(colors)],
            line=dict(color=colors[i % len(colors)]),
            fillcolor=colors[i % len(colors)].replace("#", "rgba(").replace(
                "#", "") + "30" if False else
                f"rgba({int(colors[i%len(colors)][1:3],16)},"
                f"{int(colors[i%len(colors)][3:5],16)},"
                f"{int(colors[i%len(colors)][5:7],16)},0.18)",
            boxpoints="outliers",
            hovertemplate=f"<b>{name}</b><br>Score: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0e1420",
        font=dict(family="monospace", color="#e2e8f0", size=10),
        height=440,
        title=dict(
            text="DISTRIBUCIÓN DE SCORES POR PARTIDA — BOX PLOT",
            font=dict(size=12), x=0,
        ),
        xaxis=dict(tickangle=-28, gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=9)),
        yaxis=dict(title="Score por partida (normalizado)",
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9)),
        margin=dict(l=12, r=12, t=52, b=80),
        showlegend=False,
    )
    return fig


def tab_statistics():
    """Tab 7: Análisis estadístico ANOVA del torneo."""
    if "anova" not in st.session_state or st.session_state.anova is None:
        st.info(
            "👈 Ejecuta el torneo primero (Tab 🏆 Ranking) — "
            "el análisis ANOVA se calculará automáticamente."
        )
        return

    anova = st.session_state.anova

    if "error" in anova:
        st.error(f"⚠ {anova['error']}")
        return

    # Banner de resultado
    sig   = anova["sig"]
    bc    = "#34d399" if sig else "#fbbf24"
    st.markdown(f"""
    <div style="background:rgba(14,20,45,0.7);border:1px solid rgba(56,189,248,0.18);
                border-radius:6px;padding:14px 20px;margin-bottom:16px;">
      <p style="font-family:monospace;font-size:10px;letter-spacing:.12em;
                color:#64748b;text-transform:uppercase;margin:0 0 4px 0;">
        Análisis ANOVA de una vía
      </p>
      <p style="font-family:monospace;font-size:13px;color:#e2e8f0;margin:0 0 4px 0;">
        F&thinsp;=&thinsp;<b>{anova['F']:.4f}</b>
        &emsp;p&thinsp;=&thinsp;<b>{anova['p']:.2e}</b>
        &emsp;<span style="color:{bc};">
        {'Significativo ✅' if sig else 'No significativo ⚠️'}</span>
      </p>
      <p style="font-family:monospace;font-size:11px;color:#64748b;margin:0;">
        {anova['interp']}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Métricas rápidas
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Estrategias comparadas", anova["n_strategies"])
    mc2.metric("Partidas totales",       anova["n_games_total"])
    mc3.metric("F estadístico",          f"{anova['F']:.4f}")

    st.markdown("---")

    # Tabla ANOVA
    st.plotly_chart(fig_anova_table(anova), use_container_width=True)

    st.markdown("---")

    # Box plot distribución
    per_game = st.session_state.get("per_game", {})
    selected = list(anova["desc_df"]["Estrategia"].tolist())
    if per_game:
        st.plotly_chart(
            fig_scores_distribution(per_game, selected),
            use_container_width=True,
        )

    st.markdown("---")

    # Tabla descriptiva
    st.plotly_chart(fig_desc_table(anova), use_container_width=True)

    # Expander metodología
    with st.expander("ℹ️ Sobre el análisis ANOVA", expanded=False):
        st.markdown("""
        **¿Qué mide el ANOVA aquí?**
        Compara los scores promedio *por partida* (no el score total del torneo) entre todas las
        estrategias seleccionadas.

        **H₀ (hipótesis nula):** todas las estrategias obtienen el mismo score medio por partida.

        **H₁ (hipótesis alternativa):** al menos una estrategia difiere significativamente.

        **Interpretación del p-valor:**
        - p < 0.05 → diferencias estadísticamente significativas al 95 % de confianza.
        - p ≥ 0.05 → las diferencias podrían deberse al azar dado el número de partidas.

        > **Nota:** el ANOVA asume normalidad e igualdad de varianzas. Con torneos pequeños
        (pocas partidas), el test puede tener poca potencia. Aumenta el número de juegos por par
        en el sidebar para resultados más robustos.
        """)



def tab_rng_tests():
    """
    Anexo Técnico: Tests de Calidad del RNG (PCG64, seed=42).
    Requisitos del PDF: KS test uniformidad, lag plot autocorrelación,
    histograma de 10^4 muestras.
    """
    st.markdown("### 🎲 Anexo Técnico — Tests de Calidad RNG")
    st.markdown("""
    <div style="background:rgba(14,20,45,0.8);border:1px solid rgba(56,189,248,0.18);
                border-radius:6px;padding:12px 18px;margin-bottom:14px;">
      <p style="font-size:11px;color:#94a3b8;margin:0;">
        <b style="color:#e2e8f0;">Generador:</b> numpy.random.Generator(PCG64(seed=42))
        &nbsp;·&nbsp;
        <b style="color:#e2e8f0;">Implementación:</b> np.random.default_rng(np.random.PCG64(42))
        &nbsp;·&nbsp;
        <b style="color:#e2e8f0;">Muestras:</b> 10,000
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Generate test samples
    _rng_test = np.random.default_rng(np.random.PCG64(42))
    N = 10_000
    samples = _rng_test.uniform(0, 1, N)

    # ── Test 1: Kolmogorov-Smirnov ───────────────────────────
    st.markdown("#### Test 1 — Kolmogorov-Smirnov (uniformidad)")
    from scipy import stats as _st
    ks_stat, ks_p = _st.kstest(samples, 'uniform')
    sig_color = "#34d399" if ks_p > 0.05 else "#f87171"
    result_text = "✅ No se rechaza H₀ — distribución uniforme confirmada"                   if ks_p > 0.05 else "❌ Se rechaza H₀ — posible sesgo"

    col1, col2, col3 = st.columns(3)
    col1.metric("Estadístico KS", f"{ks_stat:.6f}")
    col2.metric("p-valor", f"{ks_p:.4f}")
    col3.metric("Resultado (α=0.05)", "PASA ✅" if ks_p > 0.05 else "FALLA ❌")

    st.markdown(f"""
    <p style="font-size:11px;color:{sig_color};margin:4px 0 12px 0;">
    {result_text} &nbsp;·&nbsp; p-valor = {ks_p:.4f} {'>' if ks_p > 0.05 else '<'} 0.05
    </p>
    """, unsafe_allow_html=True)

    # ── Test 2: Histograma 10^4 muestras ─────────────────────
    st.markdown("#### Test 2 — Histograma de 10,000 muestras")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=samples, nbinsx=50,
        name="Frecuencia observada",
        marker_color="rgba(56,189,248,0.7)",
        marker_line=dict(color="rgba(56,189,248,0.3)", width=0.5),
        hovertemplate="Bin: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    expected = N / 50
    fig_hist.add_hline(y=expected, line_dash="dot",
                       line_color="rgba(251,191,36,0.7)", line_width=1.5,
                       annotation_text=f"Esperado = {expected:.0f}",
                       annotation_font=dict(size=9, color="#fbbf24"))
    fig_hist.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0e1420",
        font=dict(family="monospace", color="#e2e8f0", size=10),
        height=300,
        title=dict(text="HISTOGRAMA — 10,000 muestras U(0,1) con PCG64(seed=42)",
                   font=dict(size=11), x=0),
        xaxis=dict(title="Valor", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Frecuencia", gridcolor="rgba(255,255,255,0.04)"),
        margin=dict(l=12, r=12, t=48, b=12),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Test 3: Lag plot (autocorrelación) ───────────────────
    st.markdown("#### Test 3 — Lag Plot (autocorrelación)")
    st.caption("Un buen RNG no debe mostrar estructura en el lag plot — los puntos deben distribuirse aleatoriamente.")
    lag_sample = samples[:500]
    fig_lag = go.Figure()
    fig_lag.add_trace(go.Scatter(
        x=lag_sample[:-1], y=lag_sample[1:],
        mode="markers",
        marker=dict(color="rgba(56,189,248,0.35)", size=3),
        hovertemplate="x(t): %{x:.3f}<br>x(t+1): %{y:.3f}<extra></extra>",
        name="Par (x_t, x_{t+1})",
    ))
    fig_lag.update_layout(
        paper_bgcolor="#080b12", plot_bgcolor="#0e1420",
        font=dict(family="monospace", color="#e2e8f0", size=10),
        height=320,
        title=dict(text="LAG PLOT — x(t) vs x(t+1)  ·  sin estructura = buen RNG",
                   font=dict(size=11), x=0),
        xaxis=dict(title="x(t)", gridcolor="rgba(255,255,255,0.04)", range=[0,1]),
        yaxis=dict(title="x(t+1)", gridcolor="rgba(255,255,255,0.04)", range=[0,1]),
        margin=dict(l=12, r=12, t=48, b=12),
    )
    st.plotly_chart(fig_lag, use_container_width=True)

    # ── Correlación de lag ───────────────────────────────────
    lag_corr = float(np.corrcoef(lag_sample[:-1], lag_sample[1:])[0,1])
    st.metric("Correlación lag-1", f"{lag_corr:.6f}",
              help="Debe ser cercano a 0. |r| < 0.02 indica buena independencia.")

    # ── Test 4: Chi-cuadrado ─────────────────────────────────
    st.markdown("#### Test 4 — Chi-cuadrado (frecuencias por bin)")
    observed, _ = np.histogram(samples, bins=10)
    chi2_stat, chi2_p = _st.chisquare(observed)
    c1, c2, c3 = st.columns(3)
    c1.metric("Estadístico χ²", f"{chi2_stat:.4f}")
    c2.metric("p-valor", f"{chi2_p:.4f}")
    c3.metric("Resultado (α=0.05)", "PASA ✅" if chi2_p > 0.05 else "FALLA ❌")

    st.markdown("---")
    st.markdown("""
    **Resumen metodológico:**
    El generador PCG64 (Permuted Congruential Generator, 64-bit) de NumPy es un
    generador de período extremadamente largo (2¹²⁸) con excelentes propiedades
    estadísticas. La semilla fija (seed=42) garantiza reproducibilidad total —
    cualquier persona que ejecute el código obtendrá exactamente los mismos resultados.
    """)

    # ── Exportar resultados ──────────────────────────────────
    import io as _io
    _buf = _io.StringIO()
    pd.DataFrame({
        "sample_index": range(N),
        "value": samples,
    }).to_csv(_buf, index=False)
    st.download_button(
        "⬇ Descargar muestras RNG (CSV)",
        data=_buf.getvalue(),
        file_name="rng_samples_pcg64_seed42.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# Dashboard Streamlit — main()
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Prisoner's Dilemma Tournament",
        page_icon="🎲",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎲 Torneo del Dilema del Prisionero")
    st.markdown("**Round-Robin · 15 Estrategias · Dashboard Interactivo**")

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parámetros")

        st.subheader("Matriz de Pagos")
        col1, col2 = st.columns(2)
        with col1:
            T = st.number_input("T (Traición)",  min_value=1, max_value=20, value=5, step=1, key='T')
            R = st.number_input("R (Recompensa)", min_value=0, max_value=19, value=3, step=1, key='R')
        with col2:
            P = st.number_input("P (Castigo)",    min_value=0, max_value=18, value=1, step=1, key='P')
            S = st.number_input("S (Chivo exp.)", min_value=0, max_value=17, value=0, step=1, key='S')

        valid_payoff = (T > R > P >= S)
        if not valid_payoff:
            st.error("❌ Se requiere: T > R > P ≥ S")
        else:
            st.success("✅ T > R > P ≥ S")

        st.subheader("Parámetros del Torneo")
        w        = st.slider("w (prob. interacción futura)", 0.0, 1.0, 0.995, 0.001)
        n_games  = st.slider("Juegos por par",   1, 10,  5)
        n_rounds = st.slider("Rondas por juego", 50, 500, 200, step=50)

        st.subheader("Selección de Estrategias")
        selected = st.multiselect(
            "Estrategias a incluir",
            options=STRATEGY_NAMES,
            default=STRATEGY_NAMES,
        )
        run_btn = st.button("▶ Ejecutar Torneo", type="primary", use_container_width=True)

    # ── Session state ─────────────────────────────────────────────
    if "results"  not in st.session_state: st.session_state.results  = None
    if "anova"    not in st.session_state: st.session_state.anova    = None
    if "per_game" not in st.session_state: st.session_state.per_game = {}

    if run_btn:
        if not valid_payoff:
            st.error("Corrige la matriz de pagos antes de ejecutar.")
        elif len(selected) < 2:
            st.error("Selecciona al menos 2 estrategias.")
        else:
            with st.spinner("Ejecutando torneo..."):
                payoff_matrix, ranking, h2h_data, per_game = run_tournament_with_anova(
                    selected, T, R, P, S, w, n_games, n_rounds
                )
                anova_result = compute_anova(per_game)
            st.session_state.results = {
                "payoff_matrix": payoff_matrix,
                "ranking":       ranking,
                "h2h_data":      h2h_data,
                "selected":      selected,
                "params":        dict(T=T, R=R, P=P, S=S, w=w),
            }
            st.session_state.anova    = anova_result
            st.session_state.per_game = per_game
            st.success("✅ Torneo completado.")

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏆 Ranking",
        "🌡️ Heatmap",
        "⚔️ Head-to-Head",
        "📊 Distribución",
        "ℹ️ Estrategias",
        "🧪 Análisis Estadístico",
        "⚡ Crisis Energética Rusia–UE",
        "🔬 Anexo: Tests RNG",
    ])

    # ── Tab 1: Ranking ────────────────────────────────────────────
    with tab1:
        if not st.session_state.results:
            st.info("👈 Configura los parámetros en el sidebar y pulsa **▶ Ejecutar Torneo**.")
        else:
            res      = st.session_state.results
            ranking  = res["ranking"]
            selected = res["selected"]

            st.subheader("🏆 Top 3 Estrategias")
            top3 = ranking.head(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.metric(
                        label=f"{'🥇🥈🥉'[i]} #{i+1}",
                        value=row["Estrategia"],
                        delta=f"Score: {row['Score Total']:.3f}",
                    )
            st.divider()
            st.subheader("Tabla Completa")
            disp = ranking.copy()
            disp.index = disp.index.map(lambda x: f"#{x}")
            st.dataframe(disp, use_container_width=True)

            import io as _io_r
            _buf_r = _io_r.StringIO()
            ranking.to_csv(_buf_r, index=True)
            st.download_button(
                "⬇ Descargar ranking (CSV)",
                data=_buf_r.getvalue(),
                file_name="ranking_torneo.csv",
                mime="text/csv",
            )

    # ── Tab 2: Heatmap ────────────────────────────────────────────
    with tab2:
        if not st.session_state.results:
            st.info("Ejecuta el torneo primero.")
        else:
            payoff_matrix = st.session_state.results["payoff_matrix"]
            st.subheader("🌡️ Mapa de Calor — Pago Promedio")
            fig_heat = go.Figure(data=go.Heatmap(
                z=payoff_matrix.values,
                x=payoff_matrix.columns.tolist(),
                y=payoff_matrix.index.tolist(),
                colorscale="RdYlGn",
                hovertemplate=(
                    "<b>Jugador (fila):</b> %{y}<br>"
                    "<b>vs Oponente (col):</b> %{x}<br>"
                    "<b>Pago promedio:</b> %{z:.3f}<extra></extra>"
                ),
                text=np.round(payoff_matrix.values, 2),
                texttemplate="%{text}",
                showscale=True,
            ))
            fig_heat.update_layout(
                height=600,
                xaxis_title="Oponente",
                yaxis_title="Estrategia",
                font=dict(size=11),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Head-to-Head ───────────────────────────────────────
    with tab3:
        if not st.session_state.results:
            st.info("Ejecuta el torneo primero.")
        else:
            res      = st.session_state.results
            h2h_data = res["h2h_data"]
            selected = res["selected"]
            payoff_matrix = res["payoff_matrix"]

            st.subheader("⚔️ Comparativa Head-to-Head")
            c1, c2 = st.columns(2)
            s1_name = c1.selectbox("Estrategia 1", selected, key="h2h_s1")
            others  = [s for s in selected if s != s1_name]
            s2_name = c2.selectbox("Estrategia 2", others, key="h2h_s2") if others else None

            if s2_name:
                key = (s1_name, s2_name) if (s1_name, s2_name) in h2h_data                       else (s2_name, s1_name)
                if key in h2h_data:
                    h1r, h2r = h2h_data[key]
                    if key == (s2_name, s1_name):
                        h1r, h2r = h2r, h1r
                    rlen = len(h1r)
                    ridx = list(range(1, rlen + 1))
                    h1n  = [1 if m == "C" else 0 for m in h1r]
                    h2n  = [1 if m == "C" else 0 for m in h2r]

                    fig_h2h = go.Figure()
                    for name_h, hist_n, hist_r, color, dash in [
                        (s1_name, h1n, h1r, "royalblue", "solid"),
                        (s2_name, h2n, h2r, "tomato",    "dash"),
                    ]:
                        fig_h2h.add_trace(go.Scatter(
                            x=ridx, y=hist_n, mode="lines+markers", name=name_h,
                            line=dict(color=color, dash=dash),
                            marker=dict(
                                symbol=["circle" if m == "C" else "x" for m in hist_r],
                                size=6,
                            ),
                            hovertemplate=f"Ronda %{{x}}: {name_h} jugó %{{customdata}}<extra></extra>",
                            customdata=hist_r,
                        ))
                    fig_h2h.update_layout(
                        height=400,
                        yaxis=dict(tickvals=[0, 1], ticktext=["D", "C"], title="Acción"),
                        xaxis_title="Ronda",
                        title=f"{s1_name} vs {s2_name} — Último Juego",
                        legend=dict(orientation="h", y=1.1),
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(fig_h2h, use_container_width=True)

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric(f"% C de {s1_name}", f"{h1r.count('C')/rlen*100:.1f}%")
                    rc2.metric(f"% C de {s2_name}", f"{h2r.count('C')/rlen*100:.1f}%")
                    rc3.metric("Rondas jugadas",    rlen)
                    rc4.metric("Pago prom. (fila)", f"{payoff_matrix.loc[s1_name, s2_name]:.3f}")

    # ── Tab 4: Distribución ───────────────────────────────────────
    with tab4:
        if not st.session_state.results:
            st.info("Ejecuta el torneo primero.")
        else:
            ranking = st.session_state.results["ranking"]
            st.subheader("📊 Distribución de Scores Totales")
            sc_list = ranking["Score Total"].tolist()
            nm_list = ranking["Estrategia"].tolist()

            fig_b = go.Figure(go.Bar(
                x=nm_list, y=sc_list,
                marker_color=[
                    f"rgba({int(255*(1-i/len(nm_list)))},{int(100+155*(i/len(nm_list)))},100,0.85)"
                    for i in range(len(nm_list))
                ],
                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
            ))
            fig_b.update_layout(
                height=450,
                xaxis_title="Estrategia",
                yaxis_title="Score Total",
                xaxis_tickangle=-30,
                margin=dict(l=20, r=20, t=20, b=80),
            )
            st.plotly_chart(fig_b, use_container_width=True)

            st.subheader("Estadísticas")
            st.dataframe(pd.DataFrame({
                "Métrica": ["Media", "Mediana", "Std", "Mín", "Máx"],
                "Valor":   [f"{f(sc_list):.3f}" for f in
                            [np.mean, np.median, np.std, np.min, np.max]],
            }), use_container_width=True, hide_index=True)

    # ── Tab 5: Info estrategias ───────────────────────────────────
    with tab5:
        selected_now = st.session_state.results["selected"] if st.session_state.results else STRATEGY_NAMES
        st.subheader("ℹ️ Descripción de Estrategias")
        descs = {
            "TIT FOR TAT":      "Empieza cooperando y luego imita el último movimiento del oponente.",
            "GRIM":             "Coopera hasta que el oponente traiciona; luego traiciona para siempre.",
            "PAVLOV":           "Repite su acción si ganó, cambia si perdió (Win-Stay, Lose-Shift).",
            "ALL-D":            "Siempre traiciona.",
            "ALL-C":            "Siempre coopera.",
            "TIT FOR TWO TATS": "Solo traiciona si el oponente traicionó dos veces seguidas.",
            "RANDOM":           "Coopera o traiciona aleatoriamente con prob. 50/50.",
            "JOSS":             "Como TFT pero traiciona con prob. 0.1 tras cooperación del oponente.",
            "GRADUAL":          "Tras cada traición del oponente, castiga N veces y luego calma.",
            "ADAPTIVE":         "Empieza CCCCCDDDDD; luego elige la acción con mayor payoff histórico.",
            "EVOLVED-NN":       "Red neuronal 2 capas (6→8→1) con las últimas 3 jugadas como input.",
            "PSO-PLAYER":       "Probabilidades de cooperación optimizadas por PSO.",
            "MEMORY-3":         "Traiciona si el oponente traicionó ≥2 de las últimas 3 rondas.",
            "FRIEDMAN":         "Idéntico a GRIM: traición permanente tras el primer D del oponente.",
            "TESTER":           "Traiciona ronda 1; si el oponente perdona → explotar; si no → TFT.",
        }
        for name, desc in descs.items():
            if name in selected_now:
                with st.expander(f"**{name}**"):
                    st.write(desc)

    # ── Tab 6: Análisis Estadístico ANOVA ────────────────────────
    with tab6:
        tab_statistics()

    # ── Tab 7: Crisis Energética Rusia–UE ────────────────────────
    with tab7:
        render_energy_crisis_tab()

    # ── Tab 8: Anexo Tests RNG ────────────────────────────────
    with tab8:
        tab_rng_tests()


if __name__ == "__main__":
    main()
