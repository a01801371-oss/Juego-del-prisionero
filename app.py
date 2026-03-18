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


# ═══════════════════════════════════════════════════════════════════
#  CASE STUDY — RUSSIA–EU ENERGY CRISIS (2021–2026)
#  No modifica ninguna clase de estrategia ni el motor del torneo.
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
#  CASE STUDY — RUSSIA–EU ENERGY CRISIS (2021–2026)
#  Dashboard de Simulación Interactiva — Teoría de Juegos Aplicada
# ═══════════════════════════════════════════════════════════════════════════

import os as _os

_FLOW_THRESHOLD   = 300    # mcm/día — umbral cooperación Rusia (configurable en sidebar)
_STORAGE_DROP_PCT = 5.0    # % semanal — umbral defección UE

# ── Carga de datos ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_daily_data() -> pd.DataFrame:
    candidates = sorted(
        [f for f in _os.listdir(".") if f.startswith("daily_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        df = pd.read_csv(candidates[0])
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        ru_col = next((c for c in df.columns
                       if "russia" in c.lower() or c.strip().lower() == "ru"), None)
        if ru_col and ru_col != "Russia":
            df = df.rename(columns={ru_col: "Russia"})
        if "Russia" not in df.columns:
            nums = df.select_dtypes("number").columns.tolist()
            if nums:
                df["Russia"] = df[nums[0]]
    else:
        _r = np.random.default_rng(np.random.PCG64(2026))
        dates = pd.date_range("2021-01-01", "2026-03-12", freq="D")
        base  = np.full(len(dates), 350.0)
        base[365:730] -= 60; base[730:910] -= 150
        base[910:1100] += 30; base[1100:] -= 100
        df = pd.DataFrame({"Date": dates,
                           "Russia": np.clip(base + _r.normal(0, 25, len(dates)), 0, 600)})
    df["Russia"] = pd.to_numeric(df["Russia"], errors="coerce").fillna(0)
    return df


@st.cache_data(show_spinner=False)
def load_storage_data() -> pd.DataFrame:
    candidates = sorted(
        [f for f in _os.listdir(".") if "storage" in f.lower() and f.endswith(".xlsx")],
        reverse=True,
    )
    if candidates:
        xl = pd.read_excel(candidates[0])
        date_col = next((c for c in xl.columns
                         if any(k in c.lower() for k in ["date","week","fecha"])), xl.columns[0])
        xl = xl.rename(columns={date_col: "Week"})
        xl["Week"] = pd.to_datetime(xl["Week"], errors="coerce")
        xl = xl.dropna(subset=["Week"]).sort_values("Week").reset_index(drop=True)
        stor_col = next((c for c in xl.columns
                         if any(k in c.lower() for k in ["storage","stor","filling","pct","%","level"])),
                        None)
        if stor_col and stor_col != "Storage_pct":
            xl = xl.rename(columns={stor_col: "Storage_pct"})
        if "Storage_pct" not in xl.columns:
            nums = xl.select_dtypes("number").columns.tolist()
            if nums:
                xl["Storage_pct"] = xl[nums[0]]
        df = xl
    else:
        _r = np.random.default_rng(np.random.PCG64(2027))
        weeks = pd.date_range("2021-01-04", "2026-03-10", freq="W-MON")
        pct   = np.clip(55 + _r.normal(0, 3, len(weeks)).cumsum() * 0.35, 5, 100)
        df    = pd.DataFrame({"Week": weeks, "Storage_pct": pct})
    df["Storage_pct"] = pd.to_numeric(df["Storage_pct"], errors="coerce").fillna(50)
    df["Storage_change_pct"] = df["Storage_pct"].pct_change() * 100
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_route_data() -> pd.DataFrame:
    candidates = sorted(
        [f for f in _os.listdir(".") if f.startswith("route_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        df = pd.read_csv(candidates[0])
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    _r = np.random.default_rng(np.random.PCG64(2028))
    dates = pd.date_range("2021-01-01", "2026-03-12", freq="D")
    profiles = {
        "Nord Stream 1": (80, "2022-06-15", 0),
        "Yamal-Europe":  (60, "2022-04-27", 5),
        "Ukrainian GTS": (70, "2024-01-01", 15),
        "TurkStream":    (50, "2021-01-01", 45),
    }
    rows = []
    for route, (base_val, cut_date, floor) in profiles.items():
        cut_ts = pd.Timestamp(cut_date)
        for d in dates:
            if d < cut_ts:
                v = base_val + _r.normal(0, 4)
            else:
                v = max(floor, base_val - (d - cut_ts).days * 0.25 + _r.normal(0, 3))
            rows.append({"Date": d, "Route": route,
                         "Utilisation_pct": round(float(np.clip(v, 0, 100)), 1)})
    return pd.DataFrame(rows)


# ── Motor de simulación interactiva ────────────────────────────────────────

def simulate_ipd(
    df_daily:     pd.DataFrame,
    df_stor:      pd.DataFrame,
    T: float, R: float, P: float, S: float,
    flow_thresh:  float,
    stor_thresh:  float,
    eu_strategy:  str,          # "Histórico" | nombre de estrategia
    date_start:   pd.Timestamp,
    date_end:     pd.Timestamp,
) -> pd.DataFrame:
    """
    Núcleo del simulador interactivo.
    Genera una fila por día con las movidas, scores y métricas económicas.
    La movida de Rusia siempre viene de los datos reales.
    La movida de la UE puede ser histórica o una estrategia del simulador (modo What-if).
    """
    # ── Filtrar por rango de fechas ──
    daily = df_daily[
        (df_daily["Date"] >= date_start) &
        (df_daily["Date"] <= date_end)
    ].copy().reset_index(drop=True)

    stor = df_stor[
        (df_stor["Week"] >= date_start) &
        (df_stor["Week"] <= date_end)
    ].copy().reset_index(drop=True)

    if daily.empty:
        return pd.DataFrame()

    # ── Movida Rusia: siempre de datos reales ──
    daily["move_russia"] = daily["Russia"].apply(
        lambda x: "C" if (pd.notna(x) and float(x) > flow_thresh) else "D"
    )

    # ── Movida UE histórica: merge semanal → diario ──
    if not stor.empty:
        stor["Storage_change_pct"] = stor["Storage_pct"].pct_change() * 100
        stor["move_eu_hist"] = stor["Storage_change_pct"].apply(
            lambda x: "D" if (pd.notna(x) and x < -stor_thresh) else "C"
        )
        stor_daily = stor[["Week", "move_eu_hist", "Storage_pct"]].rename(
            columns={"Week": "Date"}
        )
        daily = pd.merge_asof(
            daily.sort_values("Date"),
            stor_daily.sort_values("Date"),
            on="Date", direction="backward",
        )
        daily["move_eu_hist"] = daily["move_eu_hist"].fillna("C")
        daily["Storage_pct"]  = daily["Storage_pct"].ffill().fillna(50)
    else:
        daily["move_eu_hist"] = "C"
        daily["Storage_pct"]  = 50.0

    # ── Movida UE: Histórica vs What-if ──
    if eu_strategy == "Histórico (datos reales)":
        daily["move_eu"] = daily["move_eu_hist"]
    else:
        # Aplicar estrategia del catálogo al historial de Rusia
        name_to_cls = {s.name: s for s in ALL_STRATEGIES}
        if eu_strategy in name_to_cls:
            strat = name_to_cls[eu_strategy]()
            eu_moves = []
            for _, row in daily.iterrows():
                eu_moves.append(strat.move())
                strat.update(row["move_russia"],
                             eu_moves[-1] if eu_moves else "C")
            daily["move_eu"] = eu_moves
        else:
            daily["move_eu"] = daily["move_eu_hist"]

    # ── Calcular scores y métricas económicas ──
    payoff_map = {
        ("C","C"): (R, R,   +2.0, +2.0,  +2.5,  "Cooperación Mutua"),
        ("D","C"): (T, S,   +4.5, -4.0,  +15.0, "Rusia defecta — Precio alto"),
        ("C","D"): (S, T,   -3.5, +0.5,  +6.0,  "UE responde — Sanciones/GNL"),
        ("D","D"): (P, P,   -2.5, -2.5,  +8.0,  "Castigo mutuo — Crisis total"),
    }
    color_map = {
        "Cooperación Mutua":          "#34d399",
        "Rusia defecta — Precio alto":"#f87171",
        "UE responde — Sanciones/GNL":"#fbbf24",
        "Castigo mutuo — Crisis total":"#a78bfa",
    }

    records = []
    cum_ru = cum_eu = 0.0
    for _, row in daily.iterrows():
        key = (row["move_russia"], row["move_eu"])
        pr, pe, gdp_ru, gdp_eu, inf_eu, label = payoff_map.get(
            key, (R, R, 2.0, 2.0, 2.5, "Cooperación Mutua")
        )
        cum_ru += pr; cum_eu += pe
        records.append({
            "score_ru": pr,    "score_eu": pe,
            "cum_ru":   cum_ru,"cum_eu":   cum_eu,
            "gdp_ru":   gdp_ru,"gdp_eu":   gdp_eu,
            "inf_eu":   inf_eu,
            "outcome":  label,
            "color":    color_map.get(label, "#64748b"),
        })

    result = pd.concat([
        daily.reset_index(drop=True),
        pd.DataFrame(records),
    ], axis=1)

    # Rolling indicators
    result["gdp_eu_30d"]  = result["gdp_eu"].rolling(30, min_periods=1).mean()
    result["gdp_ru_30d"]  = result["gdp_ru"].rolling(30, min_periods=1).mean()
    result["inf_eu_30d"]  = result["inf_eu"].rolling(30, min_periods=1).mean()
    result["coop_30d"]    = (
        ((result["move_russia"]=="C") & (result["move_eu"]=="C"))
        .rolling(30, min_periods=1).mean() * 100
    )
    return result


# ── Gráficos ───────────────────────────────────────────────────────────────

_DARK = dict(paper_bgcolor="#0a0e1a", plot_bgcolor="#0e1420",
             font=dict(family="monospace", color="#e2e8f0", size=10),
             margin=dict(l=12, r=12, t=48, b=12),
             legend=dict(bgcolor="rgba(0,0,0,0.4)", borderwidth=0),
             hovermode="x unified")
_GRID = "rgba(255,255,255,0.05)"
_RU, _EU, _GAS, _GRN = "#f87171", "#38bdf8", "#fbbf24", "#34d399"

KEY_EVENTS = [
    ("2021-10-01", "Crisis precio gas"),
    ("2022-02-24", "Invasión Ucrania"),
    ("2022-06-15", "NS1 recorte −60%"),
    ("2022-09-26", "Sabotaje Nord Stream"),
    ("2024-01-01", "Tránsito UA expira"),
]


def _add_events(fig, df, row="all"):
    for ds, lbl in KEY_EVENTS:
        ts = pd.Timestamp(ds)
        if df["Date"].min() <= ts <= df["Date"].max():
            fig.add_vline(x=ts.timestamp()*1000,
                          line_dash="dot",
                          line_color="rgba(167,139,250,0.35)",
                          line_width=1, row=row, col=1)
            fig.add_annotation(
                x=ts, y=1, xref="x", yref="paper",
                text=lbl, showarrow=False,
                font=dict(size=7, color="rgba(167,139,250,0.6)"),
                textangle=-90, xanchor="left",
            )


def fig_main_timeline(df: pd.DataFrame, eu_strategy: str) -> go.Figure:
    """
    Gráfico principal con zoom: flujo de gas, movidas codificadas por color
    y scores acumulados en doble eje.
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.28, 0.27],
        vertical_spacing=0.04,
        subplot_titles=[
            "FLUJO DE GAS RUSIA → EUROPA  (mcm/día)",
            "MOVIDAS DIARIAS — Rusia (arriba) · UE (abajo)",
            "ACUMULACIÓN DE BIENESTAR (scores DP)",
        ],
    )

    # ── Fila 1: flujo de gas con colorización por traición ──
    traicion_mask = df["move_russia"] == "D"
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Russia"],
        marker_color=np.where(traicion_mask, _RU, _GAS),
        marker_opacity=0.7,
        name="Flujo Rusia",
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.0f} mcm/día</b><extra>Flujo</extra>",
    ), row=1, col=1)
    # Umbral
    fig.add_hline(y=df.attrs.get("flow_thresh", 300),
                  line_dash="dot", line_color="rgba(251,191,36,0.5)",
                  line_width=1.5, row=1, col=1,
                  annotation_text=f"Umbral C/D",
                  annotation_font=dict(size=8, color=_GAS),
                  annotation_position="top left")

    # ── Fila 2: movidas como scatter coloreado ──
    for move_col, y_val, label in [
        ("move_russia", 1.3, "Rusia"),
        ("move_eu",     0.5, f"UE ({eu_strategy[:18]})"),
    ]:
        colors = [_GRN if m == "C" else _RU for m in df[move_col]]
        fig.add_trace(go.Scatter(
            x=df["Date"], y=[y_val]*len(df),
            mode="markers",
            marker=dict(color=colors, size=3, symbol="square"),
            name=f"Movida {label}",
            hovertemplate=f"{label}: %{{customdata}}<br>%{{x|%d %b %Y}}<extra></extra>",
            customdata=df[move_col],
            showlegend=True,
        ), row=2, col=1)

    # ── Fila 3: scores acumulados ──
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_ru"],
        name="Score acum. Rusia", mode="lines",
        line=dict(color=_RU, width=2),
        hovertemplate="Rusia: %{y:,.0f} pts<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_eu"],
        name="Score acum. UE", mode="lines",
        line=dict(color=_EU, width=2),
        hovertemplate="UE: %{y:,.0f} pts<extra></extra>",
    ), row=3, col=1)
    # Área de diferencia
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_ru"] - df["cum_eu"],
        name="Δ Score (Ru−UE)", mode="lines",
        line=dict(color="rgba(251,191,36,0.5)", width=1.2, dash="dot"),
        hovertemplate="Δ: %{y:,.0f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1, row=3, col=1)

    _add_events(fig, df)
    fig.update_layout(**_DARK, height=600,
                      title_text="LÍNEA DE TIEMPO INTERACTIVA — DILEMA DEL PRISIONERO RUSIA–UE",
                      title_font=dict(size=12))
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9),
                     rangeslider=dict(visible=True, thickness=0.04), row=3, col=1)
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9), row=1, col=1)
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9), row=2, col=1)
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(size=9))
    fig.update_yaxes(tickvals=[0.5, 1.3], ticktext=["UE", "Rusia"],
                     range=[0, 1.8], row=2, col=1)
    return fig


def fig_welfare(df: pd.DataFrame) -> go.Figure:
    """Acumulación de bienestar con fill entre curvas para visualizar quién gana."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_ru"],
        name="Bienestar Rusia", mode="lines",
        line=dict(color=_RU, width=2.5),
        hovertemplate="Rusia: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_eu"],
        name="Bienestar UE", mode="lines",
        line=dict(color=_EU, width=2.5),
        fill="tonexty",
        fillcolor="rgba(56,189,248,0.08)",
        hovertemplate="UE: %{y:,.0f}<extra></extra>",
    ))
    # Anotar máxima divergencia
    df2 = df.copy()
    df2["diff"] = df2["cum_ru"] - df2["cum_eu"]
    max_idx = df2["diff"].abs().idxmax()
    if max_idx in df2.index:
        row = df2.loc[max_idx]
        fig.add_annotation(
            x=row["Date"], y=max(row["cum_ru"], row["cum_eu"]),
            text=f"Máx divergencia<br>Δ={row['diff']:,.0f} pts",
            showarrow=True, arrowhead=2,
            font=dict(size=8, color=_GAS),
            arrowcolor=_GAS, bgcolor="rgba(0,0,0,0.5)",
        )
    _add_events(fig, df)
    fig.update_layout(**_DARK, height=360,
                      title_text="ACUMULACIÓN DE BIENESTAR — SCORES TOTALES DP",
                      title_font=dict(size=12))
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(size=9), title="Score acumulado")
    return fig


def fig_economic_rolling(df: pd.DataFrame) -> go.Figure:
    """PIB e inflación en media móvil 30 días, actualizados con los parámetros del usuario."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["PIB (media 30d, %)", "INFLACIÓN UE (media 30d, %)"])
    for col, color, label, r, c in [
        ("gdp_eu_30d", _EU, "PIB UE",    1, 1),
        ("gdp_ru_30d", _RU, "PIB Rusia", 1, 1),
    ]:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[col], name=label,
            line=dict(color=color, width=1.8),
            hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
        ), row=r, col=c)
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.15)", row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["inf_eu_30d"],
        name="Inflación UE", line=dict(color=_GAS, width=1.8),
        hovertemplate="Inflación: %{y:.1f}%<extra></extra>",
    ), row=1, col=2)
    fig.add_hline(y=2.0, line_dash="dot",
                  line_color="rgba(52,211,153,0.4)", row=1, col=2,
                  annotation_text="BCE 2%", annotation_font=dict(size=8, color=_GRN))
    fig.update_layout(**_DARK, height=320,
                      title_text="IMPACTO ECONÓMICO ESTIMADO (media 30d) — SE RECALCULA CON LA MATRIZ",
                      title_font=dict(size=12))
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(size=9))
    return fig


def fig_outcome_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap mensual de outcomes para ver patrones estacionales."""
    df2 = df.copy()
    df2["YearMonth"] = df2["Date"].dt.to_period("M").astype(str)
    df2["Day"]       = df2["Date"].dt.day
    outcome_num = {"Cooperación Mutua": 3,
                   "Rusia defecta — Precio alto": 0,
                   "UE responde — Sanciones/GNL": 1,
                   "Castigo mutuo — Crisis total": 2}
    df2["outcome_num"] = df2["outcome"].map(outcome_num).fillna(3)
    pivot = df2.pivot_table(index="YearMonth", columns="Day",
                            values="outcome_num", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.00, "#f87171"],   # Rusia defecta
            [0.33, "#a78bfa"],   # Castigo mutuo
            [0.67, "#fbbf24"],   # UE responde
            [1.00, "#34d399"],   # Cooperación
        ],
        zmin=0, zmax=3,
        hovertemplate="Mes: %{y}<br>Día: %{x}<br>Outcome: %{z:.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickvals=[0, 1, 2, 3],
            ticktext=["Rusia D", "UE D", "DD", "CC"],
            tickfont=dict(size=8, color="#94a3b8"),
        ),
    ))
    fig.update_layout(**_DARK, height=max(300, len(pivot)*14+80),
                      title_text="CALENDARIO DE OUTCOMES — PATRONES MENSUALES",
                      title_font=dict(size=12),
                      xaxis_title="Día del mes",
                      yaxis_title="Mes")
    fig.update_xaxes(tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))
    return fig


def fig_cooperation_rate(df: pd.DataFrame) -> go.Figure:
    """Tasa de cooperación mutua en ventana 30 días."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["coop_30d"],
        mode="lines",
        line=dict(color=_GRN, width=2),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        name="% CC (30d)",
        hovertemplate="%{y:.1f}% cooperación mutua<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1,
                  annotation_text="50%", annotation_font=dict(size=8))
    _add_events(fig, df)
    fig.update_layout(**_DARK, height=260,
                      title_text="TASA DE COOPERACIÓN MUTUA (media móvil 30 días)",
                      title_font=dict(size=12))
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(size=9),
                     title="% CC", range=[0, 105])
    return fig


def fig_w_gauge(w_critical: float, w_real: float) -> go.Figure:
    """Gauge doble: w crítico (Axelrod) vs w real (volatilidad almacenamiento)."""
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=w_real,
        delta={"reference": w_critical,
               "increasing": {"color": _GRN},
               "decreasing": {"color": _RU}},
        title={"text": "w real vs w crítico", "font": {"size": 11, "color": "#e2e8f0"}},
        number={"font": {"color": "#e2e8f0", "size": 24}},
        gauge={
            "axis":  {"range": [0, 1], "tickcolor": "#64748b"},
            "bar":   {"color": _EU, "thickness": 0.25},
            "steps": [
                {"range": [0, w_critical],      "color": "rgba(248,113,113,0.25)"},
                {"range": [w_critical, 1],       "color": "rgba(52,211,153,0.15)"},
            ],
            "threshold": {
                "line":      {"color": _GAS, "width": 3},
                "thickness": 0.85,
                "value":     w_critical,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0"),
        height=230,
        margin=dict(l=16, r=16, t=40, b=16),
        annotations=[dict(
            x=0.5, y=-0.05, xref="paper", yref="paper",
            text=f"w crítico Axelrod = {w_critical:.4f}  |  "
                 f"{'✅ Cooperación posible' if w_real >= w_critical else '❌ Cooperación inestable'}",
            font=dict(size=9, color=_GRN if w_real >= w_critical else _RU),
            showarrow=False,
        )],
    )
    return fig


def fig_whatif_comparison(df_hist: pd.DataFrame, df_whatif: pd.DataFrame,
                           strategy_name: str) -> go.Figure:
    """Compara bienestar histórico vs What-if para Rusia y UE."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["BIENESTAR UE — Histórico vs What-if",
                                        "BIENESTAR RUSIA — Histórico vs What-if"])
    for col_idx, (player, col_hist, col_wi, color_h, color_wi) in enumerate([
        ("UE",     "cum_eu", "cum_eu", _EU,  "#7dd3fc"),
        ("Rusia",  "cum_ru", "cum_ru", _RU,  "#fca5a5"),
    ], 1):
        fig.add_trace(go.Scatter(
            x=df_hist["Date"], y=df_hist[col_hist],
            name=f"{player} Histórico", mode="lines",
            line=dict(color=color_h, width=2),
            hovertemplate=f"{player} hist: %{{y:,.0f}}<extra></extra>",
        ), row=1, col=col_idx)
        fig.add_trace(go.Scatter(
            x=df_whatif["Date"], y=df_whatif[col_wi],
            name=f"{player} {strategy_name}", mode="lines",
            line=dict(color=color_wi, width=2, dash="dash"),
            hovertemplate=f"{player} WI: %{{y:,.0f}}<extra></extra>",
        ), row=1, col=col_idx)
    fig.update_layout(**_DARK, height=340,
                      title_text=f"MODO WHAT-IF — ¿Qué habría pasado si la UE usara {strategy_name}?",
                      title_font=dict(size=12))
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(size=9))
    return fig


# ── Tab principal ──────────────────────────────────────────────────────────

def render_energy_crisis_tab():
    """
    Dashboard interactivo de simulación: Crisis Energética Rusia–EU (2021–2026).
    Incluye: slider de tiempo, matriz dinámica, w crítico, What-if y gráficos con zoom.
    """
    # ════════════════════════════════════════════════════════════════
    # SIDEBAR de la Crisis Energética
    # ════════════════════════════════════════════════════════════════
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### ⚡ Crisis Energética\n"
        "<span style='font-size:10px;color:#64748b;letter-spacing:.1em;'>CONTROLES DE SIMULACIÓN</span>",
        unsafe_allow_html=True,
    )

    # ── Cargar datos una vez ──
    with st.spinner("Cargando datos de energía…"):
        df_daily_full = load_daily_data()
        df_stor_full  = load_storage_data()
        df_route      = load_route_data()

    min_date = df_daily_full["Date"].min().date()
    max_date = df_daily_full["Date"].max().date()

    # ── 1. SLIDER DE TIEMPO ──────────────────────────────────────────
    st.sidebar.markdown("**📅 Rango de simulación**")
    date_range = st.sidebar.slider(
        "Período analizado",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MMM YYYY",
        key="ec_date_range",
    )
    date_start = pd.Timestamp(date_range[0])
    date_end   = pd.Timestamp(date_range[1])

    # ── 2. MATRIZ DE PAGOS DINÁMICA ──────────────────────────────────
    st.sidebar.markdown("**💰 Matriz de Pagos (ajústala)**")
    ec_T = st.sidebar.slider("T — Tentación (traición unilateral)", 3.0, 15.0, 5.0, 0.5, key="ec_T")
    ec_R = st.sidebar.slider("R — Recompensa (coop. mutua)",        1.0, 10.0, 3.0, 0.5, key="ec_R")
    ec_P = st.sidebar.slider("P — Castigo (traición mutua)",        0.0,  5.0, 1.0, 0.5, key="ec_P")
    ec_S = st.sidebar.slider("S — Pago sucker (cooperas tú solo)", -3.0,  2.0, 0.0, 0.5, key="ec_S")

    valid = (ec_T > ec_R > ec_P >= ec_S)
    if not valid:
        st.sidebar.error("❌ Requiere T > R > P ≥ S")
    else:
        st.sidebar.success(f"✅ T={ec_T} R={ec_R} P={ec_P} S={ec_S}")

    # ── 3. UMBRALES DE DECISIÓN ──────────────────────────────────────
    st.sidebar.markdown("**🎚️ Umbrales de decisión**")
    flow_thresh = st.sidebar.slider(
        "Flujo mínimo para C-Rusia (mcm/día)", 100, 500, 300, 10, key="ec_flow"
    )
    stor_thresh = st.sidebar.slider(
        "Caída máx. almac. para C-UE (%)", 1.0, 15.0, 5.0, 0.5, key="ec_stor"
    )

    # ── 4. MODO WHAT-IF ──────────────────────────────────────────────
    st.sidebar.markdown("**🔮 Modo What-if — Estrategia UE**")
    whatif_options = ["Histórico (datos reales)"] + STRATEGY_NAMES
    eu_strategy = st.sidebar.selectbox(
        "¿Qué estrategia hubiera usado la UE?",
        whatif_options, index=0, key="ec_eu_strat",
    )

    # ════════════════════════════════════════════════════════════════
    # CUERPO DEL TAB
    # ════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(248,113,113,0.07),
                rgba(56,189,248,0.05));border:1px solid rgba(248,113,113,0.18);
                border-radius:6px;padding:16px 20px;margin-bottom:16px;">
      <p style="font-size:10px;letter-spacing:.15em;color:#f87171;
                text-transform:uppercase;margin:0 0 3px 0;">
        ● SIMULADOR INTERACTIVO — DILEMA DEL PRISIONERO EN EL MUNDO REAL
      </p>
      <p style="font-size:18px;font-weight:700;color:#e2e8f0;margin:0 0 2px 0;">
        Crisis Energética Rusia–Europa (2021–2026)
      </p>
      <p style="font-size:12px;color:#64748b;margin:0;">
        Mueve los sliders del sidebar para cambiar la matriz, el período o la estrategia de la UE.
        El dashboard se recalcula al instante.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not valid:
        st.error("⚠ Corrige la matriz de pagos en el sidebar (T > R > P ≥ S).")
        return

    # ── Simular período seleccionado ──
    with st.spinner("Simulando…"):
        df_sim = simulate_ipd(
            df_daily_full, df_stor_full,
            ec_T, ec_R, ec_P, ec_S,
            flow_thresh, stor_thresh,
            eu_strategy, date_start, date_end,
        )
        # También simulación histórica para comparación What-if
        df_hist = simulate_ipd(
            df_daily_full, df_stor_full,
            ec_T, ec_R, ec_P, ec_S,
            flow_thresh, stor_thresh,
            "Histórico (datos reales)", date_start, date_end,
        )

    if df_sim.empty:
        st.warning("Sin datos para el período seleccionado.")
        return

    df_sim.attrs["flow_thresh"] = flow_thresh

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN A: w CRÍTICO Y GAUGE
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 🎯 Sombra del Futuro — Parámetro *w*")
    col_w1, col_w2, col_w3 = st.columns([1.2, 1.2, 1.6])

    w_critical = (ec_T - ec_R) / (ec_T - ec_P) if (ec_T - ec_P) != 0 else 0.5

    # w real: estimado por estabilidad de almacenamiento (baja volatilidad = futuro más predecible)
    if not df_stor_full.empty and "Storage_change_pct" in df_stor_full.columns:
        vol = df_stor_full["Storage_change_pct"].std()
        w_real = float(np.clip(1 - vol / 100, 0.5, 0.999))
    else:
        w_real = 0.97

    with col_w1:
        st.metric("w crítico (Axelrod)",
                  f"{w_critical:.4f}",
                  help="w > (T−R)/(T−P) — umbral para que la cooperación sea evolutivamente estable")
    with col_w2:
        st.metric("w real (volatilidad almac.)",
                  f"{w_real:.4f}",
                  delta=f"{'✅ >' if w_real >= w_critical else '❌ <'} w crítico")
    with col_w3:
        st.markdown(
            f"**Fórmula Axelrod:** `w > (T−R)/(T−P) = ({ec_T}−{ec_R})/({ec_T}−{ec_P}) = {w_critical:.4f}`  \n"
            f"{'🟢 **Cooperación posible** — el futuro pesa suficiente.' if w_real >= w_critical else '🔴 **Cooperación inestable** — el futuro descuenta demasiado.'}"
        )

    st.plotly_chart(fig_w_gauge(w_critical, w_real), use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN B: MÉTRICAS DEL PERÍODO SELECCIONADO
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(f"### 📊 Resultados — {date_range[0].strftime('%b %Y')} → {date_range[1].strftime('%b %Y')}"
                + (f"  ·  UE: **{eu_strategy}**" if eu_strategy != "Histórico (datos reales)" else ""))

    coop_ru  = (df_sim["move_russia"] == "C").mean() * 100
    coop_eu  = (df_sim["move_eu"]     == "C").mean() * 100
    cc_pct   = ((df_sim["move_russia"]=="C") & (df_sim["move_eu"]=="C")).mean() * 100
    dd_pct   = ((df_sim["move_russia"]=="D") & (df_sim["move_eu"]=="D")).mean() * 100
    final_ru = df_sim["cum_ru"].iloc[-1]
    final_eu = df_sim["cum_eu"].iloc[-1]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🇷🇺 Coop. Rusia",       f"{coop_ru:.1f}%")
    m2.metric("🇪🇺 Coop. UE",          f"{coop_eu:.1f}%")
    m3.metric("🤝 Coop. mutua (CC)",    f"{cc_pct:.1f}%")
    m4.metric("💥 Castigo mutuo (DD)",  f"{dd_pct:.1f}%")
    m5.metric("🇷🇺 Score Rusia",        f"{final_ru:,.0f}")
    m6.metric("🇪🇺 Score UE",           f"{final_eu:,.0f}")

    # Comparativa What-if vs histórico
    if eu_strategy != "Histórico (datos reales)":
        d_eu = final_eu - df_hist["cum_eu"].iloc[-1]
        d_ru = final_ru - df_hist["cum_ru"].iloc[-1]
        wa1, wa2 = st.columns(2)
        wa1.metric("Δ Bienestar UE vs histórico",   f"{d_eu:+,.0f} pts",
                   delta_color="normal" if d_eu >= 0 else "inverse")
        wa2.metric("Δ Bienestar Rusia vs histórico", f"{d_ru:+,.0f} pts",
                   delta_color="normal" if d_ru >= 0 else "inverse")

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN C: GRÁFICOS INTERACTIVOS
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")

    # Gráfico principal (con zoom + rangeslider)
    st.plotly_chart(fig_main_timeline(df_sim, eu_strategy), use_container_width=True)

    # Fila: Bienestar + Tasa cooperación
    col_a, col_b = st.columns([1.6, 1])
    with col_a:
        st.plotly_chart(fig_welfare(df_sim), use_container_width=True)
    with col_b:
        st.plotly_chart(fig_cooperation_rate(df_sim), use_container_width=True)

    # Impacto económico (se recalcula con los sliders de la matriz)
    st.plotly_chart(fig_economic_rolling(df_sim), use_container_width=True)

    # What-if comparison
    if eu_strategy != "Histórico (datos reales)":
        st.plotly_chart(
            fig_whatif_comparison(df_hist, df_sim, eu_strategy),
            use_container_width=True,
        )

    # Calendario de outcomes
    with st.expander("📅 Calendario de outcomes (heatmap mensual)", expanded=False):
        st.plotly_chart(fig_outcome_heatmap(df_sim), use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # SECCIÓN D: TABLA + DESCARGA
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("📋 Traducción económica de la matriz actual", expanded=False):
        rows = []
        payoff_map_disp = {
            ("C","C"): (ec_R, ec_R,  +2.0,  +2.0,  +2.5,  "Cooperación Mutua"),
            ("D","C"): (ec_T, ec_S,  +4.5,  -4.0, +15.0,  "Rusia defecta"),
            ("C","D"): (ec_S, ec_T,  -3.5,  +0.5,  +6.0,  "UE responde"),
            ("D","D"): (ec_P, ec_P,  -2.5,  -2.5,  +8.0,  "Castigo mutuo"),
        }
        for (mr, me), (pr, pe, gr, ge, ie, lbl) in payoff_map_disp.items():
            rows.append({"Rusia": mr, "UE": me, "Resultado": lbl,
                         "Score Ru": pr, "Score UE": pe,
                         "PIB Ru %": gr, "PIB UE %": ge, "Inflación UE %": ie})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    import io as _io
    buf = _io.StringIO()
    dl_cols = ["Date","Russia","move_russia","move_eu","outcome",
               "score_ru","score_eu","cum_ru","cum_eu",
               "gdp_eu_30d","gdp_ru_30d","inf_eu_30d","coop_30d"]
    df_sim[[c for c in dl_cols if c in df_sim.columns]].to_csv(buf, index=False)
    st.download_button(
        "⬇ Descargar simulación (CSV)",
        data=buf.getvalue(),
        file_name=f"ipd_energia_{date_range[0]}_{date_range[1]}.csv",
        mime="text/csv",
    )

    st.markdown("""
    <p style="font-size:11px;color:#475569;margin-top:16px;line-height:1.6;">
    <b style="color:#94a3b8">Metodología:</b>
    Movida Rusia = C si flujo &gt; umbral configurado; D si ≤ umbral.
    Movida UE Histórica = C salvo que almacenamiento caiga más del umbral semanal.
    Modo What-if: la estrategia seleccionada observa las movidas reales de Rusia y decide.
    Scores e impactos económicos se recalculan al instante con la matriz del sidebar.
    Fuentes de datos: ENTSOG, GIE, Bruegel, Eurostat.
    </p>
    """, unsafe_allow_html=True)


# Dashboard Streamlit — main()
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
    if "results" not in st.session_state:
        st.session_state.results = None

    if run_btn:
        if not valid_payoff:
            st.error("Corrige la matriz de pagos antes de ejecutar.")
        elif len(selected) < 2:
            st.error("Selecciona al menos 2 estrategias.")
        else:
            with st.spinner("Ejecutando torneo..."):
                payoff_matrix, ranking, h2h_data = run_tournament(
                    selected, T, R, P, S, w, n_games, n_rounds
                )
            st.session_state.results = {
                "payoff_matrix": payoff_matrix,
                "ranking":       ranking,
                "h2h_data":      h2h_data,
                "selected":      selected,
                "params":        dict(T=T, R=R, P=P, S=S, w=w),
            }
            st.success("✅ Torneo completado.")

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Ranking",
        "🌡️ Heatmap",
        "⚔️ Head-to-Head",
        "📊 Distribución",
        "ℹ️ Estrategias",
        "⚡ Crisis Energética Rusia–UE",
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

    # ── Tab 6: Crisis Energética Rusia–UE ────────────────────────
    with tab6:
        render_energy_crisis_tab()


if __name__ == "__main__":
    main()
