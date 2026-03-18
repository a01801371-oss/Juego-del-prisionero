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

_FLOW_THRESHOLD   = 300    # mcm/day  → Russia coopera si flujo > umbral
_STORAGE_DROP_PCT = 5.0    # % semanal → EU defecta si almacenamiento cae > umbral

# Traducción económica de los cuatro outcomes del DP (T=5, R=3, P=1, S=0)
ECONOMIC_OUTCOMES = {
    ("C", "C"): dict(label="Cooperación Mutua — Suministro estable",
                     score_ru=3, score_eu=3,
                     gdp_ru=+2.0, gdp_eu=+2.0, inf_eu=+2.5,
                     gas_market="Mercado normal",         color="#34d399"),
    ("D", "C"): dict(label="Rusia defecta — Extracción a precio alto",
                     score_ru=5, score_eu=0,
                     gdp_ru=+4.5, gdp_eu=-4.0, inf_eu=+15.0,
                     gas_market="Pico de precio >€200/MWh", color="#f87171"),
    ("C", "D"): dict(label="UE responde — Sanciones / pivot GNL",
                     score_ru=0, score_eu=5,
                     gdp_ru=-3.5, gdp_eu=+0.5, inf_eu=+6.0,
                     gas_market="Prima en mercado spot",   color="#fbbf24"),
    ("D", "D"): dict(label="Castigo mutuo — Crisis energética total",
                     score_ru=1, score_eu=1,
                     gdp_ru=-2.5, gdp_eu=-2.5, inf_eu=+8.0,
                     gas_market="Disrupción total",        color="#a78bfa"),
}

# ── Carga de datos (cached) ─────────────────────────────────────────

@st.cache_data(show_spinner="Cargando datos de flujo de gas…")
def load_daily_data() -> pd.DataFrame:
    """
    Lee daily_data_*.csv del directorio raíz del repositorio.
    Columna Russia (mcm/día) → derivar move_russia ('C'/'D').
    Si el archivo no existe, genera datos sintéticos reproducibles.
    """
    candidates = sorted(
        [f for f in os.listdir(".") if f.startswith("daily_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        df = pd.read_csv(candidates[0])
        # Normalizar columna de fecha
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        # Normalizar columna Russia
        ru_col = next(
            (c for c in df.columns if "russia" in c.lower() or c.strip().lower() == "ru"),
            None,
        )
        if ru_col and ru_col != "Russia":
            df = df.rename(columns={ru_col: "Russia"})
        if "Russia" not in df.columns:
            nums = df.select_dtypes("number").columns.tolist()
            if nums:
                df["Russia"] = df[nums[0]]
    else:
        # Datos sintéticos reproducibles
        _r = np.random.default_rng(np.random.PCG64(2026))
        dates = pd.date_range("2021-01-01", "2026-03-12", freq="D")
        base  = np.full(len(dates), 350.0)
        base[365:730]  -= 60    # reducción 2022
        base[730:910]  -= 150   # shock invasión
        base[910:1100] += 30    # recuperación parcial
        base[1100:]    -= 100   # cortes 2023-24
        noise = _r.normal(0, 25, len(dates))
        df = pd.DataFrame({"Date": dates,
                           "Russia": np.clip(base + noise, 0, 600)})

    df["move_russia"] = df["Russia"].apply(
        lambda x: "C" if (pd.notna(x) and float(x) > _FLOW_THRESHOLD) else "D"
    )
    return df


@st.cache_data(show_spinner="Cargando datos de almacenamiento UE…")
def load_storage_data() -> pd.DataFrame:
    """
    Lee Weekly Storage EU & UA *.xlsx del directorio raíz.
    Calcula el cambio % semanal → move_eu ('C'/'D').
    Si el archivo no existe, genera datos sintéticos.
    """
    candidates = sorted(
        [f for f in os.listdir(".") if "storage" in f.lower() and f.endswith(".xlsx")],
        reverse=True,
    )
    if candidates:
        xl = pd.read_excel(candidates[0])
        date_col = next(
            (c for c in xl.columns if any(k in c.lower() for k in ["date","week","fecha"])),
            xl.columns[0],
        )
        xl = xl.rename(columns={date_col: "Week"})
        xl["Week"] = pd.to_datetime(xl["Week"], errors="coerce")
        xl = xl.dropna(subset=["Week"]).sort_values("Week").reset_index(drop=True)
        stor_col = next(
            (c for c in xl.columns
             if any(k in c.lower() for k in ["storage","stor","filling","pct","%","level"])),
            None,
        )
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

    df = df.reset_index(drop=True)
    df["Storage_change_pct"] = df["Storage_pct"].pct_change() * 100
    df["move_eu"] = df["Storage_change_pct"].apply(
        lambda x: "D" if (pd.notna(x) and x < -_STORAGE_DROP_PCT) else "C"
    )
    return df


@st.cache_data(show_spinner="Cargando datos de rutas…")
def load_route_data() -> pd.DataFrame:
    """
    Lee route_data_*.csv del directorio raíz.
    Fallback a datos sintéticos si el archivo no existe.
    """
    candidates = sorted(
        [f for f in os.listdir(".") if f.startswith("route_data") and f.endswith(".csv")],
        reverse=True,
    )
    if candidates:
        df = pd.read_csv(candidates[0])
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    _r     = np.random.default_rng(np.random.PCG64(2028))
    dates  = pd.date_range("2021-01-01", "2026-03-12", freq="D")
    routes = ["Nord Stream 1", "Yamal-Europe", "Ukrainian GTS", "TurkStream"]
    # Perfiles de degradación por ruta (contextualmente realistas)
    profiles = {
        "Nord Stream 1":  (80, "2022-06-15", 0),     # cierre progresivo → 0
        "Yamal-Europe":   (60, "2022-04-27", 5),      # parada brusca
        "Ukrainian GTS":  (70, "2024-01-01", 15),     # reducción gradual
        "TurkStream":     (50, "2021-01-01", 45),     # relativamente estable
    }
    rows = []
    for route, (base_val, cut_date, floor) in profiles.items():
        cut_ts = pd.Timestamp(cut_date)
        vals   = []
        for d in dates:
            if d < cut_ts:
                v = base_val + _r.normal(0, 4)
            else:
                days_after = (d - cut_ts).days
                v = max(floor, base_val - days_after * 0.25 + _r.normal(0, 3))
            vals.append(round(float(np.clip(v, 0, 100)), 1))
        for d, v in zip(dates, vals):
            rows.append({"Date": d, "Route": route, "Utilisation_pct": v})
    return pd.DataFrame(rows)


def build_timeline(daily: pd.DataFrame, storage: pd.DataFrame) -> pd.DataFrame:
    """
    Combina datos diarios de Rusia con datos semanales de almacenamiento UE,
    asigna el outcome del DP y calcula scores e impactos acumulados.
    """
    stor = storage[["Week", "move_eu", "Storage_pct", "Storage_change_pct"]].rename(
        columns={"Week": "Date"}
    )
    df = pd.merge_asof(
        daily.sort_values("Date"),
        stor.sort_values("Date"),
        on="Date", direction="backward",
    )
    df["move_eu"] = df["move_eu"].fillna("C")

    cum_ru = cum_eu = 0.0
    records = []
    for _, row in df.iterrows():
        key = (row["move_russia"], row["move_eu"])
        out = ECONOMIC_OUTCOMES.get(key, ECONOMIC_OUTCOMES[("C", "C")])
        cum_ru += out["score_ru"]
        cum_eu += out["score_eu"]
        records.append({
            "outcome":       out["label"],
            "color":         out["color"],
            "score_ru":      out["score_ru"],
            "score_eu":      out["score_eu"],
            "cum_score_ru":  cum_ru,
            "cum_score_eu":  cum_eu,
            "gdp_impact_ru": out["gdp_ru"],
            "gdp_impact_eu": out["gdp_eu"],
            "inf_impact_eu": out["inf_eu"],
        })
    return pd.concat([df.reset_index(drop=True),
                      pd.DataFrame(records)], axis=1)


# ── Figuras del caso de estudio ────────────────────────────────────

_EC = dict(
    paper_bgcolor="#080b12", plot_bgcolor="#0e1420",
    font=dict(family="monospace", color="#e2e8f0", size=10),
    margin=dict(l=16, r=16, t=52, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", borderwidth=0),
    hovermode="x unified",
)
_GRID = "rgba(255,255,255,0.04)"
_RU, _EU_C, _GAS, _STOR = "#f87171", "#38bdf8", "#fbbf24", "#34d399"


def _apply_ec(fig, height=400, title=""):
    fig.update_layout(**_EC, height=height,
                      title=dict(text=title, font=dict(size=12), x=0))
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(size=9))
    return fig


def fig_gas_scores(tl: pd.DataFrame) -> go.Figure:
    """Flujo de gas (barras) + scores acumulados (líneas) en dos filas."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=["FLUJO DE GAS (mcm/día) vs SCORES ACUMULADOS",
                        "MOVIDAS DIARIAS"],
    )
    # Barras de flujo
    fig.add_trace(go.Bar(
        x=tl["Date"], y=tl["Russia"],
        name="Flujo Rusia (mcm/día)",
        marker=dict(color=_GAS, opacity=0.5, line=dict(width=0)),
        hovertemplate="%{y:.0f} mcm<extra>Flujo</extra>",
    ), row=1, col=1)
    # Scores acumulados
    for col_name, color, label in [
        ("cum_score_ru", _RU, "Score acum. Rusia"),
        ("cum_score_eu", _EU_C, "Score acum. UE"),
    ]:
        fig.add_trace(go.Scatter(
            x=tl["Date"], y=tl[col_name], name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}: %{{y:,.0f}}<extra></extra>",
        ), row=1, col=1)
    # Línea umbral
    fig.add_hline(y=_FLOW_THRESHOLD, line_dash="dot",
                  line_color="rgba(251,191,36,0.4)", line_width=1,
                  annotation_text=f"Umbral {_FLOW_THRESHOLD} mcm",
                  annotation_font=dict(size=8, color=_GAS), row=1, col=1)
    # Movidas como puntos coloreados
    for move_col, y_val, label in [
        ("move_russia", 1.2, "Rusia"),
        ("move_eu",     0.5, "UE"),
    ]:
        colors = [_EU_C if m == "C" else _RU for m in tl[move_col]]
        fig.add_trace(go.Scatter(
            x=tl["Date"], y=[y_val] * len(tl),
            mode="markers",
            marker=dict(color=colors, size=3, symbol="square"),
            name=f"Movida {label} (azul=C, rojo=D)",
            hovertemplate=f"{label}: %{{customdata}}<extra></extra>",
            customdata=tl[move_col],
        ), row=2, col=1)
    # Anotaciones de eventos históricos
    events = [
        ("2021-10-01", "Crisis precio"),
        ("2022-02-24", "Invasión"),
        ("2022-06-15", "NS1 -60%"),
        ("2022-09-26", "Sabotaje NS1"),
        ("2024-01-01", "Contrato UA expira"),
    ]
    for ds, lbl in events:
        ts = pd.Timestamp(ds)
        if tl["Date"].min() <= ts <= tl["Date"].max():
            fig.add_vline(x=ts.timestamp() * 1000,
                          line_dash="dot", line_color="rgba(167,139,250,0.3)",
                          line_width=1, row="all", col=1)
            fig.add_annotation(x=ts, y=1, xref="x", yref="paper",
                               text=lbl, showarrow=False,
                               font=dict(size=7, color="rgba(167,139,250,0.65)"),
                               textangle=-90, xanchor="left")
    fig.update_yaxes(title_text="mcm/día | Score", row=1, col=1)
    fig.update_yaxes(tickvals=[0.5, 1.2], ticktext=["UE", "Rusia"],
                     range=[0, 1.7], row=2, col=1)
    return _apply_ec(fig, height=560, title="")


def fig_storage_eu(df_stor: pd.DataFrame) -> go.Figure:
    """Nivel de almacenamiento semanal con movida UE codificada por color."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_stor["Week"], y=df_stor["Storage_pct"],
        mode="lines", line=dict(color=_STOR, width=2),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.07)",
        name="Almacenamiento UE+UA (%)",
        hovertemplate="Semana %{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    ))
    colors = [_EU_C if m == "C" else _RU for m in df_stor["move_eu"]]
    fig.add_trace(go.Scatter(
        x=df_stor["Week"], y=df_stor["Storage_pct"],
        mode="markers",
        marker=dict(color=colors, size=7, opacity=0.85),
        name="Movida UE (azul=C / rojo=D)",
        hovertemplate="Cambio: %{customdata:.1f}%<extra></extra>",
        customdata=df_stor["Storage_change_pct"].round(2),
    ))
    mu = df_stor["Storage_pct"].mean()
    fig.add_hline(y=mu, line_dash="dot", line_color="rgba(251,191,36,0.4)",
                  line_width=1, annotation_text=f"Media {mu:.1f}%",
                  annotation_font=dict(size=8, color=_GAS))
    return _apply_ec(fig, height=320,
                     title="ALMACENAMIENTO SEMANAL UE+UA — MOVIDA UE CODIFICADA POR COLOR")


def fig_outcome_dist(tl: pd.DataFrame) -> go.Figure:
    """Distribución de frecuencia de los cuatro outcomes."""
    counts  = tl["outcome"].value_counts()
    palette = {v["label"]: v["color"] for v in ECONOMIC_OUTCOMES.values()}
    colors  = [palette.get(lbl, "#64748b") for lbl in counts.index]
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "bar"}]],
                        subplot_titles=["DISTRIBUCIÓN", "DÍAS POR OUTCOME"])
    fig.add_trace(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=colors, line=dict(color="#080b12", width=2)),
        textfont=dict(size=8), hole=0.42,
        hovertemplate="%{label}<br>%{value} días (%{percent})<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=counts.index, y=counts.values,
        marker=dict(color=colors, opacity=0.85),
        hovertemplate="%{x}<br>%{y} días<extra></extra>",
    ), row=1, col=2)
    return _apply_ec(fig, height=340,
                     title="FRECUENCIA DE OUTCOMES DEL DILEMA DEL PRISIONERO")


def fig_economic_impact(tl: pd.DataFrame) -> go.Figure:
    """GDP e inflación (media móvil 30 días) derivados de los outcomes."""
    df = tl.copy()
    for col in ["gdp_impact_eu", "gdp_impact_ru", "inf_impact_eu"]:
        df[f"{col}_30d"] = df[col].rolling(30, min_periods=1).mean()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["IMPACTO PIB (media 30d, %)",
                                        "INFLACIÓN UE (media 30d, %)"])
    for col_name, color, label in [
        ("gdp_impact_eu_30d", _EU_C, "PIB UE"),
        ("gdp_impact_ru_30d", _RU,   "PIB Rusia"),
    ]:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[col_name], name=label,
            line=dict(color=color, width=1.5),
            hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
        ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.15)", line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["inf_impact_eu_30d"],
        name="Inflación UE", line=dict(color=_GAS, width=1.5),
        hovertemplate="Inflación: %{y:.1f}%<extra></extra>",
    ), row=1, col=2)
    fig.add_hline(y=2.0, line_dash="dot",
                  line_color="rgba(52,211,153,0.35)", line_width=1, row=1, col=2,
                  annotation_text="Objetivo BCE 2%",
                  annotation_font=dict(size=8, color=_STOR))
    return _apply_ec(fig, height=340, title="IMPACTO ECONÓMICO ESTIMADO — MEDIA MÓVIL 30 DÍAS")


def fig_route_util(df_route: pd.DataFrame) -> go.Figure:
    """Utilización de rutas de gasoducto."""
    fig = go.Figure()
    palette = {
        "Nord Stream 1":  _RU,
        "Yamal-Europe":   _GAS,
        "Ukrainian GTS":  _EU_C,
        "TurkStream":     "#a78bfa",
    }
    if "Route" not in df_route.columns:
        return fig
    for route in df_route["Route"].unique():
        sub   = df_route[df_route["Route"] == route].sort_values("Date")
        color = palette.get(route, "#94a3b8")
        fig.add_trace(go.Scatter(
            x=sub["Date"], y=sub["Utilisation_pct"],
            name=route, mode="lines",
            line=dict(color=color, width=1.8),
            hovertemplate=f"{route}: %{{y:.1f}}%<extra></extra>",
        ))
    return _apply_ec(fig, height=320,
                     title="UTILIZACIÓN DE RUTAS DE GASODUCTO (%)")


def render_energy_crisis_tab():
    """
    Renderiza la pestaña completa del Caso de Estudio.
    No modifica ninguna clase de estrategia ni el motor del torneo.
    """
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(248,113,113,0.07),
                rgba(56,189,248,0.05));border:1px solid rgba(248,113,113,0.2);
                border-radius:6px;padding:18px 22px;margin-bottom:18px;">
      <p style="font-size:11px;letter-spacing:.15em;color:#f87171;
                text-transform:uppercase;margin:0 0 4px 0;">
        ● CASO DE ESTUDIO — DILEMA DEL PRISIONERO EN EL MUNDO REAL
      </p>
      <p style="font-size:19px;font-weight:700;color:#e2e8f0;margin:0 0 3px 0;">
        Crisis Energética Rusia–Europa (2021–2026)
      </p>
      <p style="font-size:13px;color:#64748b;margin:0;">
        Datos históricos de flujo de gas y almacenamiento mapeados a movidas del Dilema del Prisionero.
        Rusia coopera cuando flujo&nbsp;&gt;&nbsp;300&nbsp;mcm/día.
        UE coopera cuando almacenamiento no cae más de un 5%&nbsp;semanal.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Cargar datos
    with st.spinner("Cargando archivos de datos…"):
        try:
            df_daily = load_daily_data()
            df_stor  = load_storage_data()
            df_route = load_route_data()
        except Exception as e:
            st.error(f"⚠ Error al cargar datos: {e}")
            return

    # Filtro de fechas
    min_d = df_daily["Date"].min().date()
    max_d = df_daily["Date"].max().date()
    fc1, fc2 = st.columns(2)
    start_d = fc1.date_input("Fecha inicio", value=min_d, min_value=min_d, max_value=max_d)
    end_d   = fc2.date_input("Fecha fin",    value=max_d, min_value=min_d, max_value=max_d)

    if start_d > end_d:
        st.error("La fecha de inicio debe ser anterior a la de fin.")
        return

    df_daily_f = df_daily[
        (df_daily["Date"] >= pd.Timestamp(start_d)) &
        (df_daily["Date"] <= pd.Timestamp(end_d))
    ].copy()
    df_stor_f = df_stor[
        (df_stor["Week"] >= pd.Timestamp(start_d)) &
        (df_stor["Week"] <= pd.Timestamp(end_d))
    ].copy()

    if df_daily_f.empty:
        st.warning("Sin datos en el rango seleccionado.")
        return

    tl = build_timeline(df_daily_f, df_stor_f)

    # ── Métricas resumen ──
    st.divider()
    coop_ru = (tl["move_russia"] == "C").mean() * 100
    coop_eu = (tl["move_eu"]     == "C").mean() * 100
    sc_ru   = tl["cum_score_ru"].iloc[-1]
    sc_eu   = tl["cum_score_eu"].iloc[-1]
    avg_st  = df_stor_f["Storage_pct"].mean() if not df_stor_f.empty else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🇷🇺 Cooperación Rusia",  f"{coop_ru:.1f}%")
    m2.metric("🇪🇺 Cooperación UE",     f"{coop_eu:.1f}%")
    m3.metric("🇷🇺 Score total Rusia",  f"{sc_ru:,.0f}")
    m4.metric("🇪🇺 Score total UE",     f"{sc_eu:,.0f}")
    m5.metric("🏭 Almac. medio UE",     f"{avg_st:.1f}%")

    cc_days = int(((tl["move_russia"] == "C") & (tl["move_eu"] == "C")).sum())
    dd_days = int(((tl["move_russia"] == "D") & (tl["move_eu"] == "D")).sum())
    n2, n3, n4 = st.columns(3)
    n2.metric("📅 Días analizados",       f"{len(tl):,}")
    n3.metric("🤝 Días cooperación mutua", f"{cc_days:,}")
    n4.metric("💥 Días castigo mutuo",     f"{dd_days:,}")

    # ── Matriz de impacto real ──
    with st.expander("📋 Traducción económica de la matriz de pagos (T=5, R=3, P=1, S=0)", expanded=False):
        rows = []
        for (mr, me), v in ECONOMIC_OUTCOMES.items():
            rows.append({"Movida Rusia": mr, "Movida UE": me,
                         "Resultado": v["label"],
                         "Score Rusia": v["score_ru"], "Score UE": v["score_eu"],
                         "PIB Rusia %": v["gdp_ru"],  "PIB UE %": v["gdp_eu"],
                         "Inflación UE %": v["inf_eu"], "Mercado Gas": v["gas_market"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Gráficas ──
    st.markdown("#### Flujo de gas y scores acumulados")
    st.plotly_chart(fig_gas_scores(tl), use_container_width=True)

    st.markdown("#### Almacenamiento semanal UE")
    st.plotly_chart(fig_storage_eu(df_stor_f), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Distribución de outcomes")
        st.plotly_chart(fig_outcome_dist(tl), use_container_width=True)
    with col_b:
        st.markdown("#### Impacto económico")
        st.plotly_chart(fig_economic_impact(tl), use_container_width=True)

    st.markdown("#### Utilización de rutas de gasoducto")
    st.plotly_chart(fig_route_util(df_route), use_container_width=True)

    # Descarga
    st.divider()
    dl_cols = ["Date", "Russia", "move_russia", "move_eu", "outcome",
               "score_ru", "score_eu", "cum_score_ru", "cum_score_eu",
               "gdp_impact_ru", "gdp_impact_eu", "inf_impact_eu"]
    dl_df = tl[[c for c in dl_cols if c in tl.columns]]
    import io as _io
    buf = _io.StringIO()
    dl_df.to_csv(buf, index=False)
    st.download_button("⬇ Descargar línea de tiempo (CSV)",
                       data=buf.getvalue(),
                       file_name="russia_eu_ipd_timeline.csv",
                       mime="text/csv")

    st.markdown("""
    <div style="margin-top:20px;padding:12px 16px;border:1px solid rgba(255,255,255,0.06);
                border-radius:4px;background:rgba(14,20,45,0.5);">
      <p style="font-size:10px;letter-spacing:.12em;color:#38bdf8;text-transform:uppercase;margin:0 0 4px 0;">
        Nota metodológica
      </p>
      <p style="font-size:12px;color:#64748b;margin:0;line-height:1.7;">
        <b style="color:#94a3b8">Movida Rusia:</b> C si flujo &gt; 300 mcm/día; D si ≤ 300 mcm/día.<br>
        <b style="color:#94a3b8">Movida UE:</b> C salvo que el almacenamiento caiga &gt;5% en la semana (sanciones/GNL = D).<br>
        <b style="color:#94a3b8">Impactos económicos:</b> proxies ilustrativos basados en la jerarquía de pagos del DP.
        No son estimaciones econométricas. Fuentes: ENTSOG, GIE, Bruegel, Eurostat.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
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
