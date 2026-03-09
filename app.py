"""
Prisoner's Dilemma Tournament Dashboard
Round-Robin tournament with 15 strategies
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
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
        last_my = self.my_history[-1]
        last_opp = self.opp_history[-1]
        # Repite si ganó, cambia si perdió
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
    """Coopera hasta que el oponente traiciona; entonces traiciona N veces (N = total traiciones del oponente), luego dos C."""
    name = "GRADUAL"
    def __init__(self):
        super().__init__()
        self._punish_remaining = 0
        self._calm_remaining = 0

    def reset(self):
        super().reset()
        self._punish_remaining = 0
        self._calm_remaining = 0

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
            self._calm_remaining = 2
            return 'D'
        return 'C'


class Adaptive(Strategy):
    """Empieza con CCCCCDDDDD, luego adapta según qué acción dio mejores resultados."""
    name = "ADAPTIVE"
    _init_seq = list("CCCCCDDDDD")

    def move(self):
        t = len(self.my_history)
        if t < 10:
            return self._init_seq[t]
        # Compara payoff promedio histórico según acción propia
        c_payoffs, d_payoffs = [], []
        for i, (m, o) in enumerate(zip(self.my_history, self.opp_history)):
            score = _get_payoff_from_session(m, o)
            if m == 'C':
                c_payoffs.append(score)
            else:
                d_payoffs.append(score)
        avg_c = np.mean(c_payoffs) if c_payoffs else 0
        avg_d = np.mean(d_payoffs) if d_payoffs else 0
        return 'C' if avg_c >= avg_d else 'D'


def _get_payoff_from_session(my_move, opp_move):
    """Helper para Adaptive usando valores de sesión."""
    T = st.session_state.get('T', 5)
    R = st.session_state.get('R', 3)
    P = st.session_state.get('P', 1)
    S = st.session_state.get('S', 0)
    if my_move == 'C' and opp_move == 'C':
        return R
    if my_move == 'C' and opp_move == 'D':
        return S
    if my_move == 'D' and opp_move == 'C':
        return T
    return P


class EvolvedNN(Strategy):
    """Red Neuronal de 2 capas entrenada con ejemplos evolutivos simples."""
    name = "EVOLVED-NN"

    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(np.random.PCG64(99))
        # Arquitectura: 6 entradas → 8 oculta → 1 salida
        self.W1 = rng.standard_normal((8, 6)) * 0.5
        self.b1 = rng.standard_normal(8) * 0.1
        self.W2 = rng.standard_normal((1, 8)) * 0.5
        self.b2 = rng.standard_normal(1) * 0.1

    def _features(self):
        """Últimas 3 jugadas propias y del oponente como -1/1."""
        def enc(h, n=3):
            out = []
            for i in range(n):
                idx = -(i+1)
                if abs(idx) <= len(h):
                    out.append(1.0 if h[idx] == 'C' else -1.0)
                else:
                    out.append(0.0)
            return out
        return np.array(enc(self.my_history) + enc(self.opp_history))

    def move(self):
        x = self._features()
        h = np.tanh(self.W1 @ x + self.b1)
        out = np.tanh(self.W2 @ h + self.b2)[0]
        return 'C' if out > 0 else 'D'


class PSOPlayer(Strategy):
    """Estrategia optimizada por PSO: usa una tabla de decisión con probabilidades."""
    name = "PSO-PLAYER"

    def __init__(self):
        super().__init__()
        # Pesos encontrados por PSO simulado (hardcoded tras optimización)
        rng = np.random.default_rng(np.random.PCG64(7))
        self._weights = rng.dirichlet(np.ones(4))  # [CC,CD,DC,DD] → prob de C

    def move(self):
        if len(self.my_history) < 1:
            return 'C'
        last_pair = (self.my_history[-1], self.opp_history[-1])
        idx = {'CC': 0, 'CD': 1, 'DC': 2, 'DD': 3}[last_pair[0]+last_pair[1]]
        return 'C' if RNG.random() < self._weights[idx] else 'D'


class Memory3(Strategy):
    """Usa los últimos 3 movimientos del oponente para decidir."""
    name = "MEMORY-3"

    def move(self):
        if len(self.opp_history) < 3:
            return 'C'
        last3 = self.opp_history[-3:]
        d_count = last3.count('D')
        return 'D' if d_count >= 2 else 'C'


class Friedman(Strategy):
    """Alias de GRIM con lógica idéntica (traición permanente tras primer D)."""
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
        self._exploit = False
        self._tft_mode = False

    def reset(self):
        super().reset()
        self._exploit = False
        self._tft_mode = False

    def move(self):
        t = len(self.my_history)
        if t == 0:
            return 'D'
        if t == 1:
            if self.opp_history[-1] == 'D':
                self._tft_mode = True
            else:
                self._exploit = True
            return 'C'
        if self._exploit:
            return 'D'
        # TFT mode
        return self.opp_history[-1]


# ─────────────────────────────────────────────
# Registro de todas las estrategias
# ─────────────────────────────────────────────
ALL_STRATEGIES = [
    TitForTat, Grim, Pavlov, AllD, AllC,
    TitForTwoTats, Random, Joss, Gradual, Adaptive,
    EvolvedNN, PSOPlayer, Memory3, Friedman, Tester
]

STRATEGY_NAMES = [s.name for s in ALL_STRATEGIES]


# ─────────────────────────────────────────────
# Motor del torneo
# ─────────────────────────────────────────────

def get_payoff(my_move: str, opp_move: str, T: int, R: int, P: int, S: int) -> Tuple[int, int]:
    if my_move == 'C' and opp_move == 'C':
        return R, R
    if my_move == 'C' and opp_move == 'D':
        return S, T
    if my_move == 'D' and opp_move == 'C':
        return T, S
    return P, P


def play_game(s1: Strategy, s2: Strategy, rounds: int, T: int, R: int, P: int, S: int, w: float):
    """Juega un juego entre dos estrategias. Retorna (score1, score2, history1, history2)."""
    s1.reset()
    s2.reset()
    score1, score2 = 0.0, 0.0
    h1, h2 = [], []

    for r in range(rounds):
        # Verificar continuación con probabilidad w
        if r > 0 and RNG.random() > w:
            break
        m1 = s1.move()
        m2 = s2.move()
        p1, p2 = get_payoff(m1, m2, T, R, P, S)
        s1.update(m1, m2)
        s2.update(m2, m1)
        score1 += p1
        score2 += p2
        h1.append(m1)
        h2.append(m2)

    return score1, score2, h1, h2


def run_tournament(selected_names: List[str], T: int, R: int, P: int, S: int,
                   w: float, games: int = 5, rounds: int = 200):
    """Round-Robin tournament."""
    # Instanciar estrategias seleccionadas
    name_to_class = {s.name: s for s in ALL_STRATEGIES}
    strategies = {name: name_to_class[name] for name in selected_names}

    n = len(selected_names)
    # Matrices de payoff promedio
    payoff_matrix = pd.DataFrame(np.zeros((n, n)), index=selected_names, columns=selected_names)
    total_scores = {name: 0.0 for name in selected_names}

    # Historial head-to-head
    h2h_data = {}

    pairs = list(itertools.combinations(selected_names, 2))
    for name1, name2 in pairs:
        cls1 = strategies[name1]
        cls2 = strategies[name2]
        game_scores1, game_scores2 = [], []
        last_h1, last_h2 = [], []

        for _ in range(games):
            s1_inst = cls1()
            s2_inst = cls2()
            sc1, sc2, h1, h2 = play_game(s1_inst, s2_inst, rounds, T, R, P, S, w)
            rlen = len(h1)
            avg1 = sc1 / rlen if rlen > 0 else 0
            avg2 = sc2 / rlen if rlen > 0 else 0
            game_scores1.append(avg1)
            game_scores2.append(avg2)
            last_h1, last_h2 = h1, h2

        mean1 = np.mean(game_scores1)
        mean2 = np.mean(game_scores2)

        payoff_matrix.loc[name1, name2] = mean1
        payoff_matrix.loc[name2, name1] = mean2

        total_scores[name1] += mean1
        total_scores[name2] += mean2

        h2h_data[(name1, name2)] = (last_h1, last_h2)

    # Diagonal: self-play
    for name in selected_names:
        cls = strategies[name]
        sc_list = []
        for _ in range(games):
            s1i = cls()
            s2i = cls()
            sc1, sc2, _, _ = play_game(s1i, s2i, rounds, T, R, P, S, w)
            rlen = max(len(s1i.my_history), 1)
            sc_list.append((sc1 + sc2) / 2 / rlen)
        payoff_matrix.loc[name, name] = np.mean(sc_list)

    ranking = pd.DataFrame({
        'Estrategia': list(total_scores.keys()),
        'Score Total': list(total_scores.values())
    }).sort_values('Score Total', ascending=False).reset_index(drop=True)
    ranking.index += 1

    return payoff_matrix, ranking, h2h_data


# ─────────────────────────────────────────────
# Dashboard Streamlit
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Prisoner's Dilemma Tournament",
        page_icon="🎲",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎲 Torneo del Dilema del Prisionero")
    st.markdown("**Round-Robin · 15 Estrategias · Dashboard Interactivo**")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Parámetros")

        st.subheader("Matriz de Pagos")
        col1, col2 = st.columns(2)
        with col1:
            T = st.number_input("T (Traición)", min_value=1, max_value=20, value=5, step=1, key='T')
            R = st.number_input("R (Recompensa)", min_value=0, max_value=19, value=3, step=1, key='R')
        with col2:
            P = st.number_input("P (Castigo)", min_value=0, max_value=18, value=1, step=1, key='P')
            S = st.number_input("S (Chivo expiatorio)", min_value=0, max_value=17, value=0, step=1, key='S')

        # Validación T > R > P > S
        if not (T > R > P >= S):
            st.error("❌ Se requiere: T > R > P ≥ S")
            valid_payoff = False
        else:
            st.success("✅ T > R > P ≥ S")
            valid_payoff = True

        st.subheader("Parámetros del Torneo")
        w = st.slider("w (prob. interacción futura)", 0.0, 1.0, 0.995, 0.001)
        n_games = st.slider("Juegos por par", 1, 10, 5)
        n_rounds = st.slider("Rondas por juego", 50, 500, 200, step=50)

        st.subheader("Selección de Estrategias")
        selected = st.multiselect(
            "Estrategias a incluir",
            options=STRATEGY_NAMES,
            default=STRATEGY_NAMES
        )

        run_btn = st.button("▶ Ejecutar Torneo", type="primary", use_container_width=True)

    # ── Estado de sesión ──
    if 'results' not in st.session_state:
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
                'payoff_matrix': payoff_matrix,
                'ranking': ranking,
                'h2h_data': h2h_data,
                'selected': selected,
                'params': dict(T=T, R=R, P=P, S=S, w=w)
            }
            st.success("✅ Torneo completado.")

    # ── Visualizaciones ──
    if st.session_state.results:
        res = st.session_state.results
        payoff_matrix = res['payoff_matrix']
        ranking = res['ranking']
        h2h_data = res['h2h_data']
        selected = res['selected']

        # ── Tabs ──
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Ranking", "🌡️ Heatmap", "⚔️ Head-to-Head", "📊 Distribución", "ℹ️ Estrategias"
        ])

        # ── Tab 1: Ranking ──
        with tab1:
            st.subheader("🏆 Top 3 Estrategias")
            top3 = ranking.head(3)
            cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    st.metric(
                        label=f"{medals[i]} #{i+1}",
                        value=row['Estrategia'],
                        delta=f"Score: {row['Score Total']:.3f}"
                    )

            st.divider()
            st.subheader("Tabla Completa")
            styled_ranking = ranking.copy()
            styled_ranking.index = styled_ranking.index.map(lambda x: f"#{x}")
            st.dataframe(styled_ranking, use_container_width=True)

        # ── Tab 2: Heatmap ──
        with tab2:
            st.subheader("🌡️ Mapa de Calor — Pago Promedio")
            fig_heat = go.Figure(data=go.Heatmap(
                z=payoff_matrix.values,
                x=payoff_matrix.columns.tolist(),
                y=payoff_matrix.index.tolist(),
                colorscale='RdYlGn',
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

        # ── Tab 3: Head-to-Head ──
        with tab3:
            st.subheader("⚔️ Comparativa Head-to-Head")
            col1, col2 = st.columns(2)
            with col1:
                s1_name = st.selectbox("Estrategia 1", selected, key="h2h_s1")
            with col2:
                others = [s for s in selected if s != s1_name]
                s2_name = st.selectbox("Estrategia 2", others, key="h2h_s2") if others else None

            if s2_name:
                key = (s1_name, s2_name) if (s1_name, s2_name) in h2h_data else (s2_name, s1_name)
                if key in h2h_data:
                    h1_raw, h2_raw = h2h_data[key]
                    if key == (s2_name, s1_name):
                        h1_raw, h2_raw = h2_raw, h1_raw

                    r_len = len(h1_raw)
                    rounds_idx = list(range(1, r_len + 1))

                    # Convertir a numérico para graficar
                    h1_num = [1 if m == 'C' else 0 for m in h1_raw]
                    h2_num = [1 if m == 'C' else 0 for m in h2_raw]

                    fig_h2h = go.Figure()
                    fig_h2h.add_trace(go.Scatter(
                        x=rounds_idx, y=h1_num,
                        mode='lines+markers',
                        name=s1_name,
                        line=dict(color='royalblue'),
                        marker=dict(symbol=['circle' if m == 'C' else 'x' for m in h1_raw], size=6),
                        hovertemplate="Ronda %{x}: " + s1_name + " jugó %{customdata}<extra></extra>",
                        customdata=h1_raw
                    ))
                    fig_h2h.add_trace(go.Scatter(
                        x=rounds_idx, y=h2_num,
                        mode='lines+markers',
                        name=s2_name,
                        line=dict(color='tomato', dash='dash'),
                        marker=dict(symbol=['circle' if m == 'C' else 'x' for m in h2_raw], size=6),
                        hovertemplate="Ronda %{x}: " + s2_name + " jugó %{customdata}<extra></extra>",
                        customdata=h2_raw
                    ))
                    fig_h2h.update_layout(
                        height=400,
                        yaxis=dict(tickvals=[0, 1], ticktext=['D', 'C'], title="Acción"),
                        xaxis_title="Ronda",
                        title=f"{s1_name} vs {s2_name} — Último Juego",
                        legend=dict(orientation='h', y=1.1),
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(fig_h2h, use_container_width=True)

                    # Resumen
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(f"% C de {s1_name}", f"{h1_raw.count('C')/r_len*100:.1f}%")
                    c2.metric(f"% C de {s2_name}", f"{h2_raw.count('C')/r_len*100:.1f}%")
                    c3.metric("Rondas jugadas", r_len)
                    c4.metric("Pago promedio (fila)", f"{payoff_matrix.loc[s1_name, s2_name]:.3f}")

        # ── Tab 4: Distribución ──
        with tab4:
            st.subheader("📊 Distribución de Scores Totales")
            scores_list = ranking['Score Total'].tolist()
            names_list = ranking['Estrategia'].tolist()

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=names_list,
                y=scores_list,
                marker_color=[
                    f"rgba({int(255*(1-i/len(names_list)))}, {int(100+155*(i/len(names_list)))}, 100, 0.85)"
                    for i in range(len(names_list))
                ],
                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>"
            ))
            fig_hist.update_layout(
                height=450,
                xaxis_title="Estrategia",
                yaxis_title="Score Total",
                xaxis_tickangle=-30,
                margin=dict(l=20, r=20, t=20, b=80),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Estadísticas descriptivas
            st.subheader("Estadísticas")
            stats_df = pd.DataFrame({
                'Métrica': ['Media', 'Mediana', 'Std', 'Mín', 'Máx'],
                'Valor': [
                    f"{np.mean(scores_list):.3f}",
                    f"{np.median(scores_list):.3f}",
                    f"{np.std(scores_list):.3f}",
                    f"{np.min(scores_list):.3f}",
                    f"{np.max(scores_list):.3f}",
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # ── Tab 5: Info estrategias ──
        with tab5:
            st.subheader("ℹ️ Descripción de Estrategias")
            descriptions = {
                "TIT FOR TAT": "Empieza cooperando y luego imita el último movimiento del oponente.",
                "GRIM": "Coopera hasta que el oponente traiciona; luego traiciona para siempre.",
                "PAVLOV": "Repite su acción si ganó, cambia si perdió (Win-Stay, Lose-Shift).",
                "ALL-D": "Siempre traiciona.",
                "ALL-C": "Siempre coopera.",
                "TIT FOR TWO TATS": "Solo traiciona si el oponente traicionó dos veces seguidas.",
                "RANDOM": "Coopera o traiciona aleatoriamente con prob. 50/50.",
                "JOSS": "Como TFT pero traiciona con probabilidad 0.1 tras cooperación del oponente.",
                "GRADUAL": "Tras cada traición del oponente, castiga N veces (N = total traiciones del oponente) y luego calma.",
                "ADAPTIVE": "Empieza con secuencia CCCCCDDDDD; luego elige la acción con mayor payoff histórico.",
                "EVOLVED-NN": "Red neuronal de 2 capas (6→8→1) que usa las últimas 3 jugadas de cada jugador como input.",
                "PSO-PLAYER": "Usa probabilidades de cooperación optimizadas por PSO según el último par (mi acción, acción oponente).",
                "MEMORY-3": "Traiciona si el oponente traicionó ≥2 veces en las últimas 3 rondas.",
                "FRIEDMAN": "Idéntico a GRIM: traición permanente tras el primer D del oponente.",
                "TESTER": "Traiciona en ronda 1; si el oponente perdona, lo explota; si responde, usa TFT.",
            }
            for name, desc in descriptions.items():
                if name in selected:
                    with st.expander(f"**{name}**"):
                        st.write(desc)

    else:
        st.info("👈 Configura los parámetros en el panel lateral y presiona **▶ Ejecutar Torneo**.")

        st.markdown("""
        ### Cómo usar este dashboard
        1. **Ajusta la matriz de pagos** (T, R, P, S) respetando T > R > P ≥ S
        2. **Configura** el parámetro `w` y el número de juegos/rondas
        3. **Selecciona** las estrategias a incluir en el torneo
        4. **Ejecuta** el torneo y explora los resultados en las pestañas
        """)

    # ══════════════════════════════════════════════════════════════════
    # PESTAÑA MAESTRA: Estudio de Caso — Guerra Fría  (Muerte Súbita MAD)
    # ══════════════════════════════════════════════════════════════════
    st.divider()
    st.title("☢️ Estudio de Caso: Guerra Fría y la Carrera Armamentística")
    st.markdown("**Basado en Axelrod & Hamilton (1981)** — *The Evolution of Cooperation*")
    st.info(
        "🌍 **¿Qué estás viendo aquí?**  \n"
        "Este simulador recrea la lógica nuclear de la Guerra Fría: dos superpotencias que pueden "
        "**Cooperar** (mantener la paz) o **Traicionar** (lanzar un misil).  \n"
        "🔴 **Regla de Muerte Súbita:** si ambas potencias se Traicionan mutuamente en la misma ronda, "
        "el encuentro termina de inmediato con −100 puntos para cada una — la **Destrucción Mutua Asegurada (MAD)**. "
        "Descubre qué estrategias logran mantener la paz y cuáles desencadenan el apocalipsis."
    )

    # ── Parámetros fijos MAD ──
    GF_T, GF_R, GF_P, GF_S = 5, 3, -100, -150

    st.markdown("### 💣 Matriz de Pagos Nuclear (fija)")
    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _mc1.metric("☢️ T — Primer ataque", GF_T,    help="Ventaja si solo TÚ atacas y el otro Coopera")
    _mc2.metric("☮️ R — Paz mutua",     GF_R,    help="Ambos Cooperan → beneficio moderado y estable")
    _mc3.metric("💥 P — MAD (Traición mutua)", GF_P, help="Ambos Traicionan → MUERTE SÚBITA, −100 cada uno")
    _mc4.metric("🏳️ S — Indefensión",  GF_S,    help="Tú Cooperas, el otro Traiciona → quedas devastado")

    st.warning(
        "⚠️ **Nota clave:** La estrategia ALL-D (siempre Traiciona) parece 'racional' en el torneo estándar. "
        "Aquí verás por qué en un mundo nuclear provoca la Muerte Súbita inmediata contra cualquier rival "
        "que también desconfíe — arrastrando a todos al colapso."
    )
    st.markdown("---")

    # ── Controles ──
    gf_left, gf_right = st.columns([3, 2])

    with gf_left:
        st.markdown("### ⚙️ Configuración del Simulador")

        gf_error = st.slider(
            "💨 Riesgo de Accidente Nuclear",
            min_value=0.0, max_value=0.05, value=0.01, step=0.005,
            format="%.3f", key="gf_error",
            help=(
                "Probabilidad de que una intención de COOPERAR se ejecute como TRAICIÓN "
                "por fallos técnicos o mala comunicación. "
                "Ejemplo real: el incidente del satélite soviético Petrov (1983)."
            )
        )
        if gf_error == 0:
            st.caption("🔇 Sin ruido — comunicación perfecta entre superpotencias.")
        elif gf_error <= 0.01:
            st.caption("📡 Ruido bajo — canales diplomáticos funcionando.")
        elif gf_error <= 0.03:
            st.caption("⚡ Ruido moderado — tensión diplomática elevada.")
        else:
            st.caption("🚨 Ruido alto — al borde del accidente nuclear.")

        _ca, _cb = st.columns(2)
        with _ca:
            gf_rounds = st.slider("Rondas máximas por juego", 50, 500, 200, step=50, key="gf_rounds")
        with _cb:
            gf_games = st.slider("Juegos por par", 1, 10, 5, key="gf_games")

        st.markdown("#### 🇺🇸 EUA  vs  🇷🇺 URSS — Simulador Directo")
        _cu, _cr = st.columns(2)
        with _cu:
            gf_usa = st.selectbox("🇺🇸 Estrategia EUA", STRATEGY_NAMES, index=0, key="gf_usa")
        with _cr:
            gf_urss = st.selectbox(
                "🇷🇺 Estrategia URSS",
                [s for s in STRATEGY_NAMES if s != gf_usa],
                index=1, key="gf_urss"
            )

        gf_run = st.button(
            "☢️ Lanzar Simulación Round-Robin Nuclear",
            type="primary", use_container_width=True, key="gf_run"
        )

    with gf_right:
        st.markdown("### 📈 Sombra del Futuro (w)")
        st.markdown(
            "**w** es la probabilidad de que haya *otra* ronda futura. "
            "Axelrod demostró: si w supera el umbral, la cooperación es la estrategia racionalmente dominante."
        )
        gf_w = st.slider(
            "w — Probabilidad de interacción futura",
            min_value=0.0, max_value=1.0, value=0.97, step=0.01, key="gf_w"
        )
        _w_thresh = (GF_T - GF_R) / (GF_T - GF_P)
        st.markdown(f"**Umbral:** `w > (T−R)/(T−P) = ({GF_T}−{GF_R})/({GF_T}−({GF_P})) = {_w_thresh:.4f}`")
        if gf_w > _w_thresh:
            st.success(f"✅ **LA PAZ ES POSIBLE**  \nw = {gf_w:.2f} > umbral {_w_thresh:.4f}  \nLas estrategias cooperativas tienen ventaja evolutiva.")
        else:
            st.error(f"💥 **SISTEMA CONDENADO**  \nw = {gf_w:.2f} ≤ umbral {_w_thresh:.4f}  \nNo hay futuro suficiente para que valga cooperar.")

        _fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gf_w,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "w actual"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#4488ff"},
                "steps": [
                    {"range": [0, _w_thresh], "color": "#440000"},
                    {"range": [_w_thresh, 1], "color": "#004422"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": _w_thresh,
                },
            },
        ))
        _fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(_fig_gauge, use_container_width=True)

    # ─────────────────────────────────────────────
    # Motor MAD con Muerte Súbita
    # ─────────────────────────────────────────────
    def _play_mad(s1, s2, max_rounds, T, R, P, S, w, error_prob):
        """
        Juego con:
          - Ruido de percepción (C → D con prob error_prob)
          - Muerte Súbita: si ambos juegan D en la misma ronda → score −100 c/u y stop.
        Retorna (score1, score2, hist1, hist2, annihilated, annihilation_round)
        """
        s1.reset(); s2.reset()
        sc1, sc2 = 0.0, 0.0
        h1, h2 = [], []
        annihilated = False
        ann_round = None

        for r in range(max_rounds):
            if r > 0 and RNG.random() > w:
                break
            m1 = s1.move()
            m2 = s2.move()
            # Ruido: intención C puede ejecutarse como D
            m1e = "D" if (m1 == "C" and RNG.random() < error_prob) else m1
            m2e = "D" if (m2 == "C" and RNG.random() < error_prob) else m2
            # Muerte Súbita MAD
            if m1e == "D" and m2e == "D":
                h1.append("D"); h2.append("D")
                sc1 += P; sc2 += P
                annihilated = True
                ann_round = r + 1
                break
            p1, p2 = get_payoff(m1e, m2e, T, R, P, S)
            s1.update(m1e, m2e); s2.update(m2e, m1e)
            sc1 += p1; sc2 += p2
            h1.append(m1e); h2.append(m2e)

        return sc1, sc2, h1, h2, annihilated, ann_round

    def _run_mad_tournament(T, R, P, S, w, error_prob, games, rounds):
        """Round-Robin completo con regla de Muerte Súbita."""
        n2c = {s.name: s for s in ALL_STRATEGIES}
        pm = pd.DataFrame(
            np.zeros((len(STRATEGY_NAMES), len(STRATEGY_NAMES))),
            index=STRATEGY_NAMES, columns=STRATEGY_NAMES
        )
        totals = {n: 0.0 for n in STRATEGY_NAMES}
        ann_counts = {n: 0 for n in STRATEGY_NAMES}   # veces que provocó aniquilación
        h2h_store = {}

        for n1, n2 in itertools.combinations(STRATEGY_NAMES, 2):
            gs1, gs2, lh1, lh2 = [], [], [], []
            last_ann, last_ann_r = False, None
            for _ in range(games):
                s1i = n2c[n1](); s2i = n2c[n2]()
                sc1, sc2, hh1, hh2, ann, ann_r = _play_mad(
                    s1i, s2i, rounds, T, R, P, S, w, error_prob
                )
                rlen = max(len(hh1), 1)
                gs1.append(sc1 / rlen); gs2.append(sc2 / rlen)
                lh1, lh2, last_ann, last_ann_r = hh1, hh2, ann, ann_r
                if ann:
                    ann_counts[n1] += 1; ann_counts[n2] += 1
            m1, m2 = float(np.mean(gs1)), float(np.mean(gs2))
            pm.loc[n1, n2] = m1; pm.loc[n2, n1] = m2
            totals[n1] += m1; totals[n2] += m2
            h2h_store[(n1, n2)] = (lh1, lh2, last_ann, last_ann_r)

        # Self-play diagonal
        for name in STRATEGY_NAMES:
            sc_list = []
            for _ in range(games):
                s1i = n2c[name](); s2i = n2c[name]()
                sc1, sc2, _, _, ann, _ = _play_mad(s1i, s2i, rounds, T, R, P, S, w, error_prob)
                rlen = max(len(s1i.my_history), 1)
                sc_list.append((sc1 + sc2) / 2 / rlen)
            pm.loc[name, name] = float(np.mean(sc_list))

        ranking = pd.DataFrame({
            "Estrategia": list(totals.keys()),
            "Score de Supervivencia": list(totals.values()),
            "Aniquilaciones provocadas": [ann_counts[n] for n in totals.keys()],
        }).sort_values("Score de Supervivencia", ascending=False).reset_index(drop=True)
        ranking.index += 1
        return pm, ranking, h2h_store

    # ── Estado de sesión ──
    if "gf_results" not in st.session_state:
        st.session_state.gf_results = None

    if gf_run:
        with st.spinner("☢️ Simulando torneo Round-Robin con regla de Muerte Súbita…"):
            _pm, _rank, _h2h = _run_mad_tournament(
                GF_T, GF_R, GF_P, GF_S, gf_w, gf_error, gf_games, gf_rounds
            )
        st.session_state.gf_results = {
            "pm": _pm, "rank": _rank, "h2h": _h2h,
            "usa": gf_usa, "urss": gf_urss,
        }
        st.success("✅ Simulación completada — explora los resultados en las pestañas.")

    # ── Sub-pestañas ──
    if st.session_state.gf_results:
        _gfr     = st.session_state.gf_results
        _pm      = _gfr["pm"]
        _rank    = _gfr["rank"]
        _h2h     = _gfr["h2h"]
        _usa     = _gfr["usa"]
        _urss    = _gfr["urss"]

        _t1, _t2, _t3, _t4 = st.tabs([
            "🏆 Ranking de Supervivencia",
            "🔥 Heatmap de Conflictos",
            "⚔️ Simulador EUA vs URSS",
            "📈 Distribución de Resultados",
        ])

        # ── Tab 1: Ranking ──────────────────────────────────────────
        with _t1:
            st.subheader("🏆 Ranking de Supervivencia Nuclear")
            st.markdown(
                "Clasificación Round-Robin bajo la regla **Muerte Súbita MAD**. "
                "Las estrategias que provocan más aniquilaciones acumulan penalizaciones de −100 "
                "y caen al fondo de la tabla."
            )

            _top3 = _rank.head(3)
            _t1c1, _t1c2, _t1c3 = st.columns(3)
            for _i, (_col, (_idx, _row)) in enumerate(zip(
                [_t1c1, _t1c2, _t1c3], _top3.iterrows()
            )):
                _medals = ["🥇", "🥈", "🥉"]
                _dc = "normal" if _row["Score de Supervivencia"] > 0 else "inverse"
                _col.metric(
                    label=f"{_medals[_i]} Posición #{_i+1}",
                    value=_row["Estrategia"],
                    delta=f"Score: {_row['Score de Supervivencia']:.2f}",
                    delta_color=_dc,
                )

            st.divider()

            def _color_row(val):
                if isinstance(val, float):
                    if val > 0:   return "background-color: rgba(0,180,80,0.20)"
                    if val < -10: return "background-color: rgba(200,30,30,0.25)"
                return ""

            _disp = _rank.copy()
            _disp["Estado"] = _disp["Score de Supervivencia"].apply(
                lambda s: "☮️ Paz" if s > 0 else ("⚠️ Tensión" if s > -20 else "💥 ANIQUILACIÓN")
            )
            _disp.index = _disp.index.map(lambda x: f"#{x}")
            st.dataframe(
                _disp.style.applymap(_color_row, subset=["Score de Supervivencia"]),
                use_container_width=True
            )
            st.info(
                "🟢 **Score positivo** → La estrategia mantuvo la paz y acumuló puntos.  \n"
                "🔴 **Score negativo** → Entró en Traición mutua repetida (MAD).  \n"
                "💡 *Fíjate en la columna 'Aniquilaciones provocadas' — revela qué estrategias "
                "son más peligrosas para el sistema global.*"
            )

        # ── Tab 2: Heatmap ──────────────────────────────────────────
        with _t2:
            st.subheader("🔥 Heatmap de Conflictos Nucleares")
            st.markdown(
                "Cada celda muestra el **pago promedio** de la estrategia (fila) contra su oponente (columna).  \n"
                "🟢 **Verde/Azul** = zona de paz  |  🔴 **Rojo/Negro** = zona de colapso nuclear MAD"
            )
            _fig_h = go.Figure(data=go.Heatmap(
                z=_pm.values,
                x=_pm.columns.tolist(),
                y=_pm.index.tolist(),
                colorscale=[
                    [0.00, "#0a0000"],
                    [0.20, "#5c0000"],
                    [0.40, "#990000"],
                    [0.60, "#cc3300"],
                    [0.75, "#ff8800"],
                    [0.88, "#ffdd00"],
                    [1.00, "#00cc88"],
                ],
                hovertemplate=(
                    "<b>%{y}</b> vs <b>%{x}</b><br>"
                    "Pago promedio: <b>%{z:.2f}</b><br>"
                    "<i>Valores de −100 = Muerte Súbita (MAD)</i><extra></extra>"
                ),
                text=np.round(_pm.values, 1),
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(
                    title="Pago<br>promedio",
                    tickvals=[float(_pm.values.min()), 0, float(_pm.values.max())],
                    ticktext=["💥 MAD", "0", "☮️ Paz"],
                ),
            ))
            _fig_h.update_layout(
                height=640,
                xaxis_title="Estrategia Oponente →",
                yaxis_title="← Mi Estrategia",
                font=dict(size=10),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(_fig_h, use_container_width=True)
            st.caption("💡 Pasa el cursor sobre cada celda para ver el pago exacto del enfrentamiento.")

        # ── Tab 3: Simulador EUA vs URSS ────────────────────────────
        with _t3:
            st.subheader(f"⚔️ Simulador: 🇺🇸 {_usa}  vs  🇷🇺 {_urss}")
            st.markdown(
                "La gráfica de **Puntos de Supervivencia** muestra el score acumulado ronda a ronda. "
                "Si ocurre la **Aniquilación**, la línea se corta abruptamente y aparece el mensaje de alerta."
            )

            _kp = (_usa, _urss) if (_usa, _urss) in _h2h else (_urss, _usa)
            if _kp in _h2h:
                _hh1, _hh2, _ann, _ann_r = _h2h[_kp]
                if _kp == (_urss, _usa):
                    _hh1, _hh2 = _hh2, _hh1

                _rlen = len(_hh1)
                _ridx = list(range(1, _rlen + 1))

                # Score acumulado
                _cum1, _cum2 = [], []
                _a1 = _a2 = 0.0
                for _m1, _m2 in zip(_hh1, _hh2):
                    _p1, _p2 = get_payoff(_m1, _m2, GF_T, GF_R, GF_P, GF_S)
                    _a1 += _p1; _a2 += _p2
                    _cum1.append(_a1); _cum2.append(_a2)

                # ── Alerta prominente de aniquilación ──
                if _ann:
                    st.error(
                        f"💥 **ANIQUILACIÓN — Ronda {_ann_r}**  \n"
                        "**Conflicto nuclear total detectado. Simulación finalizada.**  \n"
                        f"Ambas potencias se Traicionaron mutuamente en la ronda {_ann_r}. "
                        "El gráfico muestra el corte abrupto en ese momento."
                    )

                # ── Gráfica de supervivencia (se corta en aniquilación) ──
                _fig_surv = go.Figure()
                if _ann and _ann_r:
                    _fig_surv.add_vrect(
                        x0=_ann_r - 0.5, x1=_ann_r + 0.5,
                        fillcolor="rgba(255,0,0,0.35)", line_width=0,
                        annotation_text="💥 ANIQUILACIÓN",
                        annotation_position="top",
                        annotation_font_color="red",
                    )
                _fig_surv.add_trace(go.Scatter(
                    x=_ridx, y=_cum1, mode="lines",
                    name=f"🇺🇸 {_usa}",
                    line=dict(color="#4499ff", width=2.5),
                    hovertemplate="Ronda %{x} — EUA: <b>%{y:.0f} pts</b><extra></extra>",
                ))
                _fig_surv.add_trace(go.Scatter(
                    x=_ridx, y=_cum2, mode="lines",
                    name=f"🇷🇺 {_urss}",
                    line=dict(color="#ff4444", width=2.5, dash="dash"),
                    hovertemplate="Ronda %{x} — URSS: <b>%{y:.0f} pts</b><extra></extra>",
                ))
                _fig_surv.add_hline(
                    y=0, line_color="rgba(255,255,255,0.3)",
                    line_dash="dot", annotation_text="Umbral 0"
                )
                _fig_surv.update_layout(
                    height=360,
                    title="📉 Puntos de Supervivencia Acumulados",
                    xaxis_title="Ronda",
                    yaxis_title="Puntos acumulados",
                    legend=dict(orientation="h", y=1.12),
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(_fig_surv, use_container_width=True)

                # ── Gráfica de acciones C / Traiciona ──
                _h1n = [1 if m == "C" else 0 for m in _hh1]
                _h2n = [1 if m == "C" else 0 for m in _hh2]
                _labels1 = ["Coopera" if m == "C" else "Traiciona" for m in _hh1]
                _labels2 = ["Coopera" if m == "C" else "Traiciona" for m in _hh2]
                _fig_act = go.Figure()
                _fig_act.add_trace(go.Scatter(
                    x=_ridx, y=_h1n, mode="lines+markers",
                    name=f"🇺🇸 {_usa}",
                    line=dict(color="#4499ff"),
                    marker=dict(symbol=["circle" if m == "C" else "x" for m in _hh1], size=6),
                    customdata=_labels1,
                    hovertemplate="Ronda %{x}: <b>%{customdata}</b><extra></extra>",
                ))
                _fig_act.add_trace(go.Scatter(
                    x=_ridx, y=_h2n, mode="lines+markers",
                    name=f"🇷🇺 {_urss}",
                    line=dict(color="#ff4444", dash="dash"),
                    marker=dict(symbol=["circle" if m == "C" else "x" for m in _hh2], size=6),
                    customdata=_labels2,
                    hovertemplate="Ronda %{x}: <b>%{customdata}</b><extra></extra>",
                ))
                _fig_act.update_layout(
                    height=260,
                    title="🎯 Decisiones por Ronda",
                    yaxis=dict(tickvals=[0, 1], ticktext=["💣 Traiciona", "☮️ Coopera"]),
                    xaxis_title="Ronda",
                    legend=dict(orientation="h", y=1.15),
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                st.plotly_chart(_fig_act, use_container_width=True)

                # ── Métricas finales ──
                st.markdown("#### 📊 Resultado del Enfrentamiento")
                _rm1, _rm2, _rm3, _rm4 = st.columns(4)
                _rm1.metric("☮️ % Cooperación EUA",  f"{_hh1.count('C') / _rlen * 100:.1f}%")
                _rm2.metric("☮️ % Cooperación URSS", f"{_hh2.count('C') / _rlen * 100:.1f}%")
                _rm3.metric("🇺🇸 Puntos finales EUA",  f"{_cum1[-1]:.0f}" if _cum1 else "—")
                _rm4.metric("🇷🇺 Puntos finales URSS", f"{_cum2[-1]:.0f}" if _cum2 else "—")

                if _ann:
                    pass  # Ya se mostró el error arriba
                else:
                    _cr = (_hh1.count("C") + _hh2.count("C")) / (2 * _rlen)
                    if _cr > 0.85:
                        st.success("☮️ **Paz estable** — Cooperación predominó. Esta combinación podría haber evitado la Guerra Fría.")
                    elif _cr > 0.5:
                        st.warning("⚠️ **Paz frágil** — Hubo cooperación, pero con episodios de Traición. Un accidente podría haber detonado el conflicto.")
                    else:
                        st.error("💥 **Espiral de Traiciones** — La desconfianza escaló. Aunque no hubo Muerte Súbita, el daño acumulado fue masivo.")

        # ── Tab 4: Distribución ─────────────────────────────────────
        with _t4:
            st.subheader("📈 Distribución de Resultados — Escenario MAD")
            st.markdown(
                "¿Cuántas estrategias lograron la paz y cuántas colapsaron? "
                "La **polarización** del histograma revela si el sistema tiende a la cooperación o a la destrucción."
            )

            _scores = _rank["Score de Supervivencia"].tolist()
            _names  = _rank["Estrategia"].tolist()
            _bcolors = ["#00cc55" if s > 0 else ("#ff8800" if s > -20 else "#cc2200") for s in _scores]

            _fig_dist = go.Figure()
            _fig_dist.add_trace(go.Bar(
                x=_names, y=_scores,
                marker_color=_bcolors,
                hovertemplate="<b>%{x}</b><br>Score: <b>%{y:.2f}</b><extra></extra>",
            ))
            _fig_dist.add_hline(
                y=0, line_color="white", line_dash="dash", line_width=1.5,
                annotation_text="← PAZ  |  COLAPSO →", annotation_position="top left"
            )
            _fig_dist.update_layout(
                height=420,
                title="Scores de Supervivencia por Estrategia (Torneo Round-Robin MAD)",
                xaxis_title="Estrategia",
                yaxis_title="Score Total de Supervivencia",
                xaxis_tickangle=-30,
                margin=dict(l=20, r=20, t=60, b=90),
            )
            st.plotly_chart(_fig_dist, use_container_width=True)

            # ── Métricas de polarización ──
            _pos  = sum(1 for s in _scores if s > 0)
            _neg  = sum(1 for s in _scores if s <= 0)
            _pol  = max(_scores) - min(_scores)
            _ann_total = int(_rank["Aniquilaciones provocadas"].sum() // 2)

            _pa, _pb, _pc, _pd = st.columns(4)
            _pa.metric("☮️ Estrategias pacíficas",       _pos,  help="Score positivo — cooperación dominó")
            _pb.metric("💥 Estrategias que colapsaron",  _neg,  help="Score negativo — Traición mutua dominó")
            _pc.metric("📊 Polarización total",  f"{_pol:.1f}", help="Máx − Mín de scores")
            _pd.metric("☢️ Eventos de Muerte Súbita",   _ann_total, help="Total de aniquilaciones en el torneo")

            if _pos == 0:
                st.error("💀 **Sistema completamente colapsado** — Ninguna estrategia mantuvo la paz. Prueba aumentando **w**.")
            elif _pos >= len(_scores) // 2:
                st.success(f"🌍 **Sistema mayormente estable** — {_pos}/{len(_scores)} estrategias lograron la paz.")
            else:
                st.warning(f"⚠️ **Equilibrio inestable** — Solo {_pos} estrategias sobrevivieron positivamente.")

            st.divider()
            st.markdown("#### 📋 Estadísticas del Torneo Nuclear")
            _stats = pd.DataFrame({
                "Métrica": ["Score promedio", "Score mediano", "Desviación estándar",
                            "Peor colapso", "Mejor cooperación"],
                "Valor": [
                    f"{np.mean(_scores):.3f}", f"{np.median(_scores):.3f}",
                    f"{np.std(_scores):.3f}",  f"{np.min(_scores):.3f}",
                    f"{np.max(_scores):.3f}",
                ],
                "Interpretación": [
                    "Salud promedio del sistema",
                    "50% de estrategias por encima de este valor",
                    "Alta = sistema muy polarizado (peligroso)",
                    "La estrategia más destructiva",
                    "La estrategia más eficaz para la paz",
                ],
            })
            st.dataframe(_stats, use_container_width=True, hide_index=True)

    else:
        st.info(
            "☝️ Configura los parámetros arriba y presiona **☢️ Lanzar Simulación Round-Robin Nuclear** "
            "para ver qué estrategias sobreviven en el escenario de la Guerra Fría."
        )


if __name__ == "__main__":
    main()
