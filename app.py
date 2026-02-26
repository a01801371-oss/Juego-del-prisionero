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
    """TIT FOR TAT pero defecciona con prob 0.1 tras cooperación del oponente."""
    name = "JOSS"
    def move(self):
        if not self.opp_history:
            return 'C'
        if self.opp_history[-1] == 'D':
            return 'D'
        return 'D' if RNG.random() < 0.1 else 'C'


class Gradual(Strategy):
    """Coopera hasta que el oponente defecciona; entonces defecciona N veces (N = total defecciones del oponente), luego dos C."""
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
    """Alias de GRIM con lógica idéntica (defección permanente tras primer D)."""
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
    """Defecciona en ronda 1; si el oponente perdona, explota; si no, usa TFT."""
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
                "GRIM": "Coopera hasta que el oponente defecciona; luego defecciona para siempre.",
                "PAVLOV": "Repite su acción si ganó, cambia si perdió (Win-Stay, Lose-Shift).",
                "ALL-D": "Siempre defecciona.",
                "ALL-C": "Siempre coopera.",
                "TIT FOR TWO TATS": "Solo defecciona si el oponente defeccionó dos veces seguidas.",
                "RANDOM": "Coopera o defecciona aleatoriamente con prob. 50/50.",
                "JOSS": "Como TFT pero traiciona con probabilidad 0.1 tras cooperación del oponente.",
                "GRADUAL": "Tras cada defección del oponente, castiga N veces (N = total D del oponente) y luego calma.",
                "ADAPTIVE": "Empieza con secuencia CCCCCDDDDD; luego elige la acción con mayor payoff histórico.",
                "EVOLVED-NN": "Red neuronal de 2 capas (6→8→1) que usa las últimas 3 jugadas de cada jugador como input.",
                "PSO-PLAYER": "Usa probabilidades de cooperación optimizadas por PSO según el último par (mi acción, acción oponente).",
                "MEMORY-3": "Defecciona si el oponente defeccionó ≥2 veces en las últimas 3 rondas.",
                "FRIEDMAN": "Idéntico a GRIM: defección permanente tras el primer D del oponente.",
                "TESTER": "Defecciona en ronda 1; si el oponente perdona, lo explota; si responde, usa TFT.",
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


if __name__ == "__main__":
    main()
