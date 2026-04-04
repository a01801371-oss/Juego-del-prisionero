# 🎲 Torneo del Dilema del Prisionero
### Replicación Axelrod & Hamilton (1981) — Dashboard Interactivo

> **Curso:** Técnicas Computacionales Avanzadas  
> **Profesor:** Dr. Leonardo González Tejeda  
> **Modalidad:** Opción 1 — Torneo Round-Robin con 15 estrategias + Caso de estudio geopolítico

---

## 🔗 Demo en vivo

**[▶ Abrir Dashboard](https://juego-del-prisionero.streamlit.app)**

---

## 📸 Screenshots

### Tab 1 — Ranking del Torneo
<img width="1104" height="262" alt="image" src="https://github.com/user-attachments/assets/06e6c5b3-7ef4-41bb-9bf9-bcc8d81db139" />
<img width="1319" height="598" alt="image" src="https://github.com/user-attachments/assets/0988cfd8-1bf5-48c4-b2d8-336446ce187d" />

### Tab 2 — Heatmap de Pagos
<img width="1298" height="804" alt="image" src="https://github.com/user-attachments/assets/be952b37-aa8e-4b41-9ff5-859181ef11b9" />

### Tab 3 — Head-to-Head
<img width="1303" height="817" alt="image" src="https://github.com/user-attachments/assets/55c1d912-93f7-4d20-9075-ce54db2ebe8e" />

### Tab 6 — Análisis Estadístico ANOVA
<img width="1293" height="612" alt="image" src="https://github.com/user-attachments/assets/181709cf-f989-4545-9dd6-d2730e1f1552" />
<img width="1303" height="586" alt="image" src="https://github.com/user-attachments/assets/bfe5f89a-c8e7-421b-9a89-cfe9c3816fd8" />
<img width="1280" height="652" alt="image" src="https://github.com/user-attachments/assets/f09e13f4-f039-4818-be67-c4ede969ecdb" />

### Tab 7 — Crisis Energética Rusia–UE
<!-- SCREENSHOT: Dashboard geopolítico con gráficas de flujo de gas, bienestar acumulado, PIB e inflación -->
```
[Insertar screenshot aquí]
```

### Tab 7 — Validación Axelrod & Hamilton
<!-- SCREENSHOT: Comparativa histórico vs TFT puro con las 4 subgráficas -->
```
[Insertar screenshot aquí]
```

### Tab 8 — Anexo Tests RNG
<!-- SCREENSHOT: KS test, histograma y lag plot del generador PCG64 -->
```
[Insertar screenshot aquí]
```

---

## 📋 Requisitos del sistema

- Python 3.11 o superior
- pip 23+
- Conexión a internet (solo para el despliegue en Streamlit Cloud)

---

## ⚙️ Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/a01801371-oss/Juego-del-prisionero.git
cd Juego-del-prisionero
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar archivos de datos

Asegúrate de que los siguientes archivos estén en la raíz del repositorio:

```
Juego-del-prisionero/
│
├── app.py
├── requirements.txt
├── README.md
│
├── daily_data_2026-03-12.csv          # Flujo de gas diario Rusia → Europa
├── route_data_2026-03-12.csv          # Utilización de rutas de gasoducto
├── Weekly Storage EU & UA 2026-0...xlsx  # Almacenamiento semanal EU y Ucrania
└── real_macro_data.csv                # PIB e inflación UE y Rusia (Eurostat / BM)
```

> ⚠️ Si algún archivo de datos no está presente, el dashboard genera datos sintéticos reproducibles automáticamente usando PCG64(seed=42). El torneo Round-Robin funciona siempre sin archivos externos.

### 5. Ejecutar el dashboard

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`

---

## 🚀 Despliegue en Streamlit Cloud

El proyecto está configurado para desplegarse directamente desde GitHub:

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu cuenta de GitHub
3. Selecciona el repositorio `Juego-del-prisionero`
4. Archivo principal: `app.py`
5. Haz clic en **Deploy**

---

## 🎮 Guía de uso

### Torneo Round-Robin (Tabs 1–5)

1. En el **sidebar izquierdo**, configura los parámetros:
   - `T, R, P, S` — valores de la matriz de pagos (default: 5, 3, 1, 0)
   - `w` — probabilidad de interacción futura (default: 0.995)
   - Juegos por par y rondas por juego
   - Selecciona las estrategias a incluir (default: las 15)
2. Haz clic en **▶ Ejecutar Torneo**
3. Explora los resultados en cada tab

| Tab | Contenido |
|-----|-----------|
| 🏆 Ranking | Top-3 con medallas · Tabla completa · Descarga CSV |
| 🌡️ Heatmap | Matriz de pagos promedio con hover interactivo |
| ⚔️ Head-to-Head | Comparativa de movidas entre dos estrategias |
| 📊 Distribución | Histograma de scores y estadísticas descriptivas |
| ℹ️ Estrategias | Descripción de cada una de las 15 estrategias |

### Crisis Energética Rusia–UE (Tab 7)

El simulador se recalcula automáticamente con cada cambio en el sidebar:

| Control | Descripción |
|---------|-------------|
| 📅 Período | Filtra el rango de fechas analizado (2021–2026) |
| 🇪🇺 Estrategia Europa | Realista/Sanciones · Pacificadora · TFT · 15 del catálogo |
| 🇷🇺 Estrategia Rusia | Datos Históricos · TFT · Always Defect · Bully |

> 💡 **Tip:** Combina `Rusia: TFT` + `Europa: TFT` para ver el óptimo de Pareto (máxima cooperación). Combina `Rusia: Always Defect` + `Europa: Realista` para ver el peor escenario.

### Tests RNG (Tab 8)

Muestra automáticamente los 4 tests de calidad del generador PCG64(seed=42):
- Test Kolmogorov-Smirnov
- Histograma de 10,000 muestras
- Lag plot de autocorrelación
- Test Chi-cuadrado

---

## 🗂️ Estructura del proyecto

```
Juego-del-prisionero/
│
├── app.py                  # Código fuente completo del dashboard
├── requirements.txt        # Dependencias con versiones mínimas
├── README.md               # Este archivo
│
└── datos/
    ├── daily_data_*.csv    # Flujo diario de gas (Bruegel / ENTSOG)
    ├── route_data_*.csv    # Utilización de gasoductos
    ├── Weekly Storage*.xlsx # Almacenamiento EU y Ucrania (GIE)
    └── real_macro_data.csv  # Macroeconomía real (Eurostat / Banco Mundial)
```

---

## 📦 Dependencias

```
streamlit>=1.30.0
numpy>=1.26.0
pandas>=2.1.0
plotly>=5.18.0
scipy>=1.11.0
openpyxl>=3.1.0
```

---

## 🧪 Reproducibilidad

Todos los resultados estocásticos son completamente reproducibles. El proyecto utiliza:

```python
RNG = np.random.default_rng(np.random.PCG64(42))
```

Esto garantiza que el ranking del torneo, las estrategias aleatorias (RANDOM, JOSS, PSO-PLAYER) y los datos sintéticos de fallback produzcan **exactamente los mismos resultados** en cualquier máquina y ejecución.

---

## 📚 Referencias

Axelrod, R., & Hamilton, W. D. (1981). The evolution of cooperation. *Science, 211*(4489), 1390–1396. https://doi.org/10.1126/science.7466396

Anthropic. (2025). *Claude Sonnet 4.6* [Modelo de lenguaje grande]. https://www.anthropic.com

Bruegel. (2024). *European natural gas imports* [Conjunto de datos]. https://www.bruegel.org/dataset/european-natural-gas-imports

Eurostat. (2024). *GDP and main components — volumes* [namq_10_gdp]. https://ec.europa.eu/eurostat

Eurostat. (2024). *Harmonised index of consumer prices* [prc_hicp_minr]. https://ec.europa.eu/eurostat

World Bank. (2024). *GDP growth (annual %)* [NY.GDP.MKTP.KD.ZG]. https://data.worldbank.org

World Bank. (2024). *Inflation, consumer prices (annual %)* [FP.CPI.TOTL.ZG]. https://data.worldbank.org

---

## 👥 Equipo

Proyecto desarrollado para el curso **Técnicas Computacionales Avanzadas**  
Campus Santa Fe · Febrero 2026
