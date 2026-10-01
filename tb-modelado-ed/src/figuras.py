"""
Genera todas las figuras del documento.
Salida: carpeta figuras/ en la raiz del repositorio.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from modelos import ModeloA, ModeloB, euler, rk4, tabla_error
from estimacion import cargar_datos, ajustar_B, ajustar_A, estimar_B_lineal, estimar_A_lineal

SALIDA = Path(__file__).resolve().parents[1] / "figuras"
SALIDA.mkdir(exist_ok=True)

NAVY = "#16335B"; ORANGE = "#C25A0A"; GREEN = "#1F6B3B"
RED = "#9B2226"; GREY = "#8A93A0"; INK = "#2B2B2B"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#C6CBD4", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": NAVY,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def _limpiar(ax):
    ax.grid(color="#E4E7EC", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _campo(ax, modelo, t_max, y_max, n=22, y_min=0.0, largo=0.022):
    """Campo direccional de una ecuacion autonoma dI/dt = f(I).

    Cada segmento se dibuja con longitud VISUAL uniforme y con el angulo que
    corresponde a la pendiente real, convirtiendo la pendiente de unidades de
    datos a unidades de pantalla mediante la razon de aspecto de los ejes.
    """
    rx = float(t_max)
    ry = float(y_max - y_min)
    tt = np.linspace(0, t_max, n)
    yy = np.linspace(y_min, y_max, n)
    T, Y = np.meshgrid(tt, yy)
    S = modelo.f(Y)                       # pendiente en unidades de datos
    th = np.arctan(S * (rx / ry))         # angulo en pantalla
    dx = largo * rx * np.cos(th)
    dy = largo * ry * np.sin(th)
    segmentos = np.stack([
        np.stack([T - dx, Y - dy], axis=-1),
        np.stack([T + dx, Y + dy], axis=-1),
    ], axis=-2).reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segmentos, colors=GREY, linewidths=1.1, alpha=0.85))


# =====================================================================
# FIG 1 - campo direccional y curvas, Modelo A
# =====================================================================
def fig1_campo_A(A: ModeloA):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    T_MAX = 25.0
    _campo(ax, A, T_MAX, A.N)
    t = np.linspace(0, T_MAX, 700)
    for I0 in [1.0, 0.1 * A.K, 0.5 * A.K, A.K, 1.4 * A.K]:
        ax.plot(t, A.solucion(t, I0), color=NAVY, lw=2.0, solid_capstyle="round")
    ax.axhline(A.K, color=ORANGE, lw=2.2, ls="--")
    ax.annotate(f"$K = N(1-1/R_0) = {A.K:.0f}$   ATRACTOR", (T_MAX * 0.985, A.K),
                ha="right", va="bottom", color=ORANGE, fontsize=8.5, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88))
    ax.axhline(0, color=RED, lw=2.2, ls="--")
    ax.annotate("$I = 0$   REPULSOR", (T_MAX * 0.985, A.N * 0.015), ha="right", va="bottom",
                color=RED, fontsize=8.5, weight="bold")
    ax.set_title(f"Modelo A · campo direccional y curvas de solución "
                 f"($R_0={A.R0:.1f}$, $N={A.N:.0f}$)", fontsize=10, weight="bold", pad=8)
    ax.set_xlabel("t (años)"); ax.set_ylabel("I(t)  (casos activos)")
    ax.set_xlim(0, T_MAX); ax.set_ylim(-A.N * 0.03, A.N * 1.02)
    _limpiar(ax); fig.tight_layout()
    fig.savefig(SALIDA / "f1_campo_modeloA.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 2 - campo direccional y curvas, Modelo B
# =====================================================================
def fig2_campo_B(B: ModeloB):
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    T_MAX, Y_MAX = 25.0, 45.0
    _campo(ax, B, T_MAX, Y_MAX)
    t = np.linspace(0, T_MAX, 700)
    for I0 in [0.0, 10.0, 20.0, B.I_eq, 35.0, 45.0]:
        ax.plot(t, B.solucion(t, I0), color=NAVY, lw=2.0, solid_capstyle="round")
    ax.axhline(B.I_eq, color=ORANGE, lw=2.2, ls="--")
    ax.annotate(f"$I^* = \\lambda/\\delta = {B.I_eq:.1f}$   ATRACTOR ÚNICO",
                (T_MAX * 0.985, B.I_eq), ha="right", va="bottom", color=ORANGE,
                fontsize=8.5, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88))
    y_tau = float(B.solucion(B.tau, 45.0))
    ax.plot([B.tau, B.tau], [B.I_eq, y_tau], color=GREEN, lw=1.5, ls=":")
    ax.plot([B.tau], [y_tau], "o", ms=6, color=GREEN, mec="white", mew=1.3, zorder=6)
    ax.annotate(f"en $t=\\tau=1/\\delta={B.tau:.1f}$ años\nse ha cerrado el 63 % de la brecha",
                (B.tau, y_tau), textcoords="offset points", xytext=(12, 14),
                color=GREEN, fontsize=8.2, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_title("Modelo B · campo direccional y curvas de solución",
                 fontsize=10, weight="bold", pad=8)
    ax.set_xlabel("t (años)"); ax.set_ylabel("I(t)  (casos por 100 000 hab.)")
    ax.set_xlim(0, T_MAX); ax.set_ylim(0, Y_MAX)
    _limpiar(ax); fig.tight_layout()
    fig.savefig(SALIDA / "f2_campo_modeloB.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 3 - lineas de fase de los dos modelos
# =====================================================================
def fig3_fase(A: ModeloA, B: ModeloB):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))

    def dibujar(ax, puntos, y_max, titulo, etiquetas, f):
        ax.plot([0, 0], [-0.04 * y_max, 1.04 * y_max], color=INK, lw=1.6)
        bordes = sorted({-0.04 * y_max, *puntos, 1.04 * y_max})
        for a, b in zip(bordes[:-1], bordes[1:]):
            m = (a + b) / 2
            s = f(m)
            if abs(s) < 1e-12:
                continue
            d = 0.085 * y_max
            y0, y1 = (m - d, m + d) if s > 0 else (m + d, m - d)
            ax.annotate("", xy=(-0.06, y1), xytext=(-0.06, y0),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.1))
            ax.annotate(f"$f{'>' if s > 0 else '<'}0$", (-0.085, m), ha="right",
                        va="center", fontsize=8, color=ORANGE, weight="bold")
        for p, (etq, estable) in zip(puntos, etiquetas):
            ax.plot([0], [p], "o", ms=9, color=(NAVY if estable else "white"),
                    mec=NAVY, mew=2, zorder=5)
            ax.annotate(etq, (0.07, p), va="center", fontsize=8.2, weight="bold",
                        color=(GREEN if estable else RED))
        ax.set_title(titulo, fontsize=9.5, weight="bold", pad=8)
        ax.set_xlim(-0.24, 0.42); ax.set_ylim(-0.12 * y_max, 1.12 * y_max)
        ax.set_xticks([]); ax.set_ylabel("I")
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#E4E7EC", lw=0.6); ax.set_axisbelow(True)

    dibujar(axes[0], [0.0, A.K], A.N,
            f"Modelo A  ($R_0={A.R0:.1f}>1$)\ndos equilibrios",
            [("$I=0$\nrepulsor", False), (f"$I=K={A.K:.0f}$\natractor", True)], A.f)
    dibujar(axes[1], [B.I_eq], 45.0,
            "Modelo B\nun solo equilibrio, siempre atractor",
            [(f"$I^*=\\lambda/\\delta={B.I_eq:.1f}$\natractor", True)], B.f)
    fig.tight_layout()
    fig.savefig(SALIDA / "f3_lineas_fase.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 4 - ajuste del Modelo B a la serie observada
# =====================================================================
def fig4_ajuste(anio, t, inc, fitB):
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    ax = axes[0]
    tt = np.linspace(t.min(), t.max(), 500)
    ax.plot(anio, inc, "o", color=NAVY, ms=5.5, label="Incidencia observada (OMS / Banco Mundial)")
    ax.plot(2000 + tt, fitB["modelo"].solucion(tt, fitB["I0"]), "-", color=ORANGE, lw=2.4,
            label="Modelo B ajustado")
    ax.axhline(fitB["I_eq"], color=GREEN, lw=1.6, ls=":",
               label=f"equilibrio $I^*={fitB['I_eq']:.1f}$")
    ax.set_ylabel("casos por 100 000 hab.")
    ax.set_title("Modelo B ajustado a la serie de México 2000–2024",
                 fontsize=10, weight="bold", pad=8)
    ax.annotate(f"$\\lambda={fitB['lambda']:.2f}$,  $\\delta={fitB['delta']:.3f}$ año$^{{-1}}$\n"
                f"$R^2={fitB['R2']:.3f}$,  RMSE $={fitB['rmse']:.2f}$",
                (0.975, 0.93), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.6, color=NAVY, weight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#E7EDF6", ec=NAVY, lw=0.9))
    ax.set_ylim(21.5, 39)
    leg = ax.legend(loc="lower left", fontsize=8.2, frameon=True, framealpha=0.96,
                    edgecolor="#C6CBD4"); leg.get_frame().set_linewidth(0.6)
    _limpiar(ax)

    ax = axes[1]
    ax.axhline(0, color=GREY, lw=1)
    ax.vlines(anio, 0, fitB["residuales"], color=NAVY, lw=2)
    ax.plot(anio, fitB["residuales"], "o", color=NAVY, ms=4)
    ax.set_ylabel("residual"); ax.set_xlabel("año"); ax.set_ylim(-5.2, 3.6)
    _limpiar(ax)
    fig.savefig(SALIDA / "f4_ajuste_modeloB.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 5 - por que el Modelo A no describe la serie nacional
# =====================================================================
def fig5_comparacion(anio, t, inc, fitB, fitA):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    tt = np.linspace(t.min(), t.max(), 500)
    ax.plot(anio, inc, "o", color=NAVY, ms=5.5, label="Incidencia observada")
    ax.plot(2000 + tt, fitB["modelo"].solucion(tt, fitB["I0"]), "-", color=ORANGE, lw=2.4,
            label=f"Modelo B   $R^2={fitB['R2']:.3f}$")
    if fitA is not None:
        e = np.exp(fitA["r"] * tt)
        yA = fitA["K"] * fitA["I0"] * e / (fitA["K"] + fitA["I0"] * (e - 1))
        ax.plot(2000 + tt, yA, "--", color=RED, lw=2.2,
                label=f"Modelo A (logístico)   $R^2={fitA['R2']:.3f}$")
    ax.axhline(inc.mean(), color=GREY, lw=1.4, ls=":",
               label=f"media constante   RMSE $={np.sqrt(((inc-inc.mean())**2).mean()):.2f}$")
    ax.set_title("Los datos nacionales seleccionan al Modelo B",
                 fontsize=10, weight="bold", pad=8)
    ax.set_xlabel("año"); ax.set_ylabel("casos por 100 000 hab.")
    leg = ax.legend(loc="upper right", fontsize=8.2, frameon=True, framealpha=0.96,
                    edgecolor="#C6CBD4"); leg.get_frame().set_linewidth(0.6)
    _limpiar(ax); fig.tight_layout()
    fig.savefig(SALIDA / "f5_comparacion_modelos.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 6 - escenario de brote en entorno cerrado (Modelo A)
# =====================================================================
def fig6_brote(N=1000.0, gamma=1 / 3, I0=1.0, T=25.0):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    t = np.linspace(0, T, 800)
    colores = {3.0: RED, 2.0: ORANGE, 1.5: NAVY, 0.8: GREEN}
    for R0 in [3.0, 2.0, 1.5, 0.8]:
        A = ModeloA(beta=R0 * gamma, gamma=gamma, N=N)
        etq = (f"$R_0={R0}$   endémico en {A.K:.0f} casos"
               if R0 > 1 else f"$R_0={R0}$   se extingue")
        ax.plot(t, A.solucion(t, I0), lw=2.3, color=colores[R0], label=etq)
    ax.set_title(f"Modelo A · brote en un entorno cerrado ($N={N:.0f}$, un caso inicial)",
                 fontsize=10, weight="bold", pad=8)
    ax.set_xlabel("t (años)"); ax.set_ylabel("I(t)  (casos activos)")
    ax.set_xlim(0, T); ax.set_ylim(0, N * 0.75)
    leg = ax.legend(loc="upper left", fontsize=8.4, frameon=True, framealpha=0.96,
                    edgecolor="#C6CBD4"); leg.get_frame().set_linewidth(0.6)
    _limpiar(ax); fig.tight_layout()
    fig.savefig(SALIDA / "f6_brote_cerrado.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# FIG 7 - validacion numerica
# =====================================================================
def fig7_numerico(A: ModeloA, B: ModeloB, I0A=1.0, I0B=37.79, T=10.0):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, modelo, I0, nombre in [(axes[0], A, I0A, "Modelo A"),
                                   (axes[1], B, I0B, "Modelo B")]:
        exacta, filas = tabla_error(modelo, I0, T)
        h = np.array([f["h"] for f in filas])
        ee = np.array([f["error_euler"] for f in filas])
        er = np.array([f["error_rk4"] for f in filas])
        ax.loglog(h, ee, "o-", color=ORANGE, lw=2, ms=5, label="Euler  (orden 1)")
        ax.loglog(h, er, "s-", color=GREEN, lw=2, ms=5, label="RK4    (orden 4)")
        ax.set_title(f"{nombre} · error global en $t={T:.0f}$ años",
                     fontsize=9.5, weight="bold", pad=8)
        ax.set_xlabel("paso  h (años)"); ax.set_ylabel("|error|")
        leg = ax.legend(loc="lower right", fontsize=8.2, frameon=True,
                        framealpha=0.96, edgecolor="#C6CBD4")
        leg.get_frame().set_linewidth(0.6)
        ax.grid(True, which="both", color="#E4E7EC", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(SALIDA / "f7_validacion_numerica.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generar_todas():
    anio, t, inc, casos = cargar_datos()
    fitB = ajustar_B(t, inc)
    fitA = ajustar_A(t, inc)

    A_brote = ModeloA(beta=1.0, gamma=1 / 3, N=1000.0)   # R0 = 3, entorno cerrado
    B_mex = fitB["modelo"]

    fig1_campo_A(A_brote)
    fig2_campo_B(B_mex)
    fig3_fase(A_brote, B_mex)
    fig4_ajuste(anio, t, inc, fitB)
    fig5_comparacion(anio, t, inc, fitB, fitA)
    fig6_brote()
    fig7_numerico(A_brote, B_mex, I0B=fitB["I0"])
    return sorted(p.name for p in SALIDA.glob("*.png"))


if __name__ == "__main__":
    for nombre in generar_todas():
        print("  escrita:", nombre)
