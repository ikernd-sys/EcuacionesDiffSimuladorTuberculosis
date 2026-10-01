"""
Estimacion de parametros a partir de la serie de incidencia de Mexico 2000-2024.

Dos rutas:
  1. Regresion lineal sobre la forma linealizada de cada modelo (metodo de clase,
     reproducible en Excel y en la TI-84).
  2. Ajuste no lineal de la solucion analitica al nivel observado (mas preciso).
"""
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

from modelos import ModeloA, ModeloB

RUTA_DATOS = Path(__file__).resolve().parents[1] / "data" / "tb_mexico.csv"


def cargar_datos(ruta=RUTA_DATOS):
    """Lee la serie. Devuelve (anio, t, incidencia, casos)."""
    anio, t, inc, casos = [], [], [], []
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            anio.append(int(fila["anio"]))
            t.append(float(fila["t"]))
            inc.append(float(fila["incidencia_por_100k"]))
            casos.append(float(fila["casos_estimados"]))
    return (np.array(anio), np.array(t), np.array(inc), np.array(casos))


def r2(y, y_hat):
    y = np.asarray(y, float)
    return 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ---------------------------------------------------------------------
# Ruta 1: regresion lineal sobre la forma linealizada
# ---------------------------------------------------------------------
def regresion_lineal(x, y):
    """Mínimos cuadrados y = m*x + b. Devuelve (m, b, R2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b, r2(y, m * x + b)


def estimar_B_lineal(t, I):
    """Modelo B:  dI/dt = lambda - delta*I.
    Regresion de la derivada contra el nivel: pendiente = -delta, ordenada = lambda."""
    dI = np.gradient(I, t)
    m, b, R2 = regresion_lineal(I, dI)
    return {"lambda": b, "delta": -m, "I_eq": b / (-m), "R2": R2}


def estimar_A_lineal(t, I):
    """Modelo A:  (1/I)dI/dt = r - (r/K)*I.
    Regresion de la tasa per capita contra el nivel: ordenada = r, pendiente = -r/K."""
    dI = np.gradient(I, t)
    m, b, R2 = regresion_lineal(I, dI / I)
    return {"r": b, "K": -b / m, "R2": R2}


# ---------------------------------------------------------------------
# Ruta 2: ajuste no lineal de la solucion analitica al nivel
# ---------------------------------------------------------------------
def ajustar_B(t, I, p0=(30.0, 1.0, 37.0)):
    """Ajusta I(t) = lambda/delta + (I0 - lambda/delta)*e^(-delta*t)."""
    def f(tt, lam, dlt, I0):
        return lam / dlt + (I0 - lam / dlt) * np.exp(-dlt * tt)
    p, cov = curve_fit(f, t, I, p0=p0, maxfev=40000)
    lam, dlt, I0 = p
    err = np.sqrt(np.diag(cov))
    ajuste = f(t, *p)
    return {
        "modelo": ModeloB(lam=lam, delta=dlt), "I0": I0,
        "lambda": lam, "delta": dlt, "I_eq": lam / dlt,
        "err_lambda": err[0], "err_delta": err[1], "err_I0": err[2],
        "R2": r2(I, ajuste), "rmse": float(np.sqrt(((I - ajuste) ** 2).mean())),
        "residuales": I - ajuste,
    }


def ajustar_A(t, I, p0=(-0.05, 25.0, 37.0)):
    """Ajusta la logistica I(t) = K*I0*e^(rt)/(K + I0*(e^(rt)-1))."""
    def f(tt, r, K, I0):
        e = np.exp(r * tt)
        return K * I0 * e / (K + I0 * (e - 1))
    try:
        p, _ = curve_fit(f, t, I, p0=p0, maxfev=60000)
    except Exception:
        return None
    ajuste = f(t, *p)
    return {"r": p[0], "K": p[1], "I0": p[2],
            "R2": r2(I, ajuste), "rmse": float(np.sqrt(((I - ajuste) ** 2).mean())),
            "prediccion": ajuste}


def tendencia_por_tramo(t, I, tramos=((2000, 2014), (2014, 2024), (2010, 2024))):
    """Pendiente lineal de la incidencia en subperiodos (t = anio - 2000)."""
    salida = []
    for a, b in tramos:
        m_ = (t >= a - 2000) & (t <= b - 2000)
        m, c, R2 = regresion_lineal(t[m_], I[m_])
        salida.append({"tramo": f"{a}-{b}", "pendiente": m, "R2": R2})
    return salida
