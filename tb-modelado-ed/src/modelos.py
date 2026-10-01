"""
Modelos de propagacion de tuberculosis
Proyecto 1 - Ecuaciones Diferenciales - Universidad Anahuac Mexico

Modelo A (transmision):  dI/dt = beta*I*(1 - I/N) - gamma*I = r*I*(1 - I/K)
Modelo B (balance):      dI/dt = lambda - delta*I
"""
from dataclasses import dataclass
import numpy as np


# =====================================================================
# MODELO A - transmision (logistico)
# =====================================================================
@dataclass
class ModeloA:
    """dI/dt = beta*I*(1 - I/N) - gamma*I, con beta = k*N."""
    beta: float          # tasa maxima de transmision per capita (1/anio)
    gamma: float         # fraccion de casos que salen por anio (1/anio)
    N: float             # tamano de la poblacion en riesgo

    @property
    def r(self) -> float:
        """Tasa neta de crecimiento r = beta - gamma."""
        return self.beta - self.gamma

    @property
    def R0(self) -> float:
        """Numero reproductivo basico R0 = beta/gamma."""
        return self.beta / self.gamma

    @property
    def K(self) -> float:
        """Prevalencia de equilibrio K = N*(beta-gamma)/beta = N*(1 - 1/R0)."""
        return self.N * (self.beta - self.gamma) / self.beta

    def f(self, I):
        """Miembro derecho de la ecuacion diferencial."""
        return self.beta * I * (1 - I / self.N) - self.gamma * I

    def solucion(self, t, I0):
        """Solucion analitica:  I(t) = K*I0*e^(rt) / (K + I0*(e^(rt) - 1))."""
        t = np.asarray(t, dtype=float)
        if abs(self.r) < 1e-12:                 # caso degenerado r = 0
            return I0 / (1 + (self.beta / self.N) * I0 * t)
        e = np.exp(self.r * t)
        return self.K * I0 * e / (self.K + I0 * (e - 1))

    def equilibrios(self):
        """Puntos criticos y su clasificacion."""
        if self.R0 > 1:
            return {0.0: "repulsor", self.K: "atractor"}
        return {0.0: "atractor"}


# =====================================================================
# MODELO B - balance (lineal)
# =====================================================================
@dataclass
class ModeloB:
    """dI/dt = lambda - delta*I."""
    lam: float           # incidencia anual (casos nuevos por 100 000 hab por anio)
    delta: float         # fraccion anual de casos que salen del sistema (1/anio)

    @property
    def I_eq(self) -> float:
        """Prevalencia de equilibrio I* = lambda/delta."""
        return self.lam / self.delta

    @property
    def tau(self) -> float:
        """Tiempo caracteristico de aproximacion al equilibrio."""
        return 1.0 / self.delta

    def f(self, I):
        return self.lam - self.delta * I

    def solucion(self, t, I0):
        """Solucion analitica:  I(t) = lambda/delta + (I0 - lambda/delta)*e^(-delta*t)."""
        t = np.asarray(t, dtype=float)
        return self.I_eq + (I0 - self.I_eq) * np.exp(-self.delta * t)

    def equilibrios(self):
        return {self.I_eq: "atractor"}


# =====================================================================
# METODOS NUMERICOS
# =====================================================================
def euler(modelo, h, T, I0):
    """Metodo de Euler. Devuelve (t, I)."""
    n = int(round(T / h))
    t = np.linspace(0.0, n * h, n + 1)
    y = np.empty(n + 1)
    y[0] = I0
    for k in range(n):
        y[k + 1] = y[k] + h * modelo.f(y[k])
    return t, y


def rk4(modelo, h, T, I0):
    """Runge-Kutta de cuarto orden. Devuelve (t, I)."""
    n = int(round(T / h))
    t = np.linspace(0.0, n * h, n + 1)
    y = np.empty(n + 1)
    y[0] = I0
    for k in range(n):
        k1 = modelo.f(y[k])
        k2 = modelo.f(y[k] + h * k1 / 2)
        k3 = modelo.f(y[k] + h * k2 / 2)
        k4 = modelo.f(y[k] + h * k3)
        y[k + 1] = y[k] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, y


def tabla_error(modelo, I0, T, pasos=(1.0, 0.5, 0.25, 0.125, 0.0625)):
    """Compara Euler y RK4 contra la solucion analitica en t = T."""
    exacta = float(modelo.solucion(T, I0))
    filas = []
    eu_prev = rk_prev = None
    for h in pasos:
        e_eu = abs(euler(modelo, h, T, I0)[1][-1] - exacta)
        e_rk = abs(rk4(modelo, h, T, I0)[1][-1] - exacta)
        filas.append({
            "h": h, "error_euler": e_eu, "error_rk4": e_rk,
            "razon_euler": (eu_prev / e_eu) if eu_prev else None,
            "razon_rk4": (rk_prev / e_rk) if rk_prev else None,
        })
        eu_prev, rk_prev = e_eu, e_rk
    return exacta, filas
