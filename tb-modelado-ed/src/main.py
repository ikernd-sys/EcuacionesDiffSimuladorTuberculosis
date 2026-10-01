"""
Simulacion completa del Proyecto 1 - Modelado de Propagacion de la Tuberculosis.

Ejecuta todo el analisis y regenera las figuras del documento:
  1. Carga la base de datos de Mexico 2000-2024
  2. Estima parametros por regresion lineal (metodo de clase)
  3. Ajusta ambos modelos por minimos cuadrados no lineales
  4. Compara los modelos y selecciona el que describe la serie
  5. Simula un brote en entorno cerrado con el Modelo A
  6. Valida las soluciones analiticas con Euler y Runge-Kutta
  7. Genera las siete figuras en la carpeta figuras/

Uso:   python src/main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from modelos import ModeloA, ModeloB, tabla_error
from estimacion import (cargar_datos, estimar_A_lineal, estimar_B_lineal,
                        ajustar_A, ajustar_B, tendencia_por_tramo)
import figuras


def titulo(txt):
    print()
    print("=" * 74)
    print(f"  {txt}")
    print("=" * 74)


def main():
    titulo("1. BASE DE DATOS")
    anio, t, inc, casos = cargar_datos()
    print(f"  Periodo            : {int(anio[0])}-{int(anio[-1])}  ({len(t)} observaciones)")
    print(f"  Incidencia         : min {inc.min():.0f}   max {inc.max():.0f}   media {inc.mean():.2f}")
    print(f"  Media ultimos 15 a.: {inc[-15:].mean():.2f}  (desv. {inc[-15:].std(ddof=1):.2f})")
    print("  Unidades           : casos nuevos por 100 000 habitantes por anio")
    print("  NOTA: la incidencia es un FLUJO; I(t) en los modelos es un ACERVO")
    print("        (prevalencia). En equilibrio se relacionan por I* = lambda/delta.")

    titulo("2. ESTIMACION POR REGRESION LINEAL (forma linealizada)")
    rB = estimar_B_lineal(t, inc)
    rA = estimar_A_lineal(t, inc)
    print(f"  Modelo B:  delta = {rB['delta']:.4f} /anio   lambda = {rB['lambda']:.3f}"
          f"   I* = {rB['I_eq']:.2f}   R2 = {rB['R2']:.3f}")
    print(f"  Modelo A:  r     = {rA['r']:.4f} /anio   K      = {rA['K']:.2f}"
          f"                 R2 = {rA['R2']:.3f}")
    print("  -> R2 bajos: la derivada por diferencias finitas es ruidosa,")
    print(f"     pero ambos coinciden en el equilibrio ({rB['I_eq']:.1f} vs {rA['K']:.1f}).")

    titulo("3. AJUSTE NO LINEAL DE LAS SOLUCIONES ANALITICAS")
    fB = ajustar_B(t, inc)
    fA = ajustar_A(t, inc)
    print("  MODELO B   dI/dt = lambda - delta*I")
    print(f"    lambda = {fB['lambda']:7.3f} +/- {fB['err_lambda']:.3f}")
    print(f"    delta  = {fB['delta']:7.4f} +/- {fB['err_delta']:.4f} /anio")
    print(f"    I0     = {fB['I0']:7.2f} +/- {fB['err_I0']:.2f}")
    print(f"    I* = lambda/delta = {fB['I_eq']:.2f} por 100 000")
    print(f"    tau = 1/delta     = {fB['modelo'].tau:.2f} anios")
    print(f"    R2 = {fB['R2']:.4f}   RMSE = {fB['rmse']:.2f}")
    if fA:
        print("  MODELO A   dI/dt = r*I*(1 - I/K)")
        print(f"    r = {fA['r']:+.4f} /anio    K = {fA['K']:.3e}  (sin sentido fisico)")
        print(f"    R2 = {fA['R2']:.4f}   RMSE = {fA['rmse']:.2f}")
    base = float(np.sqrt(((inc - inc.mean()) ** 2).mean()))
    print(f"  REFERENCIA media constante:  RMSE = {base:.2f}")
    print()
    print(f"  SELECCION: el Modelo B (R2={fB['R2']:.3f}) describe la serie;")
    print(f"             el Modelo A (R2={fA['R2']:.3f}) no. La fase de crecimiento")
    print("             logistico no es observable en la serie nacional.")

    titulo("4. ESTRUCTURA DE LOS RESIDUALES Y TENDENCIA POR TRAMOS")
    res = fB["residuales"]
    for a, b, lab in [(0, 10, "2000-2009"), (10, 18, "2010-2017"), (18, 25, "2018-2024")]:
        print(f"  {lab}: residual medio {res[a:b].mean():+.2f}")
    for tr in tendencia_por_tramo(t, inc):
        print(f"  tramo {tr['tramo']}: pendiente {tr['pendiente']:+.3f} casos/100k por anio"
              f"   R2 = {tr['R2']:.3f}")
    print("  -> el cambio de tendencia hacia 2014-2015 exige lambda variable en el tiempo.")

    titulo("5. PREVALENCIA DE EQUILIBRIO SEGUN LA DURACION DEL CASO")
    print(f"  Con lambda = 28 casos/100k/anio  ->  I* = lambda * D")
    for D, lab in [(0.5, "6 meses (tratamiento oportuno)"), (1.0, "1 anio"),
                   (2.0, "2 anios"), (3.0, "3 anios (sin tratamiento)")]:
        print(f"    D = {lab:32s} delta = {1/D:5.3f}   I* = {28*D:5.1f} por 100 000")

    titulo("6. SIMULACION DE BROTE EN ENTORNO CERRADO (Modelo A)")
    N, gamma, I0 = 1000.0, 1 / 3, 1.0
    print(f"  Poblacion N = {N:.0f},  gamma = {gamma:.3f} /anio,  un caso inicial")
    for R0 in [3.0, 2.0, 1.5, 0.8]:
        A = ModeloA(beta=R0 * gamma, gamma=gamma, N=N)
        if R0 > 1:
            print(f"    R0 = {R0:4.1f}  ->  r = {A.r:+.3f}   K = {A.K:6.1f} casos"
                  f"  ({A.K/N*100:4.1f} % de la poblacion)")
        else:
            print(f"    R0 = {R0:4.1f}  ->  r = {A.r:+.3f}   el brote se extingue")
    A3 = ModeloA(beta=3 * gamma, gamma=gamma, N=N)
    print(f"  Trayectoria con R0 = 3:")
    for y in [1, 5, 10, 15, 20]:
        v = float(A3.solucion(y, I0))
        print(f"    t = {y:2d} anios  ->  I = {v:6.1f} casos  ({v/N*100:5.1f} %)")
    print(f"  Punto de inflexion en I = K/2 = {A3.K/2:.1f} casos (maxima velocidad de contagio)")

    titulo("7. VALIDACION NUMERICA (Euler vs Runge-Kutta 4)")
    for nombre, modelo, y0 in [("Modelo A", A3, I0), ("Modelo B", fB["modelo"], fB["I0"])]:
        exacta, filas = tabla_error(modelo, y0, 10.0)
        print(f"  {nombre} - solucion exacta en t=10: {exacta:.6f}")
        print(f"    {'h':>9} {'error Euler':>13} {'razon':>7} {'error RK4':>13} {'razon':>7}")
        for f in filas:
            rE = f"{f['razon_euler']:.2f}" if f["razon_euler"] else "  -"
            rR = f"{f['razon_rk4']:.2f}" if f["razon_rk4"] else "  -"
            print(f"    {f['h']:9.5f} {f['error_euler']:13.3e} {rE:>7}"
                  f" {f['error_rk4']:13.3e} {rR:>7}")
    print("  -> Euler reduce el error x2 (orden 1); RK4 lo reduce x16 = 2^4 (orden 4).")

    titulo("8. GENERACION DE FIGURAS")
    for nombre in figuras.generar_todas():
        print("  figuras/" + nombre)

    print()
    print("Simulacion completada.")


if __name__ == "__main__":
    main()
