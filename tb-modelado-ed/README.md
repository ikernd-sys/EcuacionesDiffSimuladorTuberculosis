# Modelado de Propagación de la Tuberculosis

Simulación del Proyecto 1 de **Ecuaciones Diferenciales: Modelación** — Universidad Anáhuac México, Facultad de Ingeniería.

Dos modelos de ecuaciones diferenciales ordinarias de primer orden aplicados a la tuberculosis, resueltos analíticamente, ajustados a la serie histórica de México 2000–2024 y validados con métodos numéricos.

---

## Los dos modelos

**Modelo A — transmisión (logístico, no lineal)**

```
dI/dt = β·I·(1 − I/N) − γ·I  =  r·I·(1 − I/K)
```

con `r = β − γ`, `K = N·(β − γ)/β` y `R₀ = β/γ`. Tiene **dos** equilibrios, `I = 0` e `I = K`, cuya estabilidad se invierte cuando `R₀` cruza el valor 1.

**Modelo B — balance (lineal)**

```
dI/dt = λ − δ·I
```

con equilibrio único `I* = λ/δ` y tiempo característico `τ = 1/δ`. Siempre atractor.

---

## Resultado principal

Ajustados a la serie nacional de México, **los datos seleccionan al Modelo B**:

| Modelo | R² | RMSE | Diagnóstico |
|---|---|---|---|
| **B (balance)** | **0.834** | **1.36** | Describe el descenso y la estabilización |
| A (logístico) | 0.500 | 2.36 | `K` diverge a ~10⁷, sin sentido físico |
| media constante | — | 3.34 | referencia |

Parámetros estimados del Modelo B: `λ = 6.75 ± 1.51`, `δ = 0.2585 ± 0.0548 año⁻¹`, equilibrio `I* = 26.1` casos por 100 000 habitantes y tiempo característico `τ = 3.9 años`.

La lectura no es que el Modelo A sea incorrecto, sino que **la fase de crecimiento logístico no es observable en la serie nacional**: México no atravesó una invasión epidémica durante el periodo, sino una relajación hacia un equilibrio más bajo. El Modelo A sí aplica en entornos cerrados de alta densidad, donde la simulación muestra que un solo caso introducido con `R₀ = 3` lleva al 36 % de la población a los diez años.

---

## Cómo ejecutarlo

### En GitHub Actions (sin instalar nada)

El repositorio incluye un flujo de integración continua que ejecuta toda la simulación y publica las figuras como artefactos descargables.

1. Pestaña **Actions** → workflow **Simulación**
2. **Run workflow** → **Run workflow**
3. Al terminar, descargar el artefacto `figuras` y leer el reporte en el **Summary** del job

El flujo también se dispara automáticamente con cada `push` a `main`.

### En Google Colab

Abrir `notebooks/simulacion.ipynb` desde Colab y ejecutar todas las celdas.

### En local

```bash
git clone <url-del-repositorio>
cd tb-modelado-ed
pip install -r requirements.txt
python src/main.py
```

La ejecución tarda unos segundos, imprime el reporte completo en consola y escribe las siete figuras en `figuras/`.

---

## Estructura

```
.
├── data/
│   └── tb_mexico.csv          Serie de incidencia 2000-2024 (OMS / Banco Mundial)
├── src/
│   ├── modelos.py             Los dos modelos, soluciones analíticas, Euler y RK4
│   ├── estimacion.py          Regresión lineal y ajuste no lineal a los datos
│   ├── figuras.py             Genera las siete figuras del documento
│   └── main.py                Ejecuta el análisis completo
├── notebooks/
│   └── simulacion.ipynb       Versión para Google Colab
├── figuras/                   Salida (se genera al ejecutar)
├── .github/workflows/
│   └── simulacion.yml         Integración continua
└── requirements.txt
```

---

## Figuras que genera

| Archivo | Contenido | Sección del documento |
|---|---|---|
| `f1_campo_modeloA.png` | Campo direccional y curvas, Modelo A | 2.4.2 |
| `f2_campo_modeloB.png` | Campo direccional y curvas, Modelo B | 2.4.3 |
| `f3_lineas_fase.png` | Líneas de fase de ambos modelos | 2.4.2 |
| `f4_ajuste_modeloB.png` | Ajuste a la serie de México y residuales | 3.3 |
| `f5_comparacion_modelos.png` | Modelo A vs Modelo B contra los datos | 3.3 |
| `f6_brote_cerrado.png` | Brote en entorno cerrado según R₀ | 3.4 |
| `f7_validacion_numerica.png` | Error de Euler y RK4 en escala log | 3.5 |

---

## Nota metodológica

La variable registrada en la base de datos es la **incidencia**, es decir un *flujo*: casos nuevos por año. La variable `I(t)` de ambos modelos es la **prevalencia**, un *acervo*: casos activos en un instante. Ambas se relacionan en el equilibrio por

```
I* = λ/δ = λ · D
```

donde `D` es la duración promedio del caso. Con una incidencia de 28 por 100 000 al año, la prevalencia de equilibrio va de 14 (tratamiento en 6 meses) a 84 por 100 000 (tres años sin tratamiento). Los parámetros estimados a partir de la serie de incidencia deben leerse como **tasas efectivas agregadas**, no como magnitudes biológicas directas.

---

## Fuentes de los datos

- Organización Mundial de la Salud, *Global Tuberculosis Programme* — https://www.who.int/teams/global-tuberculosis-programme/data
- Banco Mundial, indicador `SH.TBS.INCD` — https://datos.bancomundial.org/indicador/SH.TBS.INCD?locations=MX
- Our World in Data, *Incidence of tuberculosis* — https://ourworldindata.org/grapher/incidence-of-tuberculosis-sdgs

## Equipo

Julio César Velázquez Corona · Jorge Emiliano Ávila Correa · Iker · Bernardo García Alejos

Profesor: Dr. Santos Flores Eslava
