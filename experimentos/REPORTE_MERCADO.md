# REPORTE -- Mezcla con el spread de Vegas

Fecha: 2026-10-06. No se modifico produccion; cambio propuesto en
`propuesta_mercado.diff` (aplica limpio con `patch -p1` desde `nfl_bot/`; crea
`market_lines_2026.py` y modifica `weekly_pipeline.py`).
Scripts: `mercado_calibracion.py`, `mercado_peso.py`, `mercado_sensibilidad.py`.

## 1. Spread -> probabilidad

Datos: `nflreadpy.load_schedules` 2010-2025, temporada regular, 4,175 partidos
(13 empates excluidos del ajuste).

**Signo de `spread_line`** = margen esperado del LOCAL (positivo = local favorito).
Corr(spread, margen real) = +0.43; el local gana 67% con spread > 0 y 35% con
spread < 0; cada punto de spread = 1.06 puntos de margen real.

Ajustes (log-loss leave-one-season-out; delta vs normal sigma=13.5, SE por bloques de semana):

| modelo | parametros | log-loss LOSO | delta vs 13.5 |
|---|---|---|---|
| normal sigma=13.5 (referencia) | -- | 0.61107 | -- |
| **normal, sigma libre** | **sigma = 11.13** | **0.60889** | **-0.0022 (SE 0.0011)** |
| normal + sesgo | sigma 10.99, sesgo -0.24 | 0.60887 | -0.0022 (0.0011) |
| logistica | pendiente 0.1475 | 0.60900 | -0.0021 (0.0011) |
| logistica + sesgo | 0.1494, -0.038 | 0.60896 | -0.0021 (0.0011) |

Elegido: `spread_to_prob(m) = Phi(m / 11.13)`. El sesgo no aporta y la logistica
empata. Sigma=13.5 (la desviacion del margen) da probabilidades demasiado timidas
para la probabilidad de ganar: los favoritos de 8+ ganan mas de lo que dice.

Calibracion por tramos (n; predicho sigma 11.13 / predicho 13.5 / frecuencia real):

| spread local | n | sigma 11.13 | sigma 13.5 | real (SE) |
|---|---|---|---|---|
| <= -11 | 76 | 0.13 | 0.18 | 0.18 (0.04) |
| -10 a -8 | 141 | 0.22 | 0.27 | 0.20 (0.03) |
| -7 a -5 | 380 | 0.30 | 0.33 | 0.26 (0.02) |
| -4 a -3 | 668 | 0.39 | 0.41 | 0.40 (0.02) |
| -2 a -1 | 279 | 0.45 | 0.46 | 0.47 (0.03) |
| 0 | 4 | 0.50 | 0.50 | 0.75 (0.22) |
| +1 a +2 | 500 | 0.56 | 0.55 | 0.51 (0.02) |
| +3 a +4 | 803 | 0.62 | 0.60 | 0.62 (0.02) |
| +5 a +7 | 730 | 0.72 | 0.68 | 0.72 (0.02) |
| +8 a +10 | 329 | 0.80 | 0.76 | 0.80 (0.02) |
| >= +11 | 252 | 0.89 | 0.84 | 0.90 (0.02) |

## 2. Peso del mercado

`p_final = (1-w)*base + w*Phi(spread/sigma)`, base = 0.45 Poisson (JOINT_PROD) + 0.55 Elo,
probabilidades walk-forward de `experimentos/`. Sigma reajustado solo con 2010-2022 (11.28)
para la busqueda. Spread = linea de CIERRE.

| w | 0 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | (1.0 = solo mercado) |
|---|---|---|---|---|---|---|---|---|
| log-loss seleccion 2010-2022 (n=3,347) | 0.6200 | 0.6175 | 0.6154 | 0.6136 | 0.6121 | 0.6109 | **0.6100** | 0.6094 |
| delta vs w=0 (SE) | -- | -0.0025 (0.0003) | -0.0046 (0.0006) | -0.0064 (0.0009) | -0.0079 (0.0012) | -0.0091 (0.0015) | -0.0100 (0.0017) | -0.0106 (0.0029) |
| 2023-2025, solo informativo (n=815) | 0.6250 | 0.6215 | 0.6184 | 0.6157 | 0.6133 | 0.6113 | 0.6097 | 0.6060 |

- La curva es monotona: el minimo de la grilla esta en el borde (0.60); el optimo
  continuo es **w = 0.85**. Con lineas de cierre, el mercado casi reemplaza al modelo.
- 0.20 es valido (mejora clara, -0.0046 +- 0.0006), pero deja la mitad de la ganancia:
  w=0.60 es 0.0054 mejor que 0.20 (SE 0.0011).
- 2023-2025 dice lo mismo, pero ese periodo ya se uso como holdout: no es evidencia
  independiente para esta decision.

### Sensibilidad a lineas del martes (limitacion principal)

El spread historico de nflverse es de **cierre**; el usuario pegara lineas del martes,
menos informadas (sin lesiones de ultimo momento ni dinero tardio). No hay lineas del
martes historicas para medirlo, asi que aproxime "martes = cierre + ruido N(0, tau^2)"
(50 sorteos por tau, sigma fijo como en el pipeline). Esta aproximacion trata el
movimiento de la linea como ruido puro; en realidad el movimiento lleva informacion,
asi que el deterioro real podria parecerse a un tau algo mayor.

| tau (puntos) | w optimo (p10-p90) | delta LL con w=0.20 | delta LL con w optimo (<=0.6) |
|---|---|---|---|
| 0 (cierre) | 0.85 | -0.0046 | -0.0100 |
| 1.0 | 0.74 (0.70-0.78) | -0.0045 | -0.0092 |
| 1.5 | 0.65 (0.57-0.71) | -0.0044 | -0.0084 |
| 2.0 | 0.53 (0.48-0.59) | -0.0042 | -0.0069 |
| 2.5 | 0.45 (0.40-0.49) | -0.0041 | -0.0059 |
| 3.0 | 0.37 (0.29-0.43) | -0.0037 | -0.0047 |

**Recomendacion: w = 0.40** (es lo que pone el diff). Queda por debajo del optimo en
todos los escenarios hasta tau = 2.5, o sea, sigue siendo conservador incluso si las
lineas del martes son bastante peores que las de cierre, y captura ~70-90% de la
ganancia disponible. Si se prefiere maxima prudencia, 0.20 tambien es defendible:
mejora en todos los escenarios (-0.004), solo que deja mas ganancia sin tomar.
Un error al pegar una linea (signo invertido) pesa el doble con 0.40 que con 0.20.

## 3. Cambios (propuesta_mercado.diff)

- `market_lines_2026.py` (nuevo): `MARKET_SPREADS = {(visita, local): margen_local}`,
  semana 5 cargada. CLE @ NYJ marcado "orientacion por confirmar".
- `weekly_pipeline.py`:
  - `MARKET_BLEND_WEIGHT = 0.40` y `MARKET_SPREAD_SIGMA = 11.13` como constantes de modulo
    (antes 0.20, definida dentro del loop y sin usar).
  - `spread_to_prob(home_margin)`.
  - Si hay linea: `p = (1-w)*base + w*p_mercado` y `margen = (1-w)*margen_blend + w*spread`
    (mismo peso, asi probabilidad y margen siguen apuntando al mismo lado). Si no hay
    linea: blend base sin cambios.
  - Columna nueva al final: `prob_mercado_local` (null sin linea). Las demas columnas
    no cambian. `NEUTRAL_SITE_GAMES` intacto.

## 4. Pruebas

**No regresion** (2026 semana 5, `MARKET_SPREADS` vacio vs pipeline actual): 15/15 filas
identicas en las 5 columnas originales (`DataFrame.equals` = True);
`prob_mercado_local` = null en las 15.

**Semana 5 con lineas** (w = 0.40; `data/mercado_semana5.csv`):

| partido | modelo P(local) | margen modelo | mercado P(local) | final P(local) | margen final |
|---|---|---|---|---|---|
| BAL @ ATL | 0.469 | BAL +0.6 | 0.295 | 0.399 | BAL +2.8 |
| BUF @ LA | 0.570 | LA +1.5 | 0.589 | 0.578 | LA +1.9 |
| CHI @ GB | 0.484 | CHI +0.5 | 0.411 | 0.455 | CHI +1.3 |
| CIN @ MIA | 0.416 | CIN +1.9 | 0.265 | 0.356 | CIN +3.9 |
| CLE @ NYJ * | 0.477 | CLE +0.6 | 0.589 | 0.522 | NYJ +0.6 |
| DEN @ LAC | 0.434 | DEN +1.8 | 0.411 | 0.425 | DEN +2.1 |
| DET @ ARI | 0.362 | DET +3.1 | 0.311 | 0.341 | DET +4.0 |
| HOU @ TEN | 0.287 | HOU +4.9 | 0.295 | 0.290 | HOU +5.3 |
| IND @ PIT | 0.576 | PIT +2.0 | 0.589 | 0.581 | PIT +2.2 |
| LV @ NE | 0.735 | NE +6.0 | 0.623 | 0.690 | NE +5.0 |
| MIN @ NO | 0.382 | MIN +3.4 | 0.446 | 0.408 | MIN +2.7 |
| NYG @ WAS | 0.527 | WAS +0.5 | 0.394 | 0.474 | NYG +0.9 |
| PHI @ JAX (neutral) | 0.679 | JAX +4.2 | 0.720 | 0.695 | JAX +5.1 |
| SF @ SEA | 0.538 | SEA +1.6 | 0.606 | 0.566 | SEA +2.1 |
| TB @ DAL | 0.729 | DAL +5.5 | 0.803 | 0.759 | DAL +7.1 |

\* Orientacion de la linea por confirmar. El mercado cambia de favorito en dos
partidos: **CLE @ NYJ** (el modelo da CLE, la linea +2.5 da NYJ) y **NYG @ WAS** (el modelo
da WAS, la linea -3.0 da NYG). Si la linea de CLE @ NYJ estuviera al reves, la final
seria ~0.45 para NYJ (CLE favorito). Mayores desacuerdos: BAL @ ATL (0.17), CIN @ MIA
(0.15), NYG @ WAS (0.13).

## 5. Limitaciones

1. Lineas de cierre vs martes (seccion 2): el peso optimo con cierre sobreestima el
   valor de lineas del martes; la sensibilidad es una aproximacion, no una medicion.
2. Las probabilidades historicas del modelo vienen de `experimentos/` (Elo con historial
   desde 1999, QB titular real, sin ajuste de campo neutral); el pipeline en vivo usa Elo
   desde 2015 y titulares proyectados. El peso es aproximado para el modelo en vivo.
3. Sigma de `spread_to_prob` se ajusto con 2010-2025, que incluye 2023-2025; es un solo
   parametro (11.13 vs 11.28 con 2010-2022), efecto despreciable.
4. Partidos en campo neutral: la linea ya incorpora la sede; no hay ajuste extra.
   Historicamente son pocos y no se calibraron aparte.
5. La conversion ignora el total (over/under): con el mismo spread, un partido de total
   bajo es algo mas predecible. No se modelo (fuera de alcance).
