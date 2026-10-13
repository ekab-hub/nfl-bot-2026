# REPORTE -- QB fuera por la temporada en el simulador

Fecha: 2026-10-12. Protocolo: `PROTOCOLO_QB.md` (escrito antes de estimar). Scripts:
`qb_mercado.py`, `qb_poisson_compare.py`. No se modifico produccion; cambio propuesto en
`propuesta_qb.diff` (aplica limpio con `patch -p1` desde `nfl_bot/`).

**Resultado: la puerta PASA** (b de (ii) = 8.40, SE 0.79, p unilateral < 1e-6; mismo
signo en 2023-2025). Se implemento la parte 2.

## 1. Cuanto descuenta el mercado

y = spread_line - margen Elo (k=20, ha=35, rtm=0.40; localia 0 en campo neutral).
x = caida_visita - caida_local; caida = valor(QB habitual) - valor(titular), valor =
`qb_value_epa_dropback` antes de la temporada. b > 0 = puntos de spread que el mercado
le quita al equipo por unidad de caida, mas alla de lo que ya dice el Elo. SE agrupados
por equipo-temporada.

| especificacion | 2010-2022: b (SE), n | 2023-2025 (verificacion): b (SE), n |
|---|---|---|
| (i) todos los partidos con QB distinto del habitual | 5.25 (0.44), 2,091 | 7.80 (0.90), 590 |
| **(ii) 2 primeros partidos tras el cambio** | **8.40 (0.79), 432** | **11.09 (1.69), 107** |
| (ii-b) idem, solo cambios en temporada (semana >= 2) | 8.84 (1.03), 226 | 11.06 (2.19), 47 |

- Que (i) < (ii) es lo esperado: pasados unos partidos, el Elo ya absorbio parte de la
  baja con los resultados. (ii) es la medicion limpia y la que se usa.
- (ii-b), que es el caso de una lesion, da practicamente lo mismo que (ii).
- 2023-2025: mismo signo y algo mas grande (no virgen; solo verificacion).

**Por tercil de calidad del QB reemplazado** (muestra ii, 2010-2022):

| tercil (valor del habitual) | b (SE) |
|---|---|
| bajo (< -0.027) | 3.68 (1.39) |
| medio (-0.027 a 0.056) | 8.51 (1.92) |
| alto (> 0.056) | 10.28 (1.21) |

Perder a un buen QB pesa mas por unidad de caida: el efecto no es del todo lineal.

**Elite contra suplente tipico:** p90 del valor de QBs habituales (0.188) menos la mediana
de los reemplazos (-0.059) = 0.247 -> **2.1 puntos de spread (SE 0.2) = 52 Elo**.
Ejemplo real: Mahomes (+0.148) -> Justin Fields (-0.073) = 1.85 puntos = -46 Elo.

**Chequeo sin modelo** (muestra ii, partidos con un solo equipo con cambio; perdida de
spread neta del intercepto):

| caida | n | perdida media (SE) |
|---|---|---|
| < -0.05 (mejora) | 93 | 0.39 (0.35) |
| -0.05 a 0.05 | 68 | 1.05 (0.41) |
| 0.05 a 0.15 | 88 | 1.70 (0.36) |
| 0.15 a 0.25 | 40 | 2.81 (0.52) |
| > 0.25 | 83 | 4.17 (0.34) |

La relacion es monotona, pero tiene un "costo fijo" de ~1 punto por cualquier cambio de
QB (incluso con caida ~0) que el modelo lineal no captura. Por eso b*caida probablemente
**subestima** el efecto en ~1 punto (p.ej. caida 0.20: el modelo da 1.7, los datos 2.8).

## 2. Comparacion con el Poisson semanal (solo reporte)

Mismo cambio elite -> suplente (caida 0.247), modelo JOINT_PROD (2002-2022, lambda medio 22.2):

| | puntos de margen |
|---|---|
| Poisson: por `qb_value_live` (0.289 por unidad de valor) | -1.53 |
| Poisson: por `backup_qb` (-0.087) | -1.84 |
| Poisson total | -3.25 |
| **Poisson dentro del blend (x0.45; el Elo no sabe del QB)** | **-1.46** |
| Mercado (b de (ii)) | -2.07 (~-3 contando el costo fijo) |

La sospecha se confirma a medias: el Poisson solo descuenta MAS que el mercado, pero en el
pronostico semanal pesa 0.45 y el Elo no sabe del QB, asi que el blend descuenta ~70% de
lo que descuenta el mercado (~50% si se cuenta el costo fijo). Ademas, la mitad del efecto
del Poisson viene de `backup_qb`, que no depende de la calidad: es el mismo "costo fijo"
que se ve en la tabla de tramos. No se cambio nada del modelo semanal (fuera de alcance).

## 3. Implementacion (propuesta_qb.diff)

- `season_sim_core.simulate_one_season(..., elo_adjustments=None)`: dict equipo -> Elo,
  solo en partidos sin resultado real. Se suma al margen esperado **y** a la expectativa
  de la actualizacion de Elo; el rating guardado nunca lo incluye. Sin esto, el Elo
  simulado absorberia la penalizacion y la contaria dos veces. Chequeo (KC, -46 Elo, 3,000
  temporadas): Elo final medio sin ajuste 1619.7, propuesto 1619.4, version ingenua 1563.8
  (otros ~56 Elo de castigo acumulado).
- `apply_elo_adjustments(ratings, adj)`: suma los mismos ajustes al Elo final para los
  playoffs (la baja es por el resto de la temporada). Sin ajustes devuelve el mismo dict.
- `qb_out_2026.py` (nuevo): `QB_OUT_FOR_SEASON = {}` por defecto,
  `B_SPREAD_PER_QB_VALUE = 8.40`, `qb_elo_adjustments(season)` -> Elo = -25 * 8.40 *
  (valor lesionado - valor reemplazo), con impresion de cada ajuste.
- `division_champions.py`, `superbowl_champions.py`, `conference_champions.py`: ajustes
  solo con `use_played=True`; en modo por defecto `elo_adjustments = {}`.

## 4. Pruebas

**No regresion** (`np.random.seed(2026)`, 10,000 simulaciones, `use_played=True`; con
`QB_OUT_FOR_SEASON` vacio vs codigo actual): division, Super Bowl y conferencia, 32/32
filas identicas (`DataFrame.equals` = True). Tambien el Super Bowl en modo por defecto
(`use_played=False`, semilla 7, 2,000 simulaciones): identico.

**Prueba de humo: KC sin Mahomes** (`("Patrick Mahomes", "Justin Fields")` -> -46.4 Elo;
78 partidos jugados de 2026; misma semilla):

| equipo | P(Super Bowl) antes | despues | cambio |
|---|---|---|---|
| SF | 0.157 | 0.159 | +0.002 |
| MIN | 0.116 | 0.118 | +0.002 |
| JAX | 0.099 | 0.102 | +0.003 |
| **KC** | **0.072** | **0.048** | **-0.025** |
| BUF | 0.071 | 0.075 | +0.003 |
| SEA | 0.069 | 0.068 | -0.002 |
| BAL | 0.057 | 0.059 | +0.003 |
| NE | 0.054 | 0.056 | +0.002 |
| DEN | 0.040 | 0.042 | +0.002 |

Division AFC West: KC 0.600 -> 0.519, DEN 0.322 -> 0.381. Campeon AFC: KC 0.145 -> 0.103.
(Cambios de +-0.002 en otros equipos estan dentro del ruido Monte Carlo, ~0.003.)

## 5. Limitaciones

1. **El efecto implementado probablemente es conservador**: el modelo lineal no captura
   el costo fijo de ~1 punto por cualquier cambio de QB, y la pendiente crece con la
   calidad del QB perdido (tercil alto 10.3 vs 8.4). Para Mahomes, una cifra con costo
   fijo estaria mas cerca de ~3 puntos (~75 Elo) que de 1.85. No se cambio porque el
   modelo estaba pre-registrado; es la siguiente mejora natural.
2. **Valor del QB = antes de la temporada** (`qb_value_epa_dropback` con datos < s). No
   incorpora como jugo el QB en 2026; el mercado si.
3. **Error de medicion**: el valor de los suplentes suele salir de pocos dropbacks o del
   prior de draft. Eso atenua b (sesgo hacia cero), otra razon para pensar que el efecto
   real es algo mayor.
4. **"Habitual = lider de la temporada anterior"**: si el equipo cambio de QB en el
   offseason, el simulador ya lo trae en el Elo de forma incompleta; esta herramienta solo
   cubre bajas por el resto de la temporada que se cargan a mano.
5. **2023-2025 no es virgen** (se uso como holdout antes); solo verifica signo y magnitud.
6. El ajuste es permanente por el resto de la temporada y playoffs: no modela que el QB
   pueda volver (fuera de alcance).
7. Si el nombre no esta en el roster 2026 de nflreadpy, `qb_elo_adjustments` falla con un
   error explicito (a proposito: mejor fallar que aplicar un ajuste equivocado).
