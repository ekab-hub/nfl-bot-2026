# PROTOCOLO -- Efecto de la disponibilidad del QB en el spread (para el simulador)

Escrito el 2026-10-12 ANTES de estimar nada. Lo unico revisado antes: el codigo de
`season_sim_core.py` y los tres scripts de campeones (sin correrlos).

## Datos
- `nflreadpy.load_schedules`, temporada regular 2010-2025: `spread_line` (positivo =
  local favorito, verificado en REPORTE_MERCADO.md), `home_qb_id`/`away_qb_id`
  (titular real), `location` (`Neutral` = campo neutral).
- Elo previo al partido: `motor_elo_v1.run_elo_over_seasons(range(1999, 2026), k=20,
  home_advantage=35, regression_to_mean=0.40)` (`pre_elo_home/away`, walk-forward por
  construccion). Margen Elo = (pre_home + 35*[no neutral] - pre_away)/25.
- **y = spread_line - margen Elo** (puntos).
- Valor de QB: `qb_adjustment.qb_career_value` de produccion (`qb_value_epa_dropback`)
  sobre el pbp cacheado 2001-2025. Para un partido de la temporada s, el valor de un QB es
  su ultimo valor con temporada < s; si no tiene, prior por cubeta de draft (R1 / R2-R3 /
  R4+, UDFA = R4+) calculado SOLO con novatos de temporadas < s (la version de produccion
  usa todas las temporadas: fuga; se corrige aqui).

## Variables
- QB habitual del equipo en la temporada s = el lider en titularidades REG del equipo en
  s-1 (empate -> el que abrio mas tarde). Si no hay, se usa el titular de la semana 1 de s.
- caida_equipo = valor(habitual) - valor(titular del partido); 0 si el titular es el habitual.
- x = caida_local - caida_visita. Modelo: y = a + b*x + e. Se espera b > 0 (si el local
  pierde a su QB, el spread del local baja respecto a lo que dice el Elo... ojo: caida
  positiva del local = local peor => spread del local MENOR => b NEGATIVO con esta
  definicion de x). Para que b sea "puntos de spread que se pierden por unidad de caida",
  se define **x = caida_visita - caida_local**, y b > 0 = el mercado descuenta la baja.
- Racha (spell): secuencia de partidos consecutivos del equipo con titular != habitual.

## Especificaciones (todas OLS con intercepto; SE agrupados por equipo-temporada del
equipo con caida != 0; si ambos la tienen, el del local)
- (i) Todos los partidos donde al menos un equipo tiene titular != habitual.
- (ii) Partidos donde al menos un equipo esta en los 2 primeros partidos de una racha, y
  el otro equipo tiene su habitual o tambien esta en sus 2 primeros. Incluye rachas que
  empiezan en semana 1 (cambio de offseason).
  - (ii-b, robustez) igual, solo rachas que empiezan en semana >= 2 (cambio en temporada,
    el caso de una lesion).
- (iii) Sobre la muestra (ii): pendientes separadas por tercil del valor del QB habitual
  reemplazado (terciles calculados en la muestra de seleccion, por equipo-partido con
  caida != 0): x_t = caida_visita*1[visita en t] - caida_local*1[local en t].
- Efecto elite-contra-suplente: b(ii) * (p90 del valor de QBs habituales - mediana del
  valor de los reemplazos en (ii)), en puntos y en Elo (x25).

## Particion
- Estimacion: 2010-2022. Verificacion: 2023-2025, solo signo y orden de magnitud (esos
  anos ya se usaron como holdout en otra decision; no son virgenes).

## Puerta de decision (fija)
- PASA si: b de la especificacion (ii) en 2010-2022 es > 0 con p unilateral < 0.05, Y el
  b de (ii) en 2023-2025 tiene el mismo signo (positivo).
- Si no pasa: no se implementa nada; solo reporte.
- Si pasa: el b que se usa en el simulador es el de (ii) 2010-2022 (no se re-estima con
  2023-2025).

## Subpregunta (solo reporte)
Efecto implicito del Poisson semanal (modelo JOINT_PROD, coeficientes de
`qb_value_live` y `backup_qb` entrenado 2002-2022) en puntos de margen para el mismo
cambio elite -> suplente: lambda_equipo*(exp(beta_qb*dv + beta_backup) - 1), con lambda =
media de puntos; y su peso efectivo en el blend (x0.45; el Elo no sabe del QB).

## Implementacion (solo si pasa)
- `simulate_one_season(..., elo_adjustments=None)`: desplazamiento en puntos de Elo por
  equipo, solo en partidos sin resultado real. Para no acumular la penalizacion via las
  actualizaciones de Elo de la simulacion, la expectativa usada en la actualizacion
  tambien incluye el desplazamiento; el rating guardado nunca incluye el desplazamiento.
  Con dict vacio o None el codigo hace exactamente las mismas operaciones que hoy.
- Playoffs: los mismos desplazamientos se suman al Elo final para los partidos de
  playoffs (la baja es por el resto de la temporada).
- Scripts de campeones: ajustes solo con use_played=True.

## Desviaciones
Ninguna en la estimacion ni en la puerta. Agregados despues de ver los resultados
(descriptivos, no cambian la decision ni el b usado):
1. Tabla no parametrica de perdida de spread por tramos de caida (muestra ii).
2. Chequeo de que la penalizacion no se acumula via las actualizaciones de Elo simuladas
   (version propuesta vs version ingenua).
3. Observacion: con "habitual = lider de s-1", el 40% de los equipo-partido tienen otro
   QB, sobre todo por cambios de offseason; la spec (i) queda dominada por esos casos.
