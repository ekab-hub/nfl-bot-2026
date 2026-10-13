# PROTOCOLO -- Factores para el lambda del Poisson semanal

Escrito el 2026-10-06 ANTES de correr cualquier modelo. Lo unico ejecutado antes de
escribir esto fue un inventario de cobertura de datos (que columnas/temporadas existen
en nflverse), sin ajustar ni evaluar nada. Cualquier desviacion posterior a este
documento se registra en la seccion "Desviaciones" al final, con fecha y motivo.

## 0. Reglas de trabajo

- Todo el codigo y resultados viven en `nfl_bot/experimentos/`. No se modifica ningun
  archivo de produccion (`poisson_model.py`, `poisson_features.py`, `team_epa.py`,
  `qb_adjustment.py`, `weekly_pipeline.py`, `motor_elo_v1.py`, ...). Solo se importan
  (lectura) y los cambios se proponen como diff en `REPORTE.md`.
- Parametros globales FIJOS y explicitos en cada llamada (nunca defaults):
  Elo `k=20`, `home_advantage=35`, `regression_to_mean=0.40`. Historial Elo desde 1999.
- Todo resultado (aceptado o descartado) se escribe en `experimentos/resultados.csv`.

## 1. Datos y particion

- Fuente: nflverse via `nflreadpy` (pbp, schedules, injuries, snap_counts,
  participation, players, draft_picks). Se cachean en `experimentos/data/`.
- Solo temporada regular (`game_type == "REG"`) para entrenar y evaluar.
- Features se construyen para 2002-2025 (pbp desde 2001 como "temporada previa").
- **Walk-forward por temporada**: para la temporada de prueba `s`, el modelo Poisson se
  entrena con temporadas `2002 .. s-1` (ventana expansiva) y predice todos los partidos
  REG de `s`. Las features de cada partido usan solo informacion previa al kickoff.
- **Seleccion: 2010-2022.** Todas las decisiones de seleccion se toman aqui.
- **Holdout: 2023-2025.** Se toca UNA sola vez, al final, solo con los finalistas
  (baseline + candidatos que pasen el criterio (a) individualmente + su conjunto).
  El script de holdout escribe un archivo-candado y se niega a correr una segunda vez.
- Advertencia documentada: los parametros globales (home_advantage=35,
  regression_to_mean=0.40, K=20) y otras constantes de produccion (decay 0.85,
  blend_games=8, decay de QB 0.6, peso 0.25) fueron calibrados con datos que incluyen
  2023-2025. El holdout NO es virgen respecto a esos parametros; si lo es respecto a
  las decisiones de factores tomadas aqui.
- Candidatos con cobertura limitada (snap counts desde 2013, participation/presion
  desde 2016): base y candidato se ajustan sobre las MISMAS filas de entrenamiento
  (solo temporadas con cobertura) y se evaluan sobre las mismas filas de prueba,
  exigiendo al menos 3 temporadas de entrenamiento con cobertura. La comparacion es
  siempre pareada sobre el mismo subconjunto; se reporta el subconjunto.

## 2. Modelo

- GLM Poisson log-link, una fila por equipo-partido, target = puntos anotados.
- **Especificacion de evaluacion**: features estandarizadas con media/desviacion del
  set de entrenamiento de cada fold, `PoissonRegressor(alpha=1e-4, max_iter=1000)`.
- Produccion usa features crudas con `alpha=1.0` (penalizacion fuerte sobre features
  de escala ~0.1). Se reporta el baseline con la spec de produccion como referencia y
  el cambio de spec se evalua como candidato **S0** con las mismas reglas.
- Baseline (5 features de produccion): `team_rush_epa`, `team_qb_value`,
  `opp_def_epa_allowed`, `is_home`, `div_game`, reimplementadas vectorizadas con la
  misma logica (decay semanal 0.85, blend con temporada previa en 8 partidos, valor de
  QB con decay 0.6 por temporada y peso min(n/100,1), cubetas de draft para novatos).
  Correccion de leakage del baseline: la cubeta de novatos se calcula solo con
  temporadas `< s` (produccion usa todas, incluido el futuro). Se valida que la
  reimplementacion reproduce las features de produccion en una muestra.
- Cada candidato = baseline + sus features (de uno en uno). Luego conjunto de los
  sobrevivientes del criterio (a).

## 3. Metricas

Primaria (por equipo-partido, fuera de muestra):
- **Devianza de Poisson** media: `2*(y*log(y/mu) - (y - mu))`.
- Delta pareado `d_i = dev_cand_i - dev_base_i`; error estandar por bloques de
  semana (cluster = temporada-semana; ambos equipos de un partido caen en el mismo
  bloque): `SE = sqrt(sum_b (D_b - n_b*dbar)^2) / N`. `z = dbar/SE`, p bilateral normal.
- Tambien: RMSE del total (`lam_h + lam_a` vs total real) y del margen
  (`lam_h - lam_a` vs margen real), con delta pareado.

Secundaria:
- **Log-loss del blend** `p = w*P_poisson + (1-w)*P_elo`, `P_poisson = P(D>0)` con
  Skellam (formula de produccion). `w` (peso del Poisson) RE-OPTIMIZADO para cada
  version en la grilla `0, 0.05, ..., 0.60` sobre seleccion. Partidos empatados se
  excluyen del log-loss. Delta vs baseline (cada uno con su `w` optimo), SE por
  bloques de semana.
- En holdout se usa el `w` elegido en seleccion (no se re-optimiza; el optimo de
  holdout se reporta solo como informacion).

## 4. Regla de aceptacion (todas deben cumplirse)

(a) Devianza en seleccion mejora (`dbar < 0`) con p ajustado por Holm `< 0.01`, Holm
    sobre TODOS los candidatos de la familia (seccion 5, m = 18).
(b) El log-loss del blend mejora en holdout (delta < 0, misma direccion).
(c) La mejora del log-loss del blend en seleccion (con `w` optimo) es `>= 0.001`.
(d) La feature se conoce ANTES del kickoff: test de leakage explicito por feature
    (seccion 6). El clima observado no puede aceptarse para produccion; se reporta
    como cota optimista.

## 5. Familia de candidatos (m = 18, fija)

| id | candidato | features |
|---|---|---|
| S0 | spec estandarizada + alpha bajo (vs spec produccion) | -- |
| C1a | EPA de pase del ataque propio | EPA/dropback del equipo (decay+blend) |
| C1b | success rate del ataque propio | success rate run/pass (decay+blend) |
| C1c | defensa rival separada | reemplaza `opp_def_epa_allowed` por EPA permitido en pase y en carrera |
| C2a | entregas con regresion a la media | giveaways/jugada propio y takeaways/jugada del rival, encogidos 50% hacia la media de liga |
| C2b | sack rate | sacks/dropback permitidos propio, sacks/dropback generados por el rival |
| C2c | presion (2016+) | `was_pressure`/dropback permitido propio y generado por el rival |
| C2d | jugadas explosivas | tasa propia (carrera >=10 yd, pase >=20 yd) y tasa permitida por el rival |
| C2e | equipos especiales | EPA neto de ST por partido propio y del rival |
| C2f | ritmo | jugadas ofensivas por partido propias y del rival |
| C3a | techo / superficie | indoor (dome/closed), cesped artificial |
| C3b | clima observado (cota optimista) | viento (mph), frio = max(0, 50 - temp); indoor -> 0 |
| C4a | titulares propios inactivos (2013+) | suma de snap share de OL Out/Doubtful; snap share del WR1 si Out/Doubtful |
| C4b | titulares del rival inactivos (2013+) | suma de snap share de pass rushers (DE/OLB) rivales Out/Doubtful; CB1 rival Out/Doubtful |
| C4c | QB actualizado con EPA de la temporada | reemplaza `team_qb_value` por `(150*prior + S_temp)/(150 + N_temp)` con decay semanal 0.85 |
| C4d | QB suplente | 1 si el QB titular del partido no es el lider en titularidades del equipo en la temporada (semana 1 -> 0) |
| O1 | binomial negativa | NB2 con dispersion estimada en entrenamiento (metrica: 2*delta NLL por equipo-partido) |
| O2 | correlacion entre puntajes | Poisson bivariado (componente comun lambda3 estimado en entrenamiento), metrica 2*delta NLL conjunta / 2 |

Constantes de features fijadas aqui: decay semanal 0.85, blend 8 partidos con la
temporada previa (igual que produccion); snap share = promedio de % de snaps del
jugador en partidos previos de la temporada (si no hay, temporada previa); estatus
Out/Doubtful del ultimo reporte de la semana; WR1/CB1 = mayor snap share previo
del equipo en esa posicion.

Disponibilidad de QB titular "fuera" no es candidato aparte: el baseline ya usa el QB
titular real del partido (`home_qb_id`/`away_qb_id`, conocido al anunciarse titulares
/inactivos ~90 min antes del kickoff). En produccion se usa el titular PROYECTADO, por
lo que el backtest es optimista en ese punto (documentado como limitacion).

Referencia (fuera de la familia, NO decide produccion):
- **V1**: baseline + `log(total implicito del mercado)` (total_line/2 +- spread_line/2):
  techo de lo alcanzable. Y Poisson solo-mercado (offset).
- **V2**: residual contra el mercado: Poisson con offset `log(implicito)` + cada
  candidato; delta de devianza vs offset solo. Indica si nuestros factores agregan
  informacion que el mercado no tiene.

## 6. Test de leakage por feature

1. Features derivadas de pbp/participation/snaps/lesiones: para una muestra de
   temporada-semana (>= 40, aleatoria con semilla fija, incluye semana 1), se
   recalculan las features usando datos TRUNCADOS a jugadas/partidos/reportes
   anteriores a esa semana y se verifica igualdad exacta (tolerancia 1e-9) con las
   features calculadas con todos los datos. Falla = feature rechazada.
2. Reporte de lesiones: se verifica que el `date_modified` del ultimo reporte usado
   sea anterior a la fecha del partido (se reporta el % que no cumple).
3. Features de calendario/estadio: techo y superficie son estaticos/conocidos antes
   del partido. Temperatura y viento del calendario son OBSERVADOS -> se marcan como
   cota optimista; no aceptables sin una fuente de pronostico.
4. QB titular del partido: conocido antes del kickoff (lista de inactivos), pero en
   produccion se proyecta; se documenta.

## 7. Orden de ejecucion

1. Construir datos y features (`build_features.py`), test de leakage (`leakage_test.py`),
   validar reimplementacion del baseline contra produccion.
2. Seleccion (`seleccion.py`): baseline, S0, cada candidato de uno en uno, Holm,
   blend con `w` re-optimizado, referencias de mercado; conjunto de sobrevivientes.
   Todo a `resultados.csv`. Escribe `finalistas.json`.
3. Holdout (`holdout.py`): una sola corrida con los finalistas. Candado.
4. `REPORTE.md`.

## Desviaciones

Registradas el 2026-10-06, todas ANTES de abrir el holdout.

1. **Pass rushers (C4b)**: el reporte de lesiones no distingue OLB (vienen como "LB").
   Se usa linea defensiva DE + DT + NT. Los edge de 3-4 quedan fuera.
2. **Presion (C2c)**: 2016 no tiene temporada previa para el prior -> cobertura
   efectiva 2017+, prueba 2020-2022 (minimo 3 temporadas de entrenamiento). Ademas
   `was_pressure` cubre ~89% de los dropbacks en 2016-2022 y 100% desde 2023.
3. **Clima (C3b)**: viento/temperatura faltan en partidos al aire libre (49% en 2022,
   21% en 2023, <6% resto). Se imputa con la mediana de entrenamiento (aire libre).
4. **Snap counts**: la temporada 2012 viene vacia; disponibilidad 2013+ (como estaba
   previsto), y la semana 1 de 2013 queda en cero por falta de temporada previa.
5. **Conjunto desplegable**: el conjunto literal de sobrevivientes incluye C3b (clima
   observado), que por la regla (d) no puede ir a produccion. Se agrega como finalista
   `JOINT_PROD` = sobrevivientes que pasan (d). `JOINT` (con clima) se mantiene como
   cota optimista. Decidido despues de ver seleccion y antes de abrir el holdout.
6. **Prior faltante** (solo HOU 2002, que no existia en 2001): media de liga en vez de 0.
   Afecta solo filas de entrenamiento.
7. **Fechas del reporte de lesiones**: 2025 no trae `date_modified`, no se puede
   verificar la fecha; 2009-2024: <0.2% de reportes con fecha posterior al partido.
8. **O2**: la correlacion empirica de puntajes es -0.047; el Poisson bivariado de
   componente comun solo representa correlacion >= 0, asi que lambda3 = 0 en todos
   los folds y O2 queda identico al baseline (p = 1).
9. **Valor de QB vs produccion**: produccion agrupa por (id, nombre); si un QB aparece
   con dos grafias en una temporada (p.ej. Josh Allen), produccion lo parte en dos
   filas. La reimplementacion agrupa por id. Mas la correccion de cubeta de novatos.
   EPA de carrera y EPA defensivo rival coinciden exactamente (diff < 1e-15).
