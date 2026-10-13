# REPORTE -- Factores para el lambda del Poisson semanal

Fecha: 2026-10-06. Protocolo pre-registrado en `PROTOCOLO.md` (con 9 desviaciones
registradas antes de abrir el holdout). Todos los numeros salen de `resultados.csv`.
No se modifico ningun archivo de produccion: el cambio propuesto esta en
`propuesta.diff` (aplica limpio con `patch -p1` desde `nfl_bot/`).

## Resumen

- **Aceptados (4 factores + 1 cambio de especificacion):** EPA de pase del ataque (C1a),
  success rate (C1b), valor de QB actualizado con su EPA de la temporada (C4c), marca
  de QB suplente (C4d), y estandarizar features con penalizacion casi nula (S0).
- **Modelo propuesto** (`JOINT_PROD` + S0, peso Poisson 0.45) vs produccion actual
  (spec actual, peso 0.25):

  | | log-loss blend | accuracy | devianza (pareada) |
  |---|---|---|---|
  | Seleccion 2010-2022, produccion | 0.6291 | 0.647 | -- |
  | Seleccion 2010-2022, propuesto | 0.6200 (delta -0.0091, SE 0.0024) | 0.652 | -0.332 (SE 0.028) |
  | Holdout 2023-2025, produccion | 0.6340 | 0.647 | -- |
  | Holdout 2023-2025, propuesto | 0.6250 (delta -0.0090) | 0.654 | -- |
  | Holdout, Elo solo (referencia) | 0.6348 | 0.649 | -- |

  Ese -0.009 sale de dos partes: cambio de spec (S0, ~-0.002 a -0.004) y factores
  (`JOINT_PROD`, -0.0064 en seleccion / -0.0044 en holdout).
- **Bug de produccion encontrado (independiente de los factores):** en
  `weekly_pipeline.build_week_features`, la semana a predecir todavia no tiene jugadas,
  asi que `weekly_epa_with_blend` no genera fila para ella y **todas** las features de
  EPA (`team_rush_epa`, `opp_def_epa_allowed`) caen a 0.0 para todos los equipos.
  Verificado con 2026 semana 5: 30/30 equipos en 0.0; semana 4 (ya jugada) si se
  calcula bien. O sea, en vivo el Poisson ha estado corriendo solo con QB + localia, y
  los backtests (que usan semanas ya jugadas) no reflejaban eso. Con el modelo nuevo el
  bug seria grave (success rate = 0 esta ~12 desviaciones bajo su media), por eso el
  arreglo va en el mismo diff (`extra_weeks=[week]`).
- **Mercado:** Vegas sigue muy por encima (ver referencia). Ninguno de nuestros factores
  agrega informacion sobre el residual contra el mercado (todos p_Holm >= 0.69).

## Tabla de candidatos

Devianza: delta media por equipo-partido (candidato - baseline), fuera de muestra,
walk-forward. SE por bloques de semana. p ajustado = Holm sobre los 18 candidatos.
Log-loss del blend: cada version con su peso Poisson optimo (grilla 0-0.6) en seleccion;
en holdout se usa el peso elegido en seleccion. Baseline de seleccion: log-loss 0.6263
(peso 0.30). Baseline de holdout: 0.6294.

| id | factor | delta devianza | SE | p Holm | delta LL blend (w base -> cand) | holdout: delta dev / delta LL | decision |
|---|---|---|---|---|---|---|---|
| S0 | features estandarizadas + alpha 1e-4 (vs spec produccion alpha=1) | -0.1815 | 0.0198 | <1e-6 | -0.0022 (0.15 -> 0.30) | -0.178 / -0.0040 | **ACEPTADO** |
| C1a | EPA de pase del ataque propio | -0.1193 | 0.0176 | <1e-6 | -0.0026 (0.30 -> 0.40) | -0.105 / -0.0008 | **ACEPTADO** |
| C1b | success rate del ataque propio | -0.0829 | 0.0168 | 1e-5 | -0.0034 (0.30 -> 0.40) | -0.073 / -0.0033 | **ACEPTADO** |
| C1c | defensa rival separada pase/carrera | +0.0000 | 0.0017 | 1.00 | -0.0002 | -- | descartado (a) |
| C2a | entregas con regresion a la media | -0.0276 | 0.0099 | 0.038 | -0.0005 | -- | descartado (a) |
| C2b | sack rate propio y del rival | -0.0307 | 0.0105 | 0.031 | -0.0010 | -- | descartado (a) |
| C2c | presion (test 2020-2022) | -0.0023 | 0.0053 | 1.00 | +0.0004 | -- | descartado (a) |
| C2d | jugadas explosivas | -0.0290 | 0.0098 | 0.031 | -0.0009 | -- | descartado (a) |
| C2e | EPA de equipos especiales | -0.0102 | 0.0035 | 0.031 | -0.0006 | -- | descartado (a) |
| C2f | ritmo (jugadas por partido) | -0.0106 | 0.0054 | 0.30 | +0.0002 | -- | descartado (a) |
| C3a | techo / superficie | -0.0196 | 0.0060 | 0.0115 | -0.0000 | -- | descartado (a), por poco |
| C3b | clima OBSERVADO (cota optimista) | -0.0277 | 0.0079 | 0.0055 | -0.0000 | -0.040 / -0.0001 | rechazado (c), (d) |
| C4a | titulares propios inactivos OL/WR1 (test 2016-2022) | +0.0012 | 0.0013 | 1.00 | -0.0000 | -- | descartado (a) |
| C4b | titulares rivales inactivos DL/CB1 (test 2016-2022) | +0.0008 | 0.0047 | 1.00 | -0.0004 | -- | descartado (a) |
| C4c | QB actualizado con EPA de la temporada | -0.0898 | 0.0098 | <1e-6 | -0.0038 (0.30 -> 0.40) | -0.114 / -0.0027 | **ACEPTADO** |
| C4d | marca de QB suplente | -0.0380 | 0.0090 | 0.0003 | -0.0029 (0.30 -> 0.35) | +0.001 / -0.0020 | **ACEPTADO** (ver nota) |
| O1 | binomial negativa (metrica 2*dNLL) | -1.885 | 0.064 | <1e-6 | +0.0025 (0.30 -> 0.15) | -1.76 / +0.0032 | rechazado (b), (c) |
| O2 | correlacion entre puntajes (Poisson bivariado) | 0 | 0 | 1.00 | 0 | -- | descartado (a) |
| JOINT | C1a+C1b+C3b+C4c+C4d | -0.1739 | 0.0203 | -- | -0.0064 (0.30 -> 0.45) | -0.163 / -0.0045 | rechazado (d): usa clima observado |
| **JOINT_PROD** | **C1a+C1b+C4c+C4d** | **-0.1502** | 0.0193 | -- | **-0.0064 (0.30 -> 0.45)** | **-0.127 / -0.0044** | **ACEPTADO** |

Errores estandar del log-loss: ~0.001-0.0019 en seleccion, ~0.002-0.0037 en holdout
(815 partidos); las mejoras de holdout son de la misma direccion pero ninguna es
significativa por si sola. RMSE de total y margen por candidato en `resultados.csv`.

Notas por factor:
- **C4d (QB suplente)**: paso todas las reglas, pero su mejora en devianza NO se
  replico en holdout (+0.001, SE 0.021); solo el log-loss del blend mejoro. Es el
  aceptado mas fragil.
- **C1b dentro del conjunto**: dado C1a+C4c+C4d, su aporte marginal es nulo
  (devianza -0.003, p = 0.42). Un modelo sin C1b probablemente rinde igual, pero esa
  version no se evaluo en holdout y no la propongo sin evaluarla.
- **C1a dentro del conjunto**: aporte marginal en devianza p = 0.036 y el blend sin
  C1a es 0.0008 mejor en seleccion (se solapa con el valor de QB). Misma conclusion:
  candidato a simplificar en una iteracion futura, con holdout nuevo.
- **O1 (binomial negativa)**: describe muchisimo mejor la distribucion de puntos (los
  puntos NFL tienen varianza ~4.8x la media), pero las probabilidades de victoria
  quedan peor calibradas y el blend empeora. Util si algun dia se usan los marcadores
  exactos; no para el pronostico de ganador.
- **O2**: la correlacion empirica de puntajes es -0.047; el Poisson bivariado de
  componente comun solo representa correlacion positiva, asi que queda igual al
  baseline.

### Referencia: mercado (no decide nada)

| id | descripcion | delta devianza vs baseline | delta LL blend (w) |
|---|---|---|---|
| V1 | baseline + log(total implicito de Vegas) | -0.334 (SE 0.028) | -0.0133 (0.30 -> 0.50) |
| V1m | solo mercado | -0.327 (SE 0.027) | -0.0122 (0.30 -> 0.50) |

El modelo propuesto cierra ~45% de la brecha en devianza y ~48% en log-loss contra el
techo del mercado. **Residual contra el mercado (V2)**: con offset log(implicito),
ni las features del baseline (p = 0.12) ni ninguno de los 15 factores mejora la
devianza (minimo p crudo 0.044 para C4d; todos p_Holm >= 0.69). Nada de lo que
medimos le gana al mercado; todo lo que aporta ya esta en la linea.

## Factores aceptados: cambios propuestos

Archivo completo: `experimentos/propuesta.diff` (437 lineas, 4 archivos). Para aplicarlo:

```bash
patch -p1 < experimentos/propuesta.diff
```

Que cambia:

1. `poisson_model.py`
   - `FEATURES = [team_rush_epa, team_pass_epa, team_success, qb_value_live, backup_qb,
     opp_def_epa_allowed, is_home, div_game]` (`qb_value_live` reemplaza a
     `team_qb_value`, que queda como su prior).
   - `make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4))` en vez de
     `PoissonRegressor(alpha=1.0)` sobre features crudas (S0). Con alpha=1 los
     coeficientes de EPA quedaban encogidos ~6x (rush EPA: 0.095 vs 0.644).
   - `predict_lambda(model, features: dict)` en vez de argumentos posicionales.
   - Entrena solo con temporada regular.
2. `poisson_features.py`
   - `weekly_offense_stat_with_blend` generica (misma logica decay 0.85 + blend 8
     partidos); `weekly_rush_epa_with_blend` queda como envoltorio (misma API).
   - Nuevas: `weekly_pass_epa_with_blend`, `weekly_success_rate_with_blend`,
     `qb_value_live` (`(150*prior + S)/(150 + N)`, S y N = qb_epa y dropbacks de la
     temporada con decay 0.85/semana), `backup_qb_flags`.
   - Verificado: las features nuevas coinciden exactamente (diff < 1e-15) con las
     usadas en los experimentos (temporada 2023).
3. `team_epa.py`: `weekly_epa_with_blend(..., extra_weeks=())` -- arreglo del bug de la
   semana a predecir.
4. `weekly_pipeline.py`
   - `POISSON_BLEND_WEIGHT = 0.45` (peso optimo de `JOINT_PROD` en seleccion; en
     holdout el optimo fue 0.50, consistente).
   - Poisson entrenado con 2002..season-1 (la ventana evaluada; antes 2019+).
   - Calcula las features nuevas para la semana a predecir con `extra_weeks=[week]`.
   - `backup_qb` = el titular proyectado no es el lider de titularidades de la temporada.
   - Si el nombre de `PROJECTED_STARTERS` no esta en el roster (hoy: ATL, "Michael
     Penix"), usa el ultimo titular real en vez de valor de QB 0.0, con aviso.
   - Elo con `k=20, home_advantage=35, regression_to_mean=0.40` explicitos.

Prueba de humo (copia en scratchpad, 2026 semana 5): corre de punta a punta en ~3 min,
features con distribuciones sanas (antes: EPA en 0.0) y marcadores tipicos ~20-28.

Coeficientes del modelo propuesto (entrenado 2002-2022), por 1 desviacion estandar:
EPA defensivo rival +0.060, EPA de pase +0.050, localia +0.050, QB live +0.045,
success rate +0.031, QB suplente -0.031 (crudo: -8% de puntos esperados), EPA de
carrera +0.020, divisional -0.015.

## Limitaciones

1. **El holdout no es virgen respecto a los parametros globales.** home_advantage=35,
   regression_to_mean=0.40, K=20 y otras constantes (decay 0.85, blend 8, decay de QB
   0.6) se calibraron con datos que incluyen 2023-2025. Si es virgen respecto a las
   decisiones de este experimento (corrido una sola vez, candado en `data/HOLDOUT_LOCK`).
2. **Poco poder en holdout.** 815 partidos; las mejoras de log-loss en holdout son de
   la misma direccion pero con SE de 0.002-0.004, ninguna significativa por si sola.
   La regla (b) solo exige direccion.
3. **QB titular.** El backtest usa el titular real del partido (`home_qb_id`, conocido
   ~90 min antes del kickoff con los inactivos). En produccion es el titular PROYECTADO
   de `starters_2026.py`; `qb_value_live` y `backup_qb` dependen de que esa lista se
   actualice cada semana. Si no, el backtest es optimista.
4. **Disponibilidad por partido (C4a/C4b) se probo de forma simple**: Out/Doubtful
   del reporte final, ponderado por snap share; OLB no distinguible (se uso DL);
   Questionable ignorado; solo 2013+. Que no funcionara aqui no descarta una version
   mejor (p.ej. ponderar por EPA del jugador).
5. **Clima** observado y con datos faltantes (49% de partidos al aire libre en 2022);
   aun como cota optimista, su efecto en el blend es nulo.
6. **Presion** solo tiene 3 temporadas de prueba (2020-2022): el test tiene poco poder.
7. **Elo del experimento** usa historial desde 1999; el pipeline en vivo arranca en 2015.
   Las probabilidades Elo difieren un poco del motor en vivo.
8. **No pude reproducir exactamente el baseline citado** (accuracy 0.637, log-loss
   0.6377): el periodo no esta documentado. Mi reproduccion de produccion (spec actual,
   peso 0.25): seleccion 0.6291 / 0.647, holdout 0.6340 / 0.647.
9. `walk_forward_backtest.py` usa `poisson_weight_by_week` (peso variable por semana)
   y `weekly_pipeline.py` un peso fijo; aqui se evaluo peso fijo, como en el pipeline.
10. Los backtests de produccion tienen una fuga menor: la cubeta de draft de novatos
    usa todas las temporadas, incluido el futuro. Aqui se corrigio (no afecta a las
    predicciones en vivo).
11. La ventana de entrenamiento 2002+ (vs 2019+ en produccion) no se comparo como
    factor; el diff adopta la ventana que se evaluo.

## Archivos

- `PROTOCOLO.md` -- reglas pre-registradas + desviaciones.
- `download_data.py`, `features.py` -- datos y features (reimplementacion validada
  contra produccion en `validate_baseline.py`).
- `leakage_test.py` -- 28 features recalculadas con datos truncados en 48
  temporada-semanas: igualdad exacta en todas; clima marcado como observado.
  Resultados en `data/leakage_result.csv`, `data/leakage_injury_dates.csv`.
- `evalcore.py`, `seleccion.py`, `holdout.py` -- evaluacion.
- `resultados.csv` -- todas las corridas (seleccion, conjunto, referencia, holdout),
  incluidos los descartados.
- `finalistas.json`, `propuesta.diff`.
