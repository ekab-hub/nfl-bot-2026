"""
Grid search de regression_to_mean, mismo patron que home_advantage_search.py.
El 1/3 que usamos desde el principio era un valor prestado de modelos
publicos (tipo 538), nunca calibrado con nuestro propio backtest -- a
diferencia de home_advantage, que si se ajusto con evidencia.
"""
from motor_elo_v1 import run_elo_over_seasons
from backtest_elo_v1 import evaluate, calibration_check

SEASONS = range(2015, 2025)

if __name__ == "__main__":
    for rtm in [0.15, 0.20, 0.25, 1/3, 0.40, 0.45, 0.50]:
        print(f"\n{'='*20} regression_to_mean = {rtm:.3f} {'='*20}")
        preds, _ = run_elo_over_seasons(SEASONS, home_advantage=35, regression_to_mean=rtm)
        evaluate(preds, min_season=2018)

    print(f"\n\n{'='*20} Calibracion detallada con el mejor candidato {'='*20}")
    # Ajusta este valor al que salga con mejor logloss/brier del grid de arriba
    preds, _ = run_elo_over_seasons(SEASONS, home_advantage=35, regression_to_mean=1/3)
    calibration_check(preds, min_season=2018)