"""
Grid search rapido sobre home_advantage para corregir el sesgo de calibracion
que vimos desde el primer backtest (el modelo sobreestimaba consistentemente
la probabilidad del equipo local en el rango medio).
"""
from motor_elo_v1 import run_elo_over_seasons
from backtest_elo_v1 import evaluate, calibration_check

SEASONS = range(2015, 2025)

if __name__ == "__main__":
    for ha in [15, 25, 35, 45, 55, 65]:
        print(f"\n{'='*20} home_advantage = {ha} {'='*20}")
        preds, _ = run_elo_over_seasons(SEASONS, home_advantage=ha)
        evaluate(preds, min_season=2018)

    print(f"\n\n{'='*20} Calibracion detallada con el mejor candidato {'='*20}")
    # Ajusta este valor al que salga con mejor logloss/brier del grid de arriba,
    # y vuelve a correr para ver la tabla completa de calibracion
    preds, _ = run_elo_over_seasons(SEASONS, home_advantage=35)
    calibration_check(preds, min_season=2018)