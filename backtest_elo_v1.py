"""
Backtest del motor Elo v1.
Mide: accuracy (aciertos de ganador), log-loss, y comparacion vs. el mercado (spread/moneyline).
Se corre sobre temporadas historicas para saber si el Elo base ya sirve
antes de agregarle EPA y ajuste de QB.
"""
import numpy as np
import polars as pl
from motor_elo_v1 import run_elo_over_seasons

SEASONS_BACKTEST = range(2015, 2026)  # entrena y evalua sobre el mismo histórico continuo

def moneyline_to_prob(ml):
    if ml is None:
        return None
    if ml < 0:
        return -ml / (-ml + 100)
    else:
        return 100 / (ml + 100)

def evaluate(preds: pl.DataFrame, min_season=2018):
    """
    Evaluamos solo a partir de min_season para dar tiempo a que el Elo
    'aprenda' con años previos de warmup (2015-2017 como calentamiento).
    """
    df = preds.filter(pl.col("season") >= min_season)

    y_true = (df["home_score"] > df["away_score"]).cast(pl.Int8).to_numpy()
    y_pred = df["model_prob_home"].to_numpy()

    # Accuracy (usando 0.5 como umbral)
    acc = np.mean((y_pred > 0.5).astype(int) == y_true)

    # Log-loss (clip para evitar log(0))
    eps = 1e-10
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

    # Brier score (otra medida de calibracion, mas intuitiva que log-loss)
    brier = np.mean((y_pred - y_true) ** 2)

    print(f"=== Evaluacion desde temporada {min_season} ===")
    print(f"Partidos evaluados: {len(df)}")
    print(f"Accuracy (umbral 0.5): {acc:.3f}")
    print(f"Log-loss: {logloss:.4f}")
    print(f"Brier score: {brier:.4f}")

    # --- Comparacion contra el mercado (Vegas) ---
    market_probs = df["home_moneyline"].map_elements(moneyline_to_prob, return_dtype=pl.Float64)
    valid_market = market_probs.is_not_null()

    if valid_market.sum() > 0:
        y_true_m = y_true[valid_market.to_numpy()]
        market_p = market_probs.filter(valid_market).to_numpy()
        model_p = df["model_prob_home"].filter(valid_market).to_numpy()

        market_acc = np.mean((market_p > 0.5).astype(int) == y_true_m)
        market_logloss = -np.mean(
            y_true_m * np.log(np.clip(market_p, eps, 1-eps)) +
            (1 - y_true_m) * np.log(np.clip(1 - market_p, eps, 1-eps))
        )
        print(f"\n--- Referencia: mercado (moneyline) ---")
        print(f"Accuracy mercado: {market_acc:.3f}")
        print(f"Log-loss mercado: {market_logloss:.4f}")
        print(f"(Nuestro objetivo NO es vencer al mercado, es acercarnos razonablemente)")

    return {"accuracy": acc, "logloss": logloss, "brier": brier}


def calibration_check(preds: pl.DataFrame, min_season=2018, n_bins=10):
    """
    Revisa si cuando el modelo dice '70% de probabilidad de que gane el local',
    el local realmente gana ~70% de las veces. Esto es mas importante que accuracy
    para un bot que va a comunicar probabilidades.
    """
    df = preds.filter(pl.col("season") >= min_season)
    y_true = (df["home_score"] > df["away_score"]).cast(pl.Int8).to_numpy()
    y_pred = df["model_prob_home"].to_numpy()

    bins = np.linspace(0, 1, n_bins + 1)
    print(f"\n=== Calibracion (bins de probabilidad) ===")
    print(f"{'Rango prob.':<15}{'N':<8}{'Prob. promedio':<18}{'Win rate real':<15}")
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if mask.sum() > 0:
            avg_pred = y_pred[mask].mean()
            actual_rate = y_true[mask].mean()
            print(f"{bins[i]:.1f}-{bins[i+1]:.1f}      {mask.sum():<8}{avg_pred:<18.3f}{actual_rate:<15.3f}")


if __name__ == "__main__":
    preds, engine = run_elo_over_seasons(SEASONS_BACKTEST)
    evaluate(preds, min_season=2018)
    calibration_check(preds, min_season=2018)

    print("\n=== Top 10 ratings finales (fin de 2025) ===")
    ratings_sorted = sorted(engine.ratings.items(), key=lambda x: -x[1])
    for team, rating in ratings_sorted[:10]:
        print(f"{team}: {rating:.1f}")

    print("\n\n=== Comparacion justa: Elo SOLO temporada 2024 (285 partidos, igual que Poisson) ===")
    evaluate_2024_only = preds.filter(pl.col("season") == 2024)
    evaluate(evaluate_2024_only, min_season=2024)  # min_season=2024 aqui ya no filtra nada mas, solo es para el print