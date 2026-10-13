"""
Busca el peso constante (no dependiente de semana) que minimiza logloss
del blend Elo+Poisson, usando el cache de walk_forward_backtest.py.
Reemplaza la hipotesis de "peso por semana" que el backtest no soporto.
"""
import numpy as np
import polars as pl

def evaluate_metrics(y_true, y_pred):
    eps = 1e-10
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    acc = np.mean((y_pred > 0.5).astype(int) == y_true)
    logloss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
    brier = np.mean((y_pred - y_true) ** 2)
    return acc, logloss, brier

if __name__ == "__main__":
    merged = pl.read_parquet("elo_poisson_predictions_2021_2024.parquet")
    merged = merged.with_columns(
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("actual_home_win")
    )
    y_true = merged["actual_home_win"].to_numpy()
    elo_p = merged["elo_prob_home"].to_numpy()
    poisson_p = merged["poisson_prob_home"].to_numpy()

    print(f"{'Peso Poisson':<15}{'Accuracy':<12}{'Log-loss':<12}{'Brier':<10}")
    best_logloss = np.inf
    best_weight = None

    for w in np.arange(0.0, 0.55, 0.05):
        blend = w * poisson_p + (1 - w) * elo_p
        acc, logloss, brier = evaluate_metrics(y_true, blend)
        marker = ""
        if logloss < best_logloss:
            best_logloss = logloss
            best_weight = w
            marker = "  <- mejor logloss hasta ahora"
        print(f"{w:<15.2f}{acc:<12.3f}{logloss:<12.4f}{brier:<10.4f}{marker}")

    print(f"\nMejor peso constante encontrado: {best_weight:.2f} (logloss={best_logloss:.4f})")