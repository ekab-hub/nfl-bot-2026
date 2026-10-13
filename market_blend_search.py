"""
Evalua si agregar la probabilidad implicita del mercado (Vegas, via moneyline)
mejora el blend Elo+Poisson que ya tenemos validado. Usa el cache de
threshold_sweep.py para no recalcular el walk-forward completo.

Cobertura de moneyline: incompleta/irregular antes de 2010, completa desde
2010 en adelante -- se restringe la prueba a 2010-2025 para no meter ruido
de años con datos parciales.
"""
import numpy as np
import polars as pl
import nflreadpy as nfl
from motor_elo_v1 import normalize_team

MARKET_TEST_SEASONS = range(2010, 2026)


def moneyline_to_prob(ml):
    if ml is None:
        return None
    if ml < 0:
        return -ml / (-ml + 100)
    else:
        return 100 / (ml + 100)


def get_market_probs(seasons):
    df = nfl.load_schedules(seasons=list(seasons))
    df = df.filter(pl.col("game_type") == "REG")
    df = df.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])
    df = df.with_columns([
        pl.col("home_moneyline").map_elements(moneyline_to_prob, return_dtype=pl.Float64).alias("raw_home_prob"),
        pl.col("away_moneyline").map_elements(moneyline_to_prob, return_dtype=pl.Float64).alias("raw_away_prob"),
    ])
    # Quitar el "vig" (margen de la casa): normalizar para que sumen 1,
    # en vez de usar la probabilidad cruda del home_moneyline solo (que
    # sistematicamente suma un poco mas de 1 por el margen de la casa).
    df = df.filter(pl.col("raw_home_prob").is_not_null() & pl.col("raw_away_prob").is_not_null())
    df = df.with_columns(
        (pl.col("raw_home_prob") / (pl.col("raw_home_prob") + pl.col("raw_away_prob"))).alias("market_prob_home")
    )
    return df.select(["season", "week", "home_team", "away_team", "market_prob_home"])


def evaluate_metrics(y_true, y_pred):
    eps = 1e-10
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    acc = np.mean((y_pred > 0.5).astype(int) == y_true)
    logloss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
    brier = np.mean((y_pred - y_true) ** 2)
    return acc, logloss, brier


if __name__ == "__main__":
    cached = pl.read_parquet("threshold_sweep_predictions.parquet")
    cached = cached.filter(pl.col("season").is_in(list(MARKET_TEST_SEASONS)))

    market = get_market_probs(MARKET_TEST_SEASONS)
    merged = cached.join(market, on=["season", "week", "home_team", "away_team"], how="inner")
    merged = merged.with_columns(
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_won")
    )
    print(f"Partidos con blend + mercado cruzados (2010-2025): {merged.height}\n")

    y_true = merged["home_won"].to_numpy()
    blend_p = merged["blend_prob_home"].to_numpy()  # Elo+Poisson, ya validado
    market_p = merged["market_prob_home"].to_numpy()

    print("=== Referencia: blend actual (Elo+Poisson) solo, en esta misma ventana ===")
    acc, logloss, brier = evaluate_metrics(y_true, blend_p)
    print(f"acc={acc:.4f}  logloss={logloss:.4f}  brier={brier:.4f}\n")

    print("=== Referencia: mercado solo (sin vig) ===")
    acc, logloss, brier = evaluate_metrics(y_true, market_p)
    print(f"acc={acc:.4f}  logloss={logloss:.4f}  brier={brier:.4f}\n")

    print(f"{'Peso mercado':<15}{'Accuracy':<12}{'Log-loss':<12}{'Brier':<10}")
    best_logloss, best_weight = np.inf, None
    for w in np.arange(0.0, 1.05, 0.1):
        final_p = w * market_p + (1 - w) * blend_p
        acc, logloss, brier = evaluate_metrics(y_true, final_p)
        marker = ""
        if logloss < best_logloss:
            best_logloss, best_weight = logloss, w
            marker = "  <- mejor logloss hasta ahora"
        print(f"{w:<15.2f}{acc:<12.4f}{logloss:<12.4f}{brier:<10.4f}{marker}")

    print(f"\nMejor peso de mercado: {best_weight:.2f} (logloss={best_logloss:.4f})")