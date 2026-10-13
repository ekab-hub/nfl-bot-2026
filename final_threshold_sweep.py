"""
Recalcula la efectividad en partidos individuales con AMBOS cambios:
regression_to_mean=0.40 (ya es el default de motor_elo_v1.py) + blend de
mercado al 20%. Compara contra la tabla vieja (threshold_sweep.py original).
"""
import numpy as np
import polars as pl
from threshold_sweep import build_full_blend_dataset, TEST_SEASONS
from market_blend_search import get_market_probs, MARKET_TEST_SEASONS

MARKET_WEIGHT = 0.20
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def evaluate_thresholds(df, prob_col, thresholds):
    results = []
    for t in thresholds:
        favorite_is_home = df[prob_col] >= 0.5
        favorite_prob = pl.when(favorite_is_home).then(pl.col(prob_col)).otherwise(1 - pl.col(prob_col))
        sub = df.with_columns(favorite_prob.alias("favorite_prob"), favorite_is_home.alias("favorite_is_home"))
        sub = sub.filter(pl.col("favorite_prob") >= t)
        if sub.height == 0:
            results.append({"umbral": t, "n_partidos": 0, "accuracy": None})
            continue
        correct = (sub["favorite_is_home"] == sub["home_won"].cast(pl.Boolean))
        results.append({"umbral": t, "n_partidos": sub.height, "accuracy": round(float(correct.to_numpy().mean()), 4)})
    return pl.DataFrame(results)


if __name__ == "__main__":
    print("Recalculando blend Elo+Poisson con regression_to_mean=0.40 (esto tarda, como siempre)...")
    df = build_full_blend_dataset()  # ya usa el nuevo default de motor_elo_v1.py automaticamente

    market = get_market_probs(MARKET_TEST_SEASONS)
    df = df.join(market, on=["season", "week", "home_team", "away_team"], how="left")

    # Donde no hay dato de mercado (fuera de 2010-2025), el peso de mercado
    # simplemente no se aplica -- usamos el blend base sin cambios ahi.
    df = df.with_columns(
        pl.when(pl.col("market_prob_home").is_not_null())
          .then(MARKET_WEIGHT * pl.col("market_prob_home") + (1 - MARKET_WEIGHT) * pl.col("blend_prob_home"))
          .otherwise(pl.col("blend_prob_home"))
          .alias("final_prob_home")
    )

    print(f"\nTotal partidos ({min(TEST_SEASONS)}-{max(TEST_SEASONS)}): {df.height}")
    print(f"Partidos con dato de mercado disponible: {df.filter(pl.col('market_prob_home').is_not_null()).height}\n")

    print("=== NUEVA tabla (regression_to_mean=0.40 + 20% mercado donde hay dato) ===")
    results_new = evaluate_thresholds(df, "final_prob_home", THRESHOLDS)
    with pl.Config(tbl_rows=10):
        print(results_new)

    print("\n=== Para comparar: misma tabla SIN el 20% de mercado (solo el cambio de regression_to_mean) ===")
    results_no_market = evaluate_thresholds(df, "blend_prob_home", THRESHOLDS)
    with pl.Config(tbl_rows=10):
        print(results_no_market)