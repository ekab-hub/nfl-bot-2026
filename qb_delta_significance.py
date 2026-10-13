"""
Prueba de significancia especifica para qb_delta: la tabla de ejemplos de
calibrate_scales.py mostro varios casos donde qb_delta alto coincidio con
win_change NEGATIVO, lo cual contradice la direccion esperada. Antes de
confiar en el coeficiente, medimos significancia estadistica aislada.
"""
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from underlying_stats import build_team_season_stats
from calibrate_scales import build_qb_delta_dataset

if __name__ == "__main__":
    seasons_range = range(2016, 2026)
    stats = build_team_season_stats(range(2015, 2026))
    qb_deltas = build_qb_delta_dataset(seasons_range)

    wins_by_season = stats.select(["season", "team", "wins"])
    prev_wins = wins_by_season.rename({"season": "season_prev", "wins": "wins_prev"}) \
        .with_columns((pl.col("season_prev") + 1).alias("season"))
    merged_wins = wins_by_season.join(prev_wins, on=["season", "team"], how="inner") \
        .with_columns((pl.col("wins") - pl.col("wins_prev")).alias("win_change"))

    changed = merged_wins.join(qb_deltas, on=["season", "team"], how="inner")

    print(f"Casos con cambio real de QB (n): {changed.height}")
    r, p = pearsonr(changed["qb_delta"].to_numpy(), changed["win_change"].to_numpy())
    print(f"Correlacion qb_delta vs win_change: r={r:.3f}, p-value={p:.4f}")
    print(f"{'SIGNIFICATIVO' if p < 0.05 else 'NO SIGNIFICATIVO -- podria ser ruido'}")

    print(f"\nDistribucion completa (todos los {changed.height} casos, ordenados por qb_delta):")
    with pl.Config(tbl_rows=changed.height):
        print(changed.select(["season", "team", "qb_delta", "win_change"]).sort("qb_delta", descending=True))