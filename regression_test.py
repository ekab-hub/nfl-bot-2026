"""
Prueba empirica: la 'suerte' (record real vs. lo que el diferencial de puntos
sugiere), el record en partidos cerrados, y los turnovers de la temporada N,
predicen el CAMBIO de wins en la temporada N+1? Se prueba con TODAS las
temporadas disponibles, no un caso aislado.
"""
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from underlying_stats import build_team_season_stats

if __name__ == "__main__":
    stats = build_team_season_stats(range(2015, 2026))

    # Cruzar temporada N con N+1 (mismo equipo)
    current = stats.select([
        "season", "team", "wins", "point_diff", "luck",
        "one_score_win_pct", "turnovers_given", "pyth_win_pct"
    ])
    next_season = stats.select(["season", "team", "wins"]).rename(
        {"season": "season_next", "wins": "wins_next"}
    ).with_columns((pl.col("season_next") - 1).alias("season"))

    merged = current.join(next_season, on=["season", "team"], how="inner")
    merged = merged.with_columns((pl.col("wins_next") - pl.col("wins")).alias("win_change"))

    print(f"Total pares temporada N -> N+1: {merged.height}\n")

    variables = ["luck", "one_score_win_pct", "turnovers_given", "point_diff", "wins"]
    print(f"{'Variable':<22}{'Correlacion c/ win_change':<28}{'p-value':<12}{'Interpretacion'}")
    for var in variables:
        sub = merged.filter(pl.col(var).is_not_null())
        r, p = pearsonr(sub[var].to_numpy(), sub["win_change"].to_numpy())
        sig = "SIGNIFICATIVO" if p < 0.05 else "no significativo"
        print(f"{var:<22}{r:<28.3f}{p:<12.4f}{sig}")

    print("\n--- Ejemplo concreto: equipos con peor 'luck' en 2024, su win_change real a 2025 ---")
    example = merged.filter(pl.col("season") == 2024).sort("luck").head(8)
    print(example.select(["team", "wins", "wins_next", "win_change", "luck"]))