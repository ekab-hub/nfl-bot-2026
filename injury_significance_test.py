"""
Prueba si injury_burden (temporada N) predice win_change (N a N+1) MAS ALLA
de wins y luck ya conocidos -- mismo estandar aplicado a qb_delta, que no
paso la prueba. La hipotesis: equipos con mucha carga de lesiones deberian
'rebotar' al año siguiente si la salud se normaliza (regression to mean
de lesiones, no solo de resultado).
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from underlying_stats import build_team_season_stats
from injury_stats import build_injury_burden

if __name__ == "__main__":
    stats = build_team_season_stats(range(2015, 2026))
    burden = build_injury_burden(range(2018, 2026))  # load_injuries confiable desde 2018 en adelante

    wins_by_season = stats.select(["season", "team", "wins"])
    prev_wins = wins_by_season.rename({"season": "season_prev", "wins": "wins_prev"}) \
        .with_columns((pl.col("season_prev") + 1).alias("season"))
    merged_wins = wins_by_season.join(prev_wins, on=["season", "team"], how="inner") \
        .with_columns((pl.col("wins") - pl.col("wins_prev")).alias("win_change"))

    luck_prev = stats.select(["season", "team", "luck"]).rename(
        {"season": "season_prev2", "luck": "luck_prev"}
    ).with_columns((pl.col("season_prev2") + 1).alias("season"))

    burden_prev = burden.rename(
        {"season": "season_prev3", "injury_burden": "injury_burden_prev"}
    ).with_columns((pl.col("season_prev3") + 1).alias("season"))

    full = merged_wins.join(luck_prev, on=["season", "team"], how="inner")
    full = full.join(burden_prev, on=["season", "team"], how="inner")

    print(f"Total filas: {full.height}\n")

    print("=== Correlacion simple: injury_burden_prev vs win_change ===")
    r, p = pearsonr(full["injury_burden_prev"].to_numpy(), full["win_change"].to_numpy())
    print(f"r={r:.3f}, p-value={p:.4f} -- {'SIGNIFICATIVO' if p < 0.05 else 'NO SIGNIFICATIVO'}\n")

    print("=== Modelo: wins_prev + luck_prev (baseline ya validado) ===")
    X1 = full.select(["wins_prev", "luck_prev"]).to_numpy()
    y = full["win_change"].to_numpy()
    m1 = LinearRegression().fit(X1, y)
    pred1 = m1.predict(X1)
    r2_1 = 1 - np.sum((y - pred1)**2) / np.sum((y - y.mean())**2)
    print(f"R^2 = {r2_1:.4f}\n")

    print("=== Modelo: wins_prev + luck_prev + injury_burden_prev ===")
    X2 = full.select(["wins_prev", "luck_prev", "injury_burden_prev"]).to_numpy()
    m2 = LinearRegression().fit(X2, y)
    pred2 = m2.predict(X2)
    r2_2 = 1 - np.sum((y - pred2)**2) / np.sum((y - y.mean())**2)
    print(f"R^2 = {r2_2:.4f}")
    print(f"Coef injury_burden_prev = {m2.coef_[2]:.4f}")
    print(f"Mejora de R^2 al agregar injury_burden: {r2_2 - r2_1:.4f}")