"""
Prueba si 'luck' (y one_score_win_pct) aportan poder predictivo MAS ALLA
de lo que 'wins' ya captura por si solo -- para no confundir una señal
redundante con una señal nueva antes de meterla al modelo.
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from underlying_stats import build_team_season_stats

if __name__ == "__main__":
    stats = build_team_season_stats(range(2015, 2026))

    current = stats.select([
        "season", "team", "wins", "point_diff", "luck", "one_score_win_pct", "turnovers_given"
    ])
    next_season = stats.select(["season", "team", "wins"]).rename(
        {"season": "season_next", "wins": "wins_next"}
    ).with_columns((pl.col("season_next") - 1).alias("season"))
    merged = current.join(next_season, on=["season", "team"], how="inner")
    merged = merged.with_columns((pl.col("wins_next") - pl.col("wins")).alias("win_change"))
    merged = merged.filter(pl.col("one_score_win_pct").is_not_null())

    y = merged["win_change"].to_numpy()

    print("=== Correlacion simple entre 'wins' y 'luck' (para ver que tan redundantes son) ===")
    r = np.corrcoef(merged["wins"].to_numpy(), merged["luck"].to_numpy())[0, 1]
    print(f"corr(wins, luck) = {r:.3f}\n")

    print("=== Modelo 1: SOLO wins prediciendo win_change ===")
    X1 = merged.select(["wins"]).to_numpy()
    m1 = LinearRegression().fit(X1, y)
    pred1 = m1.predict(X1)
    r2_1 = 1 - np.sum((y - pred1)**2) / np.sum((y - y.mean())**2)
    print(f"R^2 = {r2_1:.4f}, coef wins = {m1.coef_[0]:.4f}\n")

    print("=== Modelo 2: wins + luck prediciendo win_change ===")
    X2 = merged.select(["wins", "luck"]).to_numpy()
    m2 = LinearRegression().fit(X2, y)
    pred2 = m2.predict(X2)
    r2_2 = 1 - np.sum((y - pred2)**2) / np.sum((y - y.mean())**2)
    print(f"R^2 = {r2_2:.4f}, coef wins = {m2.coef_[0]:.4f}, coef luck = {m2.coef_[1]:.4f}")
    print(f"Mejora de R^2 al agregar luck: {r2_2 - r2_1:.4f}\n")

    print("=== Modelo 3: wins + luck + one_score_win_pct + turnovers_given ===")
    X3 = merged.select(["wins", "luck", "one_score_win_pct", "turnovers_given"]).to_numpy()
    m3 = LinearRegression().fit(X3, y)
    pred3 = m3.predict(X3)
    r2_3 = 1 - np.sum((y - pred3)**2) / np.sum((y - y.mean())**2)
    print(f"R^2 = {r2_3:.4f}")
    for name, coef in zip(["wins", "luck", "one_score_win_pct", "turnovers_given"], m3.coef_):
        print(f"  coef {name} = {coef:.4f}")
    print(f"Mejora de R^2 vs solo wins: {r2_3 - r2_1:.4f}")