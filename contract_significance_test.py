"""
Prueba de significancia para las 3 variables de contract_features.py,
mismo estandar que luck/qb_delta/injury_burden: correlacion + p-value
contra el cambio de wins del año siguiente, y aporte marginal sobre
el baseline ya validado (wins + luck).
"""
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from underlying_stats import build_team_season_stats
from contract_features import (build_mega_contract_count, build_offseason_spending,
                                 build_starting_lineup_draft_capital)

SEASONS = range(2018, 2026)

TEAM_NAME_MAP = {  # contracts usa nombres largos, el resto del proyecto usa abreviaciones
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
    "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
}


def normalize_contract_team(name):
    return TEAM_NAME_MAP.get(name, name)


if __name__ == "__main__":
    stats = build_team_season_stats(range(2015, 2026))

    wins_by_season = stats.select(["season", "team", "wins"])
    prev_wins = wins_by_season.rename({"season": "season_prev", "wins": "wins_prev"}) \
        .with_columns((pl.col("season_prev") + 1).alias("season"))
    win_change = wins_by_season.join(prev_wins, on=["season", "team"], how="inner") \
        .with_columns((pl.col("wins") - pl.col("wins_prev")).alias("win_change"))

    luck_prev = stats.select(["season", "team", "luck"]).rename(
        {"season": "season_prev2", "luck": "luck_prev"}
    ).with_columns((pl.col("season_prev2") + 1).alias("season"))

    mega = build_mega_contract_count(SEASONS).with_columns(
        pl.col("team").map_elements(normalize_contract_team, return_dtype=pl.Utf8).alias("team")
    ).rename({"season": "season_prev3", "mega_contract_count": "mega_prev"}) \
     .with_columns((pl.col("season_prev3") + 1).alias("season"))

    spending = build_offseason_spending(SEASONS).with_columns(
        pl.col("team").map_elements(normalize_contract_team, return_dtype=pl.Utf8).alias("team")
    ).rename({"season": "season_prev4", "offseason_spending": "spending_prev"}) \
     .with_columns((pl.col("season_prev4") + 1).alias("season"))

    capital = build_starting_lineup_draft_capital(SEASONS).rename(
        {"season": "season_prev5", "starting_lineup_draft_capital": "capital_prev"}
    ).with_columns((pl.col("season_prev5") + 1).alias("season"))

    full = win_change.join(luck_prev, on=["season", "team"], how="inner")
    full = full.join(mega, on=["season", "team"], how="inner")
    full = full.join(spending, on=["season", "team"], how="inner")
    full = full.join(capital, on=["season", "team"], how="inner")

    print(f"Total filas para la prueba: {full.height}\n")

    y = full["win_change"].to_numpy()

    print("=== Correlacion simple de cada variable nueva vs win_change ===")
    for var in ["mega_prev", "spending_prev", "capital_prev"]:
        r, p = pearsonr(full[var].to_numpy(), y)
        sig = "SIGNIFICATIVO" if p < 0.05 else "no significativo"
        print(f"{var:<16} r={r:.3f}  p={p:.4f}  {sig}")

    print("\n=== Baseline: wins_prev + luck_prev (ya validado) ===")
    X1 = full.select(["wins_prev", "luck_prev"]).to_numpy()
    m1 = LinearRegression().fit(X1, y)
    r2_1 = 1 - np.sum((y - m1.predict(X1))**2) / np.sum((y - y.mean())**2)
    print(f"R^2 = {r2_1:.4f}")

    print("\n=== Baseline + las 3 variables de contratos juntas ===")
    X2 = full.select(["wins_prev", "luck_prev", "mega_prev", "spending_prev", "capital_prev"]).to_numpy()
    m2 = LinearRegression().fit(X2, y)
    r2_2 = 1 - np.sum((y - m2.predict(X2))**2) / np.sum((y - y.mean())**2)
    print(f"R^2 = {r2_2:.4f}")
    for name, coef in zip(["wins_prev", "luck_prev", "mega_prev", "spending_prev", "capital_prev"], m2.coef_):
        print(f"  coef {name} = {coef:.4f}")
    print(f"Mejora de R^2 sobre el baseline: {r2_2 - r2_1:.4f}")
    print("\n=== Verificacion: que tan redundante es capital_prev con wins_prev? ===")
    r_redundancy, p_redundancy = pearsonr(full["capital_prev"].to_numpy(), full["wins_prev"].to_numpy())
    print(f"corr(capital_prev, wins_prev) = {r_redundancy:.3f}  (p={p_redundancy:.4f})")