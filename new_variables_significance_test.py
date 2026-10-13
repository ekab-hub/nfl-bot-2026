"""
Prueba de significancia para las 4 variables nuevas (rest_diff, sos_diff,
tz_diff/early_game, new_coach_diff), contra el residual del Elo -- mismo
estandar que momentum: buscamos señal MAS ALLA de lo que el Elo ya sabe.
"""
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    df = pl.read_parquet("new_variables_features.parquet")
    df = df.with_columns([
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_won"),
    ])
    df = df.with_columns(
        (pl.col("home_won") - pl.col("elo_prob_home")).alias("elo_residual")
    )

    print(f"Total partidos: {df.height}\n")

    variables = ["rest_diff", "sos_diff", "tz_diff", "early_game", "away_travels_west_early", "new_coach_diff"]

    print("=== Correlacion de cada variable vs. residual del Elo ===")
    for var in variables:
        r, p = pearsonr(df[var].to_numpy(), df["elo_residual"].to_numpy())
        sig = "SIGNIFICATIVO" if p < 0.05 else "no significativo"
        print(f"{var:<26} r={r:+.4f}  p={p:.4f}  {sig}")

    y = df["home_won"].to_numpy()

    print("\n=== Baseline: solo Elo ===")
    X1 = df.select(["elo_prob_home"]).to_numpy()
    m1 = LogisticRegression().fit(X1, y)
    p1 = m1.predict_proba(X1)[:, 1]
    logloss1 = -np.mean(y * np.log(np.clip(p1, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - p1, 1e-10, 1)))
    print(f"logloss = {logloss1:.4f}")

    print("\n=== Elo + las 4 variables juntas ===")
    all_vars = ["elo_prob_home"] + variables
    X2 = df.select(all_vars).to_numpy()
    m2 = LogisticRegression().fit(X2, y)
    p2 = m2.predict_proba(X2)[:, 1]
    logloss2 = -np.mean(y * np.log(np.clip(p2, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - p2, 1e-10, 1)))
    print(f"logloss = {logloss2:.4f}")
    for name, coef in zip(all_vars, m2.coef_[0]):
        print(f"  coef {name} = {coef:+.4f}")
    print(f"\nMejora de logloss sobre el baseline: {logloss1 - logloss2:.5f} (positivo = mejora)")
    df = df.with_columns([
        (pl.col("rest_diff") <= -3).cast(pl.Int8).alias("home_short_week"),  # local jugo con semana corta
        (pl.col("rest_diff") >= 3).cast(pl.Int8).alias("home_bye_advantage"),  # local viene de bye
    ])
    for var in ["home_short_week", "home_bye_advantage"]:
        r, p = pearsonr(df[var].to_numpy(), df["elo_residual"].to_numpy())
        sig = "SIGNIFICATIVO" if p < 0.05 else "no significativo"
        print(f"{var:<20} r={r:+.4f}  p={p:.4f}  {sig}")