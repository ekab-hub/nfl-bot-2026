"""
Prueba de significancia para las 2 variables de momentum, contra el
RESIDUAL del Elo (resultado real menos lo que el Elo ya predijo) -- para
medir si el momentum aporta algo MAS ALLA de la fuerza de equipo que el
Elo ya captura, no la misma señal repetida.
"""
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    df = pl.read_parquet("momentum_features.parquet")
    df = df.filter(pl.col("home_score").is_not_null())
    df = df.with_columns([
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_won"),
        (pl.col("win_streak_home") - pl.col("win_streak_away")).alias("streak_diff"),
        (pl.col("momentum_home") - pl.col("momentum_away")).alias("momentum_diff"),
    ])
    df = df.with_columns(
        (pl.col("home_won") - pl.col("elo_prob_home")).alias("elo_residual")
    )

    print(f"Total partidos: {df.height}\n")

    print("=== Correlacion de cada variable vs. residual del Elo (lo que el Elo NO explica) ===")
    for var in ["streak_diff", "momentum_diff"]:
        r, p = pearsonr(df[var].to_numpy(), df["elo_residual"].to_numpy())
        sig = "SIGNIFICATIVO" if p < 0.05 else "no significativo"
        print(f"{var:<16} r={r:.4f}  p={p:.4f}  {sig}")

    y = df["home_won"].to_numpy()

    print("\n=== Baseline: solo Elo ===")
    X1 = df.select(["elo_prob_home"]).to_numpy()
    m1 = LogisticRegression().fit(X1, y)
    p1 = m1.predict_proba(X1)[:, 1]
    logloss1 = -np.mean(y * np.log(np.clip(p1, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - p1, 1e-10, 1)))
    print(f"logloss = {logloss1:.4f}")

    print("\n=== Elo + streak_diff + momentum_diff ===")
    X2 = df.select(["elo_prob_home", "streak_diff", "momentum_diff"]).to_numpy()
    m2 = LogisticRegression().fit(X2, y)
    p2 = m2.predict_proba(X2)[:, 1]
    logloss2 = -np.mean(y * np.log(np.clip(p2, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - p2, 1e-10, 1)))
    print(f"logloss = {logloss2:.4f}")
    print(f"coef elo_prob_home = {m2.coef_[0][0]:.4f}")
    print(f"coef streak_diff   = {m2.coef_[0][1]:.4f}")
    print(f"coef momentum_diff = {m2.coef_[0][2]:.4f}")
    print(f"\nMejora de logloss sobre el baseline: {logloss1 - logloss2:.5f} (positivo = mejora)")