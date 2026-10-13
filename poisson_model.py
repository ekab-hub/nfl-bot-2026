"""
Modelo Poisson de anotacion: entrena con sklearn.PoissonRegressor sobre las features
de poisson_features.py, y genera:
  - lambda esperado de puntos por equipo
  - probabilidad de victoria/empate via distribucion de Skellam (diferencia de 2 Poisson)
  - tabla de marcadores exactos mas probables
"""
import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import skellam, poisson
from poisson_features import build_game_features

# Validado en experimentos/REPORTE.md (walk-forward 2010-2022 + holdout 2023-2025).
# qb_value_live reemplaza a team_qb_value (lo contiene como prior).
FEATURES = ["team_rush_epa", "team_pass_epa", "team_success", "qb_value_live", "backup_qb",
            "opp_def_epa_allowed", "is_home", "div_game"]
TARGET = "points_scored"


def train_poisson_model(train_seasons, alpha=1e-4, pbp_history_start=2001):
    """
    Features estandarizadas + penalizacion casi nula. Antes: features crudas (escala ~0.1)
    con alpha=1.0, que encogia los coeficientes de EPA ~6x hacia cero.
    """
    df = build_game_features(seasons_to_build=train_seasons, pbp_history_start=pbp_history_start)
    df = df.filter(pl.col("game_id").str.contains(r"_(0[1-9]|1[0-8])_"))  # solo temporada regular
    X = df.select(FEATURES).to_numpy()
    y = df[TARGET].to_numpy()

    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000))
    model.fit(X, y)
    return model, df


def predict_lambda(model, features: dict):
    """features: dict (o fila nombrada) con todas las columnas de FEATURES."""
    X = np.array([[features[f] for f in FEATURES]])
    return model.predict(X)[0]


def game_win_probs(lambda_home, lambda_away):
    """
    Diferencia de dos Poisson independientes = distribucion de Skellam.
    Nos da P(gana local), P(empate), P(gana visitante) en forma cerrada,
    sin necesidad de simular todavia (el Monte Carlo lo dejamos para la
    simulacion de TEMPORADA completa, no para un partido individual).
    """
    prob_tie = skellam.pmf(0, lambda_home, lambda_away)
    prob_home_win = skellam.sf(0, lambda_home, lambda_away)   # P(diferencia > 0)
    prob_away_win = skellam.cdf(-1, lambda_home, lambda_away)  # P(diferencia < 0)
    return {"home_win": prob_home_win, "tie": prob_tie, "away_win": prob_away_win}


def most_likely_scores(lambda_home, lambda_away, top_n=10, max_score=60):
    """Tabla de marcadores exactos mas probables (grid de Poisson conjunto)."""
    scores = []
    for h in range(max_score):
        for a in range(max_score):
            p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            scores.append((h, a, p))
    scores.sort(key=lambda x: -x[2])
    return scores[:top_n]


def backtest_poisson(model, test_seasons):
    """
    Evalua el modelo en temporadas NO usadas para entrenar (held-out real).
    Compara contra el mismo criterio del backtest de Elo (accuracy, logloss, brier)
    para poder comparar manzanas con manzanas.
    """
    df = build_game_features(seasons_to_build=test_seasons)

    results = []
    for game_id in df["game_id"].unique().to_list():
        game_rows = df.filter(pl.col("game_id") == game_id)
        home_row = game_rows.filter(pl.col("is_home") == 1)
        away_row = game_rows.filter(pl.col("is_home") == 0)
        if home_row.height == 0 or away_row.height == 0:
            continue

        lam_home = predict_lambda(model, home_row.row(0, named=True))
        lam_away = predict_lambda(model, away_row.row(0, named=True))

        probs = game_win_probs(lam_home, lam_away)
        actual_home_win = 1 if home_row["points_scored"][0] > away_row["points_scored"][0] else 0

        results.append({
            "game_id": game_id, "lambda_home": lam_home, "lambda_away": lam_away,
            "model_prob_home": probs["home_win"], "actual_home_win": actual_home_win
        })

    res_df = pl.DataFrame(results)
    y_true = res_df["actual_home_win"].to_numpy()
    y_pred = res_df["model_prob_home"].to_numpy()

    eps = 1e-10
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    acc = np.mean((y_pred > 0.5).astype(int) == y_true)
    logloss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
    brier = np.mean((y_pred - y_true) ** 2)

    print(f"=== Backtest Poisson (temporadas {test_seasons}) ===")
    print(f"Partidos evaluados: {len(res_df)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Log-loss: {logloss:.4f}")
    print(f"Brier score: {brier:.4f}")
    return res_df


if __name__ == "__main__":
    # Entrena con 2019-2023, evalua en 2024 (temporada que el modelo NUNCA vio)
    model, train_df = train_poisson_model(train_seasons=list(range(2002, 2024)))

    reg = model[-1]
    print("Coeficientes (por 1 desviacion estandar de cada feature, en log-lambda):")
    for f, coef in zip(FEATURES, reg.coef_):
        print(f"  {f}: {coef:.4f}")
    print(f"  intercept: {reg.intercept_:.4f}\n")

    res_df = backtest_poisson(model, test_seasons=[2024])

    # Ejemplo ilustrativo: KC vs DET semana 1 de 2023 (dato real que ya vimos en la tabla de features)
    example_probs = game_win_probs(lambda_home=24.5, lambda_away=21.0)
    print(f"\nEjemplo (lambda_home=24.5, lambda_away=21.0): {example_probs}")
    print("Top 5 marcadores mas probables:", most_likely_scores(24.5, 21.0, top_n=5))