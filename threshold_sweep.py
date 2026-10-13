"""
Efectividad del bot en partidos individuales segun nivel de confianza,
sobre el maximo historico disponible (2005-2025, mismo criterio de arranque
que top_n_backtest.py: 2002 como base + calentamiento).
Usa el blend Elo+Poisson (25% Poisson, validado) en modo walk-forward real:
el modelo de cada temporada de prueba se entrena SOLO con datos anteriores.
"""
import numpy as np
import polars as pl
from motor_elo_v1 import run_elo_over_seasons
from poisson_model import train_poisson_model, predict_lambda, game_win_probs
from poisson_features import build_game_features

TEST_SEASONS = range(2005, 2026)
PBP_HISTORY_START = 2002
ELO_HISTORY_START = 2002
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BLEND_WEIGHT = 0.25  # validado en weight_search.py


def get_elo_predictions_all():
    preds, _ = run_elo_over_seasons(range(ELO_HISTORY_START, max(TEST_SEASONS) + 1), home_advantage=35)
    return preds.select(
        ["season", "week", "home_team", "away_team", "home_score", "away_score", "model_prob_home"]
    ).rename({"model_prob_home": "elo_prob_home"})


def get_poisson_predictions_for_season(test_season):
    train_seasons = list(range(PBP_HISTORY_START + 1, test_season))
    model, _ = train_poisson_model(train_seasons=train_seasons, pbp_history_start=PBP_HISTORY_START)
    df = build_game_features(seasons_to_build=[test_season], pbp_history_start=PBP_HISTORY_START)

    rows = []
    for game_id in df["game_id"].unique().to_list():
        g = df.filter(pl.col("game_id") == game_id)
        home_row = g.filter(pl.col("is_home") == 1)
        away_row = g.filter(pl.col("is_home") == 0)
        if home_row.height == 0 or away_row.height == 0:
            continue

        lam_home = predict_lambda(model, home_row["team_rush_epa"][0], home_row["team_qb_value"][0],
                                   home_row["opp_def_epa_allowed"][0], 1, home_row["div_game"][0])
        lam_away = predict_lambda(model, away_row["team_rush_epa"][0], away_row["team_qb_value"][0],
                                   away_row["opp_def_epa_allowed"][0], 0, away_row["div_game"][0])
        probs = game_win_probs(lam_home, lam_away)

        rows.append({
            "season": home_row["season"][0], "week": home_row["week"][0],
            "home_team": home_row["team"][0], "away_team": away_row["team"][0],
            "poisson_prob_home": probs["home_win"],
        })
    return pl.DataFrame(rows)


def build_full_blend_dataset():
    elo_df = get_elo_predictions_all()

    all_poisson = []
    for season in TEST_SEASONS:
        print(f"Entrenando Poisson y prediciendo temporada {season}...")
        all_poisson.append(get_poisson_predictions_for_season(season))
    poisson_df = pl.concat(all_poisson)

    merged = elo_df.join(poisson_df, on=["season", "week", "home_team", "away_team"], how="inner")
    merged = merged.with_columns([
        (BLEND_WEIGHT * pl.col("poisson_prob_home") + (1 - BLEND_WEIGHT) * pl.col("elo_prob_home"))
            .alias("blend_prob_home"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_won"),
        (pl.col("home_score") == pl.col("away_score")).alias("is_tie"),
    ])
    return merged.filter(~pl.col("is_tie"))


def evaluate_thresholds(df, thresholds):
    results = []
    for t in thresholds:
        favorite_is_home = df["blend_prob_home"] >= 0.5
        favorite_prob = pl.when(favorite_is_home).then(pl.col("blend_prob_home")) \
                          .otherwise(1 - pl.col("blend_prob_home"))
        sub = df.with_columns(favorite_prob.alias("favorite_prob"),
                               favorite_is_home.alias("favorite_is_home"))
        sub = sub.filter(pl.col("favorite_prob") >= t)

        if sub.height == 0:
            results.append({"umbral": t, "n_partidos": 0, "accuracy": None})
            continue

        correct = (sub["favorite_is_home"] == sub["home_won"].cast(pl.Boolean))
        acc = correct.to_numpy().mean()
        results.append({"umbral": t, "n_partidos": sub.height, "accuracy": round(float(acc), 4)})

    return pl.DataFrame(results)


if __name__ == "__main__":
    df = build_full_blend_dataset()
    df.write_parquet("threshold_sweep_predictions.parquet")  # cache por si tarda mucho
    print(f"\nTotal de partidos evaluados ({min(TEST_SEASONS)}-{max(TEST_SEASONS)}): {df.height}\n")

    results = evaluate_thresholds(df, THRESHOLDS)
    print("=" * 55)
    print("EFECTIVIDAD POR NIVEL DE CONFIANZA DEL BOT")
    print("=" * 55)
    with pl.Config(tbl_rows=10):
        print(results)

    print("\nNota: a umbrales mas altos, quedan menos partidos (n_partidos)")
    print("-- si n es chico, el accuracy de ese umbral es menos confiable.")