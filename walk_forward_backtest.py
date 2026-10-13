"""
Backtest walk-forward: para cada temporada de prueba, el Poisson se entrena
SOLO con datos anteriores a esa temporada (nunca ve el futuro), y se acumulan
los resultados de varias temporadas de prueba para tener una muestra mas grande
y confiable que una sola temporada suelta.
"""
import numpy as np
import polars as pl
from motor_elo_v1 import run_elo_over_seasons
from poisson_model import train_poisson_model, predict_lambda, game_win_probs
from poisson_features import build_game_features
from blend_model import poisson_weight_by_week

TEST_SEASONS = [2021, 2022, 2023, 2024]
PBP_HISTORY_START = 2015  # historial base para valores de QB y EPA
ELO_HISTORY_SEASONS = range(2015, max(TEST_SEASONS) + 1)


def get_elo_predictions_all():
    preds, _ = run_elo_over_seasons(list(ELO_HISTORY_SEASONS))
    return preds.select(
        ["season", "week", "home_team", "away_team", "home_score", "away_score", "model_prob_home"]
    ).rename({"model_prob_home": "elo_prob_home"})


def get_poisson_predictions_for_season(test_season):
    # Arranca un año despues de PBP_HISTORY_START: ese primer año solo sirve
    # como "temporada anterior" para el blend de EPA, no se puede construir
    # features completos para el mismo.
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


def evaluate_metrics(y_true, y_pred, label):
    eps = 1e-10
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    acc = np.mean((y_pred > 0.5).astype(int) == y_true)
    logloss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
    brier = np.mean((y_pred - y_true) ** 2)
    print(f"{label}: n={len(y_true)}, acc={acc:.3f}, logloss={logloss:.4f}, brier={brier:.4f}")


if __name__ == "__main__":
    elo_df = get_elo_predictions_all()

    all_poisson = []
    for season in TEST_SEASONS:
        print(f"Entrenando y prediciendo temporada {season}...")
        all_poisson.append(get_poisson_predictions_for_season(season))
    poisson_df = pl.concat(all_poisson)

    merged = elo_df.join(poisson_df, on=["season", "week", "home_team", "away_team"], how="inner")
    print(f"\nTotal partidos cruzados (todas las temporadas de prueba): {merged.height}\n")

    merged.write_parquet("elo_poisson_predictions_2021_2024.parquet")
    print(f"Guardado en cache: elo_poisson_predictions_2021_2024.parquet\n")

    merged = merged.with_columns(
        pl.struct(["week"]).map_elements(
            lambda x: poisson_weight_by_week(x["week"]), return_dtype=pl.Float64
        ).alias("poisson_weight")
    )
    merged = merged.with_columns([
        (pl.col("poisson_weight") * pl.col("poisson_prob_home") +
         (1 - pl.col("poisson_weight")) * pl.col("elo_prob_home")).alias("blend_prob_home"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("actual_home_win"),
    ])

    y_true_all = merged["actual_home_win"].to_numpy()

    print("=== Todas las temporadas de prueba juntas (2021-2024) ===")
    evaluate_metrics(y_true_all, merged["elo_prob_home"].to_numpy(), "Elo puro    ")
    evaluate_metrics(y_true_all, merged["poisson_prob_home"].to_numpy(), "Poisson puro")
    evaluate_metrics(y_true_all, merged["blend_prob_home"].to_numpy(), "Blend       ")

    print("\n=== Solo semanas 1-4 (todas las temporadas de prueba) ===")
    early = merged.filter(pl.col("week") <= 4)
    y_true_early = early["actual_home_win"].to_numpy()
    evaluate_metrics(y_true_early, early["elo_prob_home"].to_numpy(), "Elo puro    ")
    evaluate_metrics(y_true_early, early["poisson_prob_home"].to_numpy(), "Poisson puro")
    evaluate_metrics(y_true_early, early["blend_prob_home"].to_numpy(), "Blend       ")

    print("\n=== Semanas 5+ (todas las temporadas de prueba) ===")
    late = merged.filter(pl.col("week") > 4)
    y_true_late = late["actual_home_win"].to_numpy()
    evaluate_metrics(y_true_late, late["elo_prob_home"].to_numpy(), "Elo puro    ")
    evaluate_metrics(y_true_late, late["poisson_prob_home"].to_numpy(), "Poisson puro")
    evaluate_metrics(y_true_late, late["blend_prob_home"].to_numpy(), "Blend       ")

    print("\n=== Por temporada individual (accuracy del blend, para ver consistencia) ===")
    for season in TEST_SEASONS:
        season_data = merged.filter(pl.col("season") == season)
        evaluate_metrics(season_data["actual_home_win"].to_numpy(),
                          season_data["blend_prob_home"].to_numpy(), f"Blend {season}")