"""
Blend Elo + Poisson con peso dependiente de la semana de temporada.
Hipotesis a validar: Poisson (EPA/QB) debe pesar mas en semanas tempranas
(el Elo todavia no ha visto resultados de la temporada actual), y el Elo
debe ganar peso conforme avanza la temporada (ya "aprendio" del calendario real).
"""
import numpy as np
import polars as pl
from motor_elo_v1 import run_elo_over_seasons
from poisson_model import train_poisson_model, predict_lambda, game_win_probs
from poisson_features import build_game_features

TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
TEST_SEASON = 2024
ELO_HISTORY_SEASONS = range(2015, TEST_SEASON + 1)  # el Elo necesita historial continuo


def poisson_weight_by_week(week, start_weight=0.65, end_weight=0.30, ramp_weeks=8):
    """
    Semana 1 -> peso alto para Poisson (0.65).
    Semana 9+ -> peso bajo, piso (0.30) -- el Poisson nunca llega a 0 porque
    la señal de QB sigue siendo util incluso tarde en temporada.
    """
    week_frac = min((week - 1) / ramp_weeks, 1.0)
    return start_weight - (start_weight - end_weight) * week_frac


def get_elo_predictions(test_season):
    preds, _ = run_elo_over_seasons(list(ELO_HISTORY_SEASONS))
    return preds.filter(pl.col("season") == test_season).select(
        ["season", "week", "home_team", "away_team", "home_score", "away_score", "model_prob_home"]
    ).rename({"model_prob_home": "elo_prob_home"})


def get_poisson_predictions(test_season):
    model, _ = train_poisson_model(train_seasons=TRAIN_SEASONS)
    df = build_game_features(seasons_to_build=[test_season])

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


def evaluate_metrics(y_true, y_pred, label, week_filter_desc=""):
    eps = 1e-10
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)
    acc = np.mean((y_pred > 0.5).astype(int) == y_true)
    logloss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))
    brier = np.mean((y_pred - y_true) ** 2)
    print(f"{label} {week_filter_desc}: n={len(y_true)}, acc={acc:.3f}, logloss={logloss:.4f}, brier={brier:.4f}")
    return acc, logloss, brier


if __name__ == "__main__":
    elo_df = get_elo_predictions(TEST_SEASON)
    poisson_df = get_poisson_predictions(TEST_SEASON)

    merged = elo_df.join(poisson_df, on=["season", "week", "home_team", "away_team"], how="inner")
    print(f"Partidos con ambas predicciones cruzadas: {merged.height}\n")

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

    print("=== Temporada completa 2024 ===")
    evaluate_metrics(y_true_all, merged["elo_prob_home"].to_numpy(), "Elo puro    ")
    evaluate_metrics(y_true_all, merged["poisson_prob_home"].to_numpy(), "Poisson puro")
    evaluate_metrics(y_true_all, merged["blend_prob_home"].to_numpy(), "Blend       ")

    print("\n=== Solo semanas 1-4 (donde esperamos que Poisson ayude mas) ===")
    early = merged.filter(pl.col("week") <= 4)
    y_true_early = early["actual_home_win"].to_numpy()
    evaluate_metrics(y_true_early, early["elo_prob_home"].to_numpy(), "Elo puro    ")
    evaluate_metrics(y_true_early, early["poisson_prob_home"].to_numpy(), "Poisson puro")
    evaluate_metrics(y_true_early, early["blend_prob_home"].to_numpy(), "Blend       ")

    print("\n=== Semanas 5+ (donde esperamos que Elo ya haya alcanzado) ===")
    late = merged.filter(pl.col("week") > 4)
    y_true_late = late["actual_home_win"].to_numpy()
    evaluate_metrics(y_true_late, late["elo_prob_home"].to_numpy(), "Elo puro    ")
    evaluate_metrics(y_true_late, late["poisson_prob_home"].to_numpy(), "Poisson puro")
    evaluate_metrics(y_true_late, late["blend_prob_home"].to_numpy(), "Blend       ")