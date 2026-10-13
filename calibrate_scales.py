"""
Calibra QB_ELO_SCALE y WINS_TO_ELO_SCALE con evidencia real, en vez de los
valores puestos a ojo. Unifica el ajuste de QB y el de luck en la MISMA
regresion (win_change ~ luck + qb_delta), para que ambos efectos se midan
en las mismas unidades (wins) antes de convertir a Elo con un solo factor.
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from underlying_stats import build_team_season_stats
from qb_adjustment import load_pbp_clean, qb_career_value, primary_starter_per_team_season

def build_qb_delta_dataset(seasons_to_check, history_start=2015):
    """
    Para cada temporada N en seasons_to_check donde el titular cambio respecto
    a N-1, calcula el delta de valor QB (conocido ANTES de que empiece N,
    sin leakage -- mismo criterio que usa el simulador real).
    """
    pbp = load_pbp_clean(range(history_start, max(seasons_to_check) + 1))
    qb_values = qb_career_value(pbp)
    starters_hist = primary_starter_per_team_season(pbp)

    rows = []
    for season in seasons_to_check:
        for team in starters_hist.filter(pl.col("season") == season)["posteam"].unique().to_list():
            new_row = starters_hist.filter((pl.col("season") == season) & (pl.col("posteam") == team))
            prev_row = starters_hist.filter((pl.col("season") == season - 1) & (pl.col("posteam") == team))
            if new_row.height == 0 or prev_row.height == 0:
                continue
            new_qb, prev_qb = new_row["passer_player_id"][0], prev_row["passer_player_id"][0]
            if new_qb == prev_qb:
                continue

            def get_val(qb_id):
                h = qb_values.filter((pl.col("qb_id") == qb_id) & (pl.col("season") < season))
                return h.sort("season", descending=True)["qb_value_epa_dropback"][0] if h.height > 0 else None

            new_val, prev_val = get_val(new_qb), get_val(prev_qb)
            if new_val is None or prev_val is None:
                continue
            rows.append({"season": season, "team": team, "qb_delta": new_val - prev_val})

    return pl.DataFrame(rows)


if __name__ == "__main__":
    seasons_range = range(2016, 2026)
    stats = build_team_season_stats(range(2015, 2026))
    qb_deltas = build_qb_delta_dataset(seasons_range)

    # win_change: de wins(N-1) a wins(N) -- lo que el ajuste preseason busca predecir
    wins_by_season = stats.select(["season", "team", "wins"])
    prev_wins = wins_by_season.rename({"season": "season_prev", "wins": "wins_prev"}) \
        .with_columns((pl.col("season_prev") + 1).alias("season"))
    merged_wins = wins_by_season.join(prev_wins, on=["season", "team"], how="inner") \
        .with_columns((pl.col("wins") - pl.col("wins_prev")).alias("win_change"))

    # luck de la temporada PREVIA (N-1), disponible antes de que arranque N
    luck_prev = stats.select(["season", "team", "luck"]).rename(
        {"season": "season_prev2", "luck": "luck_prev"}
    ).with_columns((pl.col("season_prev2") + 1).alias("season"))

    full = merged_wins.join(luck_prev, left_on=["season", "team"], right_on=["season", "team"], how="inner")
    full = full.join(qb_deltas, on=["season", "team"], how="left").with_columns(
        pl.col("qb_delta").fill_null(0.0)  # sin cambio de QB -> delta 0, no penaliza ni beneficia
    )

    print(f"Total filas para calibracion: {full.height}\n")

    X = full.select(["luck_prev", "qb_delta"]).to_numpy()
    y = full["win_change"].to_numpy()
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)

    print(f"R^2 = {r2:.4f}")
    print(f"Coeficiente luck_prev (equivalente a LUCK_COEF): {model.coef_[0]:.4f}")
    print(f"Coeficiente qb_delta  (nuevo, calibrado con evidencia): {model.coef_[1]:.4f}")
    print(f"Intercepto: {model.intercept_:.4f}")

    # Solo entre equipos que SI cambiaron de QB, para ver el efecto aislado con mas claridad
    changed = full.filter(pl.col("qb_delta") != 0.0)
    print(f"\nCasos con cambio real de QB en la muestra: {changed.height}")
    print(changed.select(["season", "team", "qb_delta", "win_change"]).sort("qb_delta", descending=True).head(8))