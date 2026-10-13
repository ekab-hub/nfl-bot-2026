"""
4 variables nuevas, todas a nivel de PARTIDO (como momentum):
  1. rest_diff: diferencia de dias de descanso (home - away) -- ya viene en nflreadpy
  2. sos_diff: fuerza de calendario reciente (rating promedio de rivales enfrentados esta
     temporada, home vs away) -- calculado con el propio Elo del proyecto
  3. timezone/viaje: diferencia de huso horario + si es partido temprano
  4. new_coach: si el equipo cambio de head coach respecto a la temporada anterior --
     ya viene en nflreadpy (home_coach/away_coach)

NOTA sobre huso horario: uso offset ESTANDAR (sin ajustar por DST especifico de
cada semana) por simplicidad -- aproximacion razonable para un analisis
exploratorio, no exacta al minuto.
"""
import nflreadpy as nfl
import polars as pl
from motor_elo_v1 import EloEngine, load_games, normalize_team

TEAM_UTC_OFFSET = {
    "ARI": -7, "ATL": -5, "BAL": -5, "BUF": -5, "CAR": -5, "CHI": -6, "CIN": -5,
    "CLE": -5, "DAL": -6, "DEN": -7, "DET": -5, "GB": -6, "HOU": -6, "IND": -5,
    "JAX": -5, "KC": -6, "LV": -8, "LAC": -8, "LA": -8, "MIA": -5, "MIN": -6,
    "NE": -5, "NO": -6, "NYG": -5, "NYJ": -5, "PHI": -5, "PIT": -5, "SEA": -8,
    "SF": -8, "TB": -5, "TEN": -6, "WAS": -5,
}


def build_coach_by_team_season():
    """Coach de cada equipo en la semana 1 de cada temporada (para detectar cambios año a año)."""
    seasons_range = range(2015, 2026)
    df = nfl.load_schedules(seasons=list(seasons_range))
    df = df.filter((pl.col("game_type") == "REG") & (pl.col("week") == 1))
    df = df.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])

    coach_map = {}
    for row in df.iter_rows(named=True):
        coach_map[(row["season"], row["home_team"])] = row["home_coach"]
        coach_map[(row["season"], row["away_team"])] = row["away_coach"]
    return coach_map


def build_features(seasons, home_advantage=35, k=20, regression_to_mean=0.40):
    games = load_games(seasons)
    engine = EloEngine(k=k, home_advantage=home_advantage, regression_to_mean=regression_to_mean)
    coach_map = build_coach_by_team_season()

    opponent_history = {}  # team -> lista de ratings de rivales enfrentados esta temporada
    rows = []
    current_season = None

    for row in games.iter_rows(named=True):
        if current_season is not None and row["season"] != current_season:
            engine.new_season_regression()
            opponent_history = {}
        current_season = row["season"]

        home, away = row["home_team"], row["away_team"]
        r_home = engine.get_rating(home)
        r_away = engine.get_rating(away)

        home_opp_hist = opponent_history.get(home, [])
        away_opp_hist = opponent_history.get(away, [])
        sos_home = sum(home_opp_hist) / len(home_opp_hist) if home_opp_hist else 1500.0
        sos_away = sum(away_opp_hist) / len(away_opp_hist) if away_opp_hist else 1500.0

        elo_prob_home = engine.expected_win_prob(r_home + engine.home_advantage, r_away)

        tz_diff = TEAM_UTC_OFFSET.get(home, -5) - TEAM_UTC_OFFSET.get(away, -5)
        try:
            gametime_hour = int(row["gametime"].split(":")[0]) if row.get("gametime") else 13
        except Exception:
            gametime_hour = 13
        early_game = 1 if gametime_hour <= 13 else 0

        prev_coach_home = coach_map.get((current_season - 1, home))
        prev_coach_away = coach_map.get((current_season - 1, away))
        curr_coach_home = coach_map.get((current_season, home))
        curr_coach_away = coach_map.get((current_season, away))
        new_coach_home = 1 if (prev_coach_home is not None and curr_coach_home != prev_coach_home) else 0
        new_coach_away = 1 if (prev_coach_away is not None and curr_coach_away != prev_coach_away) else 0

        rows.append({
            "season": row["season"], "week": row["week"], "home_team": home, "away_team": away,
            "elo_prob_home": elo_prob_home,
            "rest_diff": (row["home_rest"] or 7) - (row["away_rest"] or 7),
            "sos_diff": sos_home - sos_away,
            "tz_diff": tz_diff, "early_game": early_game, "away_travels_west_early": tz_diff * early_game,
            "new_coach_diff": new_coach_home - new_coach_away,
            "home_score": row["home_score"], "away_score": row["away_score"],
        })

        engine.update_game(home, away, row["home_score"], row["away_score"])
        opponent_history.setdefault(home, []).append(r_away)
        opponent_history.setdefault(away, []).append(r_home)

    return pl.DataFrame(rows)


if __name__ == "__main__":
    df = build_features(range(2015, 2026))
    df = df.filter(pl.col("home_score").is_not_null())
    print(f"Total partidos: {df.height}\n")
    print(df.select(["season", "week", "home_team", "away_team", "rest_diff", "sos_diff",
                      "tz_diff", "early_game", "new_coach_diff"]).tail(10))
    df.write_parquet("new_variables_features.parquet")
    print("\nGuardado en cache: new_variables_features.parquet")