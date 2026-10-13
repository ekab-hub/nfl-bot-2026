"""
Construye metricas 'subyacentes' por equipo-temporada, mas alla del record W-L,
para probar empiricamente (no asumir) si predicen el cambio de desempeño de
la SIGUIENTE temporada. Esto es el concepto de 'luck-adjusted performance':
un equipo puede tener mal record pero buenas metricas de fondo, lo cual en
otros deportes se ha documentado que predice mejora al año siguiente.
"""
import nflreadpy as nfl
import polars as pl
import numpy as np

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
def normalize_team(t):
    return TEAM_MAP.get(t, t)


def pythagorean_expectation(points_for, points_against, exponent=2.37):
    """
    Formula estandar de wins esperados segun diferencial de puntos (no record real).
    El exponente 2.37 es el valor comunmente citado para NFL (Football Outsiders/PFR),
    lo usamos como punto de partida, no como verdad asumida.
    """
    pf_exp = points_for ** exponent
    pa_exp = points_against ** exponent
    return pf_exp / (pf_exp + pa_exp)


def build_team_season_stats(seasons):
    schedules = nfl.load_schedules(seasons=seasons)
    schedules = schedules.filter(pl.col("home_score").is_not_null())
    schedules = schedules.filter(pl.col("game_type") == "REG")
    schedules = schedules.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])

    pbp = nfl.load_pbp(seasons=seasons)
    pbp = pbp.with_columns([
        pl.col("posteam").map_elements(normalize_team, return_dtype=pl.Utf8, skip_nulls=True).alias("posteam"),
    ])
    turnovers_given = pbp.filter(
        (pl.col("interception") == 1) | (pl.col("fumble_lost") == 1)
    ).group_by(["season", "posteam"]).agg(pl.len().alias("turnovers_given"))

    rows = []
    for season in seasons:
        season_games = schedules.filter(pl.col("season") == season)
        teams = sorted(set(season_games["home_team"].to_list()) | set(season_games["away_team"].to_list()))

        for team in teams:
            team_games = season_games.filter(
                (pl.col("home_team") == team) | (pl.col("away_team") == team)
            )

            wins, losses, ties = 0, 0, 0
            pf, pa = 0, 0
            one_score_w, one_score_l, one_score_total = 0, 0, 0

            for g in team_games.iter_rows(named=True):
                is_home = g["home_team"] == team
                team_score = g["home_score"] if is_home else g["away_score"]
                opp_score = g["away_score"] if is_home else g["home_score"]
                pf += team_score
                pa += opp_score
                margin = team_score - opp_score

                if margin > 0:
                    wins += 1
                elif margin < 0:
                    losses += 1
                else:
                    ties += 1

                if abs(margin) <= 8:
                    one_score_total += 1
                    if margin > 0:
                        one_score_w += 1
                    elif margin < 0:
                        one_score_l += 1

            total_games = wins + losses + ties
            actual_win_pct = (wins + 0.5 * ties) / total_games if total_games > 0 else 0
            pyth_win_pct = pythagorean_expectation(pf, pa)
            luck = actual_win_pct - pyth_win_pct  # positivo = record mejor de lo que "merece"

            to_given = turnovers_given.filter(
                (pl.col("season") == season) & (pl.col("posteam") == team)
            )["turnovers_given"]
            to_given_val = to_given[0] if len(to_given) > 0 else 0

            rows.append({
                "season": season, "team": team,
                "wins": wins, "losses": losses, "actual_win_pct": actual_win_pct,
                "point_diff": pf - pa, "pyth_win_pct": pyth_win_pct,
                "luck": luck,  # la variable clave: record real vs. lo que "merecia" por diferencial
                "one_score_games": one_score_total, "one_score_wins": one_score_w,
                "one_score_win_pct": one_score_w / one_score_total if one_score_total > 0 else None,
                "turnovers_given": to_given_val,
            })

    return pl.DataFrame(rows)


if __name__ == "__main__":
    stats = build_team_season_stats(range(2015, 2026))
    print(stats.sort(["season", "luck"], descending=[True, True]).head(10))
    print("\n--- Equipos con peor 'luck' en 2024 (record peor que sus metricas de fondo) ---")
    print(stats.filter(pl.col("season") == 2024).sort("luck").head(5).select(
        ["team", "wins", "losses", "point_diff", "pyth_win_pct", "actual_win_pct", "luck"]
    ))