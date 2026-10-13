"""
Construye la tabla de features por partido (formato largo: una fila por equipo-partido)
para alimentar el modelo Poisson de anotacion.
Combina: EPA de carrera, EPA por dropback y success rate (equipo), valor de QB
(jugador, con shrinkage para novatos, actualizado con su EPA de la temporada), marca de
QB suplente, EPA defensivo del rival, y home field.
Factores validados en experimentos/REPORTE.md.
"""
import nflreadpy as nfl
import polars as pl
import numpy as np
from qb_adjustment import (load_pbp_clean, qb_career_value, primary_starter_per_team_season,
                            build_rookie_prior_baseline, rookie_qb_value_with_shrinkage)
from team_epa import load_offensive_plays, season_final_epa, weekly_epa_with_blend

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
def normalize_team(t):
    return TEAM_MAP.get(t, t)

def weekly_offense_stat_with_blend(pbp: pl.DataFrame, season, play_filter, value_col, out_col,
                                   decay=0.85, blend_games=8, extra_weeks=()):
    """Misma logica que team_epa.weekly_epa_with_blend, para cualquier stat ofensiva por jugada
    (EPA de carrera, EPA por dropback, success rate) sobre el subconjunto `play_filter`."""
    plays = pbp.filter(play_filter & pl.col(value_col).is_not_null() & pl.col("posteam").is_not_null())
    season_plays = plays.filter(pl.col("season") == season)
    prior_tbl = plays.filter(pl.col("season") == season - 1).group_by("posteam").agg(
        pl.col(value_col).cast(pl.Float64).mean().alias("prior")
    )

    weeks = sorted(set(season_plays["week"].unique().to_list()) | set(extra_weeks))
    results = []
    for wk in weeks:
        past = season_plays.filter(pl.col("week") < wk)
        teams = sorted(set(season_plays["posteam"].to_list()))
        for team in teams:
            tp = past.filter(pl.col("posteam") == team)
            games_played = tp["week"].n_unique() if tp.height > 0 else 0
            if tp.height > 0:
                w = decay ** (wk - tp["week"].to_numpy())
                current = np.average(tp[value_col].cast(pl.Float64).to_numpy(), weights=w)
            else:
                current = 0.0
            prior = prior_tbl.filter(pl.col("posteam") == team)["prior"]
            prior = prior[0] if len(prior) > 0 else 0.0
            blend_w = min(games_played / blend_games, 1.0)
            results.append({"season": season, "week": wk, "team": team,
                            out_col: blend_w * current + (1 - blend_w) * prior})
    return pl.DataFrame(results)


def weekly_rush_epa_with_blend(pbp: pl.DataFrame, season, decay=0.85, blend_games=8, extra_weeks=()):
    """Igual logica que team_epa.weekly_epa_with_blend pero SOLO jugadas de carrera."""
    return weekly_offense_stat_with_blend(pbp, season, pl.col("play_type") == "run", "epa", "rush_epa",
                                          decay, blend_games, extra_weeks)


def weekly_pass_epa_with_blend(pbp: pl.DataFrame, season, decay=0.85, blend_games=8, extra_weeks=()):
    """EPA por dropback del ATAQUE (equipo, no QB): linea, receptores, esquema."""
    return weekly_offense_stat_with_blend(pbp, season, pl.col("qb_dropback") == 1, "epa", "pass_epa",
                                          decay, blend_games, extra_weeks)


def weekly_success_rate_with_blend(pbp: pl.DataFrame, season, decay=0.85, blend_games=8, extra_weeks=()):
    """Success rate (jugadas con EPA > 0) del ataque, carrera + pase."""
    return weekly_offense_stat_with_blend(pbp, season, pl.lit(True), "success", "success_rate",
                                          decay, blend_games, extra_weeks)


QB_LIVE_PRIOR_DROPBACKS = 150  # peso del valor historico, en dropbacks equivalentes


def qb_value_live(prior_value, qb_id, season_dropbacks: pl.DataFrame, week, decay=0.85):
    """
    Valor del QB actualizado con su EPA de la temporada en curso (semanas < week),
    con decaimiento semanal: (150*prior + S) / (150 + N).
    season_dropbacks: jugadas qb_dropback de la temporada (passer_player_id, week, qb_epa).
    """
    plays = season_dropbacks.filter((pl.col("passer_player_id") == qb_id) & (pl.col("week") < week))
    if plays.height == 0:
        return prior_value
    w = decay ** (week - plays["week"].to_numpy())
    s = float(np.sum(w * plays["qb_epa"].to_numpy()))
    n = float(np.sum(w))
    return (QB_LIVE_PRIOR_DROPBACKS * prior_value + s) / (QB_LIVE_PRIOR_DROPBACKS + n)


def backup_qb_flags(games: pl.DataFrame, qb_cols=(("home_team", "home_qb_id"), ("away_team", "away_qb_id"))):
    """
    {(team, week): 1/0} -- 1 si el QB titular de ese partido NO es el lider de
    titularidades del equipo en semanas anteriores de la temporada (empate -> el mas
    reciente). Sin partidos previos -> 0.
    """
    by_team = {}
    for row in games.sort("week").iter_rows(named=True):
        for team_col, qb_col in qb_cols:
            by_team.setdefault(row[team_col], []).append((row["week"], row[qb_col]))
    flags = {}
    for team, seq in by_team.items():
        starts, last = {}, {}
        for wk, qb in seq:
            if starts:
                lead = max(starts, key=lambda q: (starts[q], last[q]))
                flags[(team, wk)] = int(qb is not None and qb != lead)
            else:
                flags[(team, wk)] = 0
            if qb is not None:
                starts[qb] = starts.get(qb, 0) + 1
                last[qb] = wk
    return flags


def build_game_features(seasons_to_build, pbp_history_start=2018):
    """
    Para cada temporada en seasons_to_build, arma la tabla de features por equipo-semana
    y la cruza con el calendario real para dejar una fila por equipo-partido, lista
    para el Poisson.
    """
    all_seasons = list(range(pbp_history_start, max(seasons_to_build) + 1))
    pbp = load_pbp_clean(all_seasons)                 # crudo -> solo para valores de QB
    pbp_off = load_offensive_plays(all_seasons)        # filtrado (run/pass, sin nulls) -> EPA off/def y rush

    qb_values = qb_career_value(pbp)
    starters = primary_starter_per_team_season(pbp)
    picks = nfl.load_draft_picks()
    rookie_baseline = build_rookie_prior_baseline(qb_values, picks)

    schedules = nfl.load_schedules(seasons=seasons_to_build)
    schedules = schedules.filter(pl.col("home_score").is_not_null())
    schedules = schedules.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])

    rows = []
    for season in seasons_to_build:
        off_def = weekly_epa_with_blend(
            pbp_off, season,
            *season_final_epa(pbp_off.filter(pl.col("season") == season - 1))
        )
        rush = weekly_rush_epa_with_blend(pbp_off, season)
        pass_epa = weekly_pass_epa_with_blend(pbp_off, season)
        success = weekly_success_rate_with_blend(pbp_off, season)
        season_db = pbp.filter((pl.col("season") == season) & (pl.col("qb_dropback") == 1) &
                               pl.col("qb_epa").is_not_null() & pl.col("passer_player_id").is_not_null())

        season_games = schedules.filter(pl.col("season") == season)
        backup_flags = backup_qb_flags(season_games)

        for row in season_games.iter_rows(named=True):
            for side, team_col, qb_col, opp_col in [
                ("home", "home_team", "home_qb_id", "away_team"),
                ("away", "away_team", "away_qb_id", "home_team"),
            ]:
                team = row[team_col]
                opp = row[opp_col]
                wk = row["week"]
                is_home = 1 if side == "home" else 0

                def_row = off_def.filter(
                    (pl.col("team") == opp) & (pl.col("week") == wk) & (pl.col("season") == season)
                )
                opp_def_epa = def_row["def_epa"][0] if def_row.height > 0 else 0.0

                rush_row = rush.filter((pl.col("team") == team) & (pl.col("week") == wk))
                team_rush_epa = rush_row["rush_epa"][0] if rush_row.height > 0 else 0.0
                pass_row = pass_epa.filter((pl.col("team") == team) & (pl.col("week") == wk))
                team_pass_epa = pass_row["pass_epa"][0] if pass_row.height > 0 else 0.0
                succ_row = success.filter((pl.col("team") == team) & (pl.col("week") == wk))
                team_success = succ_row["success_rate"][0] if succ_row.height > 0 else 0.0

                qb_id = row[qb_col]
                qb_hist = qb_values.filter((pl.col("qb_id") == qb_id) & (pl.col("season") < season))
                # valor mas reciente conocido de ese QB antes de esta temporada; si no existe, es rookie puro
                if qb_hist.height > 0:
                    qb_val = qb_hist.sort("season", descending=True)["qb_value_epa_dropback"][0]
                else:
                    starter_pick = picks.filter((pl.col("gsis_id") == qb_id) & (pl.col("position") == "QB"))
                    if starter_pick.height > 0:
                        rnd = starter_pick["round"][0]
                        bucket = "R1" if rnd == 1 else ("R2_R3" if rnd in (2, 3) else "R4_PLUS")
                    else:
                        bucket = "R4_PLUS"  # UDFA -> cubeta mas baja como default
                    qb_val = rookie_baseline.filter(pl.col("bucket") == bucket)["baseline_value"][0]
                qb_live = qb_value_live(qb_val, qb_id, season_db, wk)

                rows.append({
                    "season": season, "week": wk, "game_id": row["game_id"],
                    "team": team, "opponent": opp, "is_home": is_home,
                    "team_rush_epa": team_rush_epa, "team_qb_value": qb_val,
                    "team_pass_epa": team_pass_epa, "team_success": team_success,
                    "qb_value_live": qb_live, "backup_qb": backup_flags.get((team, wk), 0),
                    "opp_def_epa_allowed": opp_def_epa,
                    "points_scored": row["home_score"] if side == "home" else row["away_score"],
                    "div_game": row["div_game"],
                })

    return pl.DataFrame(rows)


if __name__ == "__main__":
    features = build_game_features(seasons_to_build=[2023, 2024])
    print(features.head(10))
    print(f"\nTotal filas: {features.height}")
    print(features.select(["team_rush_epa", "team_qb_value", "opp_def_epa_allowed", "points_scored"]).describe())