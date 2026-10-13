"""
EPA ofensivo/defensivo por equipo, por semana, con:
- Filtro a jugadas de scrimmage reales (run/pass, sin kickoffs/PATs/no-plays)
- Ventana movil dentro de temporada con decaimiento por antiguedad (mas peso a lo reciente)
- Blend con temporada anterior en semanas tempranas (para no arrancar en cero)
"""
import nflreadpy as nfl
import polars as pl
import numpy as np

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

def normalize_team(t):
    return TEAM_MAP.get(t, t)

def load_offensive_plays(seasons):
    pbp = nfl.load_pbp(seasons=seasons)
    pbp = pbp.filter(
        pl.col("play_type").is_in(["run", "pass"]) &
        pl.col("epa").is_not_null() &
        pl.col("posteam").is_not_null()
    )
    pbp = pbp.with_columns([
        pl.col("posteam").map_elements(normalize_team, return_dtype=pl.Utf8).alias("posteam"),
        pl.col("defteam").map_elements(normalize_team, return_dtype=pl.Utf8).alias("defteam"),
    ])
    return pbp

def season_final_epa(pbp_season: pl.DataFrame):
    """EPA ofensivo/defensivo promedio de la temporada completa (para blend con año siguiente)."""
    off = pbp_season.group_by("posteam").agg(pl.col("epa").mean().alias("off_epa_prior"))
    deff = pbp_season.group_by("defteam").agg(pl.col("epa").mean().alias("def_epa_prior"))
    return off.rename({"posteam": "team"}), deff.rename({"defteam": "team"})

def weekly_epa_with_blend(pbp: pl.DataFrame, season, prior_off, prior_def, decay=0.85, blend_games=8,
                          extra_weeks=()):
    """
    Para cada semana de la temporada actual, calcula el EPA ofensivo/defensivo
    de cada equipo usando SOLO semanas anteriores (sin leakage), con peso
    exponencial por antiguedad, blendeado con el EPA final de la temporada previa.
    extra_weeks: semanas SIN jugadas todavia para las que tambien se quiere el valor
    (la semana a predecir en el pipeline semanal; sin esto no hay fila y cae a 0.0).
    """
    season_pbp = pbp.filter(pl.col("season") == season)
    weeks = sorted(set(season_pbp["week"].unique().to_list()) | set(extra_weeks))
    results = []

    for wk in weeks:
        past = season_pbp.filter(pl.col("week") < wk)
        teams = sorted(set(season_pbp["posteam"].to_list()))

        for team in teams:
            off_plays = past.filter(pl.col("posteam") == team)
            def_plays = past.filter(pl.col("defteam") == team)

            games_played = off_plays["week"].n_unique() if off_plays.height > 0 else 0

            # EPA ponderado por antiguedad (semanas mas recientes pesan mas)
            if off_plays.height > 0:
                w = decay ** (wk - off_plays["week"].to_numpy())
                off_current = np.average(off_plays["epa"].to_numpy(), weights=w)
            else:
                off_current = None

            if def_plays.height > 0:
                w = decay ** (wk - def_plays["week"].to_numpy())
                def_current = np.average(def_plays["epa"].to_numpy(), weights=w)
            else:
                def_current = None

            prior_o = prior_off.filter(pl.col("team") == team)["off_epa_prior"]
            prior_o = prior_o[0] if len(prior_o) > 0 else 0.0
            prior_d = prior_def.filter(pl.col("team") == team)["def_epa_prior"]
            prior_d = prior_d[0] if len(prior_d) > 0 else 0.0

            # Peso del blend: entre mas partidos jugados esta temporada, menos pesa el prior
            blend_w = min(games_played / blend_games, 1.0)

            off_final = (blend_w * off_current if off_current is not None else 0) + (1 - blend_w) * prior_o
            def_final = (blend_w * def_current if def_current is not None else 0) + (1 - blend_w) * prior_d

            results.append({
                "season": season, "week": wk, "team": team,
                "off_epa": off_final, "def_epa": def_final,
                "games_played": games_played
            })

    return pl.DataFrame(results)


if __name__ == "__main__":
    # Prueba rapida: EPA de 2024 usando 2023 como prior
    pbp_all = load_offensive_plays([2023, 2024])
    prior_off, prior_def = season_final_epa(pbp_all.filter(pl.col("season") == 2023))
    weekly = weekly_epa_with_blend(pbp_all, 2024, prior_off, prior_def)

    print(weekly.filter(pl.col("week") == 1).sort("off_epa", descending=True).head(5))
    print(weekly.filter(pl.col("week") == 10).sort("off_epa", descending=True).head(5))