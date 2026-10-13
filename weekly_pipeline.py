"""
Pipeline semanal: dado season/week, genera las predicciones de esa semana
usando resultados reales ya jugados (Elo actualizado) + blend con Poisson
(peso 0.25, validado con walk-forward backtest).
Para partidos futuros, el QB titular sale de starters_2026.py (proyectado),
no del calendario (que solo se llena post-partido).
"""
import numpy as np
import polars as pl
import nflreadpy as nfl
from scipy.stats import norm
from motor_elo_v1 import run_elo_over_seasons, normalize_team
from poisson_model import train_poisson_model, predict_lambda, game_win_probs, most_likely_scores
from team_epa import load_offensive_plays, season_final_epa, weekly_epa_with_blend
from poisson_features import (weekly_rush_epa_with_blend, weekly_pass_epa_with_blend,
                              weekly_success_rate_with_blend, qb_value_live, backup_qb_flags)
from qb_adjustment import load_pbp_clean, qb_career_value, build_rookie_prior_baseline
from market_lines_2026 import MARKET_SPREADS

POISSON_BLEND_WEIGHT = 0.45  # re-optimizado con el Poisson nuevo: walk-forward 2010-2022 (experimentos/REPORTE.md)
NEUTRAL_SITE_GAMES = {("PHI", "JAX")}  # (visita, local) de los partidos en campo neutral esta semana
HISTORY_START = 2015
# Mezcla con el spread de Vegas (experimentos/REPORTE_MERCADO.md). Con lineas de cierre el
# optimo es ~0.85 (curva monotona, 0.6 = borde de la grilla); 0.40 es conservador porque
# el usuario pega lineas del martes, menos informadas que las de cierre.
MARKET_BLEND_WEIGHT = 0.0
MARKET_SPREAD_SIGMA = 11.13  # normal calibrada 2010-2025 (LOSO); mejor que la clasica 13.5


def spread_to_prob(home_margin):
    """P(local gana) dado el margen esperado del local segun Vegas (positivo = local favorito)."""
    return float(norm.cdf(home_margin / MARKET_SPREAD_SIGMA))
POISSON_TRAIN_START = 2002   # ventana de entrenamiento evaluada en experimentos/


def get_upcoming_week(season):
    """Primera semana de la temporada sin resultado cargado todavia."""
    df = nfl.load_schedules(seasons=[season])
    df = df.filter(pl.col("game_type") == "REG")
    unplayed = df.filter(pl.col("home_score").is_null())
    if unplayed.height == 0:
        return None  # temporada regular ya termino
    return int(unplayed["week"].min())


def get_current_elo_ratings(season):
    """Elo actualizado con todos los resultados reales ya jugados de la temporada."""
    _, engine = run_elo_over_seasons(range(HISTORY_START, season + 1), k=20, home_advantage=35,
                                     regression_to_mean=0.40)
    return dict(engine.ratings)


def get_week_games(season, week):
    df = nfl.load_schedules(seasons=[season])
    df = df.filter((pl.col("game_type") == "REG") & (pl.col("week") == week))
    df = df.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])
    return df.select(["home_team", "away_team", "div_game"]).to_dicts()


def build_week_features(season, week):
    """
    Features de EPA/QB para la semana a predecir, usando solo datos de
    semanas ANTERIORES (mismo criterio sin-leakage de siempre). El QB
    titular sale de starters_2026.py (proyectado), no del calendario.
    """
    from starters_2026 import PROJECTED_STARTERS  # nombres, no gsis_id -- se resuelve abajo

    def load_available_seasons(loader_fn, history_start, target_season):
        """Intenta incluir la temporada actual; si nflreadpy aun no tiene esos
        datos (pretemporada, cero partidos jugados), cae de vuelta al historico."""
        try:
            return loader_fn(range(history_start, target_season + 1))
        except ValueError:
            print(f"(aviso: play-by-play de {target_season} no disponible todavia -- "
                  f"usando solo historico hasta {target_season - 1})")
            return loader_fn(range(history_start, target_season))

    pbp = load_available_seasons(load_pbp_clean, HISTORY_START, season)
    pbp_off = load_available_seasons(load_offensive_plays, HISTORY_START, season)

    prior_off, prior_def = season_final_epa(pbp_off.filter(pl.col("season") == season - 1))
    season_has_pbp = pbp_off.filter(pl.col("season") == season).height > 0

    if season_has_pbp:
        # extra_weeks=[week]: la semana a predecir todavia no tiene jugadas; sin esto
        # no habia fila para ella y TODAS las features de EPA caian al default 0.0.
        off_def = weekly_epa_with_blend(pbp_off, season, prior_off, prior_def, extra_weeks=[week])
        rush = weekly_rush_epa_with_blend(pbp_off, season, extra_weeks=[week])
        pass_epa = weekly_pass_epa_with_blend(pbp_off, season, extra_weeks=[week])
        success = weekly_success_rate_with_blend(pbp_off, season, extra_weeks=[week])
    else:
        # Pretemporada: cero snaps jugados de 2026 -> el EPA de cada equipo
        # es 100% el prior de temporada pasada (equivalente matematico a
        # blend_w=0 en weekly_epa_with_blend, sin depender de su maquinaria semanal).
        print("(aviso: cero partidos jugados de 2026 -- EPA usa 100% el prior de 2025)")
        teams = sorted(set(pbp_off.filter(pl.col("season") == season - 1)["posteam"].to_list()))
        off_def = pl.DataFrame([
            {"season": season, "week": week, "team": t,
             "off_epa": prior_off.filter(pl.col("team") == t)["off_epa_prior"][0]
                        if prior_off.filter(pl.col("team") == t).height > 0 else 0.0,
             "def_epa": prior_def.filter(pl.col("team") == t)["def_epa_prior"][0]
                        if prior_def.filter(pl.col("team") == t).height > 0 else 0.0}
            for t in teams
        ])
        prior_rush = pbp_off.filter(
            (pl.col("season") == season - 1) & (pl.col("play_type") == "run") & pl.col("epa").is_not_null()
        ).group_by("posteam").agg(pl.col("epa").mean().alias("rush_epa_prior"))
        rush = pl.DataFrame([
            {"season": season, "week": week, "team": t,
             "rush_epa": prior_rush.filter(pl.col("posteam") == t)["rush_epa_prior"][0]
                         if prior_rush.filter(pl.col("posteam") == t).height > 0 else 0.0}
            for t in teams
        ])
        prev = pbp_off.filter(pl.col("season") == season - 1)
        prior_pass = prev.filter(pl.col("qb_dropback") == 1).group_by("posteam").agg(pl.col("epa").mean().alias("v"))
        prior_succ = prev.group_by("posteam").agg(pl.col("success").cast(pl.Float64).mean().alias("v"))
        def prior_of(tbl, t):
            r = tbl.filter(pl.col("posteam") == t)["v"]
            return r[0] if len(r) > 0 else 0.0
        pass_epa = pl.DataFrame([{"season": season, "week": week, "team": t, "pass_epa": prior_of(prior_pass, t)}
                                 for t in teams])
        success = pl.DataFrame([{"season": season, "week": week, "team": t, "success_rate": prior_of(prior_succ, t)}
                                for t in teams])

    qb_values = qb_career_value(pbp)
    picks = nfl.load_draft_picks()
    rookie_baseline = build_rookie_prior_baseline(qb_values, picks)

    # EPA del QB en la temporada en curso (semanas < week) y titularidades previas
    season_db = pbp.filter((pl.col("season") == season) & (pl.col("qb_dropback") == 1) &
                           pl.col("qb_epa").is_not_null() & pl.col("passer_player_id").is_not_null())
    played = nfl.load_schedules(seasons=[season]).filter(
        (pl.col("game_type") == "REG") & (pl.col("week") < week) & pl.col("home_score").is_not_null()
    ).with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])

    rosters = nfl.load_rosters(seasons=[season])
    qbs = rosters.filter(pl.col("position") == "QB").select(["team", "full_name", "gsis_id"])

    def resolve_qb_id(team):
        name = PROJECTED_STARTERS.get(team)
        match = qbs.filter((pl.col("team") == team) & (pl.col("full_name") == name))
        if match.height > 0:
            return match["gsis_id"][0]
        # Fallback: titular del ultimo partido jugado esta temporada (antes: None -> valor 0.0)
        last = [(r["week"], r["home_qb_id"]) for r in played.iter_rows(named=True) if r["home_team"] == team] + \
               [(r["week"], r["away_qb_id"]) for r in played.iter_rows(named=True) if r["away_team"] == team]
        if last:
            print(f"(aviso: '{name}' no esta en el roster de {team}; uso el ultimo titular real)")
            return max(last)[1]
        print(f"(aviso: sin QB resuelto para {team}; valor de QB = 0.0)")
        return None

    def get_qb_value(qb_id):
        if qb_id is None:
            return 0.0
        hist = qb_values.filter((pl.col("qb_id") == qb_id) & (pl.col("season") < season))
        if hist.height > 0:
            return hist.sort("season", descending=True)["qb_value_epa_dropback"][0]
        starter_pick = picks.filter((pl.col("gsis_id") == qb_id) & (pl.col("position") == "QB"))
        if starter_pick.height > 0:
            rnd = starter_pick["round"][0]
            bucket = "R1" if rnd == 1 else ("R2_R3" if rnd in (2, 3) else "R4_PLUS")
        else:
            bucket = "R4_PLUS"
        row = rookie_baseline.filter(pl.col("bucket") == bucket)
        return row["baseline_value"][0] if row.height > 0 else 0.0

    def is_backup(team, qb_id):
        """1 si el titular proyectado no es el lider de titularidades del equipo en la temporada."""
        rows_t = [(r["week"], r["home_qb_id"]) for r in played.iter_rows(named=True) if r["home_team"] == team] + \
                 [(r["week"], r["away_qb_id"]) for r in played.iter_rows(named=True) if r["away_team"] == team]
        if not rows_t or qb_id is None:
            return 0
        fake = pl.DataFrame({"week": [w for w, _ in rows_t] + [week], "home_team": [team] * (len(rows_t) + 1),
                             "home_qb_id": [q for _, q in rows_t] + [qb_id]})
        return backup_qb_flags(fake, qb_cols=(("home_team", "home_qb_id"),)).get((team, week), 0)

    games = get_week_games(season, week)
    rows = []
    for g in games:
        for side, team, opp in [("home", g["home_team"], g["away_team"]),
                                 ("away", g["away_team"], g["home_team"])]:
            def_row = off_def.filter((pl.col("team") == opp) & (pl.col("week") == week))
            opp_def_epa = def_row["def_epa"][0] if def_row.height > 0 else 0.0

            rush_row = rush.filter((pl.col("team") == team) & (pl.col("week") == week))
            team_rush_epa = rush_row["rush_epa"][0] if rush_row.height > 0 else 0.0

            pass_row = pass_epa.filter((pl.col("team") == team) & (pl.col("week") == week))
            team_pass_epa = pass_row["pass_epa"][0] if pass_row.height > 0 else 0.0
            succ_row = success.filter((pl.col("team") == team) & (pl.col("week") == week))
            team_success = succ_row["success_rate"][0] if succ_row.height > 0 else 0.0

            qb_id = resolve_qb_id(team)
            qb_val = get_qb_value(qb_id)

            rows.append({
                "home_team": g["home_team"], "away_team": g["away_team"], "div_game": g["div_game"],
                "side": side, "team": team,
                "team_rush_epa": team_rush_epa, "team_qb_value": qb_val,
                "team_pass_epa": team_pass_epa, "team_success": team_success,
                "qb_value_live": qb_value_live(qb_val, qb_id, season_db, week),
                "backup_qb": is_backup(team, qb_id),
                "opp_def_epa_allowed": opp_def_epa,
                "is_home": 1 if side == "home" else 0,
            })

    return pl.DataFrame(rows)


def predict_week(season, week=None):
    if week is None:
        week = get_upcoming_week(season)
        if week is None:
            print("Temporada regular ya termino.")
            return None

    print(f"Prediciendo temporada {season}, semana {week}...\n")

    elo_ratings = get_current_elo_ratings(season)
    poisson_model, _ = train_poisson_model(train_seasons=list(range(POISSON_TRAIN_START, season)),
                                           pbp_history_start=POISSON_TRAIN_START - 1)
    features = build_week_features(season, week)

    results = []
    for game in features.select(["home_team", "away_team", "div_game"]).unique().to_dicts():
        home, away, div = game["home_team"], game["away_team"], game["div_game"]

        home_f = features.filter((pl.col("home_team") == home) & (pl.col("side") == "home"))
        away_f = features.filter((pl.col("away_team") == away) & (pl.col("side") == "away"))

        neutral = (away, home) in NEUTRAL_SITE_GAMES
        elo_home = elo_ratings.get(home, 1500) + (0 if neutral else 35)
        elo_away = elo_ratings.get(away, 1500)
        elo_prob_home = 1.0 / (1.0 + 10 ** ((elo_away - elo_home) / 400.0))

        home_row = home_f.row(0, named=True)
        away_row = away_f.row(0, named=True)
        if neutral:
            home_row["is_home"] = 0.5
            away_row["is_home"] = 0.5
        lam_home = predict_lambda(poisson_model, home_row)
        lam_away = predict_lambda(poisson_model, away_row)
        poisson_probs = game_win_probs(lam_home, lam_away)

        base_blend_prob_home = (POISSON_BLEND_WEIGHT * poisson_probs["home_win"] +
                                 (1 - POISSON_BLEND_WEIGHT) * elo_prob_home)
        market_spread = MARKET_SPREADS.get((away, home))  # margen esperado del local; None si no hay linea
        if market_spread is not None:
            prob_mercado_local = spread_to_prob(market_spread)
            blend_prob_home = ((1 - MARKET_BLEND_WEIGHT) * base_blend_prob_home +
                               MARKET_BLEND_WEIGHT * prob_mercado_local)
        else:
            prob_mercado_local = None
            blend_prob_home = base_blend_prob_home

        # Margen esperado: blend consistente con la probabilidad ya mostrada
        # (mismo peso 0.45/0.55 que el blend de probabilidad), en vez de usar
        # solo el margen implicito del Poisson (que generaba la inconsistencia).
        elo_margin = (elo_home - elo_away) / 25.0
        poisson_margin = lam_home - lam_away
        blend_margin = POISSON_BLEND_WEIGHT * poisson_margin + (1 - POISSON_BLEND_WEIGHT) * elo_margin
        if market_spread is not None:  # mismo peso de mercado que la probabilidad
            blend_margin = (1 - MARKET_BLEND_WEIGHT) * blend_margin + MARKET_BLEND_WEIGHT * market_spread

        top_scores = most_likely_scores(lam_home, lam_away, top_n=3)
        top_scores_str = " / ".join(f"{home} {h}-{a} {away}" for h, a, _ in top_scores)

        results.append({
            "matchup": f"{away} @ {home}",
            "prob_gana_local": round(blend_prob_home, 3),
            "prob_gana_visita": round(1 - blend_prob_home, 3),
            "margen_esperado": f"{home} +{blend_margin:.1f}" if blend_margin > 0 else f"{away} +{-blend_margin:.1f}",
            "marcadores_ilustrativos": top_scores_str,  # top-3, NO es "la" prediccion -- ver nota abajo
            "prob_mercado_local": round(prob_mercado_local, 3) if prob_mercado_local is not None else None,
        })

    return pl.DataFrame(results).with_columns(pl.col("prob_mercado_local").cast(pl.Float64)).sort("matchup")


if __name__ == "__main__":
    SEASON = 2026
    results = predict_week(SEASON)  # semana automatica = la proxima sin jugar
    if results is not None:
        with pl.Config(tbl_rows=20, fmt_str_lengths=50):
            print(results)