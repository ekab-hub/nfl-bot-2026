"""
Nucleo compartido del simulador de temporada: preseason ratings (con ajuste
de luck), calendario, simulacion de temporada regular, tiebreakers, seeding
de playoffs y bracket completo. Usado por division_champions.py y
superbowl_champions.py -- no se corre directamente.
"""
import numpy as np
import polars as pl
import nflreadpy as nfl
from motor_elo_v1 import EloEngine, run_elo_over_seasons, normalize_team
from underlying_stats import build_team_season_stats

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SF", "SEA"],
}
TEAM_TO_DIVISION = {team: div for div, teams in DIVISIONS.items() for team in teams}
CONFERENCES = {
    "AFC": [d for d in DIVISIONS if d.startswith("AFC")],
    "NFC": [d for d in DIVISIONS if d.startswith("NFC")],
}

LUCK_COEF = -8.48
WINS_TO_ELO_SCALE = 0
PRESEASON_K = 20
IN_SEASON_K = 35


def get_preseason_ratings(target_season, history_start=2015, k=20, home_advantage=35):
    """
    Elo con regresion a la media entre temporadas + ajuste por 'luck' de la
    temporada anterior (unico ajuste con evidencia estadisticamente
    significativa, p<0.0001, n=320 -- ver qb_delta_significance.py e
    injury_significance_test.py para las variables probadas y descartadas).
    """
    seasons = list(range(history_start, target_season))
    _, engine = run_elo_over_seasons(seasons, k=k, home_advantage=home_advantage)
    engine.new_season_regression()
    adjusted_ratings = dict(engine.ratings)

    underlying = build_team_season_stats([target_season - 1])
    for row in underlying.iter_rows(named=True):
        team = row["team"]
        shift = LUCK_COEF * row["luck"] * WINS_TO_ELO_SCALE
        adjusted_ratings[team] = adjusted_ratings.get(team, 1500) + shift

    return adjusted_ratings


def get_season_schedule(season, include_played_results=False):
    df = nfl.load_schedules(seasons=[season])
    df = df.with_columns([
        pl.col("home_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("home_team"),
        pl.col("away_team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("away_team"),
    ])
    df = df.filter(pl.col("game_type") == "REG")
    cols = ["week", "home_team", "away_team"]
    if include_played_results:
        cols += ["home_score", "away_score"]
    return df.sort(["week"]).select(cols).to_dicts()


def apply_elo_adjustments(ratings, elo_adjustments=None):
    """Elo + desplazamiento por equipo (para playoffs). Sin ajustes devuelve el mismo dict."""
    if not elo_adjustments:
        return ratings
    return {team: r + elo_adjustments.get(team, 0.0) for team, r in ratings.items()}


def simulate_one_season(preseason_ratings, schedule, k=20, home_advantage=35, margin_std=13.5,
                        elo_adjustments=None):
    """
    elo_adjustments: dict opcional equipo -> puntos de Elo (negativo = penalizacion, p.ej.
    QB fuera por la temporada; ver qb_out_2026.py). Solo afecta partidos SIN resultado real:
    se suma al calcular el margen esperado (y la expectativa de la actualizacion, para que
    el Elo simulado no absorba la penalizacion y la cuente dos veces). El rating guardado y
    devuelto nunca incluye el desplazamiento.
    """
    engine = EloEngine(k=k, home_advantage=home_advantage)
    engine.ratings = dict(preseason_ratings)

    records = {team: {"w": 0, "l": 0, "t": 0, "pf": 0, "pa": 0, "div_record": {}, "h2h": {}}
               for team in TEAM_TO_DIVISION}

    for game in schedule:
        home, away = game["home_team"], game["away_team"]
        played = game.get("home_score") is not None and game.get("away_score") is not None

        adj_home = adj_away = 0.0
        if played:
            home_score, away_score = int(game["home_score"]), int(game["away_score"])
        else:
            if elo_adjustments:
                adj_home = elo_adjustments.get(home, 0.0)
                adj_away = elo_adjustments.get(away, 0.0)
            r_home = engine.get_rating(home) + engine.home_advantage + adj_home
            r_away = engine.get_rating(away) + adj_away
            expected_margin = (r_home - r_away) / 25.0
            actual_margin = np.random.normal(expected_margin, margin_std)
            home_score = max(0, round(20 + actual_margin / 2))
            away_score = max(0, round(20 - actual_margin / 2))
            if home_score == away_score:
                home_score += 1

        if adj_home or adj_away:
            engine.ratings[home] = engine.get_rating(home) + adj_home
            engine.ratings[away] = engine.get_rating(away) + adj_away
            engine.update_game(home, away, home_score, away_score)
            engine.ratings[home] -= adj_home
            engine.ratings[away] -= adj_away
        else:
            engine.update_game(home, away, home_score, away_score)

        for team, pf, pa in [(home, home_score, away_score), (away, away_score, home_score)]:
            records[team]["pf"] += pf
            records[team]["pa"] += pa

        if home_score == away_score:
            records[home]["t"] += 1
            records[away]["t"] += 1
            continue

        winner = home if home_score > away_score else away
        loser = away if winner == home else home
        records[winner]["w"] += 1
        records[loser]["l"] += 1

        if TEAM_TO_DIVISION[home] == TEAM_TO_DIVISION[away]:
            records[winner]["div_record"][winner] = records[winner]["div_record"].get(winner, 0) + 1
        records[winner]["h2h"][loser] = records[winner]["h2h"].get(loser, 0) + 1

    return records, dict(engine.ratings)


def resolve_division_winner(records, division_teams):
    def win_pct(team):
        r = records[team]
        total = r["w"] + r["l"] + r["t"]
        return (r["w"] + 0.5 * r["t"]) / total if total > 0 else 0

    best_pct = max(win_pct(t) for t in division_teams)
    tied = [t for t in division_teams if abs(win_pct(t) - best_pct) < 1e-9]

    if len(tied) == 1:
        return tied[0]

    for team in tied:
        beat_all = all(records[team]["h2h"].get(other, 0) > 0 for other in tied if other != team)
        if beat_all:
            return team

    div_wins = {t: sum(records[t]["div_record"].values()) for t in tied}
    best_div = max(div_wins.values())
    tied = [t for t in tied if div_wins[t] == best_div]
    if len(tied) == 1:
        return tied[0]

    return max(tied, key=lambda t: records[t]["pf"] - records[t]["pa"])


def get_all_division_winners(records):
    return {div: resolve_division_winner(records, teams) for div, teams in DIVISIONS.items()}


def seed_conference(records, conference, division_winners):
    conf_divisions = CONFERENCES[conference]
    conf_teams = [t for d in conf_divisions for t in DIVISIONS[d]]
    winners = [division_winners[d] for d in conf_divisions]

    def win_pct(team):
        r = records[team]
        total = r["w"] + r["l"] + r["t"]
        return (r["w"] + 0.5 * r["t"]) / total if total > 0 else 0

    def point_diff(team):
        r = records[team]
        return r["pf"] - r["pa"]

    winners_sorted = sorted(winners, key=lambda t: (win_pct(t), point_diff(t)), reverse=True)
    non_winners = [t for t in conf_teams if t not in winners]
    wildcards_sorted = sorted(non_winners, key=lambda t: (win_pct(t), point_diff(t)), reverse=True)[:3]

    return winners_sorted + wildcards_sorted


def simulate_playoff_game(higher_seed_team, lower_seed_team, elo_ratings,
                           home_advantage=35, margin_std=13.5, neutral=False):
    r_home = elo_ratings.get(higher_seed_team, 1500) + (0 if neutral else home_advantage)
    r_away = elo_ratings.get(lower_seed_team, 1500)
    expected_margin = (r_home - r_away) / 25.0
    actual_margin = np.random.normal(expected_margin, margin_std)
    if actual_margin == 0:
        actual_margin = np.random.choice([-0.5, 0.5])
    return higher_seed_team if actual_margin > 0 else lower_seed_team


def simulate_playoffs(records, division_winners, final_elo, home_advantage=35, margin_std=13.5):
    conf_champs = {}

    for conf in ["AFC", "NFC"]:
        seeds = seed_conference(records, conf, division_winners)
        rank = {team: i + 1 for i, team in enumerate(seeds)}
        s1, s2, s3, s4, s5, s6, s7 = seeds

        w_27 = simulate_playoff_game(s2, s7, final_elo, home_advantage, margin_std)
        w_36 = simulate_playoff_game(s3, s6, final_elo, home_advantage, margin_std)
        w_45 = simulate_playoff_game(s4, s5, final_elo, home_advantage, margin_std)

        wc_winners_sorted = sorted([w_27, w_36, w_45], key=lambda t: rank[t])
        best_wc, mid_wc, worst_wc = wc_winners_sorted

        div1_winner = simulate_playoff_game(s1, worst_wc, final_elo, home_advantage, margin_std)
        div2_home, div2_away = (best_wc, mid_wc) if rank[best_wc] < rank[mid_wc] else (mid_wc, best_wc)
        div2_winner = simulate_playoff_game(div2_home, div2_away, final_elo, home_advantage, margin_std)

        cc_home, cc_away = (div1_winner, div2_winner) if rank[div1_winner] < rank[div2_winner] else (div2_winner, div1_winner)
        conf_champs[conf] = simulate_playoff_game(cc_home, cc_away, final_elo, home_advantage, margin_std)

    sb_champion = simulate_playoff_game(
        conf_champs["AFC"], conf_champs["NFC"], final_elo, home_advantage=0, margin_std=margin_std, neutral=True
    )
    return sb_champion